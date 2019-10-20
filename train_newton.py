# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import argparse
import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
from src.model import MetaLearner
from src.model import Net
from src.model import metrics
from src.data_loader import split_omniglot_characters
from src.data_loader import load_imagenet_images
from src.data_loader import OmniglotTask
from src.data_loader import ImageNetTask
from src.data_loader import fetch_dataloaders
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='data/Omniglot',
    help="Directory containing the dataset")
parser.add_argument(
    '--model_dir',
    default='experiments/base_model',
    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    default=None,
    help="Optional, name of the file in --model_dir containing weights to \
          reload before training")  # 'best' or 'train'

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size()).cuda()
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def train_single_task(model, task_lr, loss_fn, dataloaders, params):
    """
    Train the model on a single few-shot task.
    We train the model with single or multiple gradient update.
    
    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of 
                     support set and query set
        params: (Params) hyperparameters
    """
    # extract params
    num_train_updates = params.num_train_updates

    # set model to training mode
    model.train()

    # support set and query set for a single few-shot task
    dl_sup = dataloaders['train']
    X_sup, Y_sup = dl_sup.__iter__().next()
    X_sup2, Y_sup2 = dl_sup.__iter__().next()

    # move to GPU if available
    if params.cuda:
        X_sup, Y_sup = X_sup.cuda(async=True), Y_sup.cuda(async=True)

    # compute model output and loss
    Y_sup_hat = model(X_sup)
    loss = loss_fn(Y_sup_hat, Y_sup)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    # NOTE if we want approx-MAML, change create_graph=True to False
    # optimizer.zero_grad()
    # loss.backward(create_graph=True)
    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # performs updates using calculated gradients
    # we manually compute adpated parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()  # NOTE what about just dict
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        adapted_params[key] = val - task_lr * grad
        adapted_state_dict[key] = adapted_params[key]

    for _ in range(1, num_train_updates):
        Y_sup_hat = model(X_sup, adapted_state_dict)
        loss = loss_fn(Y_sup_hat, Y_sup)
        zero_grad(adapted_params.values())
        # optimizer.zero_grad()
        # loss.backward(create_graph=True)
        grads = torch.autograd.grad(
            loss, adapted_params.values(), create_graph=True)
        for (key, val), grad in zip(adapted_params.items(), grads):
            adapted_params[key] = val - task_lr * grad
            adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict

def train_single_task_newton(model, task_lr, loss_fn, dataloaders, params):
    """
    Train the model on a single few-shot task.
    We train the model with single newton_step update.

    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of
                     support set and query set
        params: (Params) hyperparameters
    """
    # extract params
    num_train_updates = params.num_train_updates

    # set model to training mode
    model.train()

    # support set and query set for a single few-shot task
    dl_sup = dataloaders['train']
    X_sup, Y_sup = dl_sup.__iter__().next()
    X_sup2, Y_sup2 = dl_sup.__iter__().next()

    # move to GPU if available
    if params.cuda:
        X_sup, Y_sup = X_sup.cuda(async=True), Y_sup.cuda(async=True)

    # compute model output and loss
    Y_sup_hat = model(X_sup)
    loss = loss_fn(Y_sup_hat, Y_sup)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    # NOTE if we want approx-MAML, change create_graph=True to False
    # optimizer.zero_grad()
    # loss.backward(create_graph=True)
    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads = utils.flatten(grads)
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Hvp(v, damping=1e-1):
        gv = (grads * Variable(v)).sum()
        Hv = torch.autograd.grad(gv, model.parameters(), retain_graph=True)
        flat_Hv = torch.cat([grad.contiguous().view(-1) for grad in Hv]).data
        return flat_Hv + v * damping

    newton_step = utils.unflatten(conjugate_gradients(Hvp, loss_grad, 10), model.parameters())

    # performs updates using calculated step
    # we manually compute adpated parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()  # NOTE what about just dict
    adapted_params = OrderedDict()
    for (key, val), dir in zip(model.named_parameters(), newton_step):
        adapted_params[key] = val - task_lr * dir
        adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict

def train_and_evaluate(model,
                       meta_train_classes,
                       meta_test_classes,
                       task_type,
                       meta_optimizer,
                       loss_fn,
                       metrics,
                       params,
                       model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every `save_summary_steps`.

    Args:
        model: (MetaLearner) a meta-learner for MAML algorithm
        meta_train_classes: (list) the classes for meta-training
        meta_train_classes: (list) the classes for meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        meta_optimizer: (torch.optim) an meta-optimizer for MetaLearner
        loss_fn: a loss function
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    TODO Validation classes
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir,
                                    args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, meta_optimizer)

    # params information
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_inner_tasks = params.num_inner_tasks
    task_lr = params.task_lr
    meta_lr = params.meta_lr

    # TODO validation accuracy
    best_test_acc = 0.0

    # For plotting to see summerized training procedure
    plot_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    with tqdm(total=params.num_episodes) as t:
        for episode in range(params.num_episodes):
            # Run one episode
            logging.info("Episode {}/{}".format(episode + 1,
                                                params.num_episodes))

            # Run inner loops to get adapted parameters (theta_t`)
            adapted_state_dicts = []
            dataloaders_list = []
            for n_task in range(num_inner_tasks):
                task = task_type(meta_train_classes, num_classes, num_samples,
                                 num_query)
                dataloaders = fetch_dataloaders(['train', 'test', 'meta'],
                                                task)
                # Perform a gradient descent to meta-learner on the task
                a_dict = train_single_task(model, task_lr, loss_fn,
                                           dataloaders, params)
                # Store adapted parameters
                # Store dataloaders for meta-update and evaluation
                adapted_state_dicts.append(a_dict)
                dataloaders_list.append(dataloaders)

            # Update the parameters of meta-learner
            # Compute losses with adapted parameters along with corresponding tasks
            # Updated the parameters of meta-learner using sum of the losses
            meta_loss = 0
            for n_task in range(num_inner_tasks):
                dataloaders = dataloaders_list[n_task]
                dl_meta = dataloaders['meta']
                X_meta, Y_meta = dl_meta.__iter__().next()
                if params.cuda:
                    X_meta, Y_meta = X_meta.cuda(async=True), Y_meta.cuda(
                        async=True)

                a_dict = adapted_state_dicts[n_task]
                Y_meta_hat = model(X_meta, a_dict)
                loss_t = loss_fn(Y_meta_hat, Y_meta)
                meta_loss += loss_t
            meta_loss /= float(num_inner_tasks)
            # print(meta_loss.item())

            # Meta-update using meta_optimizer
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

            # Evaluate model on new task
            # Evaluate on train and test dataset given a number of tasks (params.num_steps)
            if (episode + 1) % params.save_summary_steps == 0:
                train_metrics = evaluate(model, loss_fn, meta_train_classes,
                                         task_lr, task_type, metrics, params,
                                         'train')
                test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                        task_lr, task_type, metrics, params,
                                        'test')

                train_loss = train_metrics['loss']
                test_loss = test_metrics['loss']
                train_acc = train_metrics['accuracy']
                test_acc = test_metrics['accuracy']

                is_best = test_acc >= best_test_acc

                # Save weights
                utils.save_checkpoint({
                    'episode': episode + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': meta_optimizer.state_dict()
                },
                                      is_best=is_best,
                                      checkpoint=model_dir)

                # If best_test, best_save_path
                if is_best:
                    logging.info("- Found new best accuracy")
                    best_test_acc = test_acc

                    # Save best test metrics in a json file in the model directory
                    best_train_json_path = os.path.join(
                        model_dir, "metrics_train_best_weights.json")
                    utils.save_dict_to_json(train_metrics,
                                            best_train_json_path)
                    best_test_json_path = os.path.join(
                        model_dir, "metrics_test_best_weights.json")
                    utils.save_dict_to_json(test_metrics, best_test_json_path)

                # Save latest test metrics in a json file in the model directory
                last_train_json_path = os.path.join(
                    model_dir, "metrics_train_last_weights.json")
                utils.save_dict_to_json(train_metrics, last_train_json_path)
                last_test_json_path = os.path.join(
                    model_dir, "metrics_test_last_weights.json")
                utils.save_dict_to_json(test_metrics, last_test_json_path)

                plot_history['train_loss'].append(train_loss)
                plot_history['train_acc'].append(train_acc)
                plot_history['test_loss'].append(test_loss)
                plot_history['test_acc'].append(test_acc)

                t.set_postfix(
                    tr_acc='{:05.3f}'.format(train_acc),
                    te_acc='{:05.3f}'.format(test_acc),
                    tr_loss='{:05.3f}'.format(train_loss),
                    te_loss='{:05.3f}'.format(test_loss))
                print('\n')

            t.update()

    utils.plot_training_results(args.model_dir, plot_history)

def train_and_evaluate_newton(model,
                       meta_train_classes,
                       meta_test_classes,
                       task_type,
                       meta_optimizer,
                       loss_fn,
                       metrics,
                       params,
                       model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every `save_summary_steps`.

    Args:
        model: (MetaLearner) a meta-learner for MAML algorithm
        meta_train_classes: (list) the classes for meta-training
        meta_train_classes: (list) the classes for meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        meta_optimizer: (torch.optim) an meta-optimizer for MetaLearner
        loss_fn: a loss function
        metrics: (dict) a dictionary of functions that compute a metric using
                 the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    TODO Validation classes
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir,
                                    args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, meta_optimizer)

    # params information
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_inner_tasks = params.num_inner_tasks
    task_lr = params.task_lr
    meta_lr = params.meta_lr

    # TODO validation accuracy
    best_test_acc = 0.0

    # For plotting to see summerized training procedure
    plot_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    with tqdm(total=params.num_episodes) as t:
        for episode in range(params.num_episodes):
            # Run one episode
            logging.info("Episode {}/{}".format(episode + 1,
                                                params.num_episodes))

            # Run inner loops to get adapted parameters (theta_t`)
            adapted_state_dicts = []
            dataloaders_list = []
            for n_task in range(num_inner_tasks):
                task = task_type(meta_train_classes, num_classes, num_samples,
                                 num_query)
                dataloaders = fetch_dataloaders(['train', 'test', 'meta'],
                                                task)
                # Perform a gradient descent to meta-learner on the task
                a_dict = train_single_task_newton(model, task_lr, loss_fn,
                                           dataloaders, params)
                # Store adapted parameters
                # Store dataloaders for meta-update and evaluation
                adapted_state_dicts.append(a_dict)
                dataloaders_list.append(dataloaders)


            # compute outer update directions
            # 1. grad of updated parameters -> v
            # 2. grad of (nabla^2 J * U * v)
            # 3. conjugate gradient to compute H^-1 * (2.)
            directions = 0
            for n_task in range(num_inner_tasks):
                dataloaders = dataloaders_list[n_task]
                dl_meta = dataloaders['meta']
                X_meta, Y_meta = dl_meta.__iter__().next()
                if params.cuda:
                    X_meta, Y_meta = X_meta.cuda(async=True), Y_meta.cuda(
                        async=True)
                # 1. grad of updated parameters -> grad_t
                a_dict = adapted_state_dicts[n_task]
                Y_meta_hat = model(X_meta, a_dict)
                loss_t = loss_fn(Y_meta_hat, Y_meta)
                params_g = []
                # differentiable parameters
                for a in a_dict.values():
                    if a.requires_grad:
                        params_g.append(a)
                grad_t = torch.autograd.grad(loss_t, params_g)

                # 2. grad of (nabla^2 J * u * v)
                Y_meta_hat_before = model(X_meta)
                loss_t_before = loss_fn(Y_meta_hat_before, Y_meta)
                grad_t_before = torch.autograd.grad(loss_t_before, model.parameters(), create_graph=True)
                for g in params_g: # u
                    g = g.detach()
                sum_grad = 0
                for g, u in zip(params_g, grad_t_before):
                    sum_grad += torch.dot(g.view(-1),u.view(-1))
                grad_grad_u = torch.autograd.grad(sum_grad, model.parameters(), create_graph=True)
                sum_grad = 0
                for g,v in zip(grad_grad_u, grad_t):
                    sum_grad += torch.dot(g.view(-1), v.view(-1))
                v_t = torch.autograd.grad(sum_grad, model.parameters())
                v_t = torch.cat([grad.view(-1) for grad in v_t]).data

                # 3. conjugate gradient to compute H^-1 * (2.)
                def get_grad():
                    Y_before = model(X_meta)
                    loss_before = loss_fn(Y_before, Y_meta)
                    grad_before = torch.autograd.grad(loss_before, model.parameters(), create_graph=True)
                    return grad_before
                def Hvp(v, damping=1e-1):
                    flat_gb = utils.flatten(get_grad())
                    gv = (flat_gb * Variable(v)).sum()
                    Hv = torch.autograd.grad(gv, model.parameters())
                    flat_Hv = torch.cat([g.contiguous().view(-1) for g in Hv]).data
                    return flat_Hv + v * damping
                H_v = conjugate_gradients(Hvp, v_t, 10)
                d_t = (1 - task_lr) * utils.flatten(grad_t).data + H_v
                directions += d_t

            # Meta-update
            direction = directions / num_inner_tasks
            cur_params = utils.get_flat_params_from(model)
            updated_params = cur_params - direction * meta_lr
            utils.set_flat_params_to(model, updated_params)

            # Evaluate model on new task
            # Evaluate on train and test dataset given a number of tasks (params.num_steps)
            if (episode + 1) % params.save_summary_steps == 0:
                train_metrics = evaluate(model, loss_fn, meta_train_classes,
                                         task_lr, task_type, metrics, params,
                                         'train')
                test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                        task_lr, task_type, metrics, params,
                                        'test')

                train_loss = train_metrics['loss']
                test_loss = test_metrics['loss']
                train_acc = train_metrics['accuracy']
                test_acc = test_metrics['accuracy']

                is_best = test_acc >= best_test_acc

                # Save weights
                utils.save_checkpoint({
                    'episode': episode + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': meta_optimizer.state_dict()
                },
                                      is_best=is_best,
                                      checkpoint=model_dir)

                # If best_test, best_save_path
                if is_best:
                    logging.info("- Found new best accuracy")
                    best_test_acc = test_acc

                    # Save best test metrics in a json file in the model directory
                    best_train_json_path = os.path.join(
                        model_dir, "metrics_train_best_weights.json")
                    utils.save_dict_to_json(train_metrics,
                                            best_train_json_path)
                    best_test_json_path = os.path.join(
                        model_dir, "metrics_test_best_weights.json")
                    utils.save_dict_to_json(test_metrics, best_test_json_path)

                # Save latest test metrics in a json file in the model directory
                last_train_json_path = os.path.join(
                    model_dir, "metrics_train_last_weights.json")
                utils.save_dict_to_json(train_metrics, last_train_json_path)
                last_test_json_path = os.path.join(
                    model_dir, "metrics_test_last_weights.json")
                utils.save_dict_to_json(test_metrics, last_test_json_path)

                plot_history['train_loss'].append(train_loss)
                plot_history['train_acc'].append(train_acc)
                plot_history['test_loss'].append(test_loss)
                plot_history['test_acc'].append(test_acc)

                t.set_postfix(
                    tr_acc='{:05.3f}'.format(train_acc),
                    te_acc='{:05.3f}'.format(test_acc),
                    tr_loss='{:05.3f}'.format(train_loss),
                    te_loss='{:05.3f}'.format(test_loss))
                print('\n')

            t.update()

    utils.plot_training_results(args.model_dir, plot_history)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    SEED = params.SEED
    meta_lr = params.meta_lr
    num_episodes = params.num_episodes

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # NOTE These params are only applicable to pre-specified model architecture.
    # Split meta-training and meta-testing characters
    if 'Omniglot' in args.data_dir and params.dataset == 'Omniglot':
        params.in_channels = 1
        meta_train_classes, meta_test_classes = split_omniglot_characters(
            args.data_dir, SEED)
        task_type = OmniglotTask
    elif ('miniImageNet' in args.data_dir or
          'tieredImageNet' in args.data_dir) and params.dataset == 'ImageNet':
        params.in_channels = 3
        meta_train_classes, meta_test_classes = load_imagenet_images(
            args.data_dir)
        task_type = ImageNetTask
    else:
        raise ValueError("I don't know your dataset")

    # Define the model and optimizer
    if params.cuda:
        model = MetaLearner(params).cuda()
    else:
        model = MetaLearner(params)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    model_metrics = metrics

    # Train the model
    logging.info("Starting training for {} episode(s)".format(num_episodes))
    train_and_evaluate_newton(model, meta_train_classes, meta_test_classes, task_type,
                       meta_optimizer, loss_fn, model_metrics, params,
                       args.model_dir, args.restore_file)
