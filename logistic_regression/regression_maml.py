import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from torch.autograd import Variable
import utils
import argparse

class task_generator():
    def __init__(self, num_classes, num_support, n_dim=10, scale_mean=1.0, scale_std=0.01):
        self.num_classes = num_classes
        self.num_support = num_support
        self.n_dim = n_dim
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.means = [np.random.uniform(-1*self.scale_mean, self.scale_mean, self.n_dim) for _ in range(self.num_classes)]

    def get_batch(self):
        samples = [np.random.normal(loc=m, scale=np.ones_like(m)*self.scale_std) for m in self.means for _ in range(self.num_support)]
        batch = dict()
        for i in range(len(samples)):
            batch[torch.Tensor(samples[i])] = i // self.num_support
        keys = list(batch.keys())
        random.shuffle(keys)
        return torch.stack(keys), torch.Tensor([batch[key] for key in keys]).long()

class Linear(nn.Module):
    def __init__(self, n_dim, n_class):
        super(Linear, self).__init__()
        self.linear = nn.Linear(n_dim, n_class)

    def forward(self, X, params=None):
        if params == None:
            out = self.linear(X)
        else:
            out = F.linear(X, params['linear.weight'], params['linear.bias'])
        return out

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

def train_single_task(model, task_lr, loss_fn, generator, num_train_updates):

    model.train()
    # support set and query set for a single few-shot task
    X_sup, Y_sup = generator.get_batch()
    logit = model(X_sup)
    loss = loss_fn(logit, Y_sup)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # performs updates using calculated gradients
    # we manually compute adpated parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()  # NOTE what about just dict
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        adapted_params[key] = val - task_lr * grad
        adapted_state_dict[key] = adapted_params[key]

    for _ in range(1, num_train_updates): #todo
        Y_sup_hat = model(X_sup, adapted_state_dict)
        loss = loss_fn(Y_sup_hat, Y_sup)
        zero_grad(adapted_params.values())
        grads = torch.autograd.grad(
            loss, adapted_params.values(), create_graph=True)
        for (key, val), grad in zip(adapted_params.items(), grads):
            adapted_params[key] = val - task_lr * grad
            adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict

def train_and_evaluate(model, meta_optimizer, loss_fn):

    for episode in range(100000):
        # Run inner loops to get adapted parameters (theta_t`)
        adapted_state_dicts = []
        dataloaders_list = []
        for n_task in range(num_inner_tasks):
            generator = task_generator(num_classes, num_samples, n_dim=n_dim)
            # Perform a gradient descent to meta-learner on the task
            a_dict = train_single_task(model, task_lr, loss_fn,
                                       generator, 1)
            # Store adapted parameters
            # Store dataloaders for meta-update and evaluation
            adapted_state_dicts.append(a_dict)
            dataloaders_list.append(generator)

        meta_loss = 0
        for n_task in range(num_inner_tasks):
            generator = dataloaders_list[n_task]
            X_meta, Y_meta = generator.get_batch()
            a_dict = adapted_state_dicts[n_task]
            logit = model(X_meta, a_dict)
            loss_t = loss_fn(logit, Y_meta)
            meta_loss += loss_t
        meta_loss /= float(num_inner_tasks)
        # print(meta_loss.item())

        # Meta-update using meta_optimizer
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        # Evaluate model on new task
        # Evaluate on train and test dataset given a number of tasks (params.num_steps)
        if episode % 100 == 0:
            test_g = task_generator(num_classes, num_samples, n_dim=n_dim)
            x,y = test_g.get_batch()
            net_clone = copy.deepcopy(model)
            optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
            for _ in range(num_eval_updates):
                logit = net_clone(x)
                loss = loss_fn(logit, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
            logit = net_clone(x)
            _, pred = torch.max(logit, 1)
            acc = (pred == y).sum().float() / len(y)
            print('episode:', episode, 'loss: %.3f' % meta_loss.item(), 'acc: %.2f' % acc.item())

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
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

def train_single_task_newton(model, task_lr, loss_fn, generator, num_train_updates):

    model.train()
    # support set and query set for a single few-shot task
    X_sup, Y_sup = generator.get_batch()
    logit = model(X_sup)
    loss = loss_fn(logit, Y_sup)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

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

def train_and_evaluate_newton(model, loss_fn):
    for episode in range(100000):
        # Run inner loops to get adapted parameters (theta_t`)
        adapted_state_dicts = []
        dataloaders_list = []
        for n_task in range(num_inner_tasks):
            generator = task_generator(num_classes, num_samples, n_dim=n_dim)
            # Perform a gradient descent to meta-learner on the task
            a_dict = train_single_task_newton(model, task_lr, loss_fn,
                                       generator, 1)
            # Store adapted parameters
            # Store dataloaders for meta-update and evaluation
            adapted_state_dicts.append(a_dict)
            dataloaders_list.append(generator)

        # compute outer update directions
        # 1. grad of updated parameters -> v
        # 2. grad of (nabla^2 J * U * v)
        # 3. conjugate gradient to compute H^-1 * (2.)
        directions = 0
        meta_loss = 0
        for n_task in range(num_inner_tasks):
            generator = dataloaders_list[n_task]
            X_meta, Y_meta = generator.get_batch()
            # 1. grad of updated parameters -> grad_t
            a_dict = adapted_state_dicts[n_task]
            Y_meta_hat = model(X_meta, a_dict)
            logit = model(X_meta, a_dict)
            loss_t = loss_fn(logit, Y_meta)
            meta_loss += loss_t
            params_g = []
            # differentiable parameters
            for a in a_dict.values():
                if a.requires_grad:
                    params_g.append(a)
            grad_t = torch.autograd.grad(loss_t, params_g)

            # 2. grad of (nabla^2 J * u * v)
            logit_before = model(X_meta)
            loss_t_before = loss_fn(logit_before, Y_meta)
            grad_t_before = torch.autograd.grad(loss_t_before, model.parameters(), create_graph=True)
            for g in params_g:  # u
                g = g.detach()
            sum_grad = 0
            for g, u in zip(params_g, grad_t_before):
                sum_grad += torch.dot(g.view(-1), u.view(-1))
            grad_grad_u = torch.autograd.grad(sum_grad, model.parameters(), create_graph=True)
            sum_grad = 0
            for g, v in zip(grad_grad_u, grad_t):
                sum_grad += torch.dot(g.view(-1), v.view(-1))
            v_t = torch.autograd.grad(sum_grad, model.parameters())
            v_t = torch.cat([grad.view(-1) for grad in v_t]).data

            # 3. conjugate gradient to compute H^-1 * (2.)
            def get_grad():
                logit_before_ = model(X_meta)
                loss_before = loss_fn(logit_before_, Y_meta)
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
        meta_loss /= float(num_inner_tasks)
        direction = directions / num_inner_tasks
        cur_params = utils.get_flat_params_from(model)
        updated_params = cur_params - direction * meta_lr
        utils.set_flat_params_to(model, updated_params)

        # Evaluate model on new task
        # Evaluate on train and test dataset given a number of tasks (params.num_steps)
        if episode % 100 == 0:
            test_g = task_generator(num_classes, num_samples, n_dim=n_dim)
            x,y = test_g.get_batch()
            net_clone = copy.deepcopy(model)
            for _ in range(num_eval_updates):
                a_dict = train_single_task_newton(net_clone, task_lr, criterion, test_g, 1)
                updated_params = utils.flatten(a_dict.values())
                utils.set_flat_params_to(net_clone, updated_params)
            logit = net_clone(x)
            _, pred = torch.max(logit, 1)
            acc = (pred == y).sum().float() / len(y)
            print('episode:', episode, 'loss: %.3f' % meta_loss.item(), 'acc: %.2f' % acc.item())


num_classes = 10 # num of distributions for each task
num_samples = 1 # num of samples per distribution for training
num_inner_tasks = 8 # meta batch size
task_lr = 0.1 # inner lr
num_eval_updates = 1 # num of gradient steps for evaluation
n_dim = 10 # dim of space
model = Linear(n_dim=n_dim, n_class=num_classes)
criterion = nn.CrossEntropyLoss()

parser = argparse.ArgumentParser()
parser.add_argument('-sgd', action='store_true', default='False')
args = parser.parse_args()

if args.n:
    meta_lr = 1e-2
    train_and_evaluate_newton(model, criterion)
else:
    meta_lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    train_and_evaluate(model, optimizer, criterion)




