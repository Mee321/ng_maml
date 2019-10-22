import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict

class task_generator():
    def __init__(self, num_classes, num_support, num_query, n_dim=10, scale_mean=1.0, scale_std=0.01):
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
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
    loss = criterion(logit, Y_sup)

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

def train_and_evaluate(model,
                       meta_optimizer,
                       loss_fn):

    num_classes = 10
    num_samples = 1
    num_query = 10
    num_inner_tasks = 8
    task_lr = 0.1
    num_eval_updates = 3

    for episode in range(100000):
        # Run inner loops to get adapted parameters (theta_t`)
        adapted_state_dicts = []
        dataloaders_list = []
        for n_task in range(num_inner_tasks):
            generator = task_generator(num_classes, num_samples, num_query)
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
            test_g = task_generator(num_classes, num_samples, num_query)
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

num_classes = 10
model = Linear(n_dim=10, n_class=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
train_and_evaluate(model, optimizer, criterion)




