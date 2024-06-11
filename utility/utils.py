from pathlib import Path

import numpy as np
import torch

from utility.log import Log
from model.smooth_cross_entropy import smooth_crossentropy

np.set_printoptions(linewidth=1000)


def save_model(model, opt, scheduler, path):
    p = Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    state = model.state_dict()
    opt_staet = opt.state_dict()
    scheduler_state = scheduler.state_dict()
    torch.save({'model': state, 'opt': opt_staet, 'scheduler': scheduler_state}, p)


def load_model(model, opt, scheduler, path):
    print('Load from ', path)
    state = torch.load(Path(path), map_location='cpu')
    # if hasattr(model, 'module'):
    #     model.module.load_state_dict(state['model'])
    # else:
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['opt'])
    scheduler.load_state_dict(state['scheduler'])
    return model


def test_model(model, dataset, log: Log, device, result=False, flush=False):
    model.eval()
    if flush:
        log.is_train = False
        log._reset(len(dataset))
    else:
        log.eval(len(dataset))

    with torch.no_grad():
        for batch in dataset:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(model, loss.cpu(), correct.cpu())
    if result:
        return log.flush(show=False)


def reestimate_BN(model, train_dataset, device):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    with torch.no_grad():
        for batch in train_dataset:
            inputs, targets = (b.to(device) for b in batch)
            model(inputs)
    model.train(was_training)


def forward_opt_backward(model, opt, inputs, targets):
    predictions = model(inputs)
    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
    opt.zero_grad()
    loss.mean().backward()

def grad_norm(optimizer, adaptive):
    param_groups = optimizer.param_groups
    norm = torch.norm(
        torch.stack([
            ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2)
            for group in param_groups for p in group["params"]
            if p.grad is not None
        ]),
        p=2
    )
    return norm



def get_adv_grad(optimizer, adaptive):
    norm = grad_norm(optimizer, adaptive=adaptive)
    param_groups = optimizer.param_groups
    grads = []
    for group in param_groups:
        scale = 1 / (norm + 1e-12)
        for p in group["params"]:
            if p.grad is None: continue
            e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale
            grads.append(e_w.view(-1))
    adv_grad = torch.cat(grads)
    return adv_grad
