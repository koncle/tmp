import argparse
import copy
import os
from pathlib import Path

import sys;

from example.utility.param_utils import ParamOperator

sys.path.append("..")
import torch

from example.train_LA import save_model
from example.utils import test_model, load_model
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

from sam import SAM


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.015, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

    parser.add_argument("--gpu", default='3', type=str)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_path = '../models/sam_lr{}_rho{}_largeLR/'.format(args.learning_rate, args.rho)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(save_path)

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10, filename=save_path)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    losses, acces = [], []
    start_epoch = 0

    operator = ParamOperator()

    lam = 0.5

    ahead_param = operator.extract_params(model)
    behind_param = operator.extract_params(model)

    tmp_model = copy.deepcopy(model)
    tmp_opt = torch.optim.SGD(tmp_model.parameters(), lr=0.1)

    behind_alpha = 0.1

    alpha = 5
    beta = 0.5
    # rho = 0.05
    rho = 2
    adaptive = True

    for epoch in range(start_epoch, args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            prev_param = operator.extract_params(model).clone()

            # # Lookahead
            # operator.put_parameters(tmp_model, ahead_param.clone(), data=True)
            # ahead_grad = get_grad(tmp_model, tmp_opt, inputs, targets)
            # update_grad = ahead_grad

            # Lookbehind
            # first ascend step
            operator.put_parameters(tmp_model, behind_param.clone(), data=True)
            forward_opt_backward(tmp_model, tmp_opt, inputs, targets)
            behind_grad = get_adv_grad(tmp_opt, adaptive)

            # second descend step
            new_param = prev_param + behind_grad * rho
            operator.put_parameters(tmp_model, new_param, data=True)
            forward_opt_backward(tmp_model, tmp_opt, inputs, targets)
            update_grad = operator.extract_params_with_grad(tmp_model)[1]

            operator.put_grads(model, update_grad)
            optimizer.step()

            cur_param = operator.extract_params(model)
            # ahead_param = ahead_param * beta + (1 - beta) * (prev_param + (cur_param - prev_param) * alpha)
            behind_param = cur_param

            with torch.no_grad():
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        if epoch == args.epochs * 3 / 10 - 1 or epoch == args.epochs * 6 / 10 - 1 or epoch == args.epochs * 8 / 10 - 1:
            save_model(model, save_path + 'model-{}.pt'.format(args.learning_rate, epoch))
        loss, acc = test_model(model, dataset.test, log, device, result=True)
        losses.append(loss)
        acces.append(acc)
    print(losses)
    print(acces)

    log.flush()
    save_model(model, save_path + 'model_final.pt'.format(args.learning_rate))
