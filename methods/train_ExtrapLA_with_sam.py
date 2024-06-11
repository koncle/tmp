import argparse
import copy
import functools
import os
import sys;
from pathlib import Path

from example.train_sam_multistep import InfinityLoader
from example.utility.param_utils import ParamOperator

sys.path.append("..")
import torch

from example.train_LA import save_model
from example.utils import test_model
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats


class Queue():
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = []

    def put(self, x):
        if len(self.queue) >= self.maxsize:
            self.queue.pop(0)
        self.queue.append(x)

    def get(self, idx):
        return self.queue[idx]

    def get_last(self):
        return self.queue[0]

    def get_avg(self):
        return sum(self.queue) / len(self.queue)

    def __len__(self):
        return len(self.queue)


class ExtraLAWrapper():
    def __init__(self, net, opt, alpha=5, queue_size=10, init_update_freq=5):
        # hyperparameters
        self.alpha = alpha
        self.scale_factor = 1
        self.beta = 1  # 0.5 / (0.1 * 5)
        self.init_update_freq = init_update_freq

        self.steps = 0
        self.net = net
        self.opt = opt
        self.tmp_net = copy.deepcopy(self.net)
        self.tmp_opt = torch.optim.SGD(self.tmp_net.parameters(), lr=0.0, momentum=0.9, weight_decay=0.0005)

        self.maxsize = queue_size
        self.param_queue = Queue(self.maxsize)
        self.operator = ParamOperator()

        self.init_lr = self.opt.param_groups[0]['lr']

    def _extrap_step(self, alpha, adv_grad=None):
        # obtain self.opt lr:
        self.scale_factor = self.init_lr / self.opt.param_groups[0]['lr']

        cur_param = self.operator.extract_params(self.net)
        param = self.get_init_param()
        dir = cur_param - param
        # if dir.norm() > 0:
        #     dir = dir / dir.norm()
        gamma = 0
        # if adv_grad is not None:
        #     dir = dir + adv_grad * gamma
        # new_param = param + dir * self.scale_factor * 0.05
        # new_param = cur_param + dir * 0.5
        # new_param = cur_param + adv_grad * 0.05
        # new_param = cur_param + (dir * 0.3 + adv_grad * 0.05)
        # dir = torch.randn_like(dir)*0.3 + dir
        # noise = torch.randn_like(dir)
        # noise = noise / noise.norm() #* 0.3
        new_param = cur_param + dir * 0.5
        # new_param = cur_param + dir * 0.5
        self.operator.put_parameters(self.tmp_net, new_param, data=True)

    def _insert_grad(self):
        for tmp_p, cur_p in zip(self.tmp_net.parameters(), self.net.parameters()):
            if tmp_p.grad is not None:
                cur_p.grad = tmp_p.grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def do_extrap(self):
        self.set_init_param()
        self._extrap_step(self.alpha)

    def forward(self, inputs, grad=None, **kwargs):
        ret = self.tmp_net(inputs)

        with torch.no_grad():
            enable_running_stats(self.net)
            self.net(inputs)
        return ret

    def zero_grad(self):
        self.opt.zero_grad()
        self.tmp_opt.zero_grad()

    def set_init_param(self):
        param = self.operator.extract_params(self.net).detach().clone()
        self.param_queue.put(param)

    def get_init_param(self):
        left = self.steps % self.init_update_freq
        return self.param_queue.get(len(self.param_queue) - 1 - left)

    def step(self):
        self._insert_grad()
        self.opt.step()

        cur_param = self.operator.extract_params(self.net).detach()
        param = self.get_init_param()
        new_param = param + self.beta * (cur_param - param)
        self.operator.put_parameters(self.net, new_param, data=True)

        self.steps += 1

    def forward_with_loss_test(self, inputs, targets, loss_func):
        self.set_init_param()

        self.do_analyze(inputs, targets, loss_func)

        self._extrap_step(self.alpha)
        ret = self.tmp_net(inputs)

        # self.steps += 1

        with torch.no_grad():
            enable_running_stats(self.net)
            self.net(inputs)
        return ret

    def do_analyze(self, inputs, targets, loss_func):
        alphas = [1, 2, 4, 8, 16, 32, 64]
        iters = 391

        loss_freq = 1
        self.loss_landscape(alphas, loss_freq * iters, inputs, targets, loss_func)

        grad_freq = 1
        self.grad_sim(alphas, grad_freq * iters, inputs, targets, loss_func)

        adv_grad_freq = 1
        self.adv_grad_freq(adv_grad_freq * iters, inputs, targets, loss_func)

    def loss_landscape(self, alphas, freq, inputs, targets, loss_func):
        if self.steps % freq == 0:
            with torch.no_grad():
                losses = []
                for s in alphas:
                    self._extrap_step(s)
                    y = self.tmp_net(inputs)
                    loss = loss_func(y, targets).mean().item()
                    losses.append(loss)
                print("FORWARD losses: ", losses)

    def grad_sim(self, alphas, freq, inputs, targets, loss_func):
        if self.steps % freq == 0:
            y = self.net(inputs)
            loss = loss_func(y, targets).mean()
            self.opt.zero_grad()
            loss.backward()
            init_grad = self.operator.extract_params_with_grad(self.net)[1]

            grads = []
            for s in alphas:
                self._extrap_step(s)
                y = self.tmp_net(inputs)
                loss = loss_func(y, targets).mean()
                self.tmp_opt.zero_grad()
                loss.backward()
                grad = self.operator.extract_params_with_grad(self.tmp_net)[1]
                grads.append(grad)

            cos_similarity = [torch.cosine_similarity(g, init_grad, dim=0).item() for g in grads]
            print("SIMILARITY, cos : ", cos_similarity)

    def adv_grad_freq(self, freq, inputs, targets, loss_func):
        if self.steps % freq == 0:
            y = self.net(inputs)
            loss = loss_func(y, targets).mean()
            self.opt.zero_grad()
            loss.backward()
            cur_w, init_grad = self.operator.extract_params_with_grad(self.net)
            init_grad = -init_grad

            angles = []
            for i in range(len(self.param_queue)):
                w = self.param_queue.get(len(self.param_queue) - i - 1)
                cur_grad = w - cur_w
                angle = torch.cosine_similarity(cur_grad, init_grad, dim=0).item()
                angles.append(angle)
            print("ANGLES : ", angles)


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
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")

    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

    parser.add_argument("--algo", default='vanilla', type=str)
    parser.add_argument("--alpha", default=5, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--queue_size", default=10, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--init_update_freq", default=5, type=int, help="Rho parameter for SAM.")

    parser.add_argument("--gpu", default='0', type=str)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # sam_lr{}_rho{}_diffData_AddGrad/'.format(args.learning_rate)
    save_path = '../models/extraLA_with_sam_NoNorm'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(save_path)

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10, filename=save_path)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    loader = InfinityLoader(dataset.train)
    losses, acces = [], []
    start_epoch = 0

    if args.algo == 'vanilla':
        model_wrapper = ExtraLAWrapper(model, optimizer, alpha=args.alpha, queue_size=args.queue_size, init_update_freq=args.init_update_freq)
    elif args.algo == 'constant':
        model_wrapper = ConstantExtraLAWrapper(model, optimizer, alpha=args.alpha, queue_size=args.queue_size)
    else:
        raise NotImplementedError
    # model_wrapper = AdaptiveLAWrapper(model, optimizer, alpha=args.alpha, queue_size=args.queue_size, init_update_freq=args.init_update_freq)

    print('dataset batches', len(dataset.train))

    operator = ParamOperator()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for i in range(len(dataset.train)):
            batch = next(loader)
            inputs, targets = (b.to(device) for b in batch)

            model_wrapper.do_extrap()
            predictions = model_wrapper(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            model_wrapper.tmp_opt.zero_grad()
            loss.mean().backward()

            with torch.no_grad():
                adv_grad = get_adv_grad(model_wrapper.tmp_opt, adaptive=False)
                param = operator.extract_params(model_wrapper.tmp_net)
                param = param + adv_grad * 0.05
                operator.put_parameters(model_wrapper.tmp_net, param, data=True)

            predictions = model_wrapper(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            model_wrapper.zero_grad()
            loss.mean().backward()
            model_wrapper.step()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        if epoch == args.epochs * 3 / 10 - 1 or epoch == args.epochs * 6 / 10 - 1 or epoch == args.epochs * 8 / 10 - 1:
            save_model(model, save_path + 'model-lr{}-{}.pt'.format(args.learning_rate, epoch))
        loss, acc = test_model(model, dataset.test, log, device, result=True)
        losses.append(loss)
        acces.append(acc)
    print(losses)
    print(acces)

    log.flush()
    save_model(model, save_path + 'model_final.pt'.format(args.learning_rate))
