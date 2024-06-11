import argparse
import functools
import os
import random
import sys;

sys.path.append("..")
from pathlib import Path

import numpy as np
import torch
import torchvision

from data.cifar import Cifar, ImageNet
from model.resnet import PreActResNet18, PreActResNet50  # , PreActResNet101
from model.vgg import vgg19_bn
from model.pyramidnet import pyramidnet272
from model.densenet import densenet121
from model.vit import vit_model
from utility.utils import test_model, save_model, load_model
from model.smooth_cross_entropy import smooth_crossentropy
from model.wide_res_net import WideResNet
from utility.log import Log
from utility.step_lr import StepLR
from utility.initialize import initialize


# PYTHONPATH=. screen python methods/train_SGD.py --seed 0 --save-model --save-path outputs/cifar10_res18_Epoch50_SGD --model resnet18  --epochs 50
# PYTHONPATH=. screen python methods/train_SAM.py --seed 0 --save-model --save-path outputs/cifar10_res18_Epoch50_SAM --model resnet18  --epochs 50 --gpu 4,5,6,7
# PYTHONPATH=. screen python methods/train_ExtrapSAM.py --alpha 3 --dataset cifar10 --seed 0 --save-model --save-path outputs/cifar10_res18_Epoch50_ExtrapSAM --model resnet18  --epochs 50

# PYTHONPATH=. screen bash run_one_algorithm.py  SGD
# PYTHONPATH=. screen bash run_one_algorithm.py --adam --learning_rate 1e-3

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0,1,2,3', type=str)
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument('--adam', action='store_true', help="use adam")
    parser.add_argument('--adamw', action='store_true', help="use adamw")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.") 
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--dataset", default='cifar10', type=str, help="Dataset name.")

    # optimizer
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--scheduler", default='cos', type=str, help="Scheduler name.")
    parser.add_argument("--nesterov", action='store_true', help="Use Nesterov momentum.")

    # model training
    parser.add_argument("--model", default='WRN-28-10', type=str, help="Model name")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--seed", default=1, type=int, help="Random seed to be used.")
    parser.add_argument("--save-path", default='outputs/test', type=str, help="save path")

    parser.add_argument("--save-model", action='store_true', help="save model")
    
    return parser
    

def get_model(model_name, num_classes):
    if model_name == 'wrn':
        # origin: WRN-16-8 => current: WRN-28-10
        depth, widen_factor = map(int, model_name.split('-')[1:])
        model = WideResNet(depth, widen_factor, dropout=0.0, in_channels=3, labels=num_classes)
    elif model_name == 'resnet18':
        model = PreActResNet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = PreActResNet50(num_classes=num_classes)
    elif model_name == 'vgg':
        model = vgg19_bn(num_classes)
    elif model_name == 'pyramidnet':
        model = pyramidnet272(num_classes=num_classes)
    elif model_name == 'densenet':
        model = densenet121(num_classes=num_classes)
    elif model_name == 'vit':
        model = vit_model(num_classes=num_classes)
    else:
        raise NotImplementedError
    return model


def get_dataset(dataset_name, batch_size, threads):
    if dataset_name == 'cifar10':
        dataset = Cifar(batch_size=batch_size, threads=threads)
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = Cifar(batch_size=batch_size, threads=threads, cifar100=True)
        num_classes = 100
    elif dataset_name == 'imagenet':
        dataset = ImageNet(batch_size=batch_size, threads=threads)
        num_classes = 1000
    else:
        raise NotImplementedError
    return dataset, num_classes


class S():
    def step(self):
        return
def get_scheduler(name, optimizer, epochs, iterations):
    total_iterations = epochs * iterations
    if name == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
    elif name == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(total_iterations * 0.3),
                        int(total_iterations * 0.6),
                        int(total_iterations * 0.8)],
            gamma=0.1)
    elif name == 'constant' or name=='none':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=100000000)
    else:
        raise NotImplementedError
    return scheduler


from torch.cuda.amp import autocast


def get_optimizer(model, lr, args):
    if not args.adam and not args.adamw:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.adam:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.adamw:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise NotImplementedError()
    return opt


class Trainer(object):
    def __init__(self, args):
        initialize(seed=args.seed)
        self.args = args
        self.save_path = args.save_path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()

        self.dataset, self.num_classes = get_dataset(args.dataset, args.batch_size * num_gpus, args.threads * num_gpus)
        self.log = Log(log_each=10, filename=self.save_path)
        self.model = get_model(args.model, self.num_classes).to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.optimizer = get_optimizer(self.model, args.learning_rate * np.sqrt(num_gpus), args)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer, args.epochs, len(self.dataset.train))
        self.start_epoch = 0

        self.loss_func = functools.partial(smooth_crossentropy, smoothing=args.label_smoothing)

        # print classname
        print('Class: {}'.format(self.model.__class__.__name__))
        self.log.log('Class: {}'.format(self.model.__class__.__name__))
        self.log.log(str(args))
        self.log.log('real lr: {}, batch_size: {}, threads: {}'.format(args.learning_rate * np.sqrt(num_gpus),
                                                                       args.batch_size * num_gpus,
                                                                       args.threads * num_gpus))
        # self.load_model('/data/zj/PycharmProjects/sam/outputs/sgd/sgdmodel-0.1.pt', epoch=159)

    def retrieve_batch(self, data_iter):
        batch = next(data_iter)
        inputs, targets = (b.to(self.device) for b in batch)
        return inputs, targets

    def train(self):
        args = self.args

        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                self.model.train()
                self.log.train(len_dataset=len(self.dataset.train))

                iters = iter(self.dataset.train)
                i = 0
                self.pre_epoch()
                while True:
                    try:
                        inputs, targets = self.retrieve_batch(iters)

                        # self.record_grad_norm(epoch, i, inputs, targets)

                        predictions, loss = self.train_step(epoch, i, self.model, inputs, targets, self.optimizer, iters)
                        with torch.no_grad():
                            correct = torch.argmax(predictions, 1) == targets
                            self.log(self.model, loss.cpu(), correct.cpu(), self.scheduler.get_last_lr()[0])
                        i += 1
                    except StopIteration:
                        break
                self.after_epoch() 
                if self.args.save_model and epoch == args.epochs * 8 / 10 - 1:
                    save_model(self.model, self.optimizer, self.scheduler,
                            os.path.join(self.save_path, 'ckpt-lr{}-E{}.pt'.format(args.learning_rate, epoch)))

                self.model.eval()
                loss, acc = test_model(self.model, self.dataset.test, self.log, self.device, result=True)
        except Exception as e:
            self.log.log('Error: {}'.format(e))
            raise e

        self.log.flush()
        if self.args.save_model:
            save_model(self.model, self.optimizer, self.scheduler, os.path.join(self.save_path, 'model_final.pt'))

    def record_grad_norm(self, epoch, iteration, inputs, targets):
        if iteration == 0:
            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            self.loss_func(predictions, targets).mean().backward()
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            norm = torch.norm(
                torch.stack([
                    torch.norm(g.detach(), 2) for g in grads
                ]),
                p=2
            )
            self.log.log('grad norm: {:6.6f}'.format(norm.item()))
            self.optimizer.zero_grad()

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        optimizer.zero_grad()
        # with autocast():
        predictions = model(inputs)
        loss = self.loss_func(predictions, targets)
        L = loss.mean()
        L.backward()
        optimizer.step()
        self.scheduler.step()
        return predictions, loss

    def load_model(self, model_path, epoch):
        load_model(self.model, self.optimizer, self.scheduler, model_path)
        self.scheduler.last_epoch = epoch * len(self.dataset.train)
        self.start_epoch = epoch + 1
        # self.log.epoch = 0
        # self.log._reset(len(self.dataset.train))
        print('Load from : {}, start from epoch : {}'.format(model_path, epoch))

    def pre_epoch(self):
        pass

    def after_epoch(self):
        pass


if __name__ == '__main__':
    args = get_parser().parse_args()
    trainer = Trainer(args)
    trainer.train()
        