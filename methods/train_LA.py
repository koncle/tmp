import argparse
import copy
import functools
import os
import time
from pathlib import Path

import torch
import sys;

from example.utility.param_utils import ParamOperator

sys.path.append("..")

from example.utility.utils import save_model, test_model, load_model
from example.model.wide_res_net import WideResNet
from example.model.smooth_cross_entropy import smooth_crossentropy
from example.data.cifar import Cifar
from sam import SAM
from example.utility.log import Log
from example.utility.initialize import initialize
from example.utility.step_lr import StepLR, StepLR2
from example.utility.bypass_bn import enable_running_stats, disable_running_stats #

from lookahead import Lookahead


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

    parser.add_argument("--learning_rate", default=0.5, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--length", default=5, type=int, help="Base learning rate at the start of the training.")
    parser.add_argument("--outer_lr", default=0.5, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--gpu", default='3', type=str)

    args = parser.parse_args()
    print(args.epochs)
    initialize(args, seed=42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    noise = 0.0
    update_steps = args.length
    outer_lr = args.outer_lr
    # save_path = '../models2/LA_lr{}_Olr{}_length{}_Noise{}_reverseGrad/'.format(args.learning_rate, outer_lr, update_steps, noise)
    save_path = '../models2/LA_test/'.format(args.learning_rate, outer_lr, update_steps, noise)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print('Make save path', save_path)

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10, filename=save_path)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    base_optimizer = torch.optim.SGD
    # base_optimizer = functools.partial(SAM, base_optimizer=torch.optim.SGD, rho=args.rho, adaptive=args.adaptive)
    optimizer = Lookahead(model.parameters(), base_optimizer,
                          outer_lr=args.outer_lr, update_step=update_steps, lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    print('Update alphas : {}'.format(update_steps))
    scheduler = StepLR(optimizer.base_optimizer, args.learning_rate, args.epochs)

    losses, acces = [], []
    start_epoch = 0
    # algo, lr, start_epoch = 'sam', 0.1, 119
    # load_path = '../models/{}_lr{}/model-{}.pt'.format(algo, lr, start_epoch)
    # load_model(model, load_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for i, batch in enumerate(dataset.train):
            if optimizer.can_eval(i, len(dataset.train)):
                break

            enable_running_stats(model)
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            # optimizer.remove_noise()

            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            # def closure():
            #     disable_running_stats(model)
            #     smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            # optimizer.step()#closure=closure)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        if epoch == args.epochs * 3 / 10 - 1 or epoch == args.epochs * 6 / 10 - 1 or epoch == args.epochs * 8 / 10 - 1:
            save_model(model, save_path + 'model-{}.pt'.format(epoch))
        loss, acc = test_model(model, dataset.test, log, device, result=True)
        losses.append(loss)
        acces.append(acc)
    print(losses)
    print(acces)

    log.flush()
    save_model(model, save_path + 'model.pt'.format(args.learning_rate))
