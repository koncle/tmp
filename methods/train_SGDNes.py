import argparse
import functools
import os
import random
import sys

from methods.train_SGD import Trainer;

sys.path.append("..")
from pathlib import Path

import numpy as np
import torch
import torchvision



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
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--dataset", default='cifar10', type=str, help="Dataset name.")

    # optimizer
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--scheduler", default='cos', type=str, help="Scheduler name.")
    parser.add_argument("--nesterov", default=True, action='store_false', help="Use Nesterov momentum.")

    # model training
    parser.add_argument("--model", default='WRN-28-10', type=str, help="Model name")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--seed", default=1, type=int, help="Random seed to be used.")
    parser.add_argument("--save-path", default='outputs/test', type=str, help="save path")

    parser.add_argument("--save-model", action='store_true', help="save model")
    
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    trainer = Trainer(args)
    trainer.train()
