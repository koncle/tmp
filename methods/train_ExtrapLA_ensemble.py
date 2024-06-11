import copy

import numpy as np
import torch

from methods.train_ExtrapLA import ConstantExtraLAWrapper, AlphaScheduler, ExtrapLATrainer
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats
from utility.param_utils import ParamOperator

class EnsembleLATrainer(ExtrapLATrainer):
    def __init__(self, args):
        super().__init__(args)
        self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        self.ensemble_coeffs = [0.5, 1, 2, 3]
        self.operator = ParamOperator()

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        self.alpha_scheduler(epoch, iteration, iters)

        cur_alpha = self.model_wrapper.alpha
        grads = []
        for coef in self.ensemble_coeffs:
            alpha = cur_alpha * coef
            predictions = self.model_wrapper(inputs, alpha=alpha)
            loss = self.loss_func(predictions, targets)
            self.model_wrapper.zero_grad()
            loss.mean().backward()
            grad = self.operator.extract_params_with_grad(self.model_wrapper.tmp_net)[1]
            grads.append(grad)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--algo", default='constant', type=str)
    parser.add_argument("--queue_size", default=1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--start_alpha", default=0.5, type=float, help="Rho parameter for SAM.")

    # for alpha scheduler
    parser.add_argument("--alpha", default=3, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha_scheduler", default='linear', type=str)

    # for kl loss
    parser.add_argument("--kl_lambd", default=0.5, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--T", default=5, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--kl_start_epoch", default=160, type=int, help="Rho parameter for SAM.")

    args = parser.parse_args()

    trainer = EnsembleLATrainer(args)
    trainer.train()
