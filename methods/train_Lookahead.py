import copy

import numpy as np
import torch

from extra_modules.lookahead import Lookahead
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.param_utils import ParamOperator

class LookaheadTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.steps = args.steps
        self.alpha = args.alpha
        num_gpus = torch.cuda.device_count()
        self.optimizer = Lookahead(self.model.parameters(), self.optimizer,
                          outer_lr=args.alpha, update_step=self.steps, lr=args.learning_rate * np.sqrt(num_gpus), #*1.5,
                          momentum=args.momentum, weight_decay=args.weight_decay)
        self.total_iters = len(self.dataset.train)
        self.iteration_limit = self.total_iters // self.steps * self.steps

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        if iteration >= self.iteration_limit:
            with torch.no_grad():
                predictions = model(inputs)
                loss = self.loss_func(predictions, targets)
        else:
            disable_running_stats(model)
            predictions = model(inputs)
            loss = self.loss_func(predictions, targets)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            self.scheduler.step()

            enable_running_stats(model)
            with torch.no_grad():
                predictions = model(inputs)
                loss = self.loss_func(predictions, targets)
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--steps", default=10, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Rho parameter for SAM.")

    args = parser.parse_args()

    trainer = LookaheadTrainer(args)
    trainer.train()
