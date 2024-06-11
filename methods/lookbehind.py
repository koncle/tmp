import copy

import torch
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.param_utils import ParamOperator


class LookbehindTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = SAM(self.model.parameters(), self.optimizer, rho=args.rho, adaptive=args.adaptive)
        self.steps = args.steps
        self.tmp_model = copy.deepcopy(self.model)
        self.tmp_opt = None # error
        self.operator = ParamOperator()

    def train_step(self,  epoch, iteration, model, inputs, targets, optimizer, iters):

        cur_param = self.operator.extract_params(model).clone()
        self.operator.put_parameters(self.tmp_model, cur_param, data=True)

        # first ascend step
        for i in range(self.steps):
            if i == 0:
                enable_running_stats(model)
            predictions = self.tmp_model(inputs)
            loss = self.loss_func(predictions, targets)
            loss.mean().backward()

            for tmp_param, param in zip(self.tmp_model.parameters(), model.parameters()):
                param.grad = tmp_param.grad.clone()
            self.optimizer.step()
            self.tmp_opt.step()

        # second forward-backward step
        disable_running_stats(model)
        self.loss_func(model(inputs), targets).mean().backward()
        optimizer.second_step(zero_grad=True)

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--steps", default=1, type=int, help="Number of steps for SAM.")
    args = parser.parse_args()

    trainer = LookbehindTrainer(args)
    trainer.train()
