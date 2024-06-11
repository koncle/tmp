import torch
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


class SAGMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = SAM(self.model.parameters(), self.optimizer, rho=args.rho, adaptive=args.adaptive)
        self.alpha = args.alpha

    def train_step(self,  epoch, iteration, model, inputs, targets, optimizer, iters):
        # first ascend step
        enable_running_stats(model)
        predictions = model(inputs)
        loss = self.loss_func(predictions, targets)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)#, SAGM_alpha=self.alpha)

        # second forward-backward step
        disable_running_stats(model)
        predictions = model(inputs)
        self.loss_func(predictions, targets).mean().backward()
        optimizer.second_step(zero_grad=True)

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.005, type=float, help="SAGM alpha")
    args = parser.parse_args()

    trainer = SAGMTrainer(args)
    trainer.train()
