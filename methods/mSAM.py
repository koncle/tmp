import torch

from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


class mSAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = SAM(self.model.parameters(), self.optimizer, rho=args.rho, adaptive=args.adaptive)
        self.m = args.m

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        splits = len(inputs) // self.m
        inputs_list = inputs.split(self.m)
        targets_list = targets.split(self.m)

        # first ascend step
        disable_running_stats(model)
        for i in range(splits):
            predictions = model(inputs_list[i])
            loss = self.loss_func(predictions, targets_list[i])
            loss.mean().backward()

            optimizer.first_step(zero_grad=True)

            # second descend step
            self.loss_func(model(inputs), targets).mean().backward()
            optimizer.save_grad(put_data_back=True)

        optimizer.second_step(zero_grad=False)

        with torch.no_grad():
            enable_running_stats(model)
            predictions = model(inputs)
            loss = self.loss_func(predictions, targets)
        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--m", default=64, type=int, help="Number of steps for SAM.")
    args = parser.parse_args()

    assert args.batch_size % args.m == 0
    trainer = mSAMTrainer(args)
    trainer.train()
