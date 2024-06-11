import torch

from extra_modules.esam import ESAM
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


class EfficientSAMTrainer(Trainer):
    """
    https://github.com/dydjw9/Efficient_SAM/blob/main/train.py
    """

    def __init__(self, args):
        super().__init__(args)
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9,
                                         weight_decay=args.weight_decay)
        self.optimizer = ESAM(self.model.parameters(), base_optimizer, rho=args.rho, beta=args.beta, gamma=args.gamma,
                              adaptive=args.isASAM, nograd_cutoff=args.nograd_cutoff)
        optimizer0 = self.optimizer.base_optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=args.epochs)
        self.epoch = 0

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, label_smoothing):
        # first forward-backward step

        def defined_backward(loss):
            loss.backward()

        paras = [inputs, targets, smooth_crossentropy, model, defined_backward]
        optimizer.paras = paras
        optimizer.step()
        predictions, loss = optimizer.returnthings

        if epoch != self.epoch:
            self.scheduler.step()
            self.epoch = epoch
        return loss, predictions


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
