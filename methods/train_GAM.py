import torch

from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from extra_modules.gam import GSAM, CosineScheduler, ProportionScheduler


class GSAMTrainer(Trainer):
    """
    https://github.com/juntang-zhuang/GSAM/blob/main/example/train.py
    """

    def __init__(self, args):
        super().__init__(args)
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = CosineScheduler(T_max=args.epochs * len(self.dataset.train),
                                         max_value=args.learning_rate, min_value=0.0, optimizer=base_optimizer)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=self.scheduler, max_lr=args.learning_rate, min_lr=0.0,
                                            max_value=args.rho_max, min_value=args.rho_min)
        self.optimizer = GSAM(params=self.model.parameters(), base_optimizer=base_optimizer,
                              model=self.model, gsam_alpha=args.alpha,
                              rho_scheduler=rho_scheduler, adaptive=args.adaptive)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, label_smoothing):
        def loss_fn(predictions, targets):
            return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()

        optimizer.set_closure(loss_fn, inputs, targets)
        predictions, loss = optimizer.step()

        self.scheduler.step()
        optimizer.update_rho_t()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho_max", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=int, help="Rho parameter for SAM.")
    args = parser.parse_args()

    trainer = GSAMTrainer(args)
    trainer.train()
