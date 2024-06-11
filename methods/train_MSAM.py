import torch

from extra_modules.msam import MSAM
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer, get_scheduler
from utility.bypass_bn import enable_running_stats, disable_running_stats


class MSAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        self.optimizer = MSAM(self.model.parameters(), lr=args.learning_rate, rho=0.3, weight_decay=1e-3)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer, args.epochs, len(self.dataset.train))

    def pre_epoch(self):
        self.optimizer.move_up_to_momentumAscent()

    def after_epoch(self):
        self.optimizer.move_back_from_momentumAscent()

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        # first forward-backward step
        enable_running_stats(model)
        predictions = model(inputs)
        loss = self.loss_func(predictions, targets)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--noise_scale", default=0.1, type=float, help="Noise scale for RSAM.")
    args = parser.parse_args()

    trainer = MSAMTrainer(args)
    trainer.train()
