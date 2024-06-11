import torch
from extra_modules.sam import SAM
from extra_modules.vasso import VASSO
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


class VASSOTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # self.load_model('/data/sam-main/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        # self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)

        self.optimizer = VASSO(self.model.parameters(), self.optimizer, rho=args.rho, theta=args.theta)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        # first ascend step
        enable_running_stats(model)
        predictions = model(inputs)
        loss = self.loss_func(predictions, targets)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(model)
        self.loss_func(model(inputs), targets).mean().backward()
        optimizer.second_step(zero_grad=True)

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument('--theta', type=float, default=0.9, help='Moving average for VASSO')
    args = parser.parse_args()

    trainer = VASSOTrainer(args)
    trainer.train()
