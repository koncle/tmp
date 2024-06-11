import torch

from extra_modules.lookaheadsam import LookaheadSAM, LookSAM
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer, get_scheduler
from utility.bypass_bn import enable_running_stats, disable_running_stats


class LookSAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # self.load_model('/data/zj/PycharmProjects/sam/methods/outputs/test/ckpt-lr0.1-E159.pt', epoch=159)
        self.optimizer = LookaheadSAM(self.model.parameters(), self.optimizer, rho=args.rho,
                                      adaptive=args.adaptive, coef=0.9, alpha=5)
        self.do_extrapolation = True
        print('LookaheadSAM, rho: {}, adaptive: {}, do_extrapolation: {}'.format(
            args.rho, args.adaptive, self.do_extrapolation))

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        # if self.do_extrapolation:
        #     optimizer.extrapolate()
        # first ascend step
        enable_running_stats(model)
        # loss, predictions = self.optimizer.step(model, inputs, targets, self.loss_func)
        predictions = model(inputs)
        loss = self.loss_func(predictions, targets)
        loss.mean().backward()
        if optimizer.first_step(zero_grad=True):
            # second forward-backward step
            disable_running_stats(model)
            self.loss_func(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

        self.scheduler.step()
        return predictions, loss



if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    args = parser.parse_args()


    trainer = LookSAMTrainer(args)
    trainer.train()
