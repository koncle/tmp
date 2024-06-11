import torch

from extra_modules.lookahead import ExtrapLookahead
from methods.train_ExtrapLA import AlphaScheduler
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.param_utils import ParamOperator


class ExtrapLATrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = ExtrapLookahead(self.model.parameters(), self.optimizer,
                                         alpha=args.alpha, queue_size=args.queue_size,
                                         init_type=args.init_type)
        self.alpha_scheduler = AlphaScheduler(self.optimizer, args.alpha_scheduler,
                                              args.start_alpha, args.alpha, args.epochs, self.log)
        self.train_type = args.train_type
        print('training type: ', self.train_type)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        self.alpha_scheduler(epoch, iteration, iters)

        def get_loss_func(enable_stats=True):
            if enable_stats:
                enable_running_stats(self.model)
            else:
                disable_running_stats(self.model)
            predictions = self.model(inputs)
            loss = self.loss_func(predictions, targets)
            return predictions, loss

        adaptive = True
        sam_alpha = 0.05 if adaptive else 0.05
        if self.train_type == 'LA':
            predictions, loss = self.optimizer.LA_step(get_loss_func)
            with torch.no_grad():
                enable_running_stats(self.model)
                self.model(inputs)
        elif self.train_type == 'SAM':
            predictions, loss = self.optimizer.SAM_step(get_loss_func, alpha=sam_alpha, adaptive=adaptive)
        else:
            predictions, loss = self.optimizer.SAM_LA_step(get_loss_func, sam_alpha=sam_alpha, adaptive=adaptive)

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--queue_size", default=1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--start_alpha", default=0.5, type=float, help="Rho parameter for SAM.")

    # for alpha scheduler
    parser.add_argument("--alpha", default=3, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha_scheduler", default='linear', type=str)

    parser.add_argument("--train_type", default='SAM_LA', type=str)
    parser.add_argument("--init_type", default='queue', type=str)

    args = parser.parse_args()

    trainer = ExtrapLATrainer(args)
    trainer.train()
