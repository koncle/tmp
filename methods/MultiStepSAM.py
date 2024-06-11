from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


class SAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = SAM(self.model.parameters(), self.optimizer, rho=args.rho, adaptive=args.adaptive)
        self.steps = args.steps

    def train_step(self,  epoch, iteration, model, inputs, targets, optimizer, label_smoothing):
        # first ascend step
        for i in range(self.steps):
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True, save=i==0)

        # second descend step
        disable_running_stats(model)
        smooth_crossentropy(model(inputs), targets, smoothing=label_smoothing).mean().backward()
        optimizer.second_step(zero_grad=True)

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--steps", default=2, type=int, help="Number of steps for SAM.")
    args = parser.parse_args()

    trainer = SAMTrainer(args)
    trainer.train()
