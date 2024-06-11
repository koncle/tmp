import torch
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


class RSAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = SAM(self.model.parameters(), self.optimizer, rho=args.rho, adaptive=args.adaptive)
        self.noise_scale = args.noise_scale

    def add_noise(self, model):
        noises = []
        for param in model.parameters():
            # add filter-wise noise
            noise = torch.randn_like(param)
            noise = self.noise_scale * noise / noise.norm(p=2) * param.norm()
            noises.append(noise)
            param.data.add_(noise)
        return noises

    def remove_noise(self, model, noises):
        for param, noise in zip(model.parameters(), noises):
            param.data.sub_(noise)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        noises = self.add_noise(model)

        # first forward-backward step
        enable_running_stats(model)
        predictions = model(inputs)
        loss = self.loss_func(predictions, targets)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(model)
        predictions = model(inputs)
        self.loss_func(predictions, targets).mean().backward()
        optimizer.second_step(zero_grad=True)

        self.remove_noise(model, noises)

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--noise_scale", default=0.1, type=float, help="Noise scale for RSAM.")
    args = parser.parse_args()

    trainer = RSAMTrainer(args)
    trainer.train()
