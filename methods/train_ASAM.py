import torch
from methods.train_SAM import SAMTrainer
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", action='store_true', default=True, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--steps", default=1, type=int, help="Number of steps for SAM.")
    parser.add_argument("--accu_grad", default=False, type=bool,
                        help="True if you want to accumulate gradient for SAM.")
    args = parser.parse_args()

    if args.accu_grad:
        assert args.steps > 1

    trainer = SAMTrainer(args)
    trainer.train()
