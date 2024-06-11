import torch

from extra_modules.esam import ESAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer


class EfficientSAMTrainer(Trainer):
    """
    https://github.com/dydjw9/Efficient_SAM/blob/main/train.py
    #python -m torch.distributed.launch --nproc_per_node=2 train.py --beta 0.6 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --name R18C10_01 --batch_size 64 --dataset cifar10
    #python -m torch.distributed.launch --nproc_per_node=2 train.py --beta 0.6 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --name R18C100_01 --batch_size 64 --dataset cifar100

    python -m torch.distributed.launch --nproc_per_node=2 train.py --beta 0.5 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 1024  --arch wideresnet18 --name wideC10_b2048 --dataset cifar10
    python -m torch.distributed.launch --nproc_per_node=2 train.py --beta 0.5 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 1024  --arch wideresnet18 --name wideC100_b2048 --dataset cifar100
    """

    def __init__(self, args):
        super().__init__(args)
        base_optimizer = self.optimizer
        self.optimizer = ESAM(self.model.parameters(), base_optimizer, rho=args.rho, beta=args.beta, gamma=args.gamma,
                              adaptive=args.isASAM, nograd_cutoff=args.nograd_cutoff)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        def defined_backward(loss):
            loss.backward()

        paras = [inputs, targets, self.loss_func, model, defined_backward]
        optimizer.paras = paras
        optimizer.step()
        predictions, loss = optimizer.returnthings

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument('--isSAM', type=str, default='True')
    parser.add_argument('--isASAM', type=str, default='False')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument("--nograd_cutoff", default=0.05, type=float, help="Grad Cutoff.")

    args = parser.parse_args()

    trainer = EfficientSAMTrainer(args)
    trainer.train()
