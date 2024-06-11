import copy

import numpy as np
import torch

from extra_modules.extrap_lookahead import ExtrapLA
from methods.train_ExtrapLA import AlphaScheduler
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.param_utils import ParamOperator
from utility.utils import get_adv_grad


class ExtrapLATrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)

        self.optimizer = ExtrapLA(self.model.parameters(), self.optimizer, rho=0.05)
        self.alpha_scheduler = AlphaScheduler(self.optimizer, args.alpha_scheduler,
                                              args.start_alpha, args.alpha, args.epochs, self.log)
        self.LA = True
        self.SAM = False
        print("Using LA: {}, SAM: {}".format(self.LA, self.SAM))

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        self.alpha_scheduler(epoch, iteration, iters)

        def get_loss_func():
            predictions = self.model(inputs)
            loss = self.loss_func(predictions, targets)
            return predictions, loss

        # if self.SAM:
        #     enable_running_stats(self.model)
        #     predictions, loss = get_loss_func()
        #     loss.mean().backward()

        if self.LA:
            disable_running_stats(self.model)
            self.optimizer.first_step(zero_grad=False, extrap_key='last_p')#, alpha=3)

        # if self.SAM:
        #     self.optimizer.first_step(zero_grad=True, extrap_key='grad', alpha=2, adaptive=True)

        disable_running_stats(self.model)
        predictions, loss = get_loss_func()
        loss.mean().backward()
        self.optimizer.second_step(zero_grad=True)

        # if self.LA and not self.SAM:
        if True:
            with torch.no_grad():
                enable_running_stats(self.model)
                self.model(inputs)

        self.scheduler.step()
        return predictions, loss

    def record_grad_norm(self, epoch, iteration, inputs, targets):
        alphas = [-4, -2, -1, 0, 1, 2, 4]
        if epoch == 0 and iteration == 0:
            self.log.log("Test alphas: {}".format(alphas))

        if iteration == 1:
            def formatted_log(prefix, s_list):
                s_list = ["{:3.4f}".format(s) for s in s_list]
                s_list = " ".join(s_list)
                self.log.log(prefix + " " + s_list)

            losses = self.model_wrapper.loss_landscape(alphas, inputs, targets, self.loss_func)
            formatted_log("FORWARD losses: ", losses)

            # cos_similarity, grad_norms = self.model_wrapper.grad_sim(alphas, inputs, targets, self.loss_func)
            # grad_sims = self.model_wrapper.queue_grad_sim(alphas, inputs, targets, self.loss_func)
            # grad_sims = self.model_wrapper.adv_grad_sim(alphas, inputs, targets, self.loss_func)
            # formatted_log("Queue Similarity, cos : ", grad_sims)
            # formatted_log("Grad, norm : ", grad_norms)
            # formatted_log("adv grad sim : ", grad_sims)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--queue_size", default=1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--start_alpha", default=0.5, type=float, help="Rho parameter for SAM.")

    # for alpha scheduler
    parser.add_argument("--alpha", default=3, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha_scheduler", default='linear', type=str)
    parser.add_argument("--coeff", default=0.9, type=float, help="Rho parameter for SAM.")

    # for kl loss
    parser.add_argument("--kl_lambd", default=0.5, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--T", default=10, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--kl_start_epoch", default=160, type=int, help="Rho parameter for SAM.")

    args = parser.parse_args()

    trainer = ExtrapLATrainer(args)
    trainer.train()
