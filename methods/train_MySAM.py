import copy

import torch

from extra_modules.sam import SAM
from methods.train_ExtrapLA import AlphaScheduler
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.param_utils import ParamOperator


def forward_opt_backward(model, opt, inputs, targets, loss_func):
    predictions = model(inputs)
    loss = loss_func(predictions, targets)
    opt.zero_grad()
    loss.mean().backward()
    return predictions, loss

def grad_norm(optimizer, adaptive):
    param_groups = optimizer.param_groups
    norm = torch.norm(
        torch.stack([
            ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2)
            for group in param_groups for p in group["params"]
            if p.grad is not None
        ]),
        p=2
    )
    return norm


def get_adv_grad(optimizer, adaptive):
    norm = grad_norm(optimizer, adaptive=adaptive)
    param_groups = optimizer.param_groups
    grads = []
    for group in param_groups:
        scale = 1 / (norm + 1e-12)
        for p in group["params"]:
            if p.grad is None: continue
            e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale
            grads.append(e_w.view(-1))
    adv_grad = torch.cat(grads)
    return adv_grad


class ExtraSAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, alpha=5):
        super().__init__(params, base_optimizer, rho, adaptive)
        self.alpha = alpha
        self.lam = 0.5
        self.behind_alpha = 0.1
        self.alpha = 4
        self.beta = 0.5

    @torch.no_grad()
    def extrapolate(self):
        for group in self.param_groups:
            for p in group["params"]:
                if 'old_p' in self.state[p]:
                    p.data = p.data + (p - self.state[p]["old_p"]) * self.alpha
                self.state[p]["old_p"] = p.data.clone()

    @torch.no_grad()
    def first_step(self, zero_grad=False, save_p=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                if save_p:
                    self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

class MySAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        self.optimizer = ExtraSAM(self.model.parameters(), self.optimizer, rho=args.rho, adaptive=args.adaptive,
                                  alpha=args.alpha)
        self.do_extrap = True
        self.alpha_scheduler = AlphaScheduler(self.optimizer, 'linear',
                                              0.5, args.alpha, args.epochs, self.log)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        self.alpha_scheduler(epoch, iteration, iters)

        if self.do_extrap:
            self.optimizer.extrapolate()
        if epoch > 160:
            predictions, loss = forward_opt_backward(model, optimizer, inputs, targets, self.loss_func)
            self.optimizer.first_step(zero_grad=True, save_p=not self.do_extrap)
        forward_opt_backward(model, optimizer, inputs, targets, self.loss_func)
        self.optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            predictions = model(inputs)
            loss = self.loss_func(predictions, targets)
            self.scheduler.step()
        return predictions, loss


"""
        # Lookahead
        self.operator.put_parameters(self.tmp_model, self.ahead_param, data=True)
        forward_opt_backward(self.tmp_model, self.tmp_opt, inputs, targets)
        ahead_grad = self.operator.extract_params_with_grad(self.tmp_model)[1]

        # Lookbehind
        # first ascend step
        self.operator.put_parameters(self.tmp_model, self.behind_param, data=True)
        forward_opt_backward(self.tmp_model, self.tmp_opt, inputs, targets)
        behind_grad = get_adv_grad(self.tmp_opt, self.adaptive)

        # second descend step
        new_param = prev_param + behind_grad * self.rho
        self.operator.put_parameters(self.tmp_model, new_param, data=True)
        forward_opt_backward(self.tmp_model, self.tmp_opt, inputs, targets)
        update_grad = self.operator.extract_params_with_grad(self.tmp_model)[1]

        self.operator.put_grads(model, update_grad)
        optimizer.step()

        cur_param = self.operator.extract_params(model)
        ahead_param = (self.ahead_param * self.beta + 
                       (1 - self.beta) * (prev_param + (cur_param - prev_param) * self.alpha))
        self.behind_param = cur_param
"""

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=3.0, type=float, help="Alpha parameter for SAM.")
    args = parser.parse_args()

    trainer = MySAMTrainer(args)
    trainer.train()
