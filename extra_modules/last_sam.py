import torch

import torch
from collections import defaultdict
from typing import Callable, List, Optional, Union



class LastSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super(LastSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.param_groups[0].update(defaults)
        self.defaults.update(self.base_optimizer.defaults)
        self.steps = 0

    @torch.no_grad()
    def ascend(self, zero_grad=False):
        if self.steps > 0:
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = (group["rho"] / (grad_norm + 1e-12))
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def descend(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if self.steps == 0:
                    self.state[p]['last_grad'] = p.grad.clone()
                else:
                    self.state[p]['last_grad'] = p.grad * 0.1 + 0.9 * self.state[p]['last_grad']
                    p.data = self.state[p]["old_p"].clone()  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()
        self.steps += 1

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        (self.state[p]['last_grad']).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if self.state[p]['last_grad'] is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class DPSAT:
    def __init__(self, optimizer, model, rho=0.5):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascend(self):
        rho = self.rho
        grads = []
        self.grads_ascent = []

        for n, p in self.model.named_parameters():
            prev_grad = self.state[p].get("prev_grad")
            if prev_grad is None:
                prev_grad = torch.zeros_like(p)
                self.state[p]["prev_grad"] = prev_grad
            self.grads_ascent.append(self.state[p]["prev_grad"].flatten())
            grads.append(torch.norm(prev_grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2)
        if grad_norm != 0:
            grad_norm = grad_norm + 1.e-16
        else:
            grad_norm.fill_(1)

        for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
            eps = self.state[p].get("prev_grad")
            self.state[p]["eps"] = eps

            eps.mul_(rho / grad_norm)
            p.add_(eps)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def descend(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if True:
            self.grads_descent = []
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
                prev_grad = p.grad.clone().detach()
                self.state[p]["prev_grad"] = prev_grad
                self.grads_descent.append(self.state[p]["prev_grad"].flatten())

            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()


class DPSATMomentum:
    def __init__(self, optimizer, model, rho=0.5):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        rho = self.rho
        grads = []
        for n, p in self.model.named_parameters():
            prev_p = self.state[p].get("prev")
            if prev_p is None:
                prev_p = torch.clone(p).detach()
                self.state[p]["prev"] = prev_p
            grads.append(torch.norm(prev_p - p, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2)
        if grad_norm != 0:
            grad_norm = grad_norm + 1.e-16
        else:
            grad_norm.fill_(1)

        for n, p in self.model.named_parameters():
            prev_p = self.state[p].get("prev")
            eps = prev_p - p
            self.state[p]["eps"] = eps

            eps.mul_(rho / grad_norm)
            p.add_(eps)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
            prev_p = torch.clone(p).detach()
            self.state[p]["prev"] = prev_p
        self.optimizer.step()
        self.optimizer.zero_grad()

