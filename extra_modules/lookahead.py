from collections import defaultdict

import torch
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from utility.bypass_bn import disable_running_stats


class Lookahead(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, outer_lr=0.5, update_step=5, **kwargs):
        defaults = dict(outer_lr=outer_lr, **kwargs)
        super(Lookahead, self).__init__(params, defaults)
        self.kwargs = kwargs
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.param_groups[0].update(defaults)
        self.defaults.update(self.base_optimizer.defaults)

        self.cur_step = 0
        self.update_step = update_step
        self.lr = kwargs['lr']
        self.outer_lr = outer_lr

        print('Reptile optimizer with lr : {}, outer LR : {}, Step : {}'.format(kwargs['lr'], outer_lr, update_step))
        self.avg_num = 0
        self.clear()

    def clear(self):
        self.cur_step = 0
        for group in self.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                self.state[p]["old_p"] = None
                self.state[p]["avg_p"] = None
                self.state[p]['last_grad'] = None
                # self.state[p]["grad"] = None

    def recover_weights(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]['old_p'].data.clone()

    def get_weight(self, key='old_p'):
        res = []
        for group in self.param_groups:
            for p in group["params"]:
                if key == 'p':
                    r = p.data.view(-1).clone()
                else:
                    r = self.state[p][key].view(-1).clone()
                r.requires_grad = False
                res.append(r)
        res = torch.cat(res)
        return res

    @torch.no_grad()
    def step(self, epoch=0, force=False, closure=None):
        self.cur_step += 1

        # fast weights update
        if self.cur_step == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()

        self.base_optimizer.step(closure)

        if self.cur_step >= self.update_step or force:
            # slow weights update
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    grad = self.state[p]["old_p"] - p
                    p.data = (self.state[p]["old_p"] - self.outer_lr * grad).data.clone()

            self.cur_step = 0

    @torch.no_grad()
    def stepXX(self, epoch=0, force=False, closure=None):
        self.cur_step += 1

        # fast weights update
        if self.cur_step == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.grad = -p.grad

        self.base_optimizer.step(closure)

        if self.cur_step >= self.update_step or force:
            # slow weights update
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    grad = - (self.state[p]["old_p"] - p)
                    p.data = (self.state[p]["old_p"] - self.outer_lr * grad).data.clone()

            self.cur_step = 0

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def can_eval(self, cur_iter, total_iters):
        return cur_iter == (total_iters // self.update_step) * self.update_step

    def update_bn(self, loader, iters, model, device):
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        with torch.no_grad():
            inner_loops = len(loader) if iters is None else iters
            loader = iter(loader)
            for i in range(inner_loops):
                try:
                    inputs, targets = (b.to(device) for b in next(loader))
                    model.step(inputs)
                except:
                    break

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)

    def add_noise(self, strength, noise_type='uniform'):
        for group in self.param_groups:
            for p in group["params"]:
                # filter-wise noise
                if noise_type == 'uniform':
                    noise = (torch.ones_like(p).uniform_() * 2. - 1.).to(p.device)
                elif noise_type == 'normal':
                    noise = torch.ones_like(p).normal_().to(p.device)
                else:
                    raise ValueError('Unkown --noise-type')

                noise_shape = noise.shape
                noise_norms = noise.view(noise_shape[0], -1).norm(p=2, dim=1) + 1.0e-6
                p_norms = p.view(noise_shape[0], -1).norm(p=2, dim=1) + 1.0e-6
                if len(noise_shape) == 4:
                    noise_norms = noise_norms[:, None, None, None]
                    p_norms = p_norms[:, None, None, None]
                elif len(noise_shape) == 2:
                    noise_norms = noise_norms[:, None]
                    p_norms = p_norms[:, None]
                noise = noise / noise_norms * p_norms.data * strength
                p.data.add_(noise)
                self.state[p]['noise'] = noise

    def remove_noise(self):
        for group in self.param_groups:
            for p in group["params"]:
                if 'noise' in self.state[p]:
                    p.data.sub_(self.state[p]['noise'])
                    del self.state[p]['noise']


class Queue():
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = []

    def put(self, x):
        if len(self.queue) >= self.maxsize:
            self.queue.pop(0)
        self.queue.append(x)

    def get(self, idx):
        return self.queue[idx]

    def get_last(self):
        return self.queue[0]

    def get_avg(self):
        return sum(self.queue) / len(self.queue)

    def __len__(self):
        return len(self.queue)


class ExtrapLookahead(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, alpha=1, queue_size=1, init_type='queue', **kwargs):
        defaults = dict(alpha=alpha, queue_size=queue_size, **kwargs)
        super(ExtrapLookahead, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.param_groups[0].update(defaults)
        self.defaults.update(self.base_optimizer.defaults)

        self.cur_step = 0
        self.alpha = alpha
        self.queue_size = queue_size + 1

        self.init_type = init_type
        self.fixed_freq = 10
        self.exp_factor = 0.9
        self.clear()

    def clear(self):
        self.cur_step = 0
        for group in self.param_groups:
            for p in group["params"]:
                if self.init_type == 'exp':
                    self.state[p]["exp_param"] = None
                elif self.init_type == 'queue':
                    self.state[p]['param_queue'] = Queue(self.queue_size + 1)
                elif self.init_type == 'fixed':
                    self.state[p]['fixed_param'] = None
                else:
                    raise NotImplementedError

    def recover_weights(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]['old_p'].data.clone()

    def setup_init_param(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue

                if self.init_type == 'exp':
                    exp_factor = self.exp_factor
                    self.state[p]["exp_param"] = (
                            p.data.clone() * (1 - exp_factor) + exp_factor * self.state[p]["exp_param"])
                elif self.init_type == 'queue':
                    self.state[p]['param_queue'].put(p.data.clone())
                elif self.init_type == 'fixed':
                    if self.cur_step % self.fixed_freq == 0:
                        self.state[p]['fixed_param'] = p.data.clone()

    @torch.no_grad()
    def grad_extrapolate(self, alpha, adaptive, save=True):
        grad_norm = self._grad_norm(adaptive)
        scale = (alpha / (grad_norm + 1e-12))
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                if save: self.state[p]["old_p"] = p.data.clone()

                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                grads.append(e_w.view(-1))
        grads =torch.cat(grads)
        return grads

    @torch.no_grad()
    def extrapolate(self, alpha, save=True):
        dirs = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                self.state[p]["old_p"] = p.data.clone()

                if self.init_type == 'exp':
                    dir = p - self.state[p]["exp_param"]
                elif self.init_type == 'queue':
                    dir = p - self.state[p]['param_queue'].get(0)
                elif self.init_type == 'fixed':
                    dir = p - self.state[p]['fixed_param']
                else:
                    raise NotImplementedError
                dirs.append(dir.view(-1))
        dirs = torch.cat(dirs)
        norm = dirs.norm()
        # print(norm)

        scale = alpha / (norm + 1e-12)  # norm = 0-6
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                if save: self.state[p]["old_p"] = p.data.clone()

                if self.init_type == 'exp':
                    p.data = p.data + (p - self.state[p]["exp_param"]) * scale
                elif self.init_type == 'queue':
                    p.data = p.data + (p - self.state[p]['param_queue'].get(0)) * scale
                elif self.init_type == 'fixed':
                    p.data = p.data + (p - self.state[p]['fixed_param']) * scale
                else:
                    raise NotImplementedError
        return dirs

    @torch.no_grad()
    def step(self, epoch=0, force=False, closure=None):

        self.setup_init_param()

        self.base_optimizer.step(closure)

        if self.cur_step >= self.update_step or force:
            # slow weights update
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    grad = self.state[p]["old_p"] - p
                    p.data = (self.state[p]["old_p"] - self.outer_lr * grad).data.clone()

        self.cur_step += 1
        # PYTHONPATH=. python methods/train_ExtrapSAMOpt.py --learning_rate 0.05 --gpu 4,5,6,7 --model resnet50 --alpha 1 --train_type LA --queue_size 5 --alpha_scheduler constant

    def second_step(self, closure):
        pred, loss = closure(enable_stats=False)
        self.zero_grad()
        loss.mean().backward()
        self.recover_weights()
        self.base_optimizer.step()
        self.cur_step += 1
        return pred, loss

    def LA_step(self, closure, alpha=None):
        alpha = alpha if alpha is not None else self.alpha
        self.setup_init_param()
        self.extrapolate(alpha)
        pred, loss = self.second_step(closure)
        return pred, loss

    def SAM_step(self, closure, alpha=None, adaptive=False):
        alpha = alpha if alpha is not None else self.alpha
        pred, loss = closure(enable_stats=True)
        self.zero_grad()
        loss.mean().backward()
        self.grad_extrapolate(alpha, adaptive)
        pred, loss = self.second_step(closure)
        return pred, loss

    def SAM_LA_step(self, closure, sam_alpha=None, alpha=None, adaptive=False):
        sam_alpha = sam_alpha if sam_alpha is not None else self.alpha
        alpha = alpha if alpha is not None else self.alpha

        self.setup_init_param()

        pred, loss = closure(enable_stats=True)
        self.zero_grad()
        loss.mean().backward()
        sam_dir = self.grad_extrapolate(sam_alpha, adaptive)

        la_dir = self.extrapolate(alpha, save=False)
        # print(torch.cosine_similarity(sam_dir, la_dir, dim=0))

        pred, loss = self.second_step(closure)

        return pred, loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def _grad_norm(self, adaptive):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
