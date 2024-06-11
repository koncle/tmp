import torch


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


class ExtrapLA(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=3, adaptive=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super(ExtrapLA, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.param_groups[0].update(defaults)
        self.defaults.update(self.base_optimizer.defaults)
        self.alpha = alpha
        self.queue_size = 2
        self.init()

    def init(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['last_p'] = Queue(self.queue_size)

    @torch.no_grad()
    def first_step(self, zero_grad=False, extrap_key='grad', alpha=None, adaptive=False, save=True):
        norm = self._grad_norm(adaptive) if extrap_key == 'grad' else self._key_norm(extrap_key)
        # save = False if extrap_key == 'grad' else True
        if alpha is None: alpha = self.alpha
        scale = alpha / (norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                # if not p.requires_grad: continue

                if save: self.state[p]['last_p'].put(p.data.clone())

                # if p.grad is None: continue

                grad = p.grad if extrap_key == 'grad' else (p - self.state[p][extrap_key].get(0))
                e_w =  (torch.pow(p, 2) if adaptive else 1.0) * grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if len(self.state[p]["last_p"]) == 0: continue
                p.data = self.state[p]["last_p"].get(-1).clone()  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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

    def _key_norm(self, key):
        p = self.param_groups[0]["params"][0]
        if len(self.state[p][key]) == 0:
            return 1

        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * (p - self.state[p][key].get(0))).norm(p=2).to(
                    shared_device)
                for group in self.param_groups for p in group["params"]
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
