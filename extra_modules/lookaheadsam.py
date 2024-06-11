import torch

from utility.bypass_bn import enable_running_stats, disable_running_stats


class LookaheadSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, coef=0.9, alpha=3):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super(LookaheadSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.param_groups[0].update(defaults)
        self.defaults.update(self.base_optimizer.defaults)

        self.saved = False
        self.coef = coef
        self.alpha = alpha
        self.steps = 0
        print('LookaheadSAM, coef: {}, alpha: {}'.format(self.coef, self.alpha))

    @torch.no_grad()
    def extrapolate(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                if 'ema_e_w' in self.state[p]:
                    dir = self.state[p]["ema_e_w"] / self.cur_norm
                    p.data.sub_(dir * self.alpha)
        self.saved = True

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        if self.steps % 5 == 0:
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad
                    self.state[p]["e_w"] = p.grad.clone()
                    p.add_(e_w * scale.to(p))  # climb to the local maximum "w + e(w)"

            if zero_grad: self.zero_grad()
            return True
        else:
            self.second_step(zero_grad=zero_grad)
            return False

    def obtain_grad(self, key):
        l = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if key == 'grad':
                    l.append(p.grad.view(-1))
                elif key == 'e_w':
                    l.append(self.state[p][key].view(-1))
        return torch.cat(l)

    @torch.no_grad()
    def second_step(self, zero_grad=False):

        if self.steps % 5 == 0:
            gs = self.obtain_grad('grad')
            g = self.obtain_grad('e_w')
            scalar = torch.dot(gs, g) / (torch.dot(g, g) + 1e-6)
            gs_list, gv_list, gh_list = [], [], []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

                    # new line
                    # proj p.grad to e_w without inplace operation
                    g = self.state[p]["e_w"]
                    g_s = p.grad
                    g_h = scalar * g
                    g_v = g_s - g_h

                    gs_list.append(g_s.view(-1))
                    gv_list.append(g_v.view(-1))
                    gh_list.append(g_h.view(-1))

                    self.state[p]['gv'] = g_v

            gv = torch.cat(gv_list)
            gh = torch.cat(gh_list)
            gs = torch.cat(gs_list)

            self.gv_norm = torch.norm(gv, p=2)

            # if hasattr(self, 'gs'):
            #     prev_gs = self.gs
            #     prev_gh = self.gh
            #     prev_gv = self.gv
            #     gs_sim = torch.cosine_similarity(gs, prev_gs, dim=0)
            #     gh_sim = torch.cosine_similarity(gh, prev_gh, dim=0)
            #     gv_sim = torch.cosine_similarity(gv, prev_gv, dim=0)
            #     gs_l2_diff = torch.norm(gs - prev_gs, p=2)
            #     gh_l2_diff = torch.norm(gh - prev_gh, p=2)
            #     gv_l2_diff = torch.norm(gv - prev_gv, p=2)
            #     print('gs_sim: {}, gh_sim: {}, gv_sim: {}, gs_l2_diff: {}, gh_l2_diff: {}, gv_l2_diff: {}'.format(
            #         gs_sim, gh_sim, gv_sim, gs_l2_diff, gh_l2_diff, gv_l2_diff
            #     ))
            #
            # self.gs = gs
            # self.gh = gh
            # self.gv = gv
        else:
            g = self.obtain_grad('grad')
            g_norm = torch.norm(g, p=2)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.grad = 0.2 * self.state[p]["gv"] / self.gv_norm * g_norm + p.grad

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        self.saved = False
        if zero_grad: self.zero_grad()
        self.steps +=  1

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# see https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
# see https://arxiv.org/abs/2203.02714
class LookSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            alpha: float = 0.2,
            k: int = 5,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            alpha=alpha,
            k=k,
        )
        super(LookSAM, self).__init__(params, defaults)

        # init momentum buffer to zeros
        # needed to make implementation of first ascent step cleaner (before SGD.step() was ever called)
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

        for group in self.param_groups:
            group["inverse_norm_buffer"] = [0,]

        self.k = k
        self.iteration_counter = k

    @torch.no_grad()
    def move_up(self):
        norm = self._grad_norm()

        for group in self.param_groups:
            rho = group['rho']
            scale = rho / (norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                e_w = p.grad * scale

                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()
                self.state[p]["grad"] = p.grad.detach().clone()

    @torch.no_grad()
    def move_back(self):
        scalar_product = self._scalar_product()
        grad_norm = self._calc_norm(get_var_from_p = lambda x: self.state[x]["grad"])
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.sub_(self.state[p]["e_w"])

                #not normalized yet! (see next loop)
                self.state[p]["normalized_orthogonal_gradient"] = (p.grad - scalar_product.to(p)/(grad_norm.to(p)**2+1e-12) * self.state[p]["grad"]).detach().clone()

        #normalize orthogonal gradient
        grad_v_norm = self._calc_norm(get_var_from_p = lambda x: self.state[x]["normalized_orthogonal_gradient"])
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["normalized_orthogonal_gradient"].div_(grad_v_norm.to(p))

    @torch.no_grad()
    def update_gradient(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.grad.add_(self.state[p]["normalized_orthogonal_gradient"], alpha = group['alpha'] * grad_norm.to(p))

    def step(self, model, inputs, targets, loss_func):
        predictions = model(inputs)

        loss = loss_func(predictions, targets)
        loss.mean().backward()

        if self.iteration_counter == self.k:
            self.iteration_counter = 0

            self.move_up()
            self.zero_grad()

            # second forward-backward step
            enable_running_stats(model)
            loss_intermediate = loss_func(model(inputs), targets)
            disable_running_stats(model)
            loss_intermediate.mean().backward()

            self.move_back()
        else:
            self.update_gradient()


        self.SGD_step()
        self.zero_grad()

        self.iteration_counter += 1

        return loss, predictions

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def _calc_norm(self, get_var_from_p):
        """generic norm calculator (could be use form _grad_norm too)

        Args:
            get_var_from_p (callabe): getter for variable to compute norm of
        """
        norm = torch.norm(
                    torch.stack([
                        get_var_from_p(p).norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def _scalar_product(self):
        dot_prod = torch.sum(
                    torch.stack([
                        (p.grad*self.state[p]["grad"]).sum() for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
               )
        return dot_prod



    @torch.no_grad()
    def SGD_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    nesterov=nesterov,
                    )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

def sgd(params,
        d_p_list,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0: #@TODO decouple weight decay from momentum?
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)

