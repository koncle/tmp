import copy

import numpy as np
import torch

from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.param_utils import ParamOperator
from utility.utils import get_adv_grad


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


class ExtrapLAWrapper():
    def __init__(self, net, opt, args):
        # hyperparameters
        self.alpha = args.alpha
        self.scale_factor = 1
        self.beta = 1  # 0.5 / (0.1 * 5)

        self.steps = 0
        self.net = net
        self.opt = opt

        self.maxsize = args.queue_size + 1
        self.param_queue = Queue(self.maxsize)
        self.operator = ParamOperator()

        self.init_lr = self.opt.param_groups[0]['lr']
        self.noise = False

    def normalize_dir(self, dir, cur_p, adaptive=False):
        if adaptive:
            dir = (torch.pow(cur_p, 2)) * dir
        norm = dir.norm()
        return dir / (norm + 1e-12)

    def get_LA_dir(self, adaptive=False, rand=False):
        cur_param = self.operator.extract_params(self.net)
        param = self.get_init_param()
        dir = cur_param - param
        dir = self.normalize_dir(dir, cur_param, adaptive=adaptive)
        if rand:
            noise = torch.randn_like(dir)
            noise = noise / noise.norm() * 0.1
            dir = dir + noise
        return cur_param, dir

    def get_SAM_dir(self, adaptive):
        cur_param, dir = self.operator.extract_params_with_grad(self.net)
        dir = self.normalize_dir(dir, cur_param, adaptive=adaptive)
        return cur_param, dir

    @torch.no_grad()
    def _extrap_step(self, alpha, do_LA=True, rand=False, update=True):
        if do_LA:
            cur_param, dir = self.get_LA_dir(adaptive=False)
        else:
            cur_param, dir = self.get_SAM_dir(adaptive=True)

        if rand:
            noise = torch.randn_like(dir)
            noise = noise / noise.norm() * 0.1
            dir = dir + noise

        if update:
            new_param = cur_param + dir * alpha
            self.operator.put_parameters(self.net, new_param, data=True)
        return cur_param, dir

    def _insert_grad(self):
        grads = []
        for tmp_p, cur_p in zip(self.tmp_net.parameters(), self.net.parameters()):
            if tmp_p.grad is not None:
                cur_p.grad = tmp_p.grad
                grads.append(tmp_p.grad.view(-1))
        grads = torch.cat(grads, dim=0)
        return grads

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, alpha=None, do_forward=True, **kwargs):
        self.set_init_param()
        alpha = alpha if alpha is not None else self.alpha
        self.prev_param, dir = self._extrap_step(alpha)
        disable_running_stats(self.net)
        ret = self.net(*args, **kwargs)

        if do_forward:
            with torch.no_grad():
                enable_running_stats(self.net)
                self.net(*args, **kwargs)
        return ret

    def zero_grad(self):
        self.opt.zero_grad()

    def set_init_param(self):
        param = self.operator.extract_params(self.net).detach().clone()
        self.param_queue.put(param)

    def get_init_param(self):
        return self.param_queue.get(0)

    def step(self):
        # grad = self._insert_grad()
        self.operator.put_parameters(self.net, self.prev_param, data=True)
        self.opt.step()
        self.steps += 1

    def do_analyze(self, inputs, targets, loss_func):
        alphas = [-1, 0, 1, 2, 4, 8, 16]
        iters = 391 // 4

        loss_freq = 1
        self.loss_landscape(alphas, loss_freq * iters, inputs, targets, loss_func)

        grad_freq = 5
        self.grad_sim(alphas, grad_freq * iters, inputs, targets, loss_func)

        # adv_grad_freq = 1
        # self.adv_grad_freq(adv_grad_freq * iters, inputs, targets, loss_func)

    @torch.no_grad()
    def loss_landscape(self, alphas, inputs, targets, loss_func):
        # line search
        alpha_lower_bound = 0.1
        alpha_upper_bound = 10

        def get_loss(alpha):
            self._extrap_step(alpha)
            y = self.tmp_net(inputs)
            loss = loss_func(y, targets).mean().item()
            return loss

        lower_bound_loss = get_loss(alpha_lower_bound)
        mid_bound_loss = get_loss((alpha_lower_bound + alpha_upper_bound) / 2)
        while True:
            if lower_bound_loss > mid_bound_loss:
                alpha_upper_bound = (alpha_lower_bound + alpha_upper_bound) / 2
                upper_bound_loss = mid_bound_loss
                mid_bound_loss = get_loss((alpha_lower_bound + alpha_upper_bound) / 2)
            else:
                alpha_lower_bound = (alpha_lower_bound + alpha_upper_bound) / 2
                lower_bound_loss = mid_bound_loss
                mid_bound_loss = get_loss((alpha_lower_bound + alpha_upper_bound) / 2)
            if alpha_upper_bound - alpha_lower_bound < 0.1:
                break
        print("searched alpha : ", alpha_lower_bound)


    def adv_grad_sim(self, alphas, inputs, targets, loss_func):
        alphas = [0, 0.5, 1, 2, 3]
        grads = []
        original_param = self.operator.extract_params(self.tmp_net)
        for alpha in alphas:
            self.operator.put_parameters(self.tmp_net, original_param.clone(), data=True)

            self._extrap_step(alpha)
            y1 = self.tmp_net(inputs)
            loss1 = loss_func(y1, targets).mean()
            self.tmp_opt.zero_grad()
            loss1.backward()
            adv_grad = get_adv_grad(self.tmp_opt, adaptive=True)

            assign_adv_grad(self.tmp_net, adv_grad, rho=2)
            y2 = self.tmp_net(inputs)
            loss2 = loss_func(y2, targets).mean()
            self.tmp_opt.zero_grad()
            loss2.backward()
            grad = self.operator.extract_params_with_grad(self.tmp_net)[1]

            grads.append(grad.view(-1))
        # get sims
        sims = [torch.cosine_similarity(g, grads[0], dim=0).item() for g in grads[1:]]
        return sims

    def grad_sim(self, alphas, inputs, targets, loss_func):
        if True:
            y = self.net(inputs)
            loss = loss_func(y, targets).mean()
            self.opt.zero_grad()
            loss.backward()
            init_grad = self.operator.extract_params_with_grad(self.net)[1]

            grads = []
            for s in alphas:
                self._extrap_step(s)
                y = self.tmp_net(inputs)
                loss = loss_func(y, targets).mean()
                self.tmp_opt.zero_grad()
                loss.backward()
                grad = self.operator.extract_params_with_grad(self.tmp_net)[1]
                grads.append(grad)

            cos_similarity = [torch.cosine_similarity(g, init_grad, dim=0).item() for g in grads]
            grad_norms = [g.norm().item() for g in grads]
            return cos_similarity, grad_norms

    def queue_grad_sim(self, alphas, inputs, targets, loss_func):
        if True:
            y = self.net(inputs)
            loss = loss_func(y, targets).mean()
            self.opt.zero_grad()
            loss.backward()
            init_param, init_grad = self.operator.extract_params_with_grad(self.net)

            grads = []
            for param in self.param_queue.queue:
                grad = torch.cosine_similarity(param - init_param, init_grad, dim=0).item()
                grads.append(grad)
            return grads

    def adv_grad_freq(self, freq, inputs, targets, loss_func):
        if self.steps % freq == 0:
            y = self.net(inputs)
            loss = loss_func(y, targets).mean()
            self.opt.zero_grad()
            loss.backward()
            cur_w, init_grad = self.operator.extract_params_with_grad(self.net)
            init_grad = -init_grad

            angles = []
            for i in range(len(self.param_queue)):
                w = self.param_queue.get(len(self.param_queue) - i - 1)
                cur_grad = w - cur_w
                angle = torch.cosine_similarity(cur_grad, init_grad, dim=0).item()
                angles.append(angle)
            print("ANGLES : ", angles)
            return angles


class ExpExtrapLAWrapper(ExtrapLAWrapper):
    def __init__(self, net, opt, args):
        super().__init__(net, opt, args)
        self.ema = 0
        self.coeff = args.coeff
        # swa_utils
        self.avg_func = lambda averaged_param, current_param, num_averaged:\
            averaged_param + (current_param - averaged_param) / (num_averaged + 1)
        # self.avg_func = lambda averaged_param, current_param, num_averaged:\
        #     averaged_param * self.coeff + current_param * (1 - self.coeff)
        self.num = 0

    def set_init_param(self):
        self.num += 1
        param = self.operator.extract_params(self.net).detach().clone()
        self.ema = self.avg_func(self.ema, param, self.num)

    def get_init_param(self):
        return self.ema

def assign_adv_grad(tmp_net, adv_grad, rho=2):
    for param in tmp_net.parameters():
        if param.grad is not None:
            param.data = param.data + rho * adv_grad[:param.grad.data.numel()].view(param.grad.data.shape)
            adv_grad = adv_grad[param.grad.data.numel():]


class AlphaScheduler():
    def __init__(self, model_wrapper, schedule_type, start_alpha, end_alpha, total_epochs, log):
        self.model_wrapper = model_wrapper
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.total_epochs = total_epochs
        if self.start_alpha >= self.end_alpha:
            log.log("error, start_alpha: {} >= end_alpha: {}, using constant scheduler with alpha {}"
                    .format(start_alpha, end_alpha, end_alpha))
            schedule_type = 'constant'
        self.schedule_type = schedule_type
        self.model_wrapper.alpha = end_alpha
        self.log = log
        self.log_info()

    def log_info(self):
        if self.schedule_type == 'constant':
            self.log.log('using constant alpha {}'.format(self.end_alpha))
        elif self.schedule_type == 'log':
            self.log.log('using log alpha from {} to {}'.format(self.start_alpha, self.end_alpha))
        elif self.schedule_type == 'linear':
            self.log.log('using linear alpha from {} to {}'.format(self.start_alpha, self.end_alpha))
        else:
            raise NotImplementedError

    def __call__(self, epoch, iteration, iters):
        if self.schedule_type == 'constant':
            pass
        elif self.schedule_type == 'log':
            self.schedule_alpha_log(epoch, iteration)
        elif self.schedule_type == 'linear':
            self.schedule_alpha_linear(epoch, iteration)
        elif self.schedule_type == 'linear_prob':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def schedule_alpha_log(self, epoch, iteration):
        # increasingly increase alpha
        total_epochs = 200
        k = 5  # the smaller, the sharper
        if epoch < total_epochs:
            # increase and plateau in the early phase using log
            self.model_wrapper.alpha = self.start_alpha + (
                    (2 / (1 + np.e ** (-epoch / k)) - 1) * (self.end_alpha - self.start_alpha))
        else:
            self.model_wrapper.alpha = self.end_alpha

    def schedule_alpha_linear(self, epoch, iteration):
        # increasingly increase alpha
        total_epochs = 200
        if epoch < total_epochs:
            # increase and plateau in the early phase using log
            self.model_wrapper.alpha = self.start_alpha + (epoch / total_epochs) * (self.end_alpha - self.start_alpha)
        else:
            self.model_wrapper.alpha = self.end_alpha

class ExtrapLATrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        self.model_wrapper = ExtrapLAWrapper(self.model, self.optimizer, args)
        self.alpha_scheduler = AlphaScheduler(self.model_wrapper, args.alpha_scheduler,
                                              args.start_alpha, args.alpha, args.epochs, self.log)


    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        self.alpha_scheduler(epoch, iteration, iters)

        predictions = self.model_wrapper(inputs)
        loss = self.loss_func(predictions, targets)
        self.model_wrapper.zero_grad()
        loss.mean().backward()
        self.model_wrapper.step()

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
    parser.add_argument("--T", default=5, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--kl_start_epoch", default=160, type=int, help="Rho parameter for SAM.")

    args = parser.parse_args()

    trainer = ExtrapLATrainer(args)
    trainer.train()
