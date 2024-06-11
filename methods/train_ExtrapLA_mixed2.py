import torch

from methods.train_ExtrapLA import AlphaScheduler
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.param_utils import ParamOperator


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
        # self.param_queue = Queue(self.maxsize)
        self.param_ema = 0
        self.update_type = 'last'
        self.operator = ParamOperator()

        self.init_lr = self.opt.param_groups[0]['lr']
        self.noise = False

        self.step = 0
        self.last_param = None

    def zero_grad(self):
        self.opt.zero_grad()

    def set_init_param(self):
        param = self.operator.extract_params(self.net).detach().clone()
        if self.update_type == 'last':
            if self.last_param is not None:
                self.param_ema = self.last_param
            else:
                self.param_ema = param
            self.last_param = param
        else:
            theta = 0.9
            self.param_ema = theta * self.param_ema + (1-theta) * param

    def change_update_type(self, cur_type):
        self.update_type = cur_type

    def get_init_param(self):
        # return self.param_queue.get(0)
        return self.param_ema

    def normalize_dir(self, dir, cur_p, adaptive=False):
        if adaptive:
            dir = (torch.pow(cur_p, 2)) * dir
        norm = dir.norm()
        return dir / (norm + 1e-12)

    def get_LA_dir(self, adaptive=False, rand=False):
        cur_param = self.operator.extract_params(self.net)
        param = self.get_init_param()
        dir = cur_param - param
        # print(dir)
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
    def _extrap_step(self, alpha, do_LA=True, rand=False, adaptive=False, update=True):
        if do_LA:
            cur_param, dir = self.get_LA_dir(adaptive=adaptive)
        else:
            cur_param, dir = self.get_SAM_dir(adaptive=adaptive)

        if rand:
            noise = torch.randn_like(dir)
            noise = noise / noise.norm() * 0.1
            dir = dir + noise

        if update:
            new_param = cur_param + dir * alpha
            self.operator.put_parameters(self.net, new_param, data=True)
        return cur_param, dir

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, alpha=None, do_forward=True, **kwargs):
        self.set_init_param()
        alpha = alpha if alpha is not None else self.alpha
        self._extrap_step(alpha)
        disable_running_stats(self.net)
        ret = self.net(*args, **kwargs)

        if do_forward:
            with torch.no_grad():
                enable_running_stats(self.net)
                self.net(*args, **kwargs)
        return ret

    def second_step(self, closure, prev_param):
        disable_running_stats(self.net)
        predictions, loss = closure()
        self.zero_grad()
        loss.mean().backward()
        self.operator.put_parameters(self.net, prev_param, data=True)
        self.opt.step()
        return predictions, loss

    def SAM_LA_step(self, closure, inputs, LA_alpha=None, SAM_alpha=0.05, adaptive=False):
        LA_alpha = LA_alpha if LA_alpha is not None else self.alpha
        self.set_init_param()

        enable_running_stats(self.net)
        predictions, loss = closure()
        self.opt.zero_grad()
        loss.mean().backward()

        prev_param, SAM_dir = self._extrap_step(SAM_alpha, do_LA=False, update=True, adaptive=adaptive)  # a
        _, LA_dir = self._extrap_step(LA_alpha, do_LA=True, update=True)

        self.second_step(closure, prev_param)

        return predictions, loss

    def LA_step(self, closure, inputs, alpha=None, adaptive=False):
        alpha = alpha if alpha is not None else self.alpha
        self.set_init_param()
        prev_param, dir = self._extrap_step(alpha, do_LA=True, adaptive=adaptive)

        predictions, loss = self.second_step(closure, prev_param)

        with torch.no_grad():
            enable_running_stats(self.net)
            self.net(inputs)
        return predictions, loss

    def SAM_step(self, closure, alpha=None, adaptive=True):
        alpha = alpha if alpha is not None else self.alpha
        enable_running_stats(self.net)
        predictions, loss = closure()
        loss.mean().backward()
        prev_param, dir = self._extrap_step(alpha, do_LA=False, adaptive=adaptive)

        predictions, loss = self.second_step(closure, prev_param)

        return predictions, loss

    def get_SAM_LA_dir_sim(self, closure):
        enable_running_stats(self.net)
        predictions, loss = closure()
        self.opt.zero_grad()
        loss.mean().backward()

        cur_param = self.operator.extract_params(self.net)
        param = self.get_init_param()
        dir = cur_param - param
        LA_dir = self.normalize_dir(dir, cur_param)

        cur_param, dir = self.operator.extract_params_with_grad(self.net)
        SAM_dir = self.normalize_dir(dir, cur_param)

        dir_sim = torch.cosine_similarity(SAM_dir, LA_dir, dim=0)

        self.opt.zero_grad()
        return dir_sim


class ExtrapLATrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)

        self.model_wrapper = ExtrapLAWrapper(self.model, self.optimizer, args)
        self.alpha_scheduler = AlphaScheduler(self.model_wrapper, args.alpha_scheduler,
                                              args.start_alpha, args.alpha, args.epochs, self.log)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        self.alpha_scheduler(epoch, iteration, iters)

        def get_loss_func():
            predictions = self.model(inputs)
            loss = self.loss_func(predictions, targets)
            return predictions, loss

        predictions, loss = self.model_wrapper.LA_step(get_loss_func, inputs)
        # predictions, loss = self.model_wrapper.SAM_step(get_loss_func, alpha=0.05, adaptive=True)
        # predictions, loss = self.model_wrapper.SAM_LA_step(get_loss_func, inputs, SAM_alpha=0.05, adaptive=True)

        self.scheduler.step()
        return predictions, loss

    def record_grad_norm(self, epoch, iteration, inputs, targets):
        if iteration == 1:
            def get_loss_func():
                predictions = self.model(inputs)
                loss = self.loss_func(predictions, targets)
                return predictions, loss

            sim = self.model_wrapper.get_SAM_LA_dir_sim(get_loss_func)
            print("dir sim: {}".format(sim.item()))


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--queue_size", default=1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--start_alpha", default=0.5, type=float, help="Rho parameter for SAM.")

    # for alpha scheduler
    parser.add_argument("--alpha", default=3, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha_scheduler", default='linear', type=str)

    args = parser.parse_args()

    trainer = ExtrapLATrainer(args)
    trainer.train()
