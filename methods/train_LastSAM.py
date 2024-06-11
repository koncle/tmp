import torch

from extra_modules.last_sam import LastSAM, DPSAT
from extra_modules.sam import SAM
from methods.train_ExtrapLA import Queue
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats

import copy

from utility.param_utils import ParamOperator


class ExtrapLAWrapper():
    def __init__(self, net, opt, args):
        # hyperparameters
        self.alpha = args.alpha
        self.scale_factor = 1
        self.beta = 1  # 0.5 / (0.1 * 5)

        self.steps = 0
        self.net = net
        self.opt = opt
        self.tmp_net = copy.deepcopy(self.net)
        self.tmp_opt = torch.optim.SGD(self.tmp_net.parameters(), lr=0.0, momentum=0.9, weight_decay=0.0005)

        self.maxsize = args.queue_size + 1
        self.param_queue = Queue(self.maxsize)
        self.operator = ParamOperator()

        self.init_lr = self.opt.param_groups[0]['lr']
        self.noise = False
        # self.update_num = np.sum([p.numel() for p in list(self.net.f[-1].parameters())])

    @torch.no_grad()
    def _extrap_step(self, alpha):
        # obtain self.opt lr:
        self.scale_factor = self.init_lr / self.opt.param_groups[0]['lr']

        cur_param = self.operator.extract_params(self.net)
        param = self.get_init_param()

        dir = param - cur_param
        dir = dir / (dir.norm() + 1e-12)

        if alpha > 0:
            new_param = cur_param - alpha * dir
        else:
            new_param = cur_param -  alpha * dir

        self.operator.put_parameters(self.tmp_net, new_param, data=True)

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
        self._extrap_step(alpha)
        ret = self.tmp_net(*args, **kwargs)

        if do_forward:
            with torch.no_grad():
                enable_running_stats(self.net)
                self.net(*args, **kwargs)
        return ret

    def zero_grad(self):
        self.opt.zero_grad()
        self.tmp_opt.zero_grad()

    def set_init_param(self):
        param = self.operator.extract_params(self.net).detach().clone()
        self.param_queue.put(param)

    def get_init_param(self):
        return self.param_queue.get(0)

    def step(self):
        grad = self._insert_grad()
        self.opt.step()
        self.steps += 1


class LastSAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.load_model('/data/sam-main/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        # self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        # self.optimizer = DPSAT(self.optimizer, self.model, rho=args.rho)
        self.model_wrapper = ExtrapLAWrapper(self.model, self.optimizer, args)
        self.last_batch = None

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        # if self.last_batch is not None:
        #     inputs = torch.cat([self.last_batch[0], inputs])
        #     targets = torch.cat([self.last_batch[1], targets])

        # first ascend step
        # enable_running_stats(model)
        # optimizer.ascend()
        # predictions = model(inputs)
        # loss = self.loss_func(predictions, targets)
        # loss.mean().backward()
        # optimizer.descend()

        predictions = self.model_wrapper(inputs)
        ce_loss = self.loss_func(predictions, targets)
        loss = ce_loss

        self.model_wrapper.zero_grad()
        loss.mean().backward()
        self.model_wrapper.step()

        # if self.last_batch is not None:
        #     predictions = predictions[len(self.last_batch[0]):]
        #     loss = loss[len(self.last_batch[0]):]
        # else:
        #     self.last_batch = (inputs, targets)

        self.scheduler.step()
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.1, type=float, help="Rho parameter for SAM.")

    parser.add_argument("--queue_size", default=1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=3, type=float, help="Rho parameter for SAM.")

    args = parser.parse_args()

    trainer = LastSAMTrainer(args)
    trainer.train()
