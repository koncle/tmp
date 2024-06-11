import torch
from extra_modules.sam import SAM
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer
from utility.bypass_bn import enable_running_stats, disable_running_stats


class SAMTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # self.load_model('/data/zj/PycharmProjects/sam/methods/outputs/test/ckpt-lr0.1-E159.pt', epoch=159)
        # self.load_model('/data/zj/PycharmProjects/sam/outputs/ExtrapLA/ckpt-lr0.1-E159.pt', epoch=159)
        self.optimizer = SAM(self.model.parameters(), self.optimizer, rho=args.rho, adaptive=args.adaptive)
        self.steps = args.steps
        self.accu_grad = args.accu_grad

        self.coef = 0.9
        self.g = 0
        self.gs =0
        self.g_ema = 0
        self.gs_ema = 0
        self.stepX = 0

    def obtain_gradient_and_update_mvg(self):
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        grads = torch.cat(grads, dim=0)
        return grads

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        # first ascend step
        for i in range(self.steps):
            enable_running_stats(model)
            predictions = model(inputs)
            loss = self.loss_func(predictions, targets)
            loss.mean().backward()

            # self.g = self.obtain_gradient_and_update_mvg()

            optimizer.first_step(zero_grad=True, save=i == 0)  # ascend step
            if self.accu_grad:
                optimizer.save_grad(put_data_back=False)  # keep ascend

        # second forward-backward step
        disable_running_stats(model)
        self.loss_func(model(inputs), targets).mean().backward()

        # self.gs = self.obtain_gradient_and_update_mvg()
        # self.stepX += 1
        # if not isinstance(self.g_ema, int):
        #     gh = self.g * torch.dot(self.g, self.gs) / torch.dot(self.g, self.g)
        #     gv = self.g - gh
        #     gh_ema = self.g_ema * torch.dot(self.g_ema, self.gs_ema) / torch.dot(self.g_ema, self.g_ema)
        #     gv_ema = self.g_ema - gh_ema
        #
        #     self.g_ema = self.coef * self.g_ema + (1 - self.coef) * self.g
        #
        #     if self.stepX < 50:
        #         self.gs_ema = self.coef * self.gs_ema + (1 - self.coef) * self.gs
        #
        #         g_sim = torch.nn.functional.cosine_similarity(self.gs_ema, self.g_ema, dim=0)
        #         print('g_sim: {}'.format(g_sim.item()))
        #     else:
        #
        #         g_sim = torch.nn.functional.cosine_similarity(self.gs_ema, self.g_ema, dim=0)
        #         print('New g_sim: {}'.format(g_sim.item()))
        #     print()
        #
        # else:
        #     self.g_ema = self.g
        #     self.gs_ema = self.gs

        optimizer.second_step(zero_grad=True)

        self.scheduler.step()
        return predictions, loss

    def record_grad_norm(self, epoch, iteration, inputs, targets):
        # disable_running_stats(self.model)
        if iteration == 0:
            predictions = self.model(inputs)
            loss = self.loss_func(predictions, targets)
            loss.mean().backward()
            init_grad = [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]
            init_grad = torch.cat(init_grad).clone()

            self.optimizer.first_step(zero_grad=True)
            self.loss_func(self.model(inputs), targets).mean().backward()
            second_grad = [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]
            second_grad = torch.cat(second_grad)

            cos_sim = torch.nn.functional.cosine_similarity(init_grad, second_grad, dim=0)
            init_grad_norm = init_grad.norm()
            second_grad_norm = second_grad.norm()
            self.log.log('first grad norm: {}, second grad norm: {}, cos_sim: {}'.format(
                init_grad_norm.item(), second_grad_norm.item(), cos_sim.item()))
            self.optimizer.zero_grad()


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--adaptive", action='store_true', default=False, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--steps", default=1, type=int, help="Number of steps for SAM.")
    parser.add_argument("--accu_grad", default=False, type=bool,
                        help="True if you want to accumulate gradient for SAM.")
    args = parser.parse_args()

    if args.accu_grad:
        assert args.steps > 1

    trainer = SAMTrainer(args)
    trainer.train()
