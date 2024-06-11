import copy
import torch
from utility.param_utils import ParamOperator
from model.smooth_cross_entropy import smooth_crossentropy
from train_SGD import get_parser, Trainer


class ExtraSGDTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.last_grad = 0
        self.v_t = 0
        self.alpha = 0.5
        self.u = 0.9

        self.operator = ParamOperator()
        self.tmp_model = copy.deepcopy(self.model)
        self.tmp_opt = torch.optim.SGD(self.tmp_model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                       weight_decay=args.weight_decay)

    def train_step(self, epoch, iteration, model, inputs, targets, optimizer, iters):
        with torch.no_grad():
            param = self.operator.extract_params(model)
            param_1_4 = param - self.alpha * self.last_grad
            param_1_2 = param_1_4 + self.u * self.v_t
            self.operator.put_parameters(self.tmp_model, param_1_2, data=True)

        predictions = self.tmp_model(inputs)
        loss = self.loss_func(predictions, targets)
        self.tmp_opt.zero_grad()
        loss.mean().backward()

        with torch.no_grad():
            last_grad = self.operator.extract_params_with_grad(self.tmp_model)[1]
            self.v_t = self.u * self.v_t - args.learning_rate * last_grad
            param = param + self.v_t
            self.operator.put_parameters(model, param, data=True)
            model(inputs)

        self.scheduler.step(epoch)
        return predictions, loss


if __name__ == "__main__":
    parser = get_parser()

    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")

    args = parser.parse_args()

    trainer = ExtraSGDTrainer(args)
    trainer.train()
