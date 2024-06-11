class StepLR:
    def __init__(self, optimizer, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = optimizer.param_groups[0]["lr"]

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class StepLR2:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = 1
        elif epoch < self.total_epochs * 6/10:
            lr =  0.2
        elif epoch < self.total_epochs * 8/10:
            lr = 0.2 ** 2
        else:
            lr = 0.2 ** 3

        self.optimizer.outer_lr = self.optimizer.outer_lr * lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
