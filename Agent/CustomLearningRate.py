from torch.optim.lr_scheduler import LRScheduler

class CustomLearningRateSchedule(LRScheduler):
    def __init__(self, optimizer, initial_lr, decay_rates, last_epoch=-1):
        """
        initial_lr: Learning rate ban đầu
        decay_rates: Danh sách hệ số decay (n phần tử)
        """
        self.initial_lr = initial_lr
        self.decay_rates = decay_rates

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lr = self.initial_lr * self.decay_rates ** step

        return [lr for _ in self.base_lrs]  # Trả về danh sách cho từng group của optimizer