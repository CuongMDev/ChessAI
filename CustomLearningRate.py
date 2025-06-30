from torch.optim.lr_scheduler import LRScheduler

class CustomLearningRateSchedule(LRScheduler):
    def __init__(self, optimizer, initial_lr, decay_rates, decay_steps, lr_decay_interval, last_epoch=-1):
        """
        initial_lr: Learning rate ban đầu
        decay_rates: Danh sách hệ số decay (n phần tử)
        decay_steps: Danh sách step ứng với decay (n-1 phần tử)
        lr_decay_interval: Khoảng cách giữa các lần decay
        """
        assert len(decay_rates) - 1 == len(decay_steps), "Số lượng decay_steps phải ít hơn decay_rates một phần tử"

        self.initial_lr = initial_lr
        self.decay_rates = decay_rates
        self.decay_steps = decay_steps
        self.lr_decay_interval = lr_decay_interval

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lr = self.initial_lr
        prev_decay_step = 0

        # Áp dụng các decay ban đầu
        for i in range(len(self.decay_steps)):
            decay_factor = self.decay_rates[i]
            num_intervals = max((min(step, self.decay_steps[i]) - prev_decay_step) // self.lr_decay_interval, 0)
            lr *= decay_factor ** num_intervals
            prev_decay_step = self.decay_steps[i]

        # Áp dụng decay cuối cùng mãi mãi
        decay_factor = self.decay_rates[-1]
        num_intervals = max((step - prev_decay_step) // self.lr_decay_interval, 0)
        lr *= decay_factor ** num_intervals

        return [lr for _ in self.base_lrs]  # Trả về danh sách cho từng group của optimizer