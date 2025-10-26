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

class LinearWarmupDecayScheduler:
    """
    Linear warm-up + linear decay scheduler (có lr_min).

    Usage:
        scheduler = LinearWarmupDecayScheduler(
            lr_max=3e-4,
            total_epochs=150,
            warmup_ratio=0.1,
            lr_min=1e-5
        )
        for epoch in range(total_epochs):
            lr = scheduler.step()
            print(f"Epoch {epoch+1} | LR: {lr:.6f}")
    """

    def __init__(self, lr_max, lr_min, total_steps, warmup_ratio):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.current_step = 0

    def step(self):
        """Step the scheduler by 1 step and return current learning rate."""
        step = self.current_step

        if step < self.warmup_steps:
            # Linear warm-up: từ 0 -> lr_max
            lr = self.lr_max * (step + 1) / self.warmup_steps
        else:
            # Linear decay: từ lr_max -> lr_min
            decay_steps = self.total_steps - self.warmup_steps
            decay_progress = (step - self.warmup_steps + 1) / decay_steps
            lr = self.lr_max - (self.lr_max - self.lr_min) * decay_progress
            lr = max(lr, self.lr_min)  # tránh giảm thấp hơn lr_min

        self.current_step += 1
        return lr

    def reset(self):
        """Reset scheduler to step 0."""
        self.current_step = 0
