class WarmupScheduler:
    """Scheduler que combina warmup linear com outro scheduler."""
    def __init__(self, optimizer, base_scheduler, warmup_steps, init_lr):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            # Warmup linear: de 0 até init_lr
            factor = self.step_num / self.warmup_steps
            lr = self.init_lr * factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Após warmup, usa o scheduler base
            self.base_scheduler.step()