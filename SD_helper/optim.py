class NoamScheduler:
    """
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    """
    def __init__(self,
                 base_lr: float,
                 d_model: int,
                 warmup_steps: float,
                 step_per_epoch: int):
        self.base_lr = base_lr
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_per_epoch = step_per_epoch
    
    def calculate_lr(self, epoch, batch_idx):
        step = epoch*self.step_per_epoch + batch_idx
        last_step = step - 1
        if last_step == -1:
            return 1e-5
        last_step = max(1, last_step)
        return self.base_lr*self.d_model**(-0.5)*min(last_step**(-0.5), last_step*self.warmup_steps**(-1.5))