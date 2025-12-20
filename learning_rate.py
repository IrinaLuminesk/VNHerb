import math
from typing import Any


class PiecewiseScheduler:
    """
    Custom learning rate schedule that used a three phrases strategy. Linear warm-up phrase, sustain phrase, and exponential decay.

    This schedule increases the learning rate linearly from `start_lr` to `max_lr` over 
    `rampup_epochs`, then sustain the learning rate before gradually decreases it to `min_lr`.

    It is designed to be used with PyTorch's `LambdaLR` scheduler, which expects a callable 
    that returns a scaling factor relative to the optimizer's base learning rate.
    
    Args:
        start_lr (float): Initial learning rate at the beginning of training.
        max_lr (float): Maximum learning rate reached after warm-up.
        min_lr (float): Minimum learning rate reached at the end of the exponential decay.
        rampup_epochs (int): Number of epochs for linear warm-up phase.
        sustain_epochs (int): Number of epochs to keep the LR constant at `max_lr` before decay.
        exp_decay (float): Exponential decay factor (e.g., 0.8 means 20% LR reduction per epoch).
    """
    def __init__(self, start_lr, max_lr, min_lr, rampup_epochs, sustain_epochs, exp_decay):
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.rampup_epochs = rampup_epochs
        self.sustain_epochs = sustain_epochs
        self.exp_decay = exp_decay

    def __call__(self, epoch):
        if epoch < self.rampup_epochs:
            lr = (self.max_lr - self.start_lr)/self.rampup_epochs * epoch + self.start_lr
        elif epoch < self.rampup_epochs + self.sustain_epochs:
            lr = self.max_lr
        else:
            lr = (self.max_lr - self.min_lr) * self.exp_decay**(epoch - self.rampup_epochs - self.sustain_epochs) + self.min_lr
        return lr / self.max_lr  
    
class WarmupCosineScheduler:
    def __init__(self, total_epochs, warmup_epochs, min_lr, max_lr):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine