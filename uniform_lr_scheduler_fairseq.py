# uniform_lr_scheduler.py
import numpy as np
from dataclasses import dataclass, field
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler
from fairseq.dataclass import FairseqDataclass

@dataclass
class UniformLRSchedulerConfig(FairseqDataclass):
    """
    Configuration for the uniform noisy learning rate scheduler.
    """
    # min_lr: float = field(default=0.0001, metadata={"help": "minimum learning rate"})
    min_lr: float = field(default=0.00015, metadata={"help": "minimum learning rate"})
    # max_lr: float = field(default=0.0005, metadata={"help": "maximum learning rate"})
    max_lr: float = field(default=0.00045, metadata={"help": "maximum learning rate"})
    offset: float = field(default=0.0000002, metadata={"help": "offset for learning rate"})
    #for warmup -dahlia
    warmup_updates: int = field(default=5000, metadata={"help": "number of warmup steps"})
    

@register_lr_scheduler("uniform", dataclass=UniformLRSchedulerConfig)
class UniformNoisyLR(FairseqLRScheduler):
    """
    Uniform Noisy LR Scheduler: randomly samples LR from [min_lr, max_lr] every update
    """
    def __init__(self, cfg: UniformLRSchedulerConfig, optimizer):
        super().__init__(cfg, optimizer)
        self.min_lr = cfg.min_lr
        self.max_lr = cfg.max_lr
        self.offset = cfg.offset
        #for warmup-dahlia
        self.warmup_updates = cfg.warmup_updates

    # def step_update(self, num_updates):
    #     """
    #     Called after every optimizer step
    #     """
    #     lr = np.random.uniform(self.min_lr - self.offset, self.max_lr - self.offset) + self.offset 
       
    #     self.optimizer.set_lr(lr)
        
    #     return lr
    #for warmup-dahlia
    def step_update(self, num_updates):
        """
        Called after every optimizer step
        """
        if num_updates < self.warmup_updates:
            # Linear warmup
            warmup_frac = num_updates / max(1, self.warmup_updates)
            lr = self.min_lr + warmup_frac * (self.max_lr - self.min_lr)
        else:
            # Uniform random LR after warmup
            lr = np.random.uniform(self.min_lr - self.offset, self.max_lr - self.offset) + self.offset
       
        self.optimizer.set_lr(lr)
        return lr

    def step(self, epoch, val_loss=None):
        """
        Called at epoch boundaries (we do nothing here)
        """
        super().step(epoch, val_loss)
        return self.optimizer.get_lr()
