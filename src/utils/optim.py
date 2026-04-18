# --------------------------------------------------------
# NuScenes-QA Optimizer
# Supports: Adam, AdamW, Step Decay, Cosine Annealing
# --------------------------------------------------------

import math
import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size, warmup_epoch):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch


    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.25):
            r = self.lr_base * 1/(self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.5):
            r = self.lr_base * 2/(self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.75):
            r = self.lr_base * 3/(self.warmup_epoch + 1)
        else:
            r = self.lr_base

        return r


class CosineWarmupOptimizer(object):
    """
    Warmup + Cosine Annealing LR scheduler.
    During warmup: linearly ramps LR from 0 to lr_base.
    After warmup: cosine decay from lr_base to lr_min.
    """
    def __init__(self, lr_base, optimizer, data_size, batch_size,
                 warmup_epoch, max_epoch, lr_min=1e-6):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self.lr_min = lr_min
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.steps_per_epoch = max(1, data_size // batch_size)
        self.warmup_steps = self.steps_per_epoch * warmup_epoch
        self.total_steps = self.steps_per_epoch * max_epoch

    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= self.warmup_steps:
            # Linear warmup
            r = self.lr_base * (step / max(1, self.warmup_steps))
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            r = self.lr_min + 0.5 * (self.lr_base - self.lr_min) * (1 + math.cos(math.pi * progress))

        return r


def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE

    opt_name = getattr(__C, 'OPT', 'Adam')
    std_optim = getattr(Optim, opt_name)
    params = filter(lambda p: p.requires_grad, model.parameters())
    eval_str = 'params, lr=0'
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])

    raw_optimizer = eval('std_optim' + '(' + eval_str + ')')

    # Choose scheduler type
    lr_schedule = getattr(__C, 'LR_SCHEDULE', 'step')

    if lr_schedule == 'cosine':
        max_epoch = getattr(__C, 'MAX_EPOCH', 20)
        lr_min = getattr(__C, 'LR_MIN', 1e-6)
        optim = CosineWarmupOptimizer(
            lr_base, raw_optimizer, data_size, __C.BATCH_SIZE,
            __C.WARMUP_EPOCH, max_epoch, lr_min
        )
        print(f"  [Optim] {opt_name} + CosineAnnealing(warmup={__C.WARMUP_EPOCH}, "
              f"lr_base={lr_base}, lr_min={lr_min})")
    else:
        optim = WarmupOptimizer(
            lr_base, raw_optimizer, data_size, __C.BATCH_SIZE,
            __C.WARMUP_EPOCH
        )
        print(f"  [Optim] {opt_name} + StepDecay(warmup={__C.WARMUP_EPOCH}, "
              f"decay_at={getattr(__C, 'LR_DECAY_LIST', [])})")

    return optim


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r