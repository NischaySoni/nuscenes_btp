# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

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
        # Capture the original relative learning rates set during init
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self._step += 1

        rate = self.rate()
        for i, p in enumerate(self.optimizer.param_groups):
            # Scale each parameter group by its original relative scale to lr_base
            scale = self.initial_lrs[i] / self.lr_base
            p['lr'] = rate * scale
            
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


def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE

    std_optim = getattr(Optim, __C.OPT)
    
    # --- Layer-Wise Learning Rate (LLR) Separation ---
    adapter_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # The visual adapter operates on noisy YOLO inputs, needs fast adaptation
        if 'annot_adapter' in name or 'yolo_adapter' in name or 'proj_norm' in name:
            adapter_params.append(param)
        else:
            base_params.append(param)
            
    # Language backbone gets microscopic learning rate to prevent forgetting
    base_lr = lr_base / 100.0 if not getattr(__C, 'FREEZE_BACKBONE', False) and getattr(__C, 'VISUAL_FEATURE', '') == 'detected' else lr_base
    
    param_groups = [
        {'params': adapter_params, 'lr': lr_base},
        {'params': base_params, 'lr': base_lr}
    ]

    kwargs = {}
    for key in __C.OPT_PARAMS:
        kwargs[key] = __C.OPT_PARAMS[key]

    optimizer = std_optim(param_groups, **kwargs)

    optim = WarmupOptimizer(
        lr_base,
        optimizer,
        data_size,
        __C.BATCH_SIZE,
        __C.WARMUP_EPOCH
    )

    return optim


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r