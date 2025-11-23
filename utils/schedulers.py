import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def get_inverse_sqrt_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step/warmup_steps
        else:
            return (warmup_steps ** 0.5) / (step ** 0.5)
    return LambdaLR(optimizer, lr_lambda)


def exponential_ramp_up_scheduler(current_epoch, cooldown_epochs, total_epochs, curvature = 3.0):
    if current_epoch < cooldown_epochs:
        return 0.0
    else:
        current_epoch = current_epoch - cooldown_epochs
        total_epochs = total_epochs - cooldown_epochs
        x = current_epoch / total_epochs
        ret = (np.exp(curvature*x) - 1) / (np.exp(curvature) - 1)
        return float(np.clip(ret, 0.0, 1.0))