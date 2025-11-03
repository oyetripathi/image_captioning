from torch.optim.lr_scheduler import LambdaLR

def get_inverse_sqrt_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step/warmup_steps
        else:
            return (warmup_steps ** 0.5) / (step ** 0.5)
    return LambdaLR(optimizer, lr_lambda)