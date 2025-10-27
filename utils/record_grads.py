def hook_backward(module_name, grads):
    def hook(grad):
        grads[module_name] = grad.mean().detach().cpu().item() if grad is not None else None
    return hook

def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        if output.requires_grad:
            output.register_hook(hook_backward(module_name, grads))
    return hook

def register_hooks(model):
    grads = {}
    for name, layer in model.named_modules():
        if any(layer.children()) is False:
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return grads