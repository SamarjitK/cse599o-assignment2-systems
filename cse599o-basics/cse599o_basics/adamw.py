import torch
import math
from typing import Optional, Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        """
        Initialize your optimizer.
        Arguments:
        params: iterable
        A collection of parameters to optimize, or parameter groups
        (for applying different hyperparameters to different parts of the model).
        ...
        Additional arguments depending on the optimizer
        (e.g., learning rate is common).
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Make one update of the parameters.
        This is called after the backward pass in training, so gradients
        are available for each parameter.
        For each parameter tensor p:
            - Access its gradient in p.grad (if it exists).
            - Update p.data in-place based on p.grad.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group['params']:
                p: torch.nn.Parameter
                if p.grad is None:
                    continue
                state: dict = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                # update first moment estimate
                m = state.get("m", torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                state["m"] = m

                # update second moment estimate
                v = state.get("v", torch.zeros_like(p.data))
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                state["v"] = v

                # update p
                adjusted_lr = lr * (math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1)))
                p.data -= adjusted_lr * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
        return loss