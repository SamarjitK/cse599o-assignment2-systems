import torch
import einops
import math
from typing import Iterable

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    g = 0
    for p in parameters:
        if p.grad is not None:
            g += torch.sum(p.grad ** 2)
    g = math.sqrt(g)
    if g > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad *= (max_l2_norm / (g + eps))

def lr_scheduler(t: int, max_lr: float, min_lr: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return max_lr * (t / T_w)
    elif t > T_c:
        return min_lr
    else:
        coeff = (0.5) * (1 + math.cos(math.pi * (t - T_w) / (T_c-T_w)))
        return min_lr + coeff * (max_lr - min_lr)

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
        - Subtract the largest element for numerical stability.
        - Cancel out log and exp whenever possible.
        - Handle any additional batch dimensions and return the average across the batch. 
            As with Section 3.3, we assume batch-like dimensions always come first, 
            before the vocabulary size dimension.

    Args:
        inputs Float(batch_size vocab_size): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets Int(batch_size): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float(): The average cross-entropy loss across examples.
    """
    # need to collapse batches
    inputs = einops.rearrange(inputs, '... v -> (... ) v') # (batch_size, vocab_size)
    targets = einops.rearrange(targets, '... -> (...)') # (batch_size,)
    inputs = inputs - einops.reduce(inputs, '... vocab -> ... 1', 'max') # (batch_size, vocab_size)
    # targets = einops.rearrange(targets, '... b -> ... b 1') # (batch_size, 1)
    targets = targets.unsqueeze(-1) # (batch_size, 1)
    sliced = torch.gather(inputs, dim=-1, index=targets) # (batch_size,)
    return (torch.logsumexp(inputs, dim=-1) - sliced.squeeze(-1)).mean() # scalar

def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply the softmax function to an input tensor along a specified dimension.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    max_val, _ = torch.max(in_features, dim=dim, keepdim=True)
    exp_in_features = torch.exp(in_features - max_val)
    return exp_in_features / torch.sum(exp_in_features, dim=dim, keepdim=True)