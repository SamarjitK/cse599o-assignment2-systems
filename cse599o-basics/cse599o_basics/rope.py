import torch
import einops


class RoPE(torch.nn.Module):
    # to fix linting
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        Args:
        theta: Theta value for RoPE.
        d_k: Dimension of query/key vectors (should be even).
        max_seq_len: Maximum sequence length that will be inputted.
        device: torch.device | None. Device to store the buffers on.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        k_s = torch.arange(d_k // 2, dtype=torch.float32, device=device) # (d/2,)
        i_s = torch.arange(max_seq_len, dtype=torch.float32, device=device) # (max_seq_len,)

        theta_i_k = einops.repeat(i_s, 'i -> i k', k=d_k // 2) # (max_seq_len, d/2)
        theta_i_k = theta_i_k / (theta ** (2 * einops.repeat(k_s, 'k -> i k', i=max_seq_len) / d_k))

        self.register_buffer('cos', torch.cos(theta_i_k), persistent=False) # (max_seq_len, d/2)
        self.register_buffer('sin', torch.sin(theta_i_k), persistent=False) # (max_seq_len, d/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        """
        Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
        return a tensor of the same shape.
        Notes:
            - Accept x with an arbitrary number of batch dimensions.
            - token_positions has shape (..., seq_len) and gives absolute 
                positions per token along the sequence dimension.
            - Use token_positions to slice (precomputed) cos/sin tensors
                along the sequence dimension.
        """
        *batch_dims, seq_len, _ = x.shape # (..., seq_len, d_k)

        x_stacked = einops.rearrange(x, '... seq_len (k1 k2) -> ... seq_len k1 k2', k1=self.d_k//2) # (..., seq_len, d_k/2, 2)
        x1, x2 = x_stacked.unbind(dim=-1) # (..., seq_len, d_k//2), (..., seq_len, d_k//2)

        cos = self.cos[token_positions] # (..., seq_len, d_k//2)
        sin = self.sin[token_positions] # (..., seq_len, d_k//2)

        x1_r = x1 * cos - x2 * sin
        x2_r = x1 * sin + x2 * cos

        x_bind = torch.stack([x1_r, x2_r], dim=-1) # (..., seq_len, d_k/2, 2)
        return einops.rearrange(x_bind, '... seq_len k1 k2 -> ... seq_len (k1 k2)') # (..., seq_len, d_k)
