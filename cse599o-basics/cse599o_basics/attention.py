import torch
from typing import Any, Optional
import einops
from cse599o_basics.model_utils import softmax
from cse599o_basics.rope import RoPE
import torch.cuda.nvtx as nvtx


class MultiHead(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: Optional[RoPE] = None, profile: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.profile = profile
        self.w_q = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty((num_heads * self.d_k, d_model)))) # (num_heads * d_k, d_model)
        self.w_k = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty((num_heads * self.d_k, d_model)))) # (num_heads * d_k, d_model)
        self.w_v = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty((num_heads * self.d_v, d_model)))) # (num_heads * d_v, d_model)
        self.w_o = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty((d_model, num_heads * self.d_v)))) # (d_model, num_heads * d_v)
        if rope is not None:
            self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply multi-headed attention to the input.
        Args:
            x: (... sequence_length d_model) Tensor to run your implementation on.
            repo: optional
        Returns:
            (... sequence_length d_out)
        """
        w_q = einops.rearrange(self.w_q, '(h d_k) d_model -> h d_k d_model', h=self.num_heads) # (num_heads, d_k, d_model)
        w_k = einops.rearrange(self.w_k, '(h d_k) d_model -> h d_k d_model', h=self.num_heads) # (num_heads, d_k, d_model)
        w_v = einops.rearrange(self.w_v, '(h d_v) d_model -> h d_v d_model', h=self.num_heads) # (num_heads, d_v, d_model)
        w_q_x = einops.einsum(x, w_q, '... seq d_model, h d_k d_model -> ... h seq d_k') # (..., num_heads, seq, d_k)
        w_k_x = einops.einsum(x, w_k, '... seq d_model, h d_k d_model -> ... h seq d_k') # (..., num_heads, seq, d_k)
        w_v_x = einops.einsum(x, w_v, '... seq d_model, h d_v d_model -> ... h seq d_v') # (..., num_heads, seq, d_v)
        if hasattr(self, 'rope') and token_positions is not None:
            w_q_x = self.rope(w_q_x, token_positions)
            w_k_x = self.rope(w_k_x, token_positions)
        mask = ~torch.triu(torch.ones((x.shape[-2], x.shape[-2]), dtype=torch.bool), diagonal=1) # (seq, seq)
        attn_params = (w_q_x, w_k_x, w_v_x, mask)
        attention = annotated_scaled_attention(*attn_params) if self.profile else scaled_attention(*attn_params) # (..., num_heads, seq, d_v)
        multihead = einops.rearrange(attention, '... h seq d_v -> ... seq (h d_v)') # (..., seq, num_heads * d_v)
        return einops.einsum(multihead, self.w_o, '... seq h_d_v, d_model h_d_v -> ... seq d_model') # (..., seq, d_model)

@nvtx.range("scaled dot product attention")
def annotated_scaled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, compute the Scaled Dot-Product Attention (SDPA).

    Args:
        q (Float[Tensor, " ... queries d_k"]): Query tensor
        k (Float[Tensor, " ... keys d_k"]): Key tensor
        v (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    with nvtx.range("computing attention scores"):
        sqrt_d_k = torch.sqrt(torch.tensor(q.shape[-1]))
        pre_softmax = einops.einsum(q, k, '... n d_k, ... m d_k -> ... n m') / sqrt_d_k
        if mask is not None:
            if mask.device != pre_softmax.device:
                mask = mask.to(pre_softmax.device)
            pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    with nvtx.range("computing softmax"):
        post_softmax = softmax(pre_softmax, dim=-1)
    with nvtx.range("final matmul"):
        result = einops.einsum(post_softmax, v, '... n m, ... m d_v -> ... n d_v')
    return result

def scaled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, compute the Scaled Dot-Product Attention (SDPA).

    Args:
        q (Float[Tensor, " ... queries d_k"]): Query tensor
        k (Float[Tensor, " ... keys d_k"]): Key tensor
        v (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    sqrt_d_k = torch.sqrt(torch.tensor(q.shape[-1]))
    pre_softmax = einops.einsum(q, k, '... n d_k, ... m d_k -> ... n m') / sqrt_d_k
    if mask is not None:
        if mask.device != pre_softmax.device:
            mask = mask.to(pre_softmax.device)
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    post_softmax = softmax(pre_softmax, dim=-1)
    return einops.einsum(post_softmax, v, '... n m, ... m d_v -> ... n d_v')
