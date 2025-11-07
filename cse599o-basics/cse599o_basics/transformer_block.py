import torch
import einops
from cse599o_basics.rope import RoPE
from cse599o_basics.attention import MultiHead
from cse599o_basics.rmsnorm import RMSNorm
from cse599o_basics.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):
    attn: MultiHead
    ln1: RMSNorm
    ffn: SwiGLU
    ln2: RMSNorm

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE, profile: bool = False):
        """
        Initialize a Transformer block with multi-headed self-attention, RoPE (if provided),
        and a feedforward network.
        Args:
            d_model: Dimensionality of the Transformer block inputs
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the position-wise feed-forward inner layer
            rope: RoPE module
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.profile = profile
        self.attn = MultiHead(d_model, num_heads, rope, profile)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, in_features: torch.Tensor)-> torch.Tensor:
        """
        Apply the Transformer block to the input.
        Args:
            in_features: (batch, sequence_length, d_model)
        Returns:
            (batch, sequence_length, d_model) Tensor with the output of running the Transformer block
            on the input features.
        """
        token_positions = torch.arange(in_features.size(-2))
        y = in_features + self.attn(self.ln1(in_features), token_positions)
        return y + self.ffn(self.ln2(y))
        
