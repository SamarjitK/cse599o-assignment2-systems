import torch
import einops
from cse599o_basics.rope import RoPE
from cse599o_basics.embedding import Embedding
from cse599o_basics.transformer_block import TransformerBlock
from cse599o_basics.rmsnorm import RMSNorm
from cse599o_basics.linear import Linear

class TransformerLM(torch.nn.Module):
    token_embeddings: Embedding
    layers: torch.nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(self, 
                 vocab_size: int, context_length: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int, rope_theta: float,
                 profile: bool = False):
        """
        Initialize a Transformer block with multi-headed self-attention, RoPE (if provided),
        and a feedforward network.
        Args:
            vocab_size: The size of the vocabulary, necessary for determining 
                the dimensionality of the token embedding matrix
            context_length: The maximum context length, necessary for determining
                the dimensionality of the position embedding matrix
            num_layers: The number of Transformer blocks to use
            d_model: Dimensionality of the Transformer block inputs
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the position-wise feed-forward inner layer
            rope: RoPE module
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.profile = profile

        rope = RoPE(rope_theta, d_model // num_heads, context_length)

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope, profile) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_features: torch.Tensor)-> torch.Tensor:
        """
        Apply the Transformer block to the input.
        Args:
            in_features: (batch_size, sequence_length)
        Returns:
            (batch_size, sequence_length, vocab_size)
        """
        x = self.token_embeddings(in_features)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        # x = softmax(x, dim=-1) no softmax i guess!
        return x