import torch
import einops


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """
        Construct a SwiGLU transformation module.
        This function should accept the following parameters:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the up-project happening internally to your swiglu.
        device: torch.device | None = None
            Device to store the parameters on
        dtype: torch.dtype | None = None
            Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((d_ff, d_model), device=device, dtype=dtype)))
        self.w2 = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((d_model, d_ff), device=device, dtype=dtype)))
        self.w3 = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((d_ff, d_model), device=device, dtype=dtype)))

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Apply the SwiGLU transformation to the input.
        x (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.
        """
        # FFN(x) = w2 (SiLU(w1 @ x) * (w3 @ x))
        # do silu first: silu = x * torch.sigmoid(x)
        w1x = einops.einsum(x, self.w1, '... d_model, d_ff d_model -> ... d_ff')
        w3x = einops.einsum(x, self.w3, '... d_model, d_ff d_model -> ... d_ff')
        silu = w1x * torch.sigmoid(w1x)
        x = einops.einsum(silu * w3x, self.w2, '... d_ff, d_model d_ff -> ... d_model')
        return x
        