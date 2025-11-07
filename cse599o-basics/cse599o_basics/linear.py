import torch
import einops


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module.
        This function should accept the following parameters:
        
        in_features: int
            Final dimension of the input
        out_features: int
            Final dimension of the output
        device: torch.device | None = None
            Device to store the parameters on
        dtype: torch.dtype | None = None
            Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype)))


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """Apply the linear transformation to the input."""
        # printed torch.Size([4, 12, 64]) torch.Size([128, 64])
        return einops.einsum(x, self.W, '... i, o i -> ... o')