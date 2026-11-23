"""Complete tensor readouts, including irreps absent from natural-parity features."""

from __future__ import annotations

import torch


class CompleteTensorHead(torch.nn.Module):
    """Mix channels independently, then learn all allowed linear and quadratic paths.

    Independent left/right projections avoid forcing antisymmetric products to zero.
    The degree-2 odd piezoelectric block can arise from 1o x 2e; the degree-4
    elastic block can arise from 2e x 2e without raising the backbone cutoff.
    """

    def __init__(self, irreps_in: str, irreps_out: str, channels: int = 8) -> None:
        super().__init__()
        from e3nn import o3

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        types = sorted({ir for mul, ir in self.irreps_in if mul > 0})
        latent = o3.Irreps([(channels, ir) for ir in types])
        self.left = o3.Linear(self.irreps_in, latent)
        self.right = o3.Linear(self.irreps_in, latent)
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)
        self.product = o3.FullyConnectedTensorProduct(latent, latent, self.irreps_out)
        mask = torch.maximum(self.linear.output_mask, self.product.output_mask)
        if not bool(torch.all(mask > 0)):
            raise ValueError(
                f"tensor head cannot reach all of {self.irreps_out} from {self.irreps_in}"
            )
        self.register_buffer("output_mask", mask)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict irreducible tensor coefficients for each local environment."""
        return self.linear(features) + self.product(self.left(features), self.right(features))
