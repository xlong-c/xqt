"""Minimal RMSNorm facade for FFN composition."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Minimal RMSNorm facade for FFN composition."""

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            out = out * self.weight.to(device=x.device, dtype=x.dtype)
        return out
