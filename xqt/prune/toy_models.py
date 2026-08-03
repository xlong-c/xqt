"""Small smoke models owned by the pruning subsystem."""

from __future__ import annotations

import torch
from torch import nn


class StructuredPruningToyCNN(nn.Module):
    """Small Conv-BN-ReLU CNN used by structured-pruning smoke tests."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: int = 8,
        out_channels: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


__all__ = ["StructuredPruningToyCNN"]
