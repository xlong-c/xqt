"""Small toy models used by XQT quantization smoke recipes."""

from __future__ import annotations

import torch
from torch import nn


class HeteroQuantToyModel(nn.Module):
    """Tiny chain model with named components for hetero quantization tests."""

    def __init__(
        self,
        in_features: int = 4,
        hidden_features: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(in_features, hidden_features)
        self.projector = nn.Linear(hidden_features, hidden_features)
        self.decoder = nn.Linear(hidden_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = torch.relu(self.vision_encoder(x))
        projected = torch.relu(self.projector(encoded))
        return self.decoder(projected)


def build_hetero_quant_toy_model(
    in_features: int = 4,
    hidden_features: int = 4,
    num_classes: int = 2,
) -> HeteroQuantToyModel:
    """Build the named-component toy model used by multi-component quant recipes."""

    return HeteroQuantToyModel(
        in_features=in_features,
        hidden_features=hidden_features,
        num_classes=num_classes,
    )


__all__ = [
    "HeteroQuantToyModel",
    "build_hetero_quant_toy_model",
]
