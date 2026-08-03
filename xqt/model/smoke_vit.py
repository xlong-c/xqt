"""Smoke-only deterministic ViT classifier for model-family recipes."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SmokeViTClassifier(nn.Module):
    """Tiny ViT-style classifier for synthetic CPU smoke tests."""

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 8,
        input_channels: int = 3,
        hidden_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            input_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        batch = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens.mean(dim=1))


def build_smoke_vit_classifier(
    *,
    image_size: int = 32,
    patch_size: int = 8,
    input_channels: int = 3,
    hidden_dim: int = 32,
    num_heads: int = 2,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    num_classes: int = 10,
) -> SmokeViTClassifier:
    """Build a deterministic ViT smoke model."""

    return SmokeViTClassifier(
        image_size=image_size,
        patch_size=patch_size,
        input_channels=input_channels,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
    )


__all__ = [
    "SmokeViTClassifier",
    "build_smoke_vit_classifier",
]
