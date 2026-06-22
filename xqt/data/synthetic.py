"""Synthetic CPU data for XQT smoke recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SyntheticClassificationSpec:
    """Synthetic classification data settings."""

    sample_limit: int = 8
    batch_size: int = 2
    input_dim: int = 4
    input_shape: Optional[Sequence[int]] = None
    num_classes: int = 2
    seed: int = 0


def build_synthetic_classification_loader(
    spec: SyntheticClassificationSpec,
) -> DataLoader:
    """Build a deterministic synthetic classification dataloader."""

    if spec.sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    if spec.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if spec.input_shape is None and spec.input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if spec.input_shape is not None and any(int(dim) <= 0 for dim in spec.input_shape):
        raise ValueError("input_shape dims must be positive")
    if spec.num_classes <= 1:
        raise ValueError("num_classes must be greater than 1")

    generator = torch.Generator().manual_seed(spec.seed)
    if spec.input_shape is None:
        feature_shape = (spec.input_dim,)
    else:
        feature_shape = tuple(int(dim) for dim in spec.input_shape)
    features = torch.randn(spec.sample_limit, *feature_shape, generator=generator)
    labels = torch.arange(spec.sample_limit) % spec.num_classes
    dataset = TensorDataset(features, labels.long())
    return DataLoader(dataset, batch_size=spec.batch_size, shuffle=False)


__all__ = ["SyntheticClassificationSpec", "build_synthetic_classification_loader"]
