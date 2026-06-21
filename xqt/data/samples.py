"""Example input generation, input signatures and synthetic data utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SyntheticClassificationSpec:
    """Specification for generating synthetic classification samples.

    When *input_shape* is provided, each input is a random image of that shape.
    Otherwise each input is a random vector of dimension *input_dim*.
    """

    sample_limit: int
    batch_size: int
    input_shape: Optional[List[int]] = None
    num_classes: int = 2
    input_dim: int = 4
    seed: int = 0


def build_synthetic_classification_loader(
    spec: SyntheticClassificationSpec,
) -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` of synthetic
    ``(inputs, targets)`` batches.

    Returns
    -------
    DataLoader
        Yields ``(inputs, targets)`` tuples.  *inputs* has shape
        ``(batch_size, *input_shape)`` or ``(batch_size, input_dim)``.
        *targets* are integer class labels in ``[0, num_classes)``.
    """
    generator = torch.Generator()
    generator.manual_seed(spec.seed)

    if spec.input_shape is not None:
        inputs = torch.randn(
            spec.sample_limit,
            *spec.input_shape,
            generator=generator,
        )
    else:
        inputs = torch.randn(
            spec.sample_limit,
            spec.input_dim,
            generator=generator,
        )
    targets = torch.randint(
        0,
        spec.num_classes,
        (spec.sample_limit,),
        generator=generator,
    )
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=spec.batch_size)


def build_example_input(
    input_shape: List[int],
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Build a single example input tensor for export / tracing.

    Parameters
    ----------
    input_shape:
        Shape of the example input (excluding batch dimension if
        *batch_size* > 1).
    batch_size:
        Batch dimension prepended to *input_shape*.
    dtype:
        Data type of the returned tensor.
    seed:
        Random seed for reproducibility.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = [batch_size] + list(input_shape)
    return torch.randn(*shape, generator=generator, dtype=dtype)


__all__ = [
    "SyntheticClassificationSpec",
    "build_example_input",
    "build_synthetic_classification_loader",
]
