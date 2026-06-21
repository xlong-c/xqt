"""Calibration dataloader utilities and sample extraction.

Provides helpers for extracting calibration samples from arbitrary
:class:`~torch.utils.data.DataLoader` instances, which is the common
denominator across quantization backends (torchao, ONNX Runtime QDQ,
NVIDIA ModelOpt, etc.).
"""

from __future__ import annotations

from typing import Any, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader


def extract_calibration_samples(
    dataloader: DataLoader,
    sample_limit: int,
    *,
    input_index: int = 0,
) -> List[Any]:
    """Extract at most *sample_limit* input batches from *dataloader*.

    Each item yielded by *dataloader* is expected to be a tensor, a tuple,
    or a mapping.  By default the first element (``item[0]`` or
    ``item["input"]``) is kept; use *input_index* to pick a different
    position or pass ``input_index=None`` to keep the raw batch.

    Returns
    -------
    list
        Collected calibration input batches (length ≤ *sample_limit*).
    """
    samples: List[Any] = []
    for batch in dataloader:
        if input_index is not None:
            batch = _extract_input(batch, input_index)
        samples.append(batch)
        if len(samples) >= sample_limit:
            break
    return samples


def calibration_sample_count(dataloader: DataLoader) -> int:
    """Return the number of batches in *dataloader* without consuming
    the iterator more than once."""
    return len(dataloader.dataset)  # type: ignore[arg-type]


def _extract_input(batch: Any, index: int) -> Any:
    """Pick the input portion from a dataloader batch."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[index]
    if isinstance(batch, dict):
        if index == 0:
            for key in ("input", "inputs", "x"):
                if key in batch:
                    return batch[key]
        return list(batch.values())[index]
    return batch


def calibration_sweep(
    dataloader: DataLoader,
    sample_limits: Sequence[int],
    *,
    input_index: int = 0,
) -> Iterator[Tuple[int, List[Any]]]:
    """Yield ``(limit, samples)`` for each value in *sample_limits*.

    Useful for evaluating how calibration sample count affects
    quantization quality without re-reading the full dataset.
    """
    max_limit = max(sample_limits) if sample_limits else 0
    all_samples = extract_calibration_samples(
        dataloader,
        max_limit,
        input_index=input_index,
    )
    for limit in sample_limits:
        yield limit, all_samples[:limit]


__all__ = [
    "calibration_sample_count",
    "calibration_sweep",
    "extract_calibration_samples",
]
