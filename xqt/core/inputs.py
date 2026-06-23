"""Shared model input helpers for XQT passes."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

DEFAULT_INPUT_KEYS = ("inputs", "input", "x", "image", "images")
DEFAULT_TARGET_KEYS = ("targets", "target", "y", "labels", "label")
DETECTION_TARGET_KEYS = ("boxes", "labels", "image_id", "orig_size", "image_shape")


@dataclass(frozen=True)
class BatchSplit:
    """Normalized model inputs and optional target tensor from one batch."""

    inputs: Any
    targets: Optional[Any]


def _batch_size_hint(value: Any) -> Optional[int]:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return None
    return int(value.shape[0])


def _looks_like_target(value: Any, reference_inputs: Sequence[Any]) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if value.ndim == 0:
        return True

    batch_sizes = {
        batch_size
        for batch_size in (_batch_size_hint(item) for item in reference_inputs)
        if batch_size is not None
    }
    if not batch_sizes:
        return False
    if int(value.shape[0]) not in batch_sizes:
        return False
    if value.ndim == 1:
        return True
    if value.ndim == 2 and value.shape[-1] == 1:
        return True
    return False


def split_batch(
    batch: Any,
    *,
    input_keys: Sequence[str] = DEFAULT_INPUT_KEYS,
    target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
    expected_input_count: Optional[int] = None,
) -> BatchSplit:
    """Best-effort split of a batch-like object into model inputs and targets."""

    if isinstance(batch, Mapping):
        inputs = None
        detection_target_present = any(
            key in batch for key in DETECTION_TARGET_KEYS if key != "labels"
        )
        for key in input_keys:
            if key in batch:
                inputs = batch[key]
                break
        if inputs is None:
            inputs = {}
            for key, value in batch.items():
                if key in target_keys:
                    continue
                if detection_target_present and key in DETECTION_TARGET_KEYS:
                    continue
                inputs[key] = value

        if detection_target_present:
            detection_targets = {
                key: batch[key]
                for key in DETECTION_TARGET_KEYS
                if key in batch
            }
            return BatchSplit(inputs=inputs, targets=detection_targets)
        target_value = None
        for key in target_keys:
            if key in batch:
                target_value = batch[key]
                break
        if target_value is not None:
            return BatchSplit(inputs=inputs, targets=target_value)
        return BatchSplit(inputs=inputs, targets=None)

    if isinstance(batch, (tuple, list)):
        items = tuple(batch)
        if not items:
            return BatchSplit(inputs=items, targets=None)
        if len(items) == 1:
            return BatchSplit(inputs=items[0], targets=None)
        if expected_input_count is not None:
            if len(items) == expected_input_count:
                return BatchSplit(inputs=items, targets=None)
            if len(items) == expected_input_count + 1 and _looks_like_target(
                items[-1],
                items[:-1],
            ):
                if expected_input_count == 1:
                    inputs: Any = items[0]
                else:
                    inputs = items[:-1]
                targets = items[-1] if isinstance(items[-1], torch.Tensor) else None
                return BatchSplit(inputs=inputs, targets=targets)

        maybe_target = items[-1]
        if _looks_like_target(maybe_target, items[:-1]):
            if len(items) == 2:
                inputs = items[0]
            else:
                inputs = items[:-1]
            targets = maybe_target if isinstance(maybe_target, torch.Tensor) else None
            return BatchSplit(inputs=inputs, targets=targets)
        return BatchSplit(inputs=items, targets=None)

    return BatchSplit(inputs=batch, targets=None)


def extract_model_inputs(
    batch: Any,
    *,
    expected_input_count: Optional[int] = None,
) -> Any:
    """Return only the model input portion of a batch-like object."""

    return split_batch(batch, expected_input_count=expected_input_count).inputs


def infer_model_input_count(model: nn.Module) -> Optional[int]:
    """Infer the required positional input count from ``model.forward`` when possible."""

    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return None

    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return None
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if parameter.default is inspect.Parameter.empty:
            count += 1
    return count or None


__all__ = [
    "BatchSplit",
    "DEFAULT_INPUT_KEYS",
    "DETECTION_TARGET_KEYS",
    "DEFAULT_TARGET_KEYS",
    "extract_model_inputs",
    "infer_model_input_count",
    "split_batch",
]
