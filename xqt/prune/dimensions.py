"""Module dimension snapshots and topology-change diff helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _module_dimensions(module: nn.Module) -> dict[str, int | list[int]]:
    """Extract dimension-like attributes from one module."""

    dims: dict[str, int | list[int]] = {}
    if isinstance(module, nn.Linear):
        dims["in_features"] = int(module.in_features)
        dims["out_features"] = int(module.out_features)
    elif isinstance(module, nn.Conv2d):
        dims["in_channels"] = int(module.in_channels)
        dims["out_channels"] = int(module.out_channels)
        dims["groups"] = int(module.groups)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        dims["num_features"] = int(module.num_features)
    elif isinstance(module, nn.LayerNorm):
        normalized_shape = module.normalized_shape
        if isinstance(normalized_shape, int):
            dims["normalized_shape"] = [normalized_shape]
        else:
            dims["normalized_shape"] = [int(value) for value in normalized_shape]

    for attribute in (
        "embed_dim",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "inner_dim",
        "vocab_size",
    ):
        value = getattr(module, attribute, None)
        if isinstance(value, int):
            dims[attribute] = value
    return dims


def snapshot_module_dimensions(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Snapshot dimension attributes of every named module."""

    snapshot: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        dims = _module_dimensions(module)
        snapshot[name] = {
            "module_type": type(module).__name__,
            "dimensions": dims,
        }
    return snapshot


def diff_module_dimensions(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Diff two snapshots into removed modules and changed dimensions.

    Returns ``(removed, changed)`` where each entry is a serializable dict.
    """

    removed: list[dict[str, Any]] = []
    for name in sorted(set(before) - set(after)):
        removed.append(
            {
                "module_name": name,
                "module_type": before[name]["module_type"],
            }
        )

    changed: list[dict[str, Any]] = []
    for name in sorted(set(before) & set(after)):
        before_dims = before[name]["dimensions"]
        after_dims = after[name]["dimensions"]
        deltas: list[dict[str, Any]] = []
        for dimension, value in after_dims.items():
            if dimension not in before_dims:
                deltas.append(
                    {
                        "dimension": dimension,
                        "before": None,
                        "after": value,
                    }
                )
            elif before_dims[dimension] != value:
                deltas.append(
                    {
                        "dimension": dimension,
                        "before": before_dims[dimension],
                        "after": value,
                    }
                )
        for dimension, value in before_dims.items():
            if dimension not in after_dims:
                deltas.append(
                    {
                        "dimension": dimension,
                        "before": value,
                        "after": None,
                    }
                )
        if deltas:
            changed.append(
                {
                    "module_name": name,
                    "module_type": after[name]["module_type"],
                    "changed_dimensions": deltas,
                }
            )
    return removed, changed


__all__ = [
    "diff_module_dimensions",
    "snapshot_module_dimensions",
]
