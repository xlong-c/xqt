"""Shared runtime helpers for operator candidate execution."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import nn

from xqt.contracts.input_utils import split_example_input
from .types import OperatorOptimizationTargetPlan


def iter_tensors(data: Any) -> Iterable[torch.Tensor]:
    """Yield tensors recursively from a supported model-input structure."""

    if isinstance(data, torch.Tensor):
        yield data
        return
    if isinstance(data, Mapping):
        for value in data.values():
            yield from iter_tensors(value)
        return
    if isinstance(data, (tuple, list)):
        for value in data:
            yield from iter_tensors(value)


def call_module(module: nn.Module, inputs: Any) -> Any:
    """Call a module with normalized positional and keyword inputs."""

    normalized = split_example_input(inputs)
    return module(*normalized.args, **normalized.kwargs)


def call_module_no_grad(module: nn.Module, inputs: Any) -> Any:
    """Call a module without recording autograd state."""

    with torch.no_grad():
        return call_module(module, inputs)


def move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move supported model-input structures to one device."""

    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [move_to_device(value, device) for value in data]
    return data


def shape_signature(inputs: Any) -> dict[str, Any]:
    """Return the stable tensor shape signature recorded in execution reports."""

    normalized = split_example_input(inputs)
    values = list(normalized.args) + list(normalized.kwargs.values())
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    return {
        "structure": (
            "mapping"
            if normalized.kwargs
            else "tuple"
            if len(normalized.args) > 1
            else "tensor"
        ),
        "input_count": len(values),
        "tensor_shapes": [list(tensor.shape) for tensor in tensors],
        "tensor_dtypes": [str(tensor.dtype) for tensor in tensors],
        "tensor_devices": [str(tensor.device) for tensor in tensors],
    }


def infer_module_device_dtype(
    module: nn.Module,
    inputs: Any,
) -> tuple[str | None, str | None]:
    """Infer execution device and preferred dtype from a module and its inputs."""

    device: str | None = None
    dtype: str | None = None
    for tensor in module.parameters():
        device = str(tensor.device)
        if tensor.is_floating_point():
            return device, str(tensor.dtype)
        if dtype is None:
            dtype = str(tensor.dtype)
    for tensor in module.buffers():
        if device is None:
            device = str(tensor.device)
        if tensor.is_floating_point():
            return device, str(tensor.dtype)
        if dtype is None:
            dtype = str(tensor.dtype)
    for tensor in iter_tensors(inputs):
        if device is None:
            device = str(tensor.device)
        if tensor.is_floating_point():
            return device, str(tensor.dtype)
        if dtype is None:
            dtype = str(tensor.dtype)
    return device, dtype


def effective_validation_thresholds(
    target: OperatorOptimizationTargetPlan,
    *,
    baseline_output: torch.Tensor,
    optimized_output: torch.Tensor,
) -> dict[str, float]:
    """Resolve target validation tolerances, including TileLang dtype defaults."""

    thresholds = {
        "atol": float(target.validate.get("atol", 1e-5)),
        "rtol": float(target.validate.get("rtol", 1e-5)),
    }
    if target.engine != "tilelang":
        return thresholds
    if list(target.patterns or []) == ["attention"]:
        thresholds["atol"] = max(thresholds["atol"], 1e-2)
        thresholds["rtol"] = max(thresholds["rtol"], 1e-2)
    from xqt.kernels.ops._impl.engines.tilelang import tilelang_validation_thresholds

    dtype_defaults = tilelang_validation_thresholds(optimized_output.dtype)
    return {
        "atol": max(float(dtype_defaults["atol"]), thresholds["atol"]),
        "rtol": max(float(dtype_defaults["rtol"]), thresholds["rtol"]),
    }


def ordered_unique(values: Iterable[str]) -> list[str]:
    """Return values in first-seen order without duplicates."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


__all__ = [
    "call_module",
    "call_module_no_grad",
    "effective_validation_thresholds",
    "infer_module_device_dtype",
    "iter_tensors",
    "move_to_device",
    "ordered_unique",
    "shape_signature",
]
