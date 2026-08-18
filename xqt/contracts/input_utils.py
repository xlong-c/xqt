"""Shared example-input helpers for XQT export paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ExampleInputSpec:
    """Normalized positional/keyword inputs for export helpers."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def split_example_input(example_input: Any) -> ExampleInputSpec:
    """Normalize Tensor, tuple/list, or mapping inputs for model/export calls."""

    if isinstance(example_input, Mapping):
        return ExampleInputSpec(args=(), kwargs=dict(example_input))
    if isinstance(example_input, tuple):
        return ExampleInputSpec(args=example_input, kwargs={})
    if isinstance(example_input, list):
        return ExampleInputSpec(args=tuple(example_input), kwargs={})
    return ExampleInputSpec(args=(example_input,), kwargs={})


def default_input_names(example_input: Any) -> list[str]:
    """Infer stable default input names from an example input structure."""

    normalized = split_example_input(example_input)
    if normalized.kwargs:
        return [str(name) for name in normalized.kwargs.keys()]
    if len(normalized.args) <= 1:
        return ["input"]
    return [f"input_{index}" for index in range(len(normalized.args))]


def call_model_with_example_input(model: nn.Module, example_input: Any) -> Any:
    """Call a module with normalized positional or keyword inputs."""

    normalized = split_example_input(example_input)
    return model(*normalized.args, **normalized.kwargs)


def first_tensor_output(output: Any) -> torch.Tensor:
    """Extract the first tensor-like output used by current diff helpers."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        "export diff currently requires a Tensor, tuple/list Tensor[0], or logits mapping"
    )


def build_onnx_feed(
    example_input: Any,
    *,
    input_names: Optional[Sequence[str]] = None,
) -> dict[str, np.ndarray]:
    """Convert an example input into an ONNX Runtime feed dict."""

    normalized = split_example_input(example_input)
    resolved_names = list(input_names or default_input_names(example_input))

    if normalized.kwargs:
        missing = [name for name in resolved_names if name not in normalized.kwargs]
        if missing:
            raise ValueError(
                f"ONNX Runtime input_names are missing from mapping example_input: {missing}"
            )
        feed: dict[str, np.ndarray] = {}
        for name in resolved_names:
            value = normalized.kwargs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"ONNX Runtime mapping input '{name}' must be a torch.Tensor"
                )
            feed[name] = value.detach().cpu().numpy()
        return feed

    if len(normalized.args) != len(resolved_names):
        raise ValueError("example_input arity must match input_names")
    feed = {}
    for name, value in zip(resolved_names, normalized.args):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"ONNX Runtime input '{name}' must be a torch.Tensor")
        feed[name] = value.detach().cpu().numpy()
    return feed


__all__ = [
    "ExampleInputSpec",
    "build_onnx_feed",
    "call_model_with_example_input",
    "default_input_names",
    "first_tensor_output",
    "split_example_input",
]
