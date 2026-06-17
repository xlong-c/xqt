"""Activation calibration helpers for quantization passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from xqt.distill.hooks import ModuleOutputCapture

from .policy import QuantizationPolicy, list_quantizable_modules


@dataclass
class ActivationStatistic:
    """Aggregated activation statistics for a module."""

    name: str
    module_type: str
    minimum: float
    maximum: float
    mean: float
    std: float
    samples: int


class _Accumulator:
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().to(dtype=torch.float32, device="cpu").flatten()
        if flat.numel() == 0:
            return
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))
        self.sum += float(flat.sum().item())
        self.sumsq += float(torch.dot(flat, flat).item())
        self.count += int(flat.numel())

    def to_statistic(self, name: str, module_type: str) -> ActivationStatistic:
        if self.count == 0:
            minimum = 0.0
            maximum = 0.0
            mean = 0.0
            std = 0.0
        else:
            mean = self.sum / self.count
            variance = max(self.sumsq / self.count - mean * mean, 0.0)
            std = variance**0.5
            minimum = self.minimum
            maximum = self.maximum
        return ActivationStatistic(
            name=name,
            module_type=module_type,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            samples=self.count,
        )


def calibrate_activation_statistics(
    model: nn.Module,
    batches: Sequence[object],
    *,
    module_names: Optional[Sequence[str]] = None,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    policy: Optional[QuantizationPolicy] = None,
) -> list[ActivationStatistic]:
    """Run a calibration loop and aggregate activation ranges."""

    names = list(module_names) if module_names is not None else [
        candidate.name
        for candidate in list_quantizable_modules(model, policy)
    ]
    if not names:
        return []

    accumulators = {name: _Accumulator() for name in names}
    modules = dict(model.named_modules())
    missing = [name for name in names if name not in modules]
    if missing:
        raise KeyError(f"Modules not found: {missing}")

    handles = []
    try:
        for name in names:
            module = modules[name]

            def make_hook(module_name: str):
                def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
                    if isinstance(output, (tuple, list)):
                        output = output[0] if output else output
                    if isinstance(output, torch.Tensor):
                        accumulators[module_name].update(output)

                return hook

            handles.append(module.register_forward_hook(make_hook(name)))

        kwargs = dict(forward_kwargs or {})
        with torch.no_grad():
            for batch in batches:
                if isinstance(batch, tuple):
                    model(*batch, **kwargs)
                elif isinstance(batch, dict):
                    model(**batch, **kwargs)
                else:
                    model(batch, **kwargs)
    finally:
        while handles:
            handles.pop().remove()

    statistics = [
        accumulators[name].to_statistic(name, type(modules[name]).__name__)
        for name in names
    ]
    return statistics


__all__ = [
    "ActivationStatistic",
    "calibrate_activation_statistics",
]
