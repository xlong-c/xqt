"""Activation calibration helpers for quantization passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from xqt.model.hooks import ModuleOutputCapture

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
    zero_ratio: float
    saturation_ratio: float
    clipping_ratio: float
    outlier_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the statistic to a plain dictionary."""

        return {
            "name": self.name,
            "module_type": self.module_type,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "samples": self.samples,
            "zero_ratio": self.zero_ratio,
            "saturation_ratio": self.saturation_ratio,
            "clipping_ratio": self.clipping_ratio,
            "outlier_ratio": self.outlier_ratio,
        }


@dataclass
class ActivationDriftRecord:
    """Reference vs candidate activation statistic drift for one module."""

    name: str
    module_type: str
    reference: ActivationStatistic
    candidate: ActivationStatistic
    minimum_delta: float
    maximum_delta: float
    mean_delta: float
    std_delta: float
    range_delta: float
    range_ratio: Optional[float]
    zero_ratio_delta: float
    saturation_ratio_delta: float
    clipping_ratio_delta: float
    outlier_ratio_delta: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the drift record to a plain dictionary."""

        return {
            "name": self.name,
            "module_type": self.module_type,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "minimum_delta": self.minimum_delta,
            "maximum_delta": self.maximum_delta,
            "mean_delta": self.mean_delta,
            "std_delta": self.std_delta,
            "range_delta": self.range_delta,
            "range_ratio": self.range_ratio,
            "zero_ratio_delta": self.zero_ratio_delta,
            "saturation_ratio_delta": self.saturation_ratio_delta,
            "clipping_ratio_delta": self.clipping_ratio_delta,
            "outlier_ratio_delta": self.outlier_ratio_delta,
        }


class _Accumulator:
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0
        self.zero_count = 0
        self.saturation_count = 0
        self.clipping_count = 0
        self.outlier_count = 0

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().to(dtype=torch.float32, device="cpu").flatten()
        if flat.numel() == 0:
            return
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))
        self.sum += float(flat.sum().item())
        self.sumsq += float(torch.dot(flat, flat).item())
        self.count += int(flat.numel())
        abs_flat = flat.abs()
        self.zero_count += int((flat == 0).sum().item())
        max_abs = float(abs_flat.max().item())
        if max_abs > 0.0:
            saturation_threshold = max_abs * 0.99
            clipping_threshold = max_abs * 0.95
            self.saturation_count += int((abs_flat >= saturation_threshold).sum().item())
            self.clipping_count += int((abs_flat >= clipping_threshold).sum().item())

        if flat.numel() > 1:
            mean = float(flat.mean().item())
            std = float(flat.std(unbiased=False).item())
            if std > 0.0:
                self.outlier_count += int((torch.abs(flat - mean) > 3.0 * std).sum().item())

    def to_statistic(self, name: str, module_type: str) -> ActivationStatistic:
        if self.count == 0:
            minimum = 0.0
            maximum = 0.0
            mean = 0.0
            std = 0.0
            zero_ratio = 0.0
            saturation_ratio = 0.0
            clipping_ratio = 0.0
            outlier_ratio = 0.0
        else:
            mean = self.sum / self.count
            variance = max(self.sumsq / self.count - mean * mean, 0.0)
            std = variance**0.5
            minimum = self.minimum
            maximum = self.maximum
            zero_ratio = self.zero_count / self.count
            saturation_ratio = self.saturation_count / self.count
            clipping_ratio = self.clipping_count / self.count
            outlier_ratio = self.outlier_count / self.count
        return ActivationStatistic(
            name=name,
            module_type=module_type,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            samples=self.count,
            zero_ratio=zero_ratio,
            saturation_ratio=saturation_ratio,
            clipping_ratio=clipping_ratio,
            outlier_ratio=outlier_ratio,
        )


def _shared_module_names(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    policy: Optional[QuantizationPolicy] = None,
) -> list[str]:
    reference_candidates = [
        candidate
        for candidate in list_quantizable_modules(reference_model, policy)
        if candidate.quantize
    ]
    candidate_names = {
        candidate.name
        for candidate in list_quantizable_modules(candidate_model, policy)
        if candidate.quantize
    }
    return [
        candidate.name
        for candidate in reference_candidates
        if candidate.name in candidate_names
    ]


def calibrate_activation_statistics(
    model: nn.Module,
    batches: Iterable[object],
    *,
    module_names: Optional[Sequence[str]] = None,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    policy: Optional[QuantizationPolicy] = None,
) -> list[ActivationStatistic]:
    """Run a calibration loop and aggregate activation ranges."""

    names = list(module_names) if module_names is not None else [
        candidate.name
        for candidate in list_quantizable_modules(model, policy)
        if candidate.quantize
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


def analyze_activation_drift(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    batches: Iterable[object],
    *,
    module_names: Optional[Sequence[str]] = None,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    policy: Optional[QuantizationPolicy] = None,
) -> list[ActivationDriftRecord]:
    """Compare activation statistics between reference and candidate models."""

    names = list(module_names) if module_names is not None else _shared_module_names(
        reference_model,
        candidate_model,
        policy,
    )
    if not names:
        return []

    cached_batches = list(batches)
    reference_stats = calibrate_activation_statistics(
        reference_model,
        cached_batches,
        module_names=names,
        forward_kwargs=forward_kwargs,
        policy=policy,
    )
    candidate_stats = calibrate_activation_statistics(
        candidate_model,
        cached_batches,
        module_names=names,
        forward_kwargs=forward_kwargs,
        policy=policy,
    )
    reference_by_name = {stat.name: stat for stat in reference_stats}
    candidate_by_name = {stat.name: stat for stat in candidate_stats}

    drifts: list[ActivationDriftRecord] = []
    for name in names:
        if name not in reference_by_name or name not in candidate_by_name:
            continue
        reference = reference_by_name[name]
        candidate = candidate_by_name[name]
        reference_range = reference.maximum - reference.minimum
        candidate_range = candidate.maximum - candidate.minimum
        if reference_range == 0.0:
            range_ratio = 1.0 if candidate_range == 0.0 else None
        else:
            range_ratio = candidate_range / reference_range
        drifts.append(
            ActivationDriftRecord(
                name=name,
                module_type=reference.module_type,
                reference=reference,
                candidate=candidate,
                minimum_delta=candidate.minimum - reference.minimum,
                maximum_delta=candidate.maximum - reference.maximum,
                mean_delta=candidate.mean - reference.mean,
                std_delta=candidate.std - reference.std,
                range_delta=candidate_range - reference_range,
                range_ratio=range_ratio,
                zero_ratio_delta=candidate.zero_ratio - reference.zero_ratio,
                saturation_ratio_delta=candidate.saturation_ratio - reference.saturation_ratio,
                clipping_ratio_delta=candidate.clipping_ratio - reference.clipping_ratio,
                outlier_ratio_delta=candidate.outlier_ratio - reference.outlier_ratio,
            )
        )

    drifts.sort(
        key=lambda record: max(
            abs(record.mean_delta),
            abs(record.std_delta),
            abs(record.range_delta),
            abs(record.saturation_ratio_delta),
            abs(record.clipping_ratio_delta),
            abs(record.outlier_ratio_delta),
        ),
        reverse=True,
    )
    return drifts


__all__ = [
    "ActivationDriftRecord",
    "ActivationStatistic",
    "analyze_activation_drift",
    "calibrate_activation_statistics",
]
