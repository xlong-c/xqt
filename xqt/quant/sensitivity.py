"""Layer sensitivity analysis for mixed-precision quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from xqt.distill.hooks import collect_module_outputs
from xqt.eval.compare import TensorDiff, compare_tensors, summarize_tensor

from .policy import QuantizationPolicy, list_quantizable_modules


@dataclass
class LayerSensitivityRecord:
    """Per-layer sensitivity result."""

    name: str
    module_type: str
    diff: TensorDiff
    parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a plain dictionary."""

        return {
            "name": self.name,
            "module_type": self.module_type,
            "diff": self.diff.to_dict(),
            "parameter_count": self.parameter_count,
        }


@dataclass
class LayerAnalysisRecord:
    """Richer per-layer record for reports and optimization suggestions."""

    name: str
    module_type: str
    diff: TensorDiff
    parameter_count: int
    reference_summary: dict[str, object]
    candidate_summary: dict[str, object]
    weight_diff: Optional[TensorDiff] = None
    recommendation: Optional[str] = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a plain dictionary."""

        return {
            "name": self.name,
            "module_type": self.module_type,
            "diff": self.diff.to_dict(),
            "parameter_count": self.parameter_count,
            "reference_summary": dict(self.reference_summary),
            "candidate_summary": dict(self.candidate_summary),
            "weight_diff": self.weight_diff.to_dict() if self.weight_diff is not None else None,
            "recommendation": self.recommendation,
            "tags": list(self.tags),
        }


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


def analyze_layer_sensitivity(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *forward_args: object,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    module_names: Optional[Sequence[str]] = None,
    policy: Optional[QuantizationPolicy] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> list[LayerSensitivityRecord]:
    """Compare named layer outputs between two models."""

    names = list(module_names) if module_names is not None else _shared_module_names(
        reference_model,
        candidate_model,
        policy,
    )
    if not names:
        return []

    reference_outputs = collect_module_outputs(
        reference_model,
        *forward_args,
        module_names=names,
        forward_kwargs=forward_kwargs,
    )
    candidate_outputs = collect_module_outputs(
        candidate_model,
        *forward_args,
        module_names=names,
        forward_kwargs=forward_kwargs,
    )

    records: list[LayerSensitivityRecord] = []
    for name in names:
        if name not in reference_outputs or name not in candidate_outputs:
            continue
        reference_output = reference_outputs[name]
        candidate_output = candidate_outputs[name]
        if not isinstance(reference_output, torch.Tensor) or not isinstance(
            candidate_output, torch.Tensor
        ):
            continue
        if reference_output.shape != candidate_output.shape:
            raise ValueError(
                f"Layer '{name}' shape mismatch: {tuple(reference_output.shape)} vs "
                f"{tuple(candidate_output.shape)}"
            )
        diff = compare_tensors(reference_output, candidate_output, atol=atol, rtol=rtol)
        module_type = type(reference_model.get_submodule(name)).__name__
        parameter_count = sum(
            parameter.numel()
            for parameter in reference_model.get_submodule(name).parameters(recurse=False)
        )
        records.append(
            LayerSensitivityRecord(
                name=name,
                module_type=module_type,
                diff=diff,
                parameter_count=parameter_count,
            )
        )

    records.sort(key=lambda record: record.diff.max_abs, reverse=True)
    return records


def _get_module_weight(module: nn.Module) -> Optional[torch.Tensor]:
    if not hasattr(module, "weight"):
        return None
    weight = getattr(module, "weight")
    if weight is None:
        return None
    if hasattr(weight, "dequantize"):
        try:
            weight = weight.dequantize()
        except (NotImplementedError, RuntimeError, TypeError):
            return None
    if not isinstance(weight, torch.Tensor):
        return None
    return weight.detach().to(dtype=torch.float32, device="cpu")


def analyze_layer_errors(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *forward_args: object,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    module_names: Optional[Sequence[str]] = None,
    policy: Optional[QuantizationPolicy] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    include_weight_diff: bool = True,
) -> list[LayerAnalysisRecord]:
    """Build a richer per-layer error table for reports and optimization hints."""

    sensitivity_records = analyze_layer_sensitivity(
        reference_model,
        candidate_model,
        *forward_args,
        forward_kwargs=forward_kwargs,
        module_names=module_names,
        policy=policy,
        atol=atol,
        rtol=rtol,
    )
    if not sensitivity_records:
        return []

    names = [record.name for record in sensitivity_records]
    reference_outputs = collect_module_outputs(
        reference_model,
        *forward_args,
        module_names=names,
        forward_kwargs=forward_kwargs,
    )
    candidate_outputs = collect_module_outputs(
        candidate_model,
        *forward_args,
        module_names=names,
        forward_kwargs=forward_kwargs,
    )

    analysis_records: list[LayerAnalysisRecord] = []
    for record in sensitivity_records:
        reference_output = reference_outputs[record.name]
        candidate_output = candidate_outputs[record.name]
        if not isinstance(reference_output, torch.Tensor) or not isinstance(
            candidate_output, torch.Tensor
        ):
            continue

        reference_module = reference_model.get_submodule(record.name)
        candidate_module = candidate_model.get_submodule(record.name)
        reference_weight = _get_module_weight(reference_module)
        candidate_weight = _get_module_weight(candidate_module)
        weight_diff: Optional[TensorDiff] = None
        if (
            include_weight_diff
            and reference_weight is not None
            and candidate_weight is not None
            and reference_weight.shape == candidate_weight.shape
        ):
            weight_diff = compare_tensors(
                reference_weight,
                candidate_weight,
                atol=atol,
                rtol=rtol,
            )

        recommendation: Optional[str] = None
        tags: list[str] = []
        if record.diff.mean_abs > atol * 10.0:
            recommendation = "consider_higher_precision"
            tags.append("high_error")
        if weight_diff is not None and weight_diff.mean_abs > atol * 10.0:
            tags.append("weight_shift")

        analysis_records.append(
            LayerAnalysisRecord(
                name=record.name,
                module_type=record.module_type,
                diff=record.diff,
                parameter_count=record.parameter_count,
                reference_summary=summarize_tensor(reference_output).to_dict(),
                candidate_summary=summarize_tensor(candidate_output).to_dict(),
                weight_diff=weight_diff,
                recommendation=recommendation,
                tags=tuple(tags),
            )
        )

    return analysis_records


def suggest_high_precision_modules(
    records: Sequence[LayerSensitivityRecord],
    *,
    top_k: Optional[int] = None,
    max_abs_threshold: Optional[float] = None,
) -> list[str]:
    """Return module names that should stay in higher precision."""

    ranked = sorted(records, key=lambda record: record.diff.max_abs, reverse=True)
    if max_abs_threshold is not None:
        ranked = [record for record in ranked if record.diff.max_abs >= max_abs_threshold]
    if top_k is not None:
        ranked = ranked[:top_k]
    return [record.name for record in ranked]


def recommend_high_precision_modules(
    records: Sequence[LayerAnalysisRecord],
    *,
    top_k: Optional[int] = None,
    mean_abs_threshold: Optional[float] = None,
    require_weight_shift: bool = False,
) -> list[str]:
    """Return module names that should stay in higher precision based on richer analysis."""

    ranked = sorted(records, key=lambda record: record.diff.mean_abs, reverse=True)
    if mean_abs_threshold is not None:
        ranked = [record for record in ranked if record.diff.mean_abs >= mean_abs_threshold]
    if require_weight_shift:
        ranked = [
            record
            for record in ranked
            if record.weight_diff is not None and record.weight_diff.mean_abs > 0.0
        ]
    if top_k is not None:
        ranked = ranked[:top_k]
    return [record.name for record in ranked]


__all__ = [
    "LayerAnalysisRecord",
    "LayerSensitivityRecord",
    "analyze_layer_errors",
    "analyze_layer_sensitivity",
    "recommend_high_precision_modules",
    "suggest_high_precision_modules",
]
