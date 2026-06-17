"""Layer sensitivity analysis for mixed-precision quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from xqt.distill.hooks import collect_module_outputs
from xqt.eval.compare import TensorDiff, compare_tensors

from .policy import QuantizationPolicy, list_quantizable_modules


@dataclass
class LayerSensitivityRecord:
    """Per-layer sensitivity result."""

    name: str
    module_type: str
    diff: TensorDiff
    parameter_count: int


def _shared_module_names(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    policy: Optional[QuantizationPolicy] = None,
) -> list[str]:
    reference_candidates = list_quantizable_modules(reference_model, policy)
    candidate_names = {candidate.name for candidate in list_quantizable_modules(candidate_model, policy)}
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


__all__ = [
    "LayerSensitivityRecord",
    "analyze_layer_sensitivity",
    "suggest_high_precision_modules",
]
