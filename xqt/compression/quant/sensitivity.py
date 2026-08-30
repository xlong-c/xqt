"""Layer sensitivity analysis for mixed-precision quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from xqt.kernels.nn.fixtures.hooks import collect_module_outputs
from xqt.analysis.compare import TensorDiff, compare_tensors, summarize_tensor

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
    candidate_modules = dict(candidate_model.named_modules())
    return [
        candidate.name
        for candidate in reference_candidates
        if candidate.name in candidate_names
        or (
            candidate.name in candidate_modules
            and (
                hasattr(candidate_modules[candidate.name], "dequantize_weight")
                or hasattr(candidate_modules[candidate.name], "weight")
            )
        )
    ]


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _get_submodule_or_key_error(model: nn.Module, name: str) -> nn.Module:
    try:
        return model.get_submodule(name)
    except AttributeError as exc:
        raise KeyError(f"Modules not found: ['{name}']") from exc


def _first_tensor_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        "layer sensitivity requires a Tensor, tuple/list Tensor[0], or logits mapping output"
    )


def _sample_tensor_pair(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    sample_budget: Optional[int],
    sample_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Tensor shapes differ: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )
    if sample_budget is None or sample_budget <= 0 or reference.numel() <= sample_budget:
        return reference, candidate

    flat_reference = reference.detach().reshape(-1)
    flat_candidate = candidate.detach().reshape(-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_seed)
    indices = torch.randperm(int(flat_reference.numel()), generator=generator)[:sample_budget]
    indices = indices.to(device=flat_reference.device)
    return (
        flat_reference.index_select(0, indices),
        flat_candidate.index_select(0, indices),
    )


def _compare_sampled_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    sample_budget: Optional[int],
    sample_seed: int,
    per_channel: bool = False,
    per_token: bool = False,
) -> TensorDiff:
    sampled_reference, sampled_candidate = _sample_tensor_pair(
        reference,
        candidate,
        sample_budget=sample_budget,
        sample_seed=sample_seed,
    )
    return compare_tensors(
        sampled_reference,
        sampled_candidate,
        atol=atol,
        rtol=rtol,
        per_channel=per_channel,
        per_token=per_token,
    )


def _collect_shared_tensor_outputs(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *forward_args: object,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    module_names: Optional[Sequence[str]] = None,
    policy: Optional[QuantizationPolicy] = None,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    names = list(module_names) if module_names is not None else _shared_module_names(
        reference_model,
        candidate_model,
        policy,
    )
    if not names:
        return [], {}, {}

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
    return names, reference_outputs, candidate_outputs


def _analyze_layer_output_drift(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *forward_args: object,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    module_names: Optional[Sequence[str]] = None,
    policy: Optional[QuantizationPolicy] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    sample_budget: Optional[int] = None,
    sample_seed: int = 0,
    per_channel: bool = False,
    per_token: bool = False,
) -> list[LayerSensitivityRecord]:
    names, reference_outputs, candidate_outputs = _collect_shared_tensor_outputs(
        reference_model,
        candidate_model,
        *forward_args,
        forward_kwargs=forward_kwargs,
        module_names=module_names,
        policy=policy,
    )
    if not names:
        return []

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
        diff = _compare_sampled_tensors(
            reference_output,
            candidate_output,
            atol=atol,
            rtol=rtol,
            sample_budget=sample_budget,
            sample_seed=sample_seed,
            per_channel=per_channel,
            per_token=per_token,
        )
        reference_module = reference_model.get_submodule(name)
        records.append(
            LayerSensitivityRecord(
                name=name,
                module_type=type(reference_module).__name__,
                diff=diff,
                parameter_count=_parameter_count(reference_module),
            )
        )

    records.sort(key=lambda record: record.diff.mean_abs, reverse=True)
    return records


def _run_model_output(
    model: nn.Module,
    *forward_args: object,
    forward_kwargs: Optional[Mapping[str, object]] = None,
) -> torch.Tensor:
    with torch.no_grad():
        output = model(*forward_args, **dict(forward_kwargs or {}))
    return _first_tensor_output(output)


def _run_with_isolated_module(
    reference_model: nn.Module,
    candidate_module: nn.Module,
    target_name: str,
    *forward_args: object,
    forward_kwargs: Optional[Mapping[str, object]] = None,
) -> torch.Tensor:
    reference_module = reference_model.get_submodule(target_name)

    def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> Any:
        with torch.no_grad():
            return candidate_module(*inputs)

    handle = reference_module.register_forward_hook(hook)
    try:
        return _run_model_output(
            reference_model,
            *forward_args,
            forward_kwargs=forward_kwargs,
        )
    finally:
        handle.remove()


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


def analyze_layer_sensitivity(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *forward_args: object,
    forward_kwargs: Optional[Mapping[str, object]] = None,
    module_names: Optional[Sequence[str]] = None,
    policy: Optional[QuantizationPolicy] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    sample_budget: Optional[int] = None,
    sample_seed: int = 0,
    per_channel: bool = False,
    per_token: bool = False,
) -> list[LayerSensitivityRecord]:
    """Measure isolated final-output drift when replacing one module at a time."""

    names = list(module_names) if module_names is not None else _shared_module_names(
        reference_model,
        candidate_model,
        policy,
    )
    if not names:
        return []

    if reference_model is candidate_model:
        records: list[LayerSensitivityRecord] = []
        for name in names:
            reference_module = _get_submodule_or_key_error(reference_model, name)
            zero_diff = compare_tensors(
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
                atol=atol,
                rtol=rtol,
            )
            records.append(
                LayerSensitivityRecord(
                    name=name,
                    module_type=type(reference_module).__name__,
                    diff=zero_diff,
                    parameter_count=_parameter_count(reference_module),
                )
            )
        return records

    reference_output = _run_model_output(
        reference_model,
        *forward_args,
        forward_kwargs=forward_kwargs,
    )

    records: list[LayerSensitivityRecord] = []
    for name in names:
        reference_module = _get_submodule_or_key_error(reference_model, name)
        candidate_module = _get_submodule_or_key_error(candidate_model, name)
        isolated_output = _run_with_isolated_module(
            reference_model,
            candidate_module,
            name,
            *forward_args,
            forward_kwargs=forward_kwargs,
        )
        diff = _compare_sampled_tensors(
            reference_output,
            isolated_output,
            atol=atol,
            rtol=rtol,
            sample_budget=sample_budget,
            sample_seed=sample_seed,
            per_channel=per_channel,
            per_token=per_token,
        )
        records.append(
            LayerSensitivityRecord(
                name=name,
                module_type=type(reference_module).__name__,
                diff=diff,
                parameter_count=_parameter_count(reference_module),
            )
        )

    records.sort(key=lambda record: record.diff.mean_abs, reverse=True)
    return records


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
    sample_budget: Optional[int] = None,
    sample_seed: int = 0,
    per_channel: bool = False,
    per_token: bool = False,
) -> list[LayerAnalysisRecord]:
    """Build a richer per-layer cumulative error table for reports."""

    drift_records = _analyze_layer_output_drift(
        reference_model,
        candidate_model,
        *forward_args,
        forward_kwargs=forward_kwargs,
        module_names=module_names,
        policy=policy,
        atol=atol,
        rtol=rtol,
        sample_budget=sample_budget,
        sample_seed=sample_seed,
        per_channel=per_channel,
        per_token=per_token,
    )
    if not drift_records:
        return []

    names = [record.name for record in drift_records]
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
    for record in drift_records:
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

    analysis_records.sort(key=lambda record: record.diff.mean_abs, reverse=True)
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
