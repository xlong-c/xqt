"""Layer analysis helpers shared by reports and practice examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch
from torch import nn


class _TensorDiffLike(Protocol):
    max_abs: float
    mean_abs: float
    mean_squared: float
    sqnr_db: Optional[float]
    cosine_similarity: Optional[float]
    argmax_mismatch_rate: Optional[float]
    reference_summary: Optional[Any]


class _LayerAnalysisRecordLike(Protocol):
    name: str
    module_type: str
    diff: _TensorDiffLike
    reference_summary: Mapping[str, object]
    weight_diff: Optional[_TensorDiffLike]
    recommendation: Optional[str]
    tags: Sequence[str]


class _LayerSensitivityRecordLike(Protocol):
    name: str
    module_type: str
    diff: _TensorDiffLike


@dataclass(frozen=True)
class AvoidListEntry:
    """One suggested avoid-list row derived from layer analysis."""

    layer: str
    reason: str
    suggested_actions: tuple[str, ...]
    used_by: str

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "reason": self.reason,
            "suggested_actions": list(self.suggested_actions),
            "used_by": self.used_by,
        }


def _split_example_input(
    example_input: object,
) -> tuple[tuple[object, ...], Optional[Mapping[str, object]]]:
    if isinstance(example_input, Mapping):
        return (), dict(example_input)
    if isinstance(example_input, tuple):
        return example_input, None
    if isinstance(example_input, list):
        return tuple(example_input), None
    return (example_input,), None


def _list_shape(value: object) -> Optional[list[int]]:
    if not isinstance(value, Mapping):
        return None
    shape = value.get("shape")
    if not isinstance(shape, list):
        return None
    normalized: list[int] = []
    for item in shape:
        if not isinstance(item, int):
            return None
        normalized.append(item)
    return normalized


def layer_error_rows(
    records: Sequence[_LayerAnalysisRecordLike],
    *,
    top_k: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
) -> list[dict[str, object]]:
    """Convert cumulative layer error records to JSONL-friendly rows."""

    selected_metrics = {str(metric) for metric in metrics} if metrics is not None else None
    ranked = sorted(records, key=lambda record: record.diff.mean_abs, reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    rows: list[dict[str, object]] = []
    for index, record in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": index,
                "layer": record.name,
                "type": record.module_type,
                "weight_shape": (
                    _list_shape(record.weight_diff.reference_summary.to_dict())
                    if record.weight_diff is not None
                    and record.weight_diff.reference_summary is not None
                    else None
                ),
                "act_shape": _list_shape(record.reference_summary),
                "weight": (
                    {
                        "mae": record.weight_diff.mean_abs,
                        "max": record.weight_diff.max_abs,
                        "cosine_similarity": record.weight_diff.cosine_similarity,
                        "snr_db": record.weight_diff.sqnr_db,
                    }
                    if record.weight_diff is not None
                    else None
                ),
                "activation": {
                    key: value
                    for key, value in {
                        "mae": record.diff.mean_abs,
                        "max": record.diff.max_abs,
                        "cosine_similarity": record.diff.cosine_similarity,
                        "snr_db": record.diff.sqnr_db,
                        "mse": record.diff.mean_squared,
                    }.items()
                    if selected_metrics is None
                    or key in selected_metrics
                    or (
                        key == "mae"
                        and "mean_abs" in selected_metrics
                    )
                    or (
                        key == "max"
                        and "max_abs" in selected_metrics
                    )
                },
                "argmax_mismatch": record.diff.argmax_mismatch_rate,
                "tags": list(record.tags),
                "recommendation": record.recommendation,
            }
        )
    return rows


def layer_sensitivity_rows(
    records: Sequence[_LayerSensitivityRecordLike],
    *,
    top_k: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
) -> list[dict[str, object]]:
    """Convert layer sensitivity records to JSONL-friendly rows."""

    selected_metrics = {str(metric) for metric in metrics} if metrics is not None else None
    ranked = sorted(records, key=lambda record: record.diff.mean_abs, reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    rows: list[dict[str, object]] = []
    for index, record in enumerate(ranked, start=1):
        recommendation = "keep_fp32" if index == 1 and record.diff.mean_abs > 0.0 else None
        rows.append(
            {
                "rank": index,
                "layer": record.name,
                "type": record.module_type,
                "sensitivity": {
                    key: value
                    for key, value in {
                        "snr_db": record.diff.sqnr_db,
                        "cosine_similarity": record.diff.cosine_similarity,
                        "mse": record.diff.mean_squared,
                        "mae": record.diff.mean_abs,
                        "max": record.diff.max_abs,
                    }.items()
                    if selected_metrics is None
                    or key in selected_metrics
                    or key == "mse"
                    or (
                        key == "mae"
                        and "mean_abs" in selected_metrics
                    )
                    or (
                        key == "max"
                        and "max_abs" in selected_metrics
                    )
                },
                "recommendation": recommendation,
            }
        )
    return rows


def _sample_flat_tensor(
    tensor: torch.Tensor,
    *,
    sample_budget: Optional[int],
    sample_seed: int,
) -> torch.Tensor:
    flat = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if sample_budget is None or sample_budget <= 0 or flat.numel() <= sample_budget:
        return flat
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_seed)
    indices = torch.randperm(int(flat.numel()), generator=generator)[:sample_budget]
    return flat.index_select(0, indices)


def _distribution_summary(
    tensor: torch.Tensor,
    *,
    histogram_bins: int = 32,
) -> dict[str, object]:
    flat = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if flat.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "histogram_bins": histogram_bins,
            "histogram_range": [0.0, 0.0],
            "histogram": [0] * histogram_bins,
        }
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
    minimum = float(flat.min().item())
    maximum = float(flat.max().item())
    centered = flat - mean
    if std > 0.0:
        normalized = centered / std
        skewness = float((normalized.pow(3).mean()).item())
        kurtosis = float((normalized.pow(4).mean()).item())
    else:
        skewness = 0.0
        kurtosis = 0.0
    if minimum == maximum:
        histogram = [0] * histogram_bins
        histogram[0] = int(flat.numel())
    else:
        histogram = [
            int(value)
            for value in torch.histc(
                flat,
                bins=histogram_bins,
                min=minimum,
                max=maximum,
            ).tolist()
        ]
    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "histogram_bins": histogram_bins,
        "histogram_range": [minimum, maximum],
        "histogram": histogram,
    }


def layer_statistics_rows(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    example_input: object,
    *,
    module_names: Sequence[str],
    sample_budget: Optional[int] = None,
    sample_seed: int = 0,
    histogram_bins: int = 32,
) -> list[dict[str, object]]:
    """Build optional per-layer activation and weight distribution rows."""

    if not module_names:
        return []

    from xqt.model.hooks import collect_module_outputs

    forward_args, forward_kwargs = _split_example_input(example_input)
    reference_outputs = collect_module_outputs(
        reference_model,
        *forward_args,
        module_names=module_names,
        forward_kwargs=forward_kwargs,
    )
    candidate_outputs = collect_module_outputs(
        candidate_model,
        *forward_args,
        module_names=module_names,
        forward_kwargs=forward_kwargs,
    )
    rows: list[dict[str, object]] = []
    for index, name in enumerate(module_names):
        reference_output = reference_outputs.get(name)
        candidate_output = candidate_outputs.get(name)
        if isinstance(reference_output, torch.Tensor) and isinstance(candidate_output, torch.Tensor):
            reference_sample = _sample_flat_tensor(
                reference_output,
                sample_budget=sample_budget,
                sample_seed=sample_seed + index,
            )
            candidate_sample = _sample_flat_tensor(
                candidate_output,
                sample_budget=sample_budget,
                sample_seed=sample_seed + index,
            )
            error_sample = candidate_sample - reference_sample
            rows.append(
                {
                    "layer": name,
                    "variable": "output",
                    "error": _distribution_summary(
                        error_sample,
                        histogram_bins=histogram_bins,
                    ),
                    "quantized": _distribution_summary(
                        candidate_sample,
                        histogram_bins=histogram_bins,
                    ),
                    "float": _distribution_summary(
                        reference_sample,
                        histogram_bins=histogram_bins,
                    ),
                }
            )

        reference_module = reference_model.get_submodule(name)
        candidate_module = candidate_model.get_submodule(name)
        reference_weight = getattr(reference_module, "weight", None)
        candidate_weight = getattr(candidate_module, "weight", None)
        if not isinstance(reference_weight, torch.Tensor) or not isinstance(candidate_weight, torch.Tensor):
            continue
        if reference_weight.shape != candidate_weight.shape:
            continue
        reference_weight_sample = _sample_flat_tensor(
            reference_weight,
            sample_budget=sample_budget,
            sample_seed=sample_seed + 10_000 + index,
        )
        candidate_weight_sample = _sample_flat_tensor(
            candidate_weight,
            sample_budget=sample_budget,
            sample_seed=sample_seed + 10_000 + index,
        )
        weight_error_sample = candidate_weight_sample - reference_weight_sample
        rows.append(
            {
                "layer": name,
                "variable": "weight",
                "error": _distribution_summary(
                    weight_error_sample,
                    histogram_bins=histogram_bins,
                ),
                "quantized": _distribution_summary(
                    candidate_weight_sample,
                    histogram_bins=histogram_bins,
                ),
                "float": _distribution_summary(
                    reference_weight_sample,
                    histogram_bins=histogram_bins,
                ),
            }
        )
    return rows


def build_avoid_list(
    layer_error_rows_value: Sequence[Mapping[str, object]],
    layer_sensitivity_rows_value: Sequence[Mapping[str, object]],
    *,
    top_k: Optional[int] = None,
    used_by: str = "quant_retry",
) -> list[dict[str, object]]:
    """Build a suggestion-only avoid list from layer error signals."""

    selected_error_rows = list(layer_error_rows_value[:top_k] if top_k is not None else layer_error_rows_value)
    sensitivity_by_layer = {
        str(row.get("layer")): row
        for row in layer_sensitivity_rows_value
        if row.get("layer") is not None
    }
    selected_sensitivity_rows = list(
        layer_sensitivity_rows_value[:top_k]
        if top_k is not None
        else layer_sensitivity_rows_value
    )
    entries: list[AvoidListEntry] = []
    rows_to_merge: list[tuple[str, Mapping[str, object], Mapping[str, object]] | tuple[str, None, Mapping[str, object]]] = []
    seen_layers: set[str] = set()
    for row in selected_error_rows:
        layer_name = row.get("layer")
        if not isinstance(layer_name, str):
            continue
        seen_layers.add(layer_name)
        rows_to_merge.append((layer_name, row, sensitivity_by_layer.get(layer_name, {})))
    for row in selected_sensitivity_rows:
        layer_name = row.get("layer")
        if not isinstance(layer_name, str) or layer_name in seen_layers:
            continue
        seen_layers.add(layer_name)
        rows_to_merge.append((layer_name, None, row))

    for layer_name, error_row, sensitivity_row in rows_to_merge:
        suggested_actions: list[str] = ["review_model_transform"]
        recommendation = error_row.get("recommendation") if error_row is not None else None
        sensitivity_recommendation = sensitivity_row.get("recommendation")
        if (
            recommendation in {"consider_higher_precision", "keep_fp32"}
            or sensitivity_recommendation == "keep_fp32"
        ):
            suggested_actions.append("keep_fp32")
        elif "prune" in used_by:
            suggested_actions.append("skip_prune")
        else:
            suggested_actions.append("skip_quant")
        reason = "high_sensitivity" if error_row is None else "high_error"
        if error_row is not None:
            tags = error_row.get("tags")
            if isinstance(tags, list) and tags:
                first_tag = tags[0]
                if isinstance(first_tag, str):
                    reason = first_tag
        entries.append(
            AvoidListEntry(
                layer=layer_name,
                reason=reason,
                suggested_actions=tuple(dict.fromkeys(suggested_actions)),
                used_by=used_by,
            )
        )
    return [entry.to_dict() for entry in entries]


def build_layer_analysis_payload(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    example_input: object,
    *,
    module_names: Optional[Sequence[str]] = None,
    policy: object = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    include_weight_diff: bool = True,
    include_sensitivity: bool = True,
    include_statistics: bool = False,
    include_avoid_list: bool = True,
    metrics: Optional[Sequence[str]] = None,
    row_top_k: Optional[int] = None,
    avoid_top_k: Optional[int] = None,
    sample_budget: Optional[int] = None,
    sample_seed: int = 0,
    runtime: str = "current_pytorch",
    avoid_used_by: str = "quant_retry",
    per_channel: bool = False,
    per_token: bool = False,
) -> dict[str, object]:
    """Build reusable layer-analysis payload for reports and JSONL events."""

    from xqt.quant.sensitivity import (
        analyze_layer_errors,
        analyze_layer_sensitivity,
    )

    forward_args, forward_kwargs = _split_example_input(example_input)
    error_records = analyze_layer_errors(
        reference_model,
        candidate_model,
        *forward_args,
        forward_kwargs=forward_kwargs,
        module_names=module_names,
        policy=policy,
        atol=atol,
        rtol=rtol,
        include_weight_diff=include_weight_diff,
        sample_budget=sample_budget,
        sample_seed=sample_seed,
        per_channel=per_channel,
        per_token=per_token,
    )
    sensitivity_records = (
        analyze_layer_sensitivity(
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
        if include_sensitivity
        else []
    )
    error_rows = layer_error_rows(error_records, top_k=row_top_k, metrics=metrics)
    sensitivity_rows_value = layer_sensitivity_rows(
        sensitivity_records,
        top_k=row_top_k,
        metrics=metrics,
    )
    statistics_rows_value = (
        layer_statistics_rows(
            reference_model,
            candidate_model,
            example_input,
            module_names=[str(row["layer"]) for row in error_rows if isinstance(row.get("layer"), str)],
            sample_budget=sample_budget,
            sample_seed=sample_seed,
        )
        if include_statistics
        else []
    )
    return {
        "runtime": runtime,
        "mode": "cumulative",
        "sample_budget": sample_budget,
        "layer_errors": error_rows,
        "layer_sensitivity": sensitivity_rows_value,
        "layer_statistics": statistics_rows_value,
        "avoid_list": (
            build_avoid_list(
                error_rows,
                sensitivity_rows_value,
                top_k=avoid_top_k,
                used_by=avoid_used_by,
            )
            if include_avoid_list
            else []
        ),
    }


def build_layer_analysis_events(
    step_name: str,
    layer_analysis: Mapping[str, object],
) -> list[dict[str, object]]:
    """Convert a layer-analysis payload into JSONL events."""

    events: list[dict[str, object]] = []
    layer_errors_value = layer_analysis.get("layer_errors")
    if isinstance(layer_errors_value, list) and layer_errors_value:
        events.append(
            {
                "schema_version": 1,
                "event": "layer_errors",
                "step": step_name,
                "runtime": layer_analysis.get("runtime", "current_pytorch"),
                "mode": layer_analysis.get("mode", "cumulative"),
                "sample_budget": layer_analysis.get("sample_budget"),
                "rows": layer_errors_value,
            }
        )
    layer_sensitivity_value = layer_analysis.get("layer_sensitivity")
    if isinstance(layer_sensitivity_value, list) and layer_sensitivity_value:
        events.append(
            {
                "schema_version": 1,
                "event": "layer_sensitivity",
                "step": step_name,
                "runtime": layer_analysis.get("runtime", "current_pytorch"),
                "mode": "isolated",
                "sample_budget": layer_analysis.get("sample_budget"),
                "rows": layer_sensitivity_value,
            }
        )
    layer_statistics_value = layer_analysis.get("layer_statistics")
    if isinstance(layer_statistics_value, list) and layer_statistics_value:
        for row in layer_statistics_value:
            if not isinstance(row, Mapping):
                continue
            events.append(
                {
                    "schema_version": 1,
                    "event": "layer_statistics",
                    "step": step_name,
                    "runtime": layer_analysis.get("runtime", "current_pytorch"),
                    **dict(row),
                }
            )
    avoid_list_value = layer_analysis.get("avoid_list")
    if isinstance(avoid_list_value, list) and avoid_list_value:
        events.append(
            {
                "schema_version": 1,
                "event": "avoid_list",
                "step": step_name,
                "rows": avoid_list_value,
            }
        )
    return events


def collect_top_layer_errors(
    scenarios: Mapping[str, object],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    """Collect top layer error rows across scenarios for run_finish/report summaries."""

    rows: list[dict[str, object]] = []
    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, Mapping):
            continue
        layer_analysis = scenario.get("layer_analysis")
        if not isinstance(layer_analysis, Mapping):
            continue
        layer_rows = layer_analysis.get("layer_errors")
        if not isinstance(layer_rows, list):
            continue
        for row in layer_rows:
            if not isinstance(row, Mapping):
                continue
            activation = row.get("activation")
            if not isinstance(activation, Mapping):
                continue
            rows.append(
                {
                    "scenario": str(scenario_name),
                    "runtime": str(layer_analysis.get("runtime") or "current_pytorch"),
                    "layer": row.get("layer"),
                    "type": row.get("type"),
                    "act_mae": activation.get("mae"),
                    "act_max": activation.get("max"),
                    "act_cosine_similarity": activation.get("cosine_similarity"),
                    "act_snr_db": activation.get("snr_db"),
                    "recommendation": row.get("recommendation"),
                    "tags": row.get("tags"),
                }
            )
    rows.sort(
        key=lambda item: (
            float(item.get("act_mae") or 0.0),
            float(item.get("act_max") or 0.0),
        ),
        reverse=True,
    )
    return rows[:top_k]


def top_layer_error_for_scenario(
    scenarios: Mapping[str, object],
    scenario_name: str,
) -> Mapping[str, object] | None:
    """Return the highest-ranked layer error row for one scenario."""

    scenario = scenarios.get(scenario_name)
    if not isinstance(scenario, Mapping):
        return None
    layer_analysis = scenario.get("layer_analysis")
    if not isinstance(layer_analysis, Mapping):
        return None
    layer_rows = layer_analysis.get("layer_errors")
    if not isinstance(layer_rows, list) or not layer_rows:
        return None
    first = layer_rows[0]
    return first if isinstance(first, Mapping) else None


__all__ = [
    "AvoidListEntry",
    "build_avoid_list",
    "build_layer_analysis_events",
    "build_layer_analysis_payload",
    "collect_top_layer_errors",
    "layer_error_rows",
    "layer_statistics_rows",
    "layer_sensitivity_rows",
    "top_layer_error_for_scenario",
]
