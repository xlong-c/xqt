"""Context field accessors, device helpers, and runtime config builders."""

from __future__ import annotations

import copy
from typing import Any, Mapping, TypeVar, cast

import torch

from xqt.core.schema import (
    AnalysisConfig,
    BenchmarkConfig,
    OperatorOptimizationConfig,
    OutputDiffConfig,
    PruneConfig,
    QuantConfig,
)
from xqt.core.types import XQTContext
from xqt.core.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    OperatorStageSpec,
    QuantStageSpec,
    stage_spec_to_config,
)


_T = TypeVar("_T")


def _require_context_field(value: _T | None, field_name: str) -> _T:
    if value is None:
        raise ValueError(f"XQTContext.{field_name} is required")
    return value


def _move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move nested batch structures onto the target device."""

    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    return data


def _context_analysis_config(context: XQTContext) -> AnalysisConfig:
    return _require_context_field(context.analysis_config, "analysis_config")


def _context_quant_config(context: XQTContext) -> QuantConfig:
    return _require_context_field(context.quant_config, "quant_config")


def _context_prune_config(context: XQTContext) -> PruneConfig:
    return _require_context_field(context.prune_config, "prune_config")


def _context_output_diff_config(context: XQTContext) -> OutputDiffConfig:
    return _require_context_field(context.output_diff_config, "output_diff_config")


def _context_operator_config(context: XQTContext) -> OperatorOptimizationConfig:
    return _require_context_field(context.operator_config, "operator_config")


def _context_benchmark_config(context: XQTContext) -> BenchmarkConfig:
    return _require_context_field(context.benchmark_config, "benchmark_config")


def _context_task_type(context: XQTContext) -> str:
    return str(context.task_type or "")


def _context_model_target(context: XQTContext) -> str | None:
    return context.model_target


def _context_model_params(context: XQTContext) -> dict[str, Any]:
    return dict(_require_context_field(context.model_params, "model_params"))


def _quant_runtime_config(
    resolved_quant: QuantConfig | QuantStageSpec,
) -> QuantConfig:
    if isinstance(resolved_quant, QuantConfig):
        return copy.deepcopy(resolved_quant)
    return stage_spec_to_config(
        resolved_quant,
        QuantConfig,
        overrides={"enabled": True},
    )


def _benchmark_runtime_config(
    spec: BenchmarkStageSpec,
    *,
    base: BenchmarkConfig | None = None,
) -> BenchmarkConfig:
    return stage_spec_to_config(spec, BenchmarkConfig, base=base)


def _analysis_runtime_config(spec: AnalyzeStageSpec) -> AnalysisConfig:
    return stage_spec_to_config(
        spec,
        AnalysisConfig,
        overrides={"enabled": True},
    )


def _operator_runtime_config(spec: OperatorStageSpec) -> OperatorOptimizationConfig:
    return stage_spec_to_config(
        spec,
        OperatorOptimizationConfig,
        overrides={"enabled": True},
        exclude={"benchmark"},
    )
