"""Operator-engine metadata and execution preflight helpers."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import nn

from xqt.core.types import XQTContext
from xqt.quant.capability import describe_quant_backend_capability

from .backends.cutile import (
    CuTileCompileSettings,
    build_cutile_artifact_metadata,
    list_cutile_kernel_specs,
)
from .backends.cutlass import (
    CutlassCompileSettings,
    build_cutlass_artifact_metadata,
    list_cutlass_kernel_specs,
)
from .backends.cute_dsl import (
    CuteDSLCompileSettings,
    build_cute_dsl_artifact_metadata,
    list_cute_dsl_kernel_specs,
)
from .backends.tilelang import (
    TileLangCompileSettings,
    build_tilelang_artifact_metadata,
    list_tilelang_kernel_specs,
    tilelang_validation_thresholds,
)
from .backends.triton import list_triton_kernel_specs
from .patterns import summarize_candidate_report
from .reference_wrappers import _ReferenceGuardedLinearWrapper
from .tilelang_wrappers import (
    _TileLangAttentionWrapper,
    _TileLangConv3dWrapper,
    _TileLangConvWrapper,
    _TileLangDequantGemmWrapper,
    _TileLangLinearWrapper,
    _TileLangNormWrapper,
    _TileLangXqtAttentionWrapper,
)
from .triton_wrappers import triton_execution_metadata
from .types import OperatorOptimizationTargetPlan


def quant_runtime_guard(
    context: XQTContext,
    plan: OperatorOptimizationTargetPlan,
) -> str | None:
    """Reject PyTorch kernel replacement for a non-PyTorch quantized runtime."""

    quant_metrics = context.metrics.get("quant")
    if not isinstance(quant_metrics, dict):
        return None
    backend = str(quant_metrics.get("backend") or "")
    component_metrics = quant_metrics.get("components", [{}])
    first_component = component_metrics[0] if isinstance(component_metrics, list) and component_metrics else {}
    runtime = str(
        first_component.get("runtime") if isinstance(first_component, Mapping) else ""
    ) or str(quant_metrics.get("runtime") or "")
    if backend == "onnxruntime_qdq" or runtime == "onnxruntime":
        return "quantized runtime artifact is onnxruntime_qdq and cannot be rewritten as a PyTorch custom kernel"
    if backend and describe_quant_backend_capability(backend).runtime != "pytorch":
        capability = describe_quant_backend_capability(backend)
        return (
            f"quantized runtime '{capability.runtime}' is not a PyTorch module runtime "
            "for operator optimization replacement"
        )
    del plan
    return None


def engine_metadata(
    target: OperatorOptimizationTargetPlan,
    *,
    dtype: str | None = None,
) -> dict[str, Any]:
    """Build serializable capability and artifact metadata for one target engine."""

    metadata: dict[str, Any] = {}
    if target.engine == "deployment_engine":
        metadata["deployment_target"] = {
            "runtime": target.options.get("runtime"),
            "stage": target.options.get("stage"),
            "semantic_target": target.options.get("semantic_target") or target.name,
            "fallback_reason": target.options.get("fallback_reason"),
        }
    if target.engine == "triton":
        metadata["kernel_registry"] = list_triton_kernel_specs()
        metadata["execution_mode"] = "not_run"
        metadata["execution_reason"] = None
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    if target.engine == "tilelang":
        settings = TileLangCompileSettings(
            target=str(target.tilelang.get("target", "cuda")),
            target_arch=target.tilelang.get("target_arch"),
            cache_dir=target.tilelang.get("cache_dir"),
            threads=int(target.tilelang.get("threads", 128)),
            num_stages=int(target.tilelang.get("num_stages", 2)),
            pass_configs=dict(target.tilelang.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_tilelang_kernel_specs()
        metadata["tilelang_artifacts"] = {
            pattern: build_tilelang_artifact_metadata(pattern, settings)
            for pattern in (target.patterns or ["attention"])
            if pattern in metadata["kernel_registry"]
        }
        thresholds = tilelang_validation_thresholds(dtype)
        thresholds.update(dict(target.validate))
        metadata["validation_thresholds"] = thresholds
        metadata["execution_mode"] = "not_run"
        metadata["execution_reason"] = None
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    if target.engine == "cutile":
        settings = CuTileCompileSettings(
            target=str(target.cutile.get("target", "cuda")),
            target_arch=target.cutile.get("target_arch"),
            cache_dir=target.cutile.get("cache_dir"),
            threads=int(target.cutile.get("threads", 128)),
            pass_configs=dict(target.cutile.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_cutile_kernel_specs()
        metadata["cutile_artifacts"] = {
            pattern: build_cutile_artifact_metadata(pattern, settings)
            for pattern in (target.patterns or ["bias_silu"])
            if pattern in metadata["kernel_registry"]
        }
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    if target.engine == "cutlass":
        tile_shape = tuple(int(value) for value in target.cutlass.get("tile_shape", [128, 128, 64]))
        cluster_shape_raw = target.cutlass.get("cluster_shape")
        settings = CutlassCompileSettings(
            target_arch=target.cutlass.get("target_arch"),
            cache_dir=target.cutlass.get("cache_dir"),
            tile_shape=tile_shape,
            cluster_shape=(tuple(int(value) for value in cluster_shape_raw) if cluster_shape_raw is not None else None),
            pass_configs=dict(target.cutlass.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_cutlass_kernel_specs()
        metadata["cutlass_artifacts"] = {
            pattern: build_cutlass_artifact_metadata(pattern, settings)
            for pattern in (target.patterns or ["gemm_epilogue"])
            if pattern in metadata["kernel_registry"]
        }
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    if target.engine == "cute_dsl":
        tile_shape = tuple(int(value) for value in target.cute_dsl.get("tile_shape", [128, 128, 64]))
        cluster_shape_raw = target.cute_dsl.get("cluster_shape")
        settings = CuteDSLCompileSettings(
            target_arch=target.cute_dsl.get("target_arch"),
            cache_dir=target.cute_dsl.get("cache_dir"),
            tile_shape=tile_shape,
            cluster_shape=(tuple(int(value) for value in cluster_shape_raw) if cluster_shape_raw is not None else None),
            pass_configs=dict(target.cute_dsl.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_cute_dsl_kernel_specs()
        metadata["cute_dsl_artifacts"] = {
            pattern: build_cute_dsl_artifact_metadata(pattern, settings)
            for pattern in (target.patterns or ["gemm_epilogue"])
            if pattern in metadata["kernel_registry"]
        }
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    return metadata


def artifact_paths_from_engine_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Extract generated engine artifact paths from metadata."""

    artifact_paths: dict[str, str] = {}
    for engine in ("tilelang", "cutile", "cutlass", "cute_dsl"):
        engine_artifacts = metadata.get(f"{engine}_artifacts")
        if not isinstance(engine_artifacts, Mapping):
            continue
        for pattern, artifact_metadata in engine_artifacts.items():
            if isinstance(artifact_metadata, Mapping) and artifact_metadata.get("artifact_path"):
                artifact_paths[f"{engine}.{pattern}"] = str(artifact_metadata["artifact_path"])
    return artifact_paths


def operator_engine_execution_metadata(
    model: nn.Module,
    *,
    engine: str,
) -> dict[str, Any]:
    """Read concrete execution metadata emitted by an engine wrapper."""

    if engine == "triton":
        return triton_execution_metadata(model)
    if engine == "tilelang":
        attached = getattr(model, "_xqt_tilelang_execution_metadata", None)
        if isinstance(attached, dict):
            return dict(attached)
        wrappers = (
            _TileLangAttentionWrapper,
            _TileLangXqtAttentionWrapper,
            _TileLangConvWrapper,
            _TileLangConv3dWrapper,
            _TileLangLinearWrapper,
            _TileLangNormWrapper,
            _TileLangDequantGemmWrapper,
        )
        for module in model.modules():
            if isinstance(module, wrappers):
                return module.execution_metadata()
        return {"execution_mode": "unknown", "execution_reason": None}
    if engine in {"cutile", "cute_dsl"}:
        for module in model.modules():
            if isinstance(module, _ReferenceGuardedLinearWrapper) and module.engine == engine:
                return module.execution_metadata()
        return {"execution_mode": "unknown", "execution_reason": None}
    return {}


def scan_candidate_report(
    scanner: Callable[[nn.Module, Any], list[Any]],
    model: nn.Module,
    example_input: Any,
) -> dict[str, Any]:
    """Run a pattern scanner while preserving scanner failure in the report."""

    try:
        report = summarize_candidate_report(scanner(model, example_input))
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "candidate_count": 0,
            "patterns": [],
            "recommended_engines": [],
            "candidates": [],
        }
    return {**report, "status": "ok", "error": None}


def torch_compile_explain_report(module: nn.Module, inputs: Any) -> dict[str, Any]:
    """Collect the stable report subset from ``torch._dynamo.explain``."""

    dynamo = getattr(torch, "_dynamo", None)
    explain = getattr(dynamo, "explain", None) if dynamo is not None else None
    if explain is None:
        return {
            "status": "unavailable",
            "error": "torch._dynamo.explain is not available",
            "graph_count": None,
            "graph_break_count": None,
            "break_reasons": [],
            "op_count": None,
            "compile_times": None,
        }
    from xqt.export.input_utils import split_example_input

    normalized = split_example_input(inputs)
    try:
        result = explain(module)(*normalized.args, **normalized.kwargs)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "graph_count": None,
            "graph_break_count": None,
            "break_reasons": [],
            "op_count": None,
            "compile_times": None,
        }
    return {
        "status": "ok",
        "error": None,
        "graph_count": getattr(result, "graph_count", None),
        "graph_break_count": getattr(result, "graph_break_count", None),
        "break_reasons": [str(reason) for reason in getattr(result, "break_reasons", []) or []],
        "op_count": getattr(result, "op_count", None),
        "compile_times": str(getattr(result, "compile_times", "")) or None,
    }


def planned_operator_skip_reason(target: OperatorOptimizationTargetPlan) -> str | None:
    """Return a future planned-engine skip reason when one is defined."""

    del target
    return None


__all__ = [
    "artifact_paths_from_engine_metadata",
    "engine_metadata",
    "operator_engine_execution_metadata",
    "planned_operator_skip_reason",
    "quant_runtime_guard",
    "scan_candidate_report",
    "torch_compile_explain_report",
]
