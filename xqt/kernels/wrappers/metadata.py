"""Operator-engine metadata and execution preflight helpers."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from xqt.core.types import XQTContext
from xqt.kernels.engine_resolve import get_engine_registration

from xqt.kernels.ops._impl.engines.cutile import (
    CuTileCompileSettings,
    build_cutile_artifact_metadata,
    list_cutile_kernel_specs,
)
from xqt.kernels.ops._impl.engines.cutlass import (
    CutlassCompileSettings,
    build_cutlass_artifact_metadata,
    list_cutlass_kernel_specs,
)
from xqt.kernels.ops._impl.engines.cute_dsl import (
    CuteDSLCompileSettings,
    build_cute_dsl_artifact_metadata,
    list_cute_dsl_kernel_specs,
)
from xqt.kernels.ops._impl.engines.tilelang import (
    TileLangCompileSettings,
    build_tilelang_artifact_metadata,
    list_tilelang_kernel_specs,
    tilelang_validation_thresholds,
)
from xqt.kernels.ops._impl.engines.triton import list_triton_kernel_specs
from .patterns import scan_candidate_report
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

    report = quant_runtime_guard_report(context, plan)
    return str(report["reason"]) if report["guard_applies"] else None


def quant_runtime_guard_report(
    context: XQTContext,
    plan: OperatorOptimizationTargetPlan,
) -> dict[str, Any]:
    """Return structured non-PyTorch quant runtime guard facts."""

    quant_metrics = context.metrics.get("quant")
    if not isinstance(quant_metrics, dict):
        return _quant_runtime_guard_payload(plan, guard_applies=False)
    backend = str(quant_metrics.get("backend") or "")
    component_metrics = quant_metrics.get("components", [{}])
    first_component = (
        component_metrics[0]
        if isinstance(component_metrics, list) and component_metrics
        else {}
    )
    runtime = str(
        first_component.get("runtime") if isinstance(first_component, Mapping) else ""
    ) or str(quant_metrics.get("runtime") or "")
    component_count = len(component_metrics) if isinstance(component_metrics, list) else 0
    runtime_source = (
        "components.0.runtime" if runtime and first_component else "quant.runtime"
    )
    if backend == "onnxruntime_qdq" or runtime == "onnxruntime":
        reason = (
            "quantized runtime artifact is onnxruntime_qdq and cannot be rewritten "
            "as a PyTorch custom kernel"
        )
        return _quant_runtime_guard_payload(
            plan,
            guard_applies=True,
            reason=reason,
            backend=backend,
            runtime=runtime or "onnxruntime",
            runtime_source=runtime_source,
            component_count=component_count,
        )
    # Known non-PyTorch quant runtimes (avoid importing quant.capability here).
    non_pytorch_runtimes = {
        "onnxruntime_qdq": "onnxruntime",
        "bitsandbytes": "bitsandbytes",
    }
    if backend in non_pytorch_runtimes:
        runtime_name = non_pytorch_runtimes[backend]
        reason = (
            f"quantized runtime '{runtime_name}' is not a PyTorch module runtime "
            "for operator optimization replacement"
        )
        return _quant_runtime_guard_payload(
            plan,
            guard_applies=True,
            reason=reason,
            backend=backend,
            runtime=runtime_name,
            runtime_source="quant.backend",
            component_count=component_count,
        )
    if runtime and runtime not in {"", "pytorch", "torch"}:
        reason = (
            f"quantized runtime '{runtime}' is not a PyTorch module runtime "
            "for operator optimization replacement"
        )
        return _quant_runtime_guard_payload(
            plan,
            guard_applies=True,
            reason=reason,
            backend=backend,
            runtime=runtime,
            runtime_source=runtime_source,
            component_count=component_count,
        )
    return _quant_runtime_guard_payload(
        plan,
        guard_applies=False,
        backend=backend,
        runtime=runtime,
        runtime_source=runtime_source if runtime else None,
        component_count=component_count,
    )


def _quant_runtime_guard_payload(
    plan: OperatorOptimizationTargetPlan,
    *,
    guard_applies: bool,
    reason: str | None = None,
    backend: str = "",
    runtime: str = "",
    runtime_source: str | None = None,
    component_count: int = 0,
) -> dict[str, Any]:
    return {
        "guard_applies": guard_applies,
        "reason": reason,
        "backend": backend,
        "runtime": runtime,
        "runtime_source": runtime_source,
        "component_count": component_count,
        "target_name": plan.name,
        "target_engine": plan.engine,
        "allowed_runtimes": ["pytorch", "torch"],
    }


def operator_numeric_validation_report(
    target: OperatorOptimizationTargetPlan,
    *,
    status: str,
    numeric_diff: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return stable numeric validation facts for an operator target."""

    diff = dict(numeric_diff or {})
    resolved_thresholds = dict(thresholds or {})
    allclose = diff.get("allclose")
    return {
        "target_name": target.name,
        "target_path": target.target_path,
        "benchmark_target_path": target.benchmark_target_path,
        "status": status,
        "reason": reason,
        "allclose": allclose if isinstance(allclose, bool) else None,
        "thresholds": resolved_thresholds,
        "max_abs": diff.get("max_abs"),
        "mean_abs": diff.get("mean_abs"),
        "cosine_similarity": diff.get("cosine_similarity"),
    }


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
    registration = get_engine_registration(engine)
    if registration is not None and registration.materializer == "reference_guarded":
        for module in model.modules():
            if isinstance(module, _ReferenceGuardedLinearWrapper) and module.engine == engine:
                return module.execution_metadata()
        return {"execution_mode": "unknown", "execution_reason": None}
    return {}


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
    from xqt.contracts.input_utils import split_example_input

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
        "break_reasons": [
            str(reason) for reason in getattr(result, "break_reasons", []) or []
        ],
        "break_details": _torch_compile_break_details(
            getattr(result, "break_reasons", []) or []
        ),
        "op_count": getattr(result, "op_count", None),
        "compile_times": str(getattr(result, "compile_times", "")) or None,
    }


def _torch_compile_break_details(reasons: list[Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for index, reason in enumerate(reasons):
        stack = getattr(reason, "user_stack", None)
        graph_break = getattr(reason, "graph_break", None)
        details.append(
            {
                "index": index,
                "reason": str(getattr(reason, "reason", reason)),
                "graph_break": bool(graph_break) if graph_break is not None else None,
                "user_stack": str(stack) if stack is not None else None,
            }
        )
    return details


def torch_compile_graph_report(
    target: OperatorOptimizationTargetPlan,
    explain: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach operator target config to a stable torch.compile graph report."""

    break_reasons = list(explain.get("break_reasons", []) or [])
    break_details = list(explain.get("break_details", []) or [])
    return {
        "backend": target.engine,
        "status": explain.get("status"),
        "error": explain.get("error"),
        "graph_count": explain.get("graph_count"),
        "graph_break_count": explain.get("graph_break_count"),
        "break_reasons": break_reasons,
        "break_details": break_details,
        "op_count": explain.get("op_count"),
        "compile_times": explain.get("compile_times"),
        "mode": target.mode,
        "dynamic": target.dynamic,
        "fullgraph": target.fullgraph,
        "target_name": target.name,
        "target_path": target.target_path,
        "benchmark_target_path": target.benchmark_target_path,
    }


def operator_planned_skip_report(
    target: OperatorOptimizationTargetPlan,
    capability: Mapping[str, Any],
    *,
    reason: str | None,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    """Return a stable planned/skip report for non-executed operator targets."""

    requires_cuda = bool(capability.get("requires_cuda"))
    if cuda_available is None:
        cuda_available = bool(torch.cuda.is_available())
    missing_requirements: list[str] = []
    if requires_cuda and not cuda_available:
        missing_requirements.append("cuda")
    if bool(capability.get("requires_exportable_graph")):
        missing_requirements.append("exportable_graph")
    if bool(capability.get("requires_calibration")):
        missing_requirements.append("calibration")

    status = str(capability.get("status") or "unknown")
    execution_status = "planned" if status == "planned" else "skipped"
    if status == "available" and not missing_requirements:
        execution_status = "available"
    return {
        "target_name": target.name,
        "engine": target.engine,
        "status": status,
        "maturity": capability.get("maturity"),
        "execution_status": execution_status,
        "reason": reason,
        "requires_cuda": requires_cuda,
        "cuda_available": cuda_available,
        "missing_requirements": missing_requirements,
        "fallback": target.fallback,
        "fallback_policy": target.fallback_policy,
        "target_path": target.target_path,
        "benchmark_target_path": target.benchmark_target_path,
    }


def planned_operator_skip_reason(target: OperatorOptimizationTargetPlan) -> str | None:
    """Return a future planned-engine skip reason when one is defined."""

    del target
    return None


__all__ = [
    "artifact_paths_from_engine_metadata",
    "engine_metadata",
    "operator_engine_execution_metadata",
    "operator_numeric_validation_report",
    "operator_planned_skip_report",
    "planned_operator_skip_reason",
    "quant_runtime_guard",
    "quant_runtime_guard_report",
    "scan_candidate_report",
    "torch_compile_graph_report",
    "torch_compile_explain_report",
]
