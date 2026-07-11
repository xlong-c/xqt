"""Operator optimization plan execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from xqt.analysis.compare import compare_tensors
from xqt.core.errors import XQTBackendError
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.schema import BenchmarkConfig
from xqt.core.types import XQTContext
from xqt.export.input_utils import first_tensor_output

from ._benchmark import (
    _benchmark_callable_for_execution,
    _benchmark_paired_batched_callables,
    _benchmark_paired_callables,
    _effective_min_speedup,
    _native_runtime_near_equal,
    _native_runtime_speedup_strategy,
    _paired_steady_state_speedup_strategy,
    _percentile,
    _tilelang_inner_iterations,
)
from .capability import describe_operator_engine_capability
from .compile_backend import compile_with_torch
from .execution_support import (
    call_module_no_grad,
    effective_validation_thresholds,
    infer_module_device_dtype,
    move_to_device,
    shape_signature,
)
from .materialize import (
    _replace_component_model,
    _resolve_component_model,
)
from .metadata import (
    artifact_paths_from_engine_metadata,
    engine_metadata as build_engine_metadata,
    operator_engine_execution_metadata,
    planned_operator_skip_reason,
    quant_runtime_guard,
    scan_candidate_report,
    torch_compile_explain_report,
)
from .reference_wrappers import build_reference_guarded_linear_candidate_model
from .tilelang_wrappers import build_tilelang_candidate_model
from .triton_wrappers import build_triton_candidate_model
from .patterns import scan_export_candidates, scan_fx_candidates
from .types import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationExecutionResult,
    OperatorOptimizationReport,
)


def _normalize_fallback_policy(policy: str | None) -> str:
    text = str(policy or "prefer_fallback").strip().lower()
    aliases = {
        "strict": "strict",
        "prefer_fallback": "prefer_fallback",
        "prefer-fallback": "prefer_fallback",
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError(
            "operator target fallback_policy must be 'strict' or 'prefer_fallback'"
        )
    return normalized


def _fallback_policy_reason(
    skip_reason: str | None,
    *,
    fallback_policy: str,
) -> str | None:
    if skip_reason is None:
        return None
    if fallback_policy == "strict":
        return f"strict policy rejected fallback: {skip_reason}"
    return skip_reason


def _runtime_fallback_reason(execution_detail: dict[str, Any]) -> str | None:
    """Return the concrete reason when a wrapper executed its reference path."""

    if execution_detail.get("execution_mode") != "reference_fallback":
        return None
    reason = execution_detail.get("execution_reason")
    if isinstance(reason, str) and reason:
        return reason
    return "operator engine executed the configured reference fallback"


def execute_operator_optimization_plan(
    context: XQTContext,
    plan: OperatorOptimizationExecutionPlan,
    *,
    benchmark_config: BenchmarkConfig | None = None,
    device: str | None = None,
) -> OperatorOptimizationExecutionResult:
    """Execute a normalized operator optimization plan and return unified reports."""

    benchmark = benchmark_config or context.benchmark_config
    if benchmark is None:
        raise ValueError("XQTContext.benchmark_config is required")
    benchmark_device = device or context.device
    target_device = torch.device(benchmark_device)
    current_model = context.require_model().to(target_device)
    context.model = current_model
    reports: list[OperatorOptimizationReport] = []
    artifacts: dict[str, Any] = {}
    if not plan.targets:
        return OperatorOptimizationExecutionResult(
            model=current_model,
            reports=[],
            artifacts={},
        )

    if context.example_inputs is None:
        raise ValueError("example_inputs are required for operator optimization")
    root_batch = context.example_inputs
    root_inputs = move_to_device(
        extract_model_inputs(
            root_batch,
            expected_input_count=infer_model_input_count(current_model),
        ),
        target_device,
    )
    candidate_reports = {
        "fx": scan_candidate_report(scan_fx_candidates, current_model, root_inputs),
        "torch_export": scan_candidate_report(
            scan_export_candidates,
            current_model,
            root_inputs,
        ),
    }
    artifacts["operator_optimization_candidates"] = candidate_reports

    for target in plan.targets:
        capability = describe_operator_engine_capability(target.engine)
        fallback_policy = _normalize_fallback_policy(target.fallback_policy)
        target_model = _resolve_component_model(current_model, target.target_path)
        module_inputs = root_inputs
        if target.target_path:
            expected_input_count = infer_model_input_count(target_model)
            module_inputs = move_to_device(
                extract_model_inputs(
                    root_batch,
                    expected_input_count=expected_input_count,
                ),
                target_device,
            )
        device, dtype = infer_module_device_dtype(target_model, module_inputs)
        engine_metadata = build_engine_metadata(target, dtype=dtype)
        artifact_paths = artifact_paths_from_engine_metadata(engine_metadata)
        compile_explain = (
            torch_compile_explain_report(target_model, module_inputs)
            if target.engine == "torch_compile"
            else {
                "status": "not_applicable",
                "error": None,
                "graph_count": None,
                "graph_break_count": None,
                "break_reasons": [],
                "op_count": None,
                "compile_times": None,
            }
        )

        skip_reason = quant_runtime_guard(context, target)
        if skip_reason is None and target.engine == "torch_compile" and not capability.available:
            skip_reason = "torch.compile is not available in the current PyTorch build"
        if skip_reason is None and target.engine in {"cutlass", "custom_cuda"}:
            if not torch.cuda.is_available():
                skip_reason = f"{target.engine} requires CUDA-capable hardware"
            else:
                skip_reason = planned_operator_skip_reason(target) or (
                    f"{target.engine} engine is configured but not implemented in the built-in executor"
                )
        if (
            skip_reason is None
            and target.engine in {"cutile", "cute_dsl"}
            and not torch.cuda.is_available()
            and target.fallback != "eager"
        ):
            skip_reason = f"{target.engine} requires CUDA-capable hardware"
        if skip_reason is None and target.engine == "deployment_engine":
            skip_reason = "deployment_engine is metadata-only in the built-in executor"
        fallback_reason = _fallback_policy_reason(
            skip_reason,
            fallback_policy=fallback_policy,
        )
        fallback_detail = {
            "engine": target.engine,
            "fallback": target.fallback,
            "fallback_policy": fallback_policy,
            "reason": fallback_reason,
            "graph_break_count": compile_explain.get("graph_break_count"),
            "graph_breaks": list(compile_explain.get("break_reasons", [])),
            "compiled_regions": compile_explain.get("graph_count"),
            "explain": compile_explain,
        }

        baseline_output = first_tensor_output(
            call_module_no_grad(target_model, module_inputs)
        )
        baseline_execution_detail: dict[str, Any] = {}
        if target.engine in {"tilelang", "triton", "cutile", "cute_dsl"}:
            baseline_execution_detail = operator_engine_execution_metadata(
                target_model,
                engine=target.engine,
            )
        latency_before, baseline_benchmark_strategy = _benchmark_callable_for_execution(
            lambda: call_module_no_grad(target_model, module_inputs),
            warmup=benchmark.warmup,
            iterations=benchmark.iterations,
            sync_cuda=benchmark.sync_cuda,
            device=benchmark_device,
            execution_detail=baseline_execution_detail,
        )

        if skip_reason is not None:
            reports.append(
                OperatorOptimizationReport(
                    target_name=target.name,
                    module_path=target.target_path,
                    engine=target.engine,
                    runtime=capability.runtime,
                    applied=False,
                    fallback=target.fallback,
                    fallback_policy=fallback_policy,
                    skip_reason=fallback_reason,
                    compile_time_ms=None,
                    latency_before=latency_before,
                    latency_after=latency_before,
                    speedup=1.0,
                    numeric_diff={
                        "allclose": True,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                    },
                    device=device,
                    dtype=dtype,
                    shape_signature=shape_signature(module_inputs),
                    exportable=capability.exportable,
                    artifact_paths=artifact_paths,
                    metadata={
                        "execution_state": "skipped",
                        "fallback_policy": fallback_policy,
                        "fallback_reason": fallback_reason,
                        "fallback_detail": fallback_detail,
                        "graph_break_report": compile_explain,
                        "options": dict(target.options),
                        "mode": target.mode,
                        "patterns": list(target.patterns),
                        "capability": capability.to_dict(),
                        **engine_metadata,
                    },
                )
            )
            continue

        compiled_model = None
        compile_time_ms = None
        try:
            if target.engine == "torch_compile":
                compiled_model, compile_time_ms = compile_with_torch(target_model, target)
            elif target.engine == "tilelang":
                compiled_model = build_tilelang_candidate_model(target_model, target)
            elif target.engine == "triton":
                compiled_model = build_triton_candidate_model(target_model, target)
            elif target.engine in {"cutile", "cute_dsl"}:
                compiled_model = build_reference_guarded_linear_candidate_model(
                    target_model,
                    target,
                    engine=target.engine,
                )
            else:
                raise XQTBackendError(
                    f"Operator optimization engine '{target.engine}' is not executable yet"
                )
        except Exception as exc:
            reports.append(
                OperatorOptimizationReport(
                    target_name=target.name,
                    module_path=target.target_path,
                    engine=target.engine,
                    runtime=capability.runtime,
                    applied=False,
                    fallback=target.fallback,
                    fallback_policy=fallback_policy,
                    skip_reason=_fallback_policy_reason(
                        str(exc),
                        fallback_policy=fallback_policy,
                    ),
                    compile_time_ms=compile_time_ms,
                    latency_before=latency_before,
                    latency_after=latency_before,
                    speedup=1.0,
                    numeric_diff={
                        "allclose": True,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                    },
                    device=device,
                    dtype=dtype,
                    shape_signature=shape_signature(module_inputs),
                    exportable=capability.exportable,
                    artifact_paths=artifact_paths,
                    metadata={
                        "execution_state": "fallback",
                        "fallback_policy": fallback_policy,
                        "fallback_reason": _fallback_policy_reason(
                            str(exc),
                            fallback_policy=fallback_policy,
                        ),
                        "fallback_detail": {
                            "engine": target.engine,
                            "fallback": target.fallback,
                            "fallback_policy": fallback_policy,
                            "reason": _fallback_policy_reason(
                                str(exc),
                                fallback_policy=fallback_policy,
                            ),
                            "graph_break_count": compile_explain.get("graph_break_count"),
                            "graph_breaks": list(compile_explain.get("break_reasons", [])),
                            "compiled_regions": compile_explain.get("graph_count"),
                            "explain": compile_explain,
                        },
                        "graph_break_report": compile_explain,
                        "options": dict(target.options),
                        "mode": target.mode,
                        "patterns": list(target.patterns),
                        "capability": capability.to_dict(),
                        **engine_metadata,
                    },
                )
            )
            continue

        candidate_target = compiled_model
        optimized_output = first_tensor_output(
            call_module_no_grad(candidate_target, module_inputs)
        )
        execution_detail: dict[str, Any] = {}
        if target.engine in {"tilelang", "triton", "cutile", "cute_dsl"}:
            execution_detail = operator_engine_execution_metadata(
                candidate_target,
                engine=target.engine,
            )
        identity_candidate = candidate_target is target_model
        effective_thresholds = effective_validation_thresholds(
            target,
            baseline_output=baseline_output,
            optimized_output=optimized_output,
        )
        if target.engine == "tilelang":
            engine_metadata["validation_thresholds"] = dict(effective_thresholds)
        numeric_diff = compare_tensors(
            baseline_output,
            optimized_output,
            atol=effective_thresholds["atol"],
            rtol=effective_thresholds["rtol"],
        ).to_dict()
        benchmark_strategy = baseline_benchmark_strategy
        speedup_metric = "mean_ms"
        speedup_statistics: dict[str, float | None] = {}
        native_speedup_strategy = _native_runtime_speedup_strategy(execution_detail)
        paired_steady_state_strategy = _paired_steady_state_speedup_strategy(execution_detail)
        if native_speedup_strategy is not None:
            if identity_candidate:
                benchmark_strategy = "identity_native_baseline"
                latency_after = dict(latency_before)
                speedup_metric = "p50_ms"
                speedup_statistics = {
                    "mean_ms": 1.0,
                    "p50_ms": 1.0,
                    "paired_ratio_p50": 1.0,
                }
                speedup = 1.0
            else:
                benchmark_strategy = native_speedup_strategy
                paired_before, paired_after, paired_speedup_ratios = _benchmark_paired_callables(
                    lambda: call_module_no_grad(target_model, module_inputs),
                    lambda: call_module_no_grad(candidate_target, module_inputs),
                    warmup=benchmark.warmup,
                    iterations=benchmark.iterations,
                    sync_cuda=benchmark.sync_cuda,
                    device=benchmark_device,
                )
                latency_before = paired_before.to_dict()
                latency_after = paired_after.to_dict()
                speedup_metric = "p50_ms"
                mean_before = float(latency_before["mean_ms"])
                mean_after = float(latency_after["mean_ms"])
                p50_before = float(latency_before["p50_ms"])
                p50_after = float(latency_after["p50_ms"])
                paired_speedup_sorted = sorted(paired_speedup_ratios)
                paired_speedup_p50 = (
                    _percentile(paired_speedup_sorted, 50)
                    if paired_speedup_sorted
                    else None
                )
                speedup_statistics = {
                    "mean_ms": (mean_before / mean_after) if mean_after > 0.0 else None,
                    "p50_ms": (p50_before / p50_after) if p50_after > 0.0 else None,
                    "paired_ratio_p50": paired_speedup_p50,
                }
                speedup = (
                    paired_speedup_p50
                    if paired_speedup_p50 is not None
                    else speedup_statistics["p50_ms"]
                )
        elif paired_steady_state_strategy is not None:
            benchmark_strategy = paired_steady_state_strategy
            inner_iterations = _tilelang_inner_iterations(execution_detail)
            latency_before, latency_after, paired_speedup_ratios = _benchmark_paired_batched_callables(
                lambda: call_module_no_grad(target_model, module_inputs),
                lambda: call_module_no_grad(candidate_target, module_inputs),
                warmup=benchmark.warmup,
                iterations=benchmark.iterations,
                sync_cuda=benchmark.sync_cuda,
                device=benchmark_device,
                inner_iterations=inner_iterations,
            )
            mean_before = float(latency_before["mean_ms"])
            mean_after = float(latency_after["mean_ms"])
            p50_before = float(latency_before["p50_ms"])
            p50_after = float(latency_after["p50_ms"])
            paired_speedup_sorted = sorted(paired_speedup_ratios)
            paired_speedup_p50 = (
                _percentile(paired_speedup_sorted, 50)
                if paired_speedup_sorted
                else None
            )
            speedup_statistics = {
                "mean_ms": (mean_before / mean_after) if mean_after > 0.0 else None,
                "p50_ms": (p50_before / p50_after) if p50_after > 0.0 else None,
                "paired_ratio_p50": paired_speedup_p50,
            }
            speedup = speedup_statistics["mean_ms"]
        else:
            latency_after, benchmark_strategy = _benchmark_callable_for_execution(
                lambda: call_module_no_grad(candidate_target, module_inputs),
                warmup=benchmark.warmup,
                iterations=benchmark.iterations,
                sync_cuda=benchmark.sync_cuda,
                device=benchmark_device,
                execution_detail=execution_detail,
            )
            mean_before = float(latency_before["mean_ms"])
            mean_after = float(latency_after["mean_ms"])
            speedup = (mean_before / mean_after) if mean_after > 0.0 else None
            speedup_statistics = {
                "mean_ms": speedup,
                "p50_ms": (
                    float(latency_before["p50_ms"]) / float(latency_after["p50_ms"])
                    if float(latency_after["p50_ms"]) > 0.0
                    else None
                ),
            }
        if target.engine in {"tilelang", "triton", "cutile", "cute_dsl"}:
            execution_detail = operator_engine_execution_metadata(
                candidate_target,
                engine=target.engine,
            )
        effective_min_speedup = _effective_min_speedup(
            target,
            execution_detail=execution_detail,
        )
        meets_numeric = bool(numeric_diff.get("allclose"))
        meets_speedup = speedup is not None and speedup >= effective_min_speedup
        if not meets_speedup and _native_runtime_near_equal(
            execution_detail,
            latency_before=latency_before,
            latency_after=latency_after,
        ):
            meets_speedup = True
        applied = meets_numeric and meets_speedup
        skip_reason = None
        if not meets_numeric:
            skip_reason = "numeric validation failed"
        elif not meets_speedup:
            skip_reason = (
                f"speedup {speedup:.4f} did not reach min_speedup {effective_min_speedup:.4f}"
                if speedup is not None
                else "latency_after is zero so speedup could not be computed"
            )
        runtime_fallback_reason = _runtime_fallback_reason(execution_detail)
        if runtime_fallback_reason is not None:
            applied = False
            skip_reason = runtime_fallback_reason
        if applied:
            current_model = _replace_component_model(
                current_model,
                target.target_path,
                compiled_model,
            )
        elif target.engine == "tilelang" and compiled_model is target_model:
            if hasattr(compiled_model, "_xqt_tilelang_execution_metadata"):
                delattr(compiled_model, "_xqt_tilelang_execution_metadata")
        fallback_reason = _fallback_policy_reason(
            skip_reason,
            fallback_policy=fallback_policy,
        )
        fallback_detail = {
            "engine": target.engine,
            "fallback": target.fallback,
            "fallback_policy": fallback_policy,
            "reason": fallback_reason,
            "graph_break_count": compile_explain.get("graph_break_count"),
            "graph_breaks": list(compile_explain.get("break_reasons", [])),
            "compiled_regions": compile_explain.get("graph_count"),
            "explain": compile_explain,
        }
        report_metadata: dict[str, Any] = {
            "execution_state": "executed" if applied else "fallback",
            "fallback_policy": fallback_policy,
            "fallback_reason": fallback_reason,
            "fallback_detail": fallback_detail,
            "graph_break_report": compile_explain,
            "options": dict(target.options),
            "mode": target.mode,
            "patterns": list(target.patterns),
            "min_speedup": target.min_speedup,
            "effective_min_speedup": effective_min_speedup,
            "benchmark_strategy": benchmark_strategy,
            "candidate_materialization": "target_only_benchmark_no_root_deepcopy",
            "speedup_metric": speedup_metric,
            "speedup_statistics": speedup_statistics,
            "capability": capability.to_dict(),
            **engine_metadata,
            **execution_detail,
        }
        module_contract = getattr(compiled_model, "_xqt_module_contract", None)
        if isinstance(module_contract, dict):
            report_metadata["module_contract"] = dict(module_contract)
        reports.append(
            OperatorOptimizationReport(
                target_name=target.name,
                module_path=target.target_path,
                engine=target.engine,
                runtime=capability.runtime,
                applied=applied,
                fallback=target.fallback,
                fallback_policy=fallback_policy,
                skip_reason=fallback_reason,
                compile_time_ms=compile_time_ms,
                latency_before=latency_before,
                latency_after=latency_after,
                speedup=speedup,
                numeric_diff=numeric_diff,
                device=device,
                dtype=dtype,
                shape_signature=shape_signature(module_inputs),
                exportable=capability.exportable,
                artifact_paths=artifact_paths,
                metadata=report_metadata,
            )
        )

    manifest_path = Path(context.artifact_dir) / "operator_optimization.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [report.to_dict() for report in reports],
                "candidates": candidate_reports,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["operator_optimization_report"] = manifest_path
    return OperatorOptimizationExecutionResult(
        model=current_model,
        reports=reports,
        artifacts=artifacts,
    )


__all__ = [
    "execute_operator_optimization_plan",
]
