"""Built-in lightweight XQT passes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from xqt.benchmark import benchmark_callable, benchmark_memory
from xqt.core.artifact import ArtifactRecord, MetricRecord
from xqt.core.imports import build_target
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.registry import register_pass
from xqt.core.types import XQTContext
from xqt.analysis import (
    build_layer_analysis_payload,
    layer_statistics_rows,
    records_to_rows,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)
from xqt.export import export_onnx
from xqt.operator_opt import (
    build_operator_optimization_plan,
    execute_operator_optimization_plan,
    summarize_operator_optimization_reports,
)
from xqt.prune import (
    collect_module_importance,
    prune_runtime_capability_from_report,
    PruningSchedule,
    apply_block_sparse_pruning,
    apply_global_l1_unstructured_pruning,
    apply_nm_structured_sparsity,
    apply_structured_pruning,
    rank_prune_candidates,
    remove_pruning_reparameterization,
    run_prune_schedule,
    run_structured_prune_schedule,
    summarize_pruning,
)
from xqt.quant import (
    analyze_activation_drift,
    analyze_layer_errors,
    build_quantization_plan,
    execute_quantization_plan,
    recommend_high_precision_modules,
    summarize_quantization_reports,
)
from xqt.quant.backends.onnx_qdq import quantize_onnx_qdq_static

from .export_pass import ExportPass, call_model as _call_model


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


def _build_quant_layer_analysis_summary(context: XQTContext) -> dict[str, object]:
    """Return layer diff and sensitivity summary for a quantized PyTorch model."""

    if context.reference_model is None:
        return {"available": False, "reason": "reference_model_missing"}
    if not isinstance(context.model, nn.Module):
        return {"available": False, "reason": "quantized_pytorch_model_missing"}
    if context.example_inputs is None:
        return {"available": False, "reason": "example_inputs_missing"}

    analysis_config = context.config.analysis
    device = torch.device(context.config.model.device)
    candidate_model = context.model.to(device).eval()
    reference_model = context.reference_model.to(device).eval()
    example_input = _move_to_device(
        extract_model_inputs(
            context.example_inputs,
            expected_input_count=infer_model_input_count(candidate_model),
        ),
        device,
    )
    try:
        payload = build_layer_analysis_payload(
            reference_model,
            candidate_model,
            example_input,
            module_names=analysis_config.module_names,
            atol=context.config.validation.output_diff.atol,
            rtol=context.config.validation.output_diff.rtol,
            include_weight_diff=analysis_config.include_weight_diff,
            include_sensitivity=True,
            include_statistics=False,
            include_avoid_list=True,
            metrics=analysis_config.metrics,
            row_top_k=analysis_config.top_k,
            avoid_top_k=analysis_config.top_k,
            sample_budget=analysis_config.sample_budget,
            sample_seed=analysis_config.sample_seed,
            runtime="quantized_pytorch",
            avoid_used_by="quant_keep_high_precision",
            per_channel=analysis_config.structured.per_channel,
            per_token=analysis_config.structured.per_token,
        )
        module_names = list(analysis_config.module_names or [])
        if not module_names:
            module_names = [
                name
                for name, module in candidate_model.named_modules()
                if name and hasattr(module, "weight")
            ]
        payload["layer_statistics"] = layer_statistics_rows(
            reference_model,
            candidate_model,
            example_input,
            module_names=module_names,
            sample_budget=analysis_config.sample_budget,
            sample_seed=analysis_config.sample_seed,
        )
        payload["available"] = True
        payload["layer_error_count"] = len(payload.get("layer_errors", []))
        payload["layer_sensitivity_count"] = len(payload.get("layer_sensitivity", []))
        payload["layer_statistics_count"] = len(payload.get("layer_statistics", []))
        if not payload["layer_errors"] and payload["layer_statistics"]:
            payload["layer_error_rows_from_statistics"] = True
        return payload
    except Exception as exc:
        return {
            "available": False,
            "reason": "analysis_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "layer_errors": [],
            "layer_sensitivity": [],
            "layer_statistics": [],
            "avoid_list": [],
            "layer_error_count": 0,
            "layer_sensitivity_count": 0,
            "layer_statistics_count": 0,
        }


def _update_structured_prune_benchmark_status(
    context: XQTContext,
    benchmark_metrics: dict[str, object],
) -> None:
    prune_metrics = context.metrics.get("prune")
    if not isinstance(prune_metrics, dict) or prune_metrics.get("method") != "structured":
        return
    prune_metrics["benchmark_status"] = {
        "attempted": True,
        "passed": True,
        "latency": dict(benchmark_metrics.get("latency", {})),
        "memory": benchmark_metrics.get("memory"),
        "capability": benchmark_metrics.get("capability"),
    }


def _prune_module_types(include_module_types: object) -> tuple[type[nn.Module], ...]:
    names = include_module_types
    if not isinstance(names, list):
        names = ["Linear", "Conv2d"]
    module_types = tuple(
        module_type
        for module_type_name, module_type in (
            ("Linear", nn.Linear),
            ("Conv2d", nn.Conv2d),
        )
        if module_type_name in names
    )
    return module_types or (nn.Linear, nn.Conv2d)


def _prune_module_names(
    model: nn.Module,
    module_types: tuple[type[nn.Module], ...],
) -> list[str]:
    return [
        name or "<root>"
        for name, module in model.named_modules()
        if isinstance(module, module_types)
    ]


def _detection_structured_prune_skip_report(
    model: nn.Module,
    *,
    method: str,
    granularity: str | None,
    scope: str,
    target_sparsity: float,
    module_types: tuple[type[nn.Module], ...],
) -> dict[str, object]:
    summary = summarize_pruning(model, module_types=module_types).to_dict()
    skip_reason = (
        "structured detection pruning is skipped because residual/CSP/C2f/SPPF/"
        "detect-head dependency rewrite is not implemented in the built-in executor"
    )
    return {
        **summary,
        "method": method,
        "granularity": granularity or "channel",
        "scope": scope,
        "target_sparsity": target_sparsity,
        "task_type": "detection",
        "applied": False,
        "execution_state": "skipped",
        "skip_reason": skip_reason,
        "skipped_modules": [
            {
                "module_name": module_name,
                "reason": skip_reason,
            }
            for module_name in _prune_module_names(model, module_types)
        ],
        "speedup_claimed": False,
        "baseline_kind": None,
    }


def _annotate_unstructured_prune_report(
    report: dict[str, object],
    *,
    task_type: str,
    target_sparsity: float,
) -> dict[str, object]:
    report["method"] = "global_l1_unstructured"
    report["target_sparsity"] = target_sparsity
    report["applied"] = True
    report["execution_state"] = "applied"
    report["task_type"] = task_type
    if task_type == "detection":
        report["baseline_kind"] = "unstructured_sparsity_report"
        report["speedup_claimed"] = False
        report["skip_reason"] = None
        report["skipped_modules"] = []
    return report


@register_pass("load_model")
class LoadModelPass:
    """Build the configured PyTorch model."""

    name = "load_model"

    def run(self, context: XQTContext) -> XQTContext:
        if context.model is not None:
            return context
        target = context.config.model.target
        if not target:
            raise ValueError("model.target is required when context.model is not set")
        model = build_target(target, context.config.model.params)
        if not isinstance(model, nn.Module):
            raise TypeError("model.target must build a torch.nn.Module")
        model.eval()
        context.model = model
        if context.reference_model is None:
            context.reference_model = copy.deepcopy(model)
        return context


@register_pass("analyze")
class AnalyzePass:
    """Analyze current model outputs against the baseline snapshot."""

    name = "analyze"

    def run(self, context: XQTContext) -> XQTContext:
        analysis_config = context.config.analysis
        if not analysis_config.enabled:
            return context
        model = context.require_model()
        if analysis_config.compare_to != "baseline":
            raise ValueError("Only analysis.compare_to=baseline is supported")
        reference_model = context.reference_model
        if reference_model is None:
            raise ValueError("reference_model is required for analyze pass")

        if context.example_inputs is None:
            raise ValueError("example_inputs are required for analyze pass")
        batch = context.example_inputs
        expected_input_count = infer_model_input_count(model)
        inputs = _move_to_device(
            extract_model_inputs(batch, expected_input_count=expected_input_count),
            torch.device(context.config.model.device),
        )
        model = model.to(context.config.model.device)
        reference_model = reference_model.to(context.config.model.device)
        records = analyze_layer_errors(
            reference_model,
            model,
            inputs,
            module_names=analysis_config.module_names,
            atol=context.config.validation.output_diff.atol,
            rtol=context.config.validation.output_diff.rtol,
            include_weight_diff=analysis_config.include_weight_diff,
            policy=None,
            per_channel=analysis_config.structured.per_channel,
            per_token=analysis_config.structured.per_token,
        )
        if analysis_config.top_k is not None:
            records = records[: analysis_config.top_k]

        rows = records_to_rows(records)
        statistics_module_names = (
            analysis_config.module_names
            if analysis_config.module_names is not None
            else [record.name for record in records]
        )
        layer_statistics = (
            layer_statistics_rows(
                reference_model,
                model,
                inputs,
                module_names=statistics_module_names,
                sample_budget=analysis_config.sample_budget,
                sample_seed=analysis_config.sample_seed,
                histogram_bins=analysis_config.histogram_bins,
            )
            if analysis_config.include_statistics
            else []
        )
        activation_drift = analyze_activation_drift(
            reference_model,
            model,
            [inputs],
            module_names=analysis_config.module_names,
        )
        importance_records = collect_module_importance(
            model,
            module_names=analysis_config.module_names,
        )
        prune_candidates: list[dict[str, object]] = []
        if analysis_config.recommendations.prune_candidates:
            prune_candidate_records = rank_prune_candidates(
                model,
                records,
                module_names=analysis_config.module_names,
                top_k=analysis_config.top_k,
            )
            prune_candidates = [record.to_dict() for record in prune_candidate_records]
        recommended_modules: list[str] = []
        if analysis_config.recommendations.mixed_precision:
            recommended_modules = recommend_high_precision_modules(
                records,
                top_k=analysis_config.top_k,
            )
        context.metrics["analysis"] = {
            "compare_to": analysis_config.compare_to,
            "metrics": list(analysis_config.metrics),
            "record_count": len(records),
            "records": [record.to_dict() for record in records],
            "rows": rows,
            "layer_statistics": layer_statistics,
            "activation_drift": [record.to_dict() for record in activation_drift],
            "importance": [record.to_dict() for record in importance_records],
            "prune_candidates": prune_candidates,
            "recommended_high_precision_modules": recommended_modules,
            "pareto_points": [],
        }

        artifact_dir = Path(context.config.project.artifact_dir)
        if analysis_config.export.json:
            json_path = write_json_report(
                context.metrics["analysis"],
                artifact_dir / "analysis.json",
            )
            context.artifacts["analysis_json"] = json_path
        if analysis_config.export.csv:
            csv_path = write_csv_report(rows, artifact_dir / "analysis.csv")
            context.artifacts["analysis_csv"] = csv_path
        if analysis_config.export.markdown:
            top_rows = rows[: min(analysis_config.top_k or len(rows), len(rows))]
            markdown_sections = {
                record.get("name", f"layer_{index}"): {
                    "module_type": record.get("module_type"),
                    "diff.max_abs": record.get("diff.max_abs"),
                    "diff.mean_abs": record.get("diff.mean_abs"),
                    "recommendation": record.get("recommendation"),
                }
                for index, record in enumerate(top_rows)
            }
            markdown_path = write_markdown_report(
                "XQT Analysis",
                markdown_sections,
                artifact_dir / "analysis.md",
            )
            context.artifacts["analysis_markdown"] = markdown_path

        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="analysis.record_count",
                    value=len(records),
                    metadata={"compare_to": analysis_config.compare_to},
                )
            )
        return context


@register_pass("prune")
class PrunePass:
    """Apply built-in pruning methods when enabled."""

    name = "prune"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        prune_config = context.config.compression.prune
        if not prune_config.enabled:
            return context
        if prune_config.method == "nm_structured":
            pattern = prune_config.selection.get("pattern") or prune_config.params.get("pattern")
            if not isinstance(pattern, (list, tuple)) or len(pattern) != 2:
                raise ValueError(
                    "compression.prune.selection.pattern or compression.prune.params.pattern "
                    "must be a two-item list like [2, 4] for method=nm_structured"
                )
            pattern_n = int(pattern[0])
            pattern_m = int(pattern[1])
            module_types = _prune_module_types(prune_config.params.get("include_module_types"))
            report = apply_nm_structured_sparsity(
                model,
                pattern_n=pattern_n,
                pattern_m=pattern_m,
                module_types=module_types,
            )
            context.metrics["prune"] = report.to_dict()
            if context.manifest is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report.sparsity,
                        metadata={
                            "method": prune_config.method,
                            "granularity": "nm",
                            "pattern": [pattern_n, pattern_m],
                            "compliance_ratio": report.compliance_ratio,
                        },
                    )
                )
            return context
        if prune_config.method == "block_sparse":
            block_shape = (
                prune_config.selection.get("block_shape")
                or prune_config.params.get("block_shape")
                or [4, 4]
            )
            if not isinstance(block_shape, (list, tuple)) or len(block_shape) != 2:
                raise ValueError(
                    "compression.prune.selection.block_shape or "
                    "compression.prune.params.block_shape must be a two-item list like [4, 4] "
                    "for method=block_sparse"
                )
            report = apply_block_sparse_pruning(
                model,
                target_sparsity=prune_config.target_sparsity,
                block_shape=(int(block_shape[0]), int(block_shape[1])),
                module_types=_prune_module_types(prune_config.params.get("include_module_types")),
            )
            context.metrics["prune"] = report.to_dict()
            if context.manifest is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="prune.block_sparsity",
                        value=report.sparsity,
                        threshold=prune_config.target_sparsity,
                        passed=report.sparsity >= prune_config.target_sparsity,
                        metadata={
                            "method": prune_config.method,
                            "granularity": "block_sparse",
                            "block_shape": list(report.block_shape),
                            "parameter_sparsity": report.parameter_sparsity,
                        },
                    )
            )
            return context
        if prune_config.method == "structured":
            module_types = _prune_module_types(prune_config.params.get("include_module_types"))
            if context.config.task.type == "detection":
                report = _detection_structured_prune_skip_report(
                    model,
                    method=prune_config.method,
                    granularity=prune_config.granularity,
                    scope=prune_config.scope,
                    target_sparsity=prune_config.target_sparsity,
                    module_types=module_types,
                )
                context.metrics["prune"] = report
                if context.manifest is not None:
                    context.manifest.add_metric(
                        MetricRecord(
                            name="prune.sparsity",
                            value=report["sparsity"],
                            threshold=prune_config.target_sparsity,
                            passed=False,
                            metadata={
                                "method": prune_config.method,
                                "granularity": prune_config.granularity,
                                "task_type": "detection",
                                "execution_state": "skipped",
                                "skip_reason": report["skip_reason"],
                                "skipped_module_count": len(report["skipped_modules"]),
                                "speedup_claimed": False,
                            },
                        )
                    )
                return context
            example_input = None
            if context.example_inputs is not None:
                expected_input_count = infer_model_input_count(model)
                example_input = _move_to_device(
                    extract_model_inputs(
                        context.example_inputs,
                        expected_input_count=expected_input_count,
                    ),
                    torch.device(context.config.model.device),
                )
            if prune_config.schedule in {"linear", "one_shot"} and int(
                prune_config.params.get("steps", 1)
            ) > 1:
                report = run_structured_prune_schedule(
                    model,
                    schedule=PruningSchedule(
                        target_sparsity=prune_config.target_sparsity,
                        steps=int(prune_config.params.get("steps", 1)),
                        start_sparsity=float(prune_config.params.get("start_sparsity", 0.0)),
                        schedule=prune_config.schedule,
                    ),
                    granularity=prune_config.granularity or "channel",
                    scope=prune_config.scope,
                    importance=dict(prune_config.importance),
                    selection=dict(prune_config.selection),
                    example_input=example_input,
                )
                context.metrics["prune"] = report.to_dict()
                if context.manifest is not None:
                    context.manifest.add_metric(
                        MetricRecord(
                            name="prune.sparsity",
                            value=report.final_sparsity,
                            threshold=prune_config.target_sparsity,
                            passed=report.final_sparsity >= prune_config.target_sparsity,
                            metadata={
                                "method": prune_config.method,
                                "granularity": prune_config.granularity,
                                "schedule": prune_config.schedule,
                                "steps": len(report.steps),
                            },
                        )
                    )
                return context
            report = apply_structured_pruning(
                model,
                prune_config.target_sparsity,
                granularity=prune_config.granularity or "channel",
                scope=prune_config.scope,
                importance=prune_config.importance,
                selection=prune_config.selection,
                example_input=example_input,
            )
            context.metrics["prune"] = report.to_dict()
            if context.manifest is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report.sparsity,
                        threshold=prune_config.target_sparsity,
                        passed=report.sparsity >= prune_config.target_sparsity,
                        metadata={
                            "method": prune_config.method,
                            "granularity": prune_config.granularity,
                            "parameter_count_before": report.parameter_count_before,
                            "parameter_count_after": report.parameter_count_after,
                            "parameter_reduction_ratio": report.parameter_reduction_ratio,
                        },
                    )
                )
            return context
        if prune_config.method != "global_l1_unstructured":
            raise ValueError(
                "Only compression.prune.method=global_l1_unstructured or "
                "compression.prune.method=structured or "
                "compression.prune.method=block_sparse or "
                "compression.prune.method=nm_structured is supported by the built-in prune pass"
            )
        if prune_config.schedule in {"linear", "one_shot"} and int(
            prune_config.params.get("steps", 1)
        ) > 1:
            report = run_prune_schedule(
                model,
                schedule=PruningSchedule(
                    target_sparsity=prune_config.target_sparsity,
                    steps=int(prune_config.params.get("steps", 1)),
                    start_sparsity=float(prune_config.params.get("start_sparsity", 0.0)),
                    schedule=prune_config.schedule,
                ),
            )
            context.metrics["prune"] = report.to_dict()
            if context.manifest is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report.final_sparsity,
                        threshold=prune_config.target_sparsity,
                        passed=report.final_sparsity >= prune_config.target_sparsity,
                        metadata={
                            "schedule": prune_config.schedule,
                            "steps": len(report.steps),
                        },
                    )
                )
            return context
        report = apply_global_l1_unstructured_pruning(
            model,
            prune_config.target_sparsity,
        )
        remove_pruning_reparameterization(model)
        report = summarize_pruning(model)
        report_dict = _annotate_unstructured_prune_report(
            report.to_dict(),
            task_type=context.config.task.type,
            target_sparsity=prune_config.target_sparsity,
        )
        context.metrics["prune"] = report_dict
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="prune.sparsity",
                    value=report.sparsity,
                    threshold=prune_config.target_sparsity,
                    passed=report.sparsity >= prune_config.target_sparsity,
                    metadata={
                        "method": prune_config.method,
                        "task_type": context.config.task.type,
                        "baseline_kind": report_dict.get("baseline_kind"),
                        "speedup_claimed": report_dict.get("speedup_claimed"),
                    },
                )
            )
        return context


@register_pass("quant")
class QuantPass:
    """Apply configured quantization backend."""

    name = "quant"

    def run(self, context: XQTContext) -> XQTContext:
        quant_config = context.config.compression.quant
        if not quant_config.enabled:
            return context
        plan = build_quantization_plan(quant_config)
        execution = execute_quantization_plan(
            context,
            plan,
            export_onnx_fn=export_onnx,
            quantize_onnx_qdq_static_fn=quantize_onnx_qdq_static,
        )
        context.model = execution.model
        context.artifacts.update(execution.artifacts)
        quant_metrics = summarize_quantization_reports(execution.reports)
        quant_metrics["layer_analysis"] = _build_quant_layer_analysis_summary(context)
        context.metrics["quant"] = quant_metrics
        if context.manifest is not None:
            layer_analysis = quant_metrics["layer_analysis"]
            context.manifest.add_metric(
                MetricRecord(
                    name="quant.layer_analysis.available",
                    value=bool(layer_analysis.get("available")),
                    passed=bool(layer_analysis.get("available")),
                    metadata={
                        "reason": layer_analysis.get("reason"),
                        "layer_error_count": layer_analysis.get("layer_error_count", 0),
                        "layer_sensitivity_count": layer_analysis.get(
                            "layer_sensitivity_count",
                            0,
                        ),
                    },
                )
            )
            for report in execution.reports:
                if report.runtime == "onnxruntime" and "path" in report.metadata:
                    context.manifest.add_artifact(
                        ArtifactRecord(
                            path=str(report.metadata["path"]),
                            format="onnx",
                            runtime="onnxruntime",
                            checksum=report.metadata.get("checksum"),
                            metadata={
                                "quantization": "qdq",
                                "component_name": report.component_name,
                                **report.metadata,
                            },
                        )
                    )
                context.manifest.add_metric(
                    MetricRecord(
                        name=f"quant.{report.component_name}.quantized_module_count",
                        value=len(report.quantized_modules),
                        metadata={
                            "backend": report.backend,
                            "strategy": report.strategy,
                        },
                    )
                )
                if report.calibration_samples is not None:
                    context.manifest.add_metric(
                        MetricRecord(
                            name=f"quant.{report.component_name}.calibration_samples",
                            value=report.calibration_samples,
                            metadata={
                                "backend": report.backend,
                                "calibration_source": report.metadata.get("calibration_source"),
                            },
                        )
                    )
        return context


@register_pass("operator_optimization")
class OperatorOptimizationPass:
    """Apply configured operator optimization engine."""

    name = "operator_optimization"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        operator_config = context.config.operator_optimization
        if not operator_config.enabled:
            return context
        plan = build_operator_optimization_plan(operator_config)
        execution = execute_operator_optimization_plan(context, plan)
        context.model = execution.model
        context.artifacts.update(execution.artifacts)
        context.metrics["operator_optimization"] = summarize_operator_optimization_reports(
            execution.reports,
            candidate_reports=execution.artifacts.get("operator_optimization_candidates"),
        )
        if context.manifest is not None:
            context.manifest.operator_optimization = dict(
                context.metrics["operator_optimization"]
            )
        if context.manifest is not None:
            report_artifact = execution.artifacts.get("operator_optimization_report")
            if isinstance(report_artifact, Path) and report_artifact.is_file():
                context.manifest.add_artifact(
                    ArtifactRecord.from_file(
                        report_artifact,
                        format="json",
                        runtime="pytorch",
                        metadata={"kind": "operator_optimization_report"},
                    )
                )
            for report in execution.reports:
                context.manifest.add_metric(
                    MetricRecord(
                        name=f"operator_optimization.{report.target_name}.applied",
                        value=report.applied,
                        metadata={
                            "engine": report.engine,
                            "runtime": report.runtime,
                            "fallback": report.fallback,
                            "skip_reason": report.skip_reason,
                            "speedup": report.speedup,
                            "compile_time_ms": report.compile_time_ms,
                            "exportable": report.exportable,
                        },
                    )
                )
                if report.compile_time_ms is not None:
                    context.manifest.add_metric(
                        MetricRecord(
                            name=f"operator_optimization.{report.target_name}.compile_time_ms",
                            value=report.compile_time_ms,
                            metadata={"engine": report.engine},
                        )
                    )
        del model
        return context


@register_pass("benchmark")
class BenchmarkPass:
    """Benchmark current model latency on the configured example inputs."""

    name = "benchmark"

    def run(self, context: XQTContext) -> XQTContext:
        model = context.require_model()
        if context.example_inputs is None:
            raise ValueError("example_inputs are required for benchmark")
        batch = context.example_inputs
        inputs = _move_to_device(
            extract_model_inputs(
                batch,
                expected_input_count=infer_model_input_count(model),
            ),
            torch.device(context.config.model.device),
        )

        def fn() -> object:
            return _call_model(model, inputs)

        report = benchmark_callable(
            fn,
            warmup=context.config.benchmark.warmup,
            iterations=context.config.benchmark.iterations,
            sync_cuda=context.config.benchmark.sync_cuda,
            device=context.config.model.device,
        )
        benchmark_metrics = report.to_dict()
        benchmark_metrics["latency"] = report.to_dict()
        memory_report = None
        if context.config.benchmark.measure_memory:
            memory_report = benchmark_memory(
                fn,
                iterations=1,
                sync_cuda=context.config.benchmark.sync_cuda,
                device=context.config.model.device,
            )
            benchmark_metrics["memory"] = memory_report.to_dict()
        context.metrics["benchmark"] = benchmark_metrics
        prune_metrics = context.metrics.get("prune")
        if isinstance(prune_metrics, dict) and prune_metrics.get("method") in {
            "nm_structured",
            "block_sparse",
        }:
            benchmark_metrics["capability"] = {
                "prune": prune_runtime_capability_from_report(
                    prune_metrics,
                    device=context.config.model.device,
                )
            }
        _update_structured_prune_benchmark_status(context, benchmark_metrics)
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="benchmark.p50_ms",
                    value=report.p50_ms,
                )
            )
            if memory_report is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="benchmark.memory.delta_bytes",
                        value=memory_report.delta_bytes,
                        metadata={"backend": memory_report.backend},
                    )
                )
        return context


@register_pass("write_reports")
class WriteReportsPass:
    """Write JSON and Markdown reports for collected metrics."""

    name = "write_reports"

    def run(self, context: XQTContext) -> XQTContext:
        artifact_dir = Path(context.config.project.artifact_dir)
        json_path = write_json_report(context.metrics, artifact_dir / "metrics.json")
        markdown_path = write_markdown_report(
            "XQT Metrics",
            {
                key: value if isinstance(value, dict) else {"value": value}
                for key, value in context.metrics.items()
            },
            artifact_dir / "metrics.md",
        )
        context.artifacts["metrics_json"] = json_path
        context.artifacts["metrics_markdown"] = markdown_path
        return context


__all__ = [
    "AnalyzePass",
    "BenchmarkPass",
    "ExportPass",
    "LoadModelPass",
    "PrunePass",
    "QuantPass",
    "WriteReportsPass",
]
