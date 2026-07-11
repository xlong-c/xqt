"""Built-in lightweight XQT passes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from xqt.analysis import (
    layer_statistics_rows,
    records_to_rows,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)
from xqt.benchmark import benchmark_callable, benchmark_memory
from xqt.core.artifact import ArtifactRecord, MetricRecord
from xqt.core.imports import build_target
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.registry import register_pass
from xqt.core.schema import (
    AnalysisConfig,
    BenchmarkConfig,
    OperatorOptimizationConfig,
    OutputDiffConfig,
)
from xqt.core.types import XQTContext
from xqt.operator_opt.execute import execute_operator_optimization_plan
from xqt.operator_opt.plan import build_operator_optimization_plan
from xqt.operator_opt.reporting import summarize_operator_optimization_reports
from xqt.prune import (
    collect_module_importance,
    prune_runtime_capability_from_report,
    rank_prune_candidates,
)
from xqt.quant import (
    analyze_activation_drift,
    analyze_layer_errors,
    recommend_high_precision_modules,
)

from .export_pass import ExportPass, call_model as _call_model, run_export_stage
from .pass_helpers.context import (
    _analysis_runtime_config,
    _benchmark_runtime_config,
    _context_analysis_config,
    _context_benchmark_config,
    _context_model_params,
    _context_model_target,
    _context_operator_config,
    _context_output_diff_config,
    _move_to_device,
    _operator_runtime_config,
)
from .pass_helpers.prune_stage import (
    _resolved_prune_config,
    _run_prune_with_resolved_config,
    _update_structured_prune_benchmark_status,
)
from .pass_helpers.quant_stage import _run_quant_with_resolved_config
from xqt.workflows.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    OperatorStageSpec,
    PruneStageSpec,
    QuantStageSpec,
)


@register_pass("load_model")
class LoadModelPass:
    """Build the configured PyTorch model."""

    name = "load_model"

    def run(self, context: XQTContext) -> XQTContext:
        if context.model is not None:
            return context
        target = _context_model_target(context)
        if not target:
            raise ValueError("model.target is required when context.model is not set")
        model = build_target(target, _context_model_params(context))
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

    def run(
        self,
        context: XQTContext,
        *,
        analysis_config: AnalysisConfig | None = None,
        output_diff: OutputDiffConfig | None = None,
    ) -> XQTContext:
        resolved_analysis = analysis_config or _context_analysis_config(context)
        resolved_output_diff = output_diff or _context_output_diff_config(context)
        if not resolved_analysis.enabled:
            return context
        model = context.require_model()
        if resolved_analysis.compare_to != "baseline":
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
            torch.device(context.device),
        )
        model = model.to(context.device)
        reference_model = reference_model.to(context.device)
        records = analyze_layer_errors(
            reference_model,
            model,
            inputs,
            module_names=resolved_analysis.module_names,
            atol=resolved_output_diff.atol,
            rtol=resolved_output_diff.rtol,
            include_weight_diff=resolved_analysis.include_weight_diff,
            policy=None,
            per_channel=resolved_analysis.structured.per_channel,
            per_token=resolved_analysis.structured.per_token,
        )
        if resolved_analysis.top_k is not None:
            records = records[: resolved_analysis.top_k]

        rows = records_to_rows(records)
        statistics_module_names = (
            resolved_analysis.module_names
            if resolved_analysis.module_names is not None
            else [record.name for record in records]
        )
        layer_statistics = (
            layer_statistics_rows(
                reference_model,
                model,
                inputs,
                module_names=statistics_module_names,
                sample_budget=resolved_analysis.sample_budget,
                sample_seed=resolved_analysis.sample_seed,
                histogram_bins=resolved_analysis.histogram_bins,
            )
            if resolved_analysis.include_statistics
            else []
        )
        activation_drift = analyze_activation_drift(
            reference_model,
            model,
            [inputs],
            module_names=resolved_analysis.module_names,
        )
        importance_records = collect_module_importance(
            model,
            module_names=resolved_analysis.module_names,
        )
        prune_candidates: list[dict[str, object]] = []
        if resolved_analysis.recommendations.prune_candidates:
            prune_candidate_records = rank_prune_candidates(
                model,
                records,
                module_names=resolved_analysis.module_names,
                top_k=resolved_analysis.top_k,
            )
            prune_candidates = [record.to_dict() for record in prune_candidate_records]
        recommended_modules: list[str] = []
        if resolved_analysis.recommendations.mixed_precision:
            recommended_modules = recommend_high_precision_modules(
                records,
                top_k=resolved_analysis.top_k,
            )
        context.metrics["analysis"] = {
            "compare_to": resolved_analysis.compare_to,
            "metrics": list(resolved_analysis.metrics),
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

        artifact_dir = Path(context.artifact_dir)
        if resolved_analysis.export.json:
            json_path = write_json_report(
                context.metrics["analysis"],
                artifact_dir / "analysis.json",
            )
            context.artifacts["analysis_json"] = json_path
        if resolved_analysis.export.csv:
            csv_path = write_csv_report(rows, artifact_dir / "analysis.csv")
            context.artifacts["analysis_csv"] = csv_path
        if resolved_analysis.export.markdown:
            top_rows = rows[: min(resolved_analysis.top_k or len(rows), len(rows))]
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
                    metadata={"compare_to": resolved_analysis.compare_to},
                )
            )
        return context


@register_pass("operator_optimization")
class OperatorOptimizationPass:
    """Apply configured operator optimization engine."""

    name = "operator_optimization"

    def run(
        self,
        context: XQTContext,
        *,
        operator_config: OperatorOptimizationConfig | None = None,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> XQTContext:
        model = context.require_model()
        resolved_operator = operator_config or _context_operator_config(context)
        resolved_benchmark = benchmark_config or _context_benchmark_config(context)
        if not resolved_operator.enabled:
            return context
        plan = build_operator_optimization_plan(resolved_operator)
        execution = execute_operator_optimization_plan(
            context,
            plan,
            benchmark_config=resolved_benchmark,
            device=context.device,
        )
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
                            "fallback_policy": report.fallback_policy,
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

    def run(
        self,
        context: XQTContext,
        *,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> XQTContext:
        model = context.require_model()
        if context.example_inputs is None:
            raise ValueError("example_inputs are required for benchmark")
        resolved_benchmark = benchmark_config or _context_benchmark_config(context)
        batch = context.example_inputs
        inputs = _move_to_device(
            extract_model_inputs(
                batch,
                expected_input_count=infer_model_input_count(model),
            ),
            torch.device(context.device),
        )

        def fn() -> object:
            return _call_model(model, inputs)

        report = benchmark_callable(
            fn,
            warmup=resolved_benchmark.warmup,
            iterations=resolved_benchmark.iterations,
            sync_cuda=resolved_benchmark.sync_cuda,
            device=context.device,
        )
        benchmark_metrics = report.to_dict()
        benchmark_metrics["latency"] = report.to_dict()
        memory_report = None
        if resolved_benchmark.measure_memory:
            memory_report = benchmark_memory(
                fn,
                iterations=1,
                sync_cuda=resolved_benchmark.sync_cuda,
                device=context.device,
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
                    device=context.device,
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


def run_prune_stage(context: XQTContext, spec: PruneStageSpec) -> XQTContext:
    """Run a prune stage from a typed stage spec."""

    return _run_prune_with_resolved_config(context, _resolved_prune_config(context, spec))


def run_quant_stage(context: XQTContext, spec: QuantStageSpec) -> XQTContext:
    """Run a quant stage from a typed stage spec."""

    return _run_quant_with_resolved_config(context, spec)


def run_operator_stage(
    context: XQTContext,
    spec: OperatorStageSpec,
    *,
    base_benchmark_config: BenchmarkConfig | None = None,
) -> XQTContext:
    """Run an operator optimization stage from a typed stage spec."""

    operator_config = _operator_runtime_config(spec)
    benchmark_config = (
        _benchmark_runtime_config(
            spec.benchmark,
            base=base_benchmark_config or _context_benchmark_config(context),
        )
        if spec.benchmark is not None
        else base_benchmark_config or _context_benchmark_config(context)
    )
    context.operator_config = copy.deepcopy(operator_config)
    context.benchmark_config = copy.deepcopy(benchmark_config)
    return OperatorOptimizationPass().run(
        context,
        operator_config=operator_config,
        benchmark_config=benchmark_config,
    )


def run_analyze_stage(context: XQTContext, spec: AnalyzeStageSpec) -> XQTContext:
    """Run an analysis stage from a typed stage spec."""

    analysis_config = _analysis_runtime_config(spec)
    context.analysis_config = copy.deepcopy(analysis_config)
    return AnalyzePass().run(context, analysis_config=analysis_config)


def run_benchmark_stage(
    context: XQTContext,
    spec: BenchmarkStageSpec,
    *,
    base_benchmark_config: BenchmarkConfig | None = None,
) -> XQTContext:
    """Run a benchmark stage from a typed stage spec."""

    benchmark_config = _benchmark_runtime_config(
        spec,
        base=base_benchmark_config or _context_benchmark_config(context),
    )
    context.benchmark_config = copy.deepcopy(benchmark_config)
    return BenchmarkPass().run(context, benchmark_config=benchmark_config)


@register_pass("write_reports")
class WriteReportsPass:
    """Write JSON and Markdown reports for collected metrics."""

    name = "write_reports"

    def run(self, context: XQTContext) -> XQTContext:
        artifact_dir = Path(context.artifact_dir)
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
    "WriteReportsPass",
    "run_analyze_stage",
    "run_benchmark_stage",
    "run_export_stage",
    "run_operator_stage",
    "run_prune_stage",
    "run_quant_stage",
]
