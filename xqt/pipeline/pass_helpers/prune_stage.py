"""Prune stage implementation helpers."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from xqt.core.artifact import MetricRecord
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.schema import PruneConfig
from xqt.core.types import XQTContext
from xqt.compression.prune import (
    PruningSchedule,
    apply_block_sparse_pruning,
    apply_global_l1_unstructured_pruning,
    apply_nm_structured_sparsity,
    apply_structured_pruning,
    describe_prune_granularity,
    estimate_model_flops,
    prune_method_report_fields,
    prune_runtime_capability_from_report,
    remove_pruning_reparameterization,
    run_prune_schedule,
    run_structured_prune_schedule,
    summarize_pruning,
)
from xqt.core.stage_specs import PruneStageSpec

from .context import _context_prune_config, _context_task_type, _move_to_device


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
        **prune_method_report_fields(method),
        **describe_prune_granularity(granularity or "channel"),
        "method": method,
        "granularity": granularity or "channel",
        "scope": scope,
        "target_sparsity": target_sparsity,
        "task_type": "detection",
        "applied": False,
        "execution_state": "skipped",
        "skip_reason": skip_reason,
        "safety": {
            "task_type": "detection",
            "passed": True,
            "checks": [
                {
                    "name": "detection_head_guard",
                    "family": "detection",
                    "passed": True,
                    "blocked_modules": [],
                    "violations": [],
                    "notes": [skip_reason],
                }
            ],
            "blocked_modules": [],
            "violations": [],
        },
        "skipped_modules": [
            {
                "module_name": module_name,
                "reason": skip_reason,
            }
            for module_name in _prune_module_names(model, module_types)
        ],
    }


def _annotate_unstructured_prune_report(
    report: dict[str, object],
    *,
    model: nn.Module,
    task_type: str,
    target_sparsity: float,
) -> dict[str, object]:
    flops_before = estimate_model_flops(model)["flops"]
    report["method"] = "global_l1_unstructured"
    report["target_sparsity"] = target_sparsity
    report["applied"] = True
    report["execution_state"] = "applied"
    report["task_type"] = task_type
    report.update(prune_method_report_fields("global_l1_unstructured"))
    report["skip_reason"] = None
    report["skipped_modules"] = []
    report["mask_only_modules"] = [
        str(entry["module_name"]) for entry in report.get("entries", [])
    ]
    report["flops_before"] = flops_before
    report["flops_after"] = flops_before
    report["flops_reduction_ratio"] = 0.0
    report["flops_estimate_kind"] = "shape_based_per_output_position"
    return report


def _resolved_prune_config(
    context: XQTContext,
    prune_config: PruneConfig | PruneStageSpec | None,
) -> PruneConfig:
    if isinstance(prune_config, PruneStageSpec):
        return PruneConfig(
            enabled=True,
            method=prune_config.method,
            granularity=prune_config.granularity,
            scope=prune_config.scope,
            target_sparsity=prune_config.target_sparsity,
            schedule=prune_config.schedule,
            importance=dict(prune_config.importance),
            selection=dict(prune_config.selection),
            rewrite=dict(prune_config.rewrite),
            params=dict(prune_config.params),
        )
    return copy.deepcopy(prune_config) if prune_config is not None else copy.deepcopy(_context_prune_config(context))


def _run_prune_with_resolved_config(
    context: XQTContext,
    resolved_prune: PruneConfig,
) -> XQTContext:
    context.prune_config = copy.deepcopy(resolved_prune)
    model = context.require_model()
    if not resolved_prune.enabled:
        return context
    if resolved_prune.method == "nm_structured":
        pattern = resolved_prune.selection.get("pattern") or resolved_prune.params.get("pattern")
        if not isinstance(pattern, (list, tuple)) or len(pattern) != 2:
            raise ValueError(
                "compression.prune.selection.pattern or compression.prune.params.pattern "
                "must be a two-item list like [2, 4] for method=nm_structured"
            )
        pattern_n = int(pattern[0])
        pattern_m = int(pattern[1])
        module_types = _prune_module_types(resolved_prune.params.get("include_module_types"))
        report = apply_nm_structured_sparsity(
            model,
            pattern_n=pattern_n,
            pattern_m=pattern_m,
            module_types=module_types,
        )
        report_dict = report.to_dict()
        report_dict["runtime_capability"] = prune_runtime_capability_from_report(
            report_dict,
            device=context.device,
        )
        context.metrics["prune"] = report_dict
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="prune.sparsity",
                    value=report.sparsity,
                    metadata={
                        "method": resolved_prune.method,
                        "granularity": "nm",
                        "pattern": [pattern_n, pattern_m],
                        "compliance_ratio": report.compliance_ratio,
                    },
                )
            )
        return context
    if resolved_prune.method == "block_sparse":
        block_shape = (
            resolved_prune.selection.get("block_shape")
            or resolved_prune.params.get("block_shape")
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
            target_sparsity=resolved_prune.target_sparsity,
            block_shape=(int(block_shape[0]), int(block_shape[1])),
            module_types=_prune_module_types(resolved_prune.params.get("include_module_types")),
        )
        report_dict = report.to_dict()
        report_dict["runtime_capability"] = prune_runtime_capability_from_report(
            report_dict,
            device=context.device,
        )
        context.metrics["prune"] = report_dict
        if context.manifest is not None:
            context.manifest.add_metric(
                MetricRecord(
                    name="prune.block_sparsity",
                    value=report.sparsity,
                    threshold=resolved_prune.target_sparsity,
                    passed=report.sparsity >= resolved_prune.target_sparsity,
                    metadata={
                        "method": resolved_prune.method,
                        "granularity": "block_sparse",
                        "block_shape": list(report.block_shape),
                        "parameter_sparsity": report.parameter_sparsity,
                    },
                )
        )
        return context
    if resolved_prune.method == "structured":
        module_types = _prune_module_types(resolved_prune.params.get("include_module_types"))
        if _context_task_type(context) == "detection":
            report = _detection_structured_prune_skip_report(
                model,
                method=resolved_prune.method,
                granularity=resolved_prune.granularity,
                scope=resolved_prune.scope,
                target_sparsity=resolved_prune.target_sparsity,
                module_types=module_types,
            )
            context.metrics["prune"] = report
            if context.manifest is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report["sparsity"],
                        threshold=resolved_prune.target_sparsity,
                        passed=False,
                        metadata={
                            "method": resolved_prune.method,
                            "granularity": resolved_prune.granularity,
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
                torch.device(context.device),
            )
        if resolved_prune.schedule in {"linear", "one_shot"} and int(
            resolved_prune.params.get("steps", 1)
        ) > 1:
            report = run_structured_prune_schedule(
                model,
                schedule=PruningSchedule(
                    target_sparsity=resolved_prune.target_sparsity,
                    steps=int(resolved_prune.params.get("steps", 1)),
                    start_sparsity=float(resolved_prune.params.get("start_sparsity", 0.0)),
                    schedule=resolved_prune.schedule,
                ),
                granularity=resolved_prune.granularity or "channel",
                scope=resolved_prune.scope,
                importance=dict(resolved_prune.importance),
                selection=dict(resolved_prune.selection),
                example_input=example_input,
                task_type=_context_task_type(context),
                structure_contract=context.structure_contract,
            )
            context.metrics["prune"] = report.to_dict()
            if context.structure_contract is not None:
                context.metrics["prune"]["structure_contract_family"] = (
                    context.structure_contract.family
                )
            if context.manifest is not None:
                context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report.final_sparsity,
                        threshold=resolved_prune.target_sparsity,
                        passed=report.final_sparsity >= resolved_prune.target_sparsity,
                        metadata={
                            "method": resolved_prune.method,
                            "granularity": resolved_prune.granularity,
                            "schedule": resolved_prune.schedule,
                            "steps": len(report.steps),
                        },
                    )
                )
            return context
        report = apply_structured_pruning(
            model,
            resolved_prune.target_sparsity,
            granularity=resolved_prune.granularity or "channel",
            scope=resolved_prune.scope,
            importance=resolved_prune.importance,
            selection=resolved_prune.selection,
            example_input=example_input,
            task_type=_context_task_type(context),
            structure_contract=context.structure_contract,
        )
        context.metrics["prune"] = report.to_dict()
        if context.structure_contract is not None:
            from xqt.contracts.model_structure import update_structure_contract_for_model

            context.structure_contract = update_structure_contract_for_model(
                context.structure_contract, model
            )
            context.metrics["prune"]["structure_contract_family"] = (
                context.structure_contract.family
            )
            if context.structure_contract.topology_fingerprint:
                context.metrics["prune"]["structure_contract_fingerprint"] = (
                    context.structure_contract.topology_fingerprint
                )
        if context.manifest is not None:
            context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report.sparsity,
                        threshold=resolved_prune.target_sparsity,
                        passed=report.sparsity >= resolved_prune.target_sparsity,
                        metadata={
                            "method": resolved_prune.method,
                            "granularity": resolved_prune.granularity,
                            "parameter_count_before": report.parameter_count_before,
                            "parameter_count_after": report.parameter_count_after,
                            "parameter_reduction_ratio": report.parameter_reduction_ratio,
                    },
                )
            )
        return context
    if resolved_prune.method != "global_l1_unstructured":
        raise ValueError(
            "Only compression.prune.method=global_l1_unstructured or "
            "compression.prune.method=structured or "
            "compression.prune.method=block_sparse or "
            "compression.prune.method=nm_structured is supported by the built-in prune pass"
        )
    if resolved_prune.schedule in {"linear", "one_shot"} and int(
        resolved_prune.params.get("steps", 1)
    ) > 1:
        report = run_prune_schedule(
            model,
            schedule=PruningSchedule(
                target_sparsity=resolved_prune.target_sparsity,
                steps=int(resolved_prune.params.get("steps", 1)),
                start_sparsity=float(resolved_prune.params.get("start_sparsity", 0.0)),
                schedule=resolved_prune.schedule,
            ),
        )
        context.metrics["prune"] = report.to_dict()
        if context.manifest is not None:
            context.manifest.add_metric(
                    MetricRecord(
                        name="prune.sparsity",
                        value=report.final_sparsity,
                        threshold=resolved_prune.target_sparsity,
                        passed=report.final_sparsity >= resolved_prune.target_sparsity,
                        metadata={
                            "schedule": resolved_prune.schedule,
                            "steps": len(report.steps),
                        },
                    )
            )
        return context
    report = apply_global_l1_unstructured_pruning(
        model,
        resolved_prune.target_sparsity,
    )
    remove_pruning_reparameterization(model)
    report = summarize_pruning(model)
    report_dict = _annotate_unstructured_prune_report(
        report.to_dict(),
        model=model,
        task_type=_context_task_type(context),
        target_sparsity=resolved_prune.target_sparsity,
    )
    context.metrics["prune"] = report_dict
    if context.manifest is not None:
        context.manifest.add_metric(
            MetricRecord(
                name="prune.sparsity",
                value=report.sparsity,
                threshold=resolved_prune.target_sparsity,
                passed=report.sparsity >= resolved_prune.target_sparsity,
                metadata={
                    "method": resolved_prune.method,
                    "task_type": _context_task_type(context),
                    "baseline_kind": report_dict.get("baseline_kind"),
                    "speedup_claimed": report_dict.get("speedup_claimed"),
                },
            )
        )
    return context
