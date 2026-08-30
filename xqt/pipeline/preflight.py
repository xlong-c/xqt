"""Preflight checks for XQT recipes."""

from __future__ import annotations

from xqt.core.config import ConfigInput
from xqt.core.schema import (
    PruneConfig,
    QuantConfig,
    TASK_TYPES,
)
from xqt.core.workflow_schema import OptimizationConfig
from xqt.core.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    DeployStageSpec,
    ExportStageSpec,
    OperatorStageSpec,
    PruneStageSpec,
    QuantStageSpec,
    stage_spec_to_config,
)

from .preflight_checks._base import (
    PreflightCheck,
    PreflightReport,
    _check_target,
)
from .preflight_checks.deploy import _check_deploy_runtime_handle
from .preflight_checks.export import _check_export_targets
from .preflight_checks.model import _check_model_device
from .preflight_checks.operator import _check_operator_targets
from .preflight_checks.prune import _check_prune_config
from .preflight_checks.quant import _check_quant_config


def preflight_optimization_config(
    config: ConfigInput | OptimizationConfig,
) -> PreflightReport:
    """Run lightweight dependency and target checks for a stage workflow."""

    from xqt.core.workflow_loader import load_optimization_config

    loaded = load_optimization_config(config)
    report = PreflightReport()
    report.add(
        "project.artifact_dir",
        True,
        "artifact directory configured",
        path=str(loaded.project.get("artifact_dir", "")),
    )
    report.add(
        "task.type",
        loaded.task.type in TASK_TYPES,
        "task type configured",
        task_type=loaded.task.type,
    )
    if loaded.task.type == "detection":
        report.add(
            "task.detection_postprocess",
            True,
            "detection postprocess configured",
            **{
                "format": loaded.task.detection_postprocess.format,
                "box_format": loaded.task.detection_postprocess.box_format,
                "score_threshold": loaded.task.detection_postprocess.score_threshold,
                "iou_threshold": loaded.task.detection_postprocess.iou_threshold,
                "max_detections": loaded.task.detection_postprocess.max_detections,
            },
        )
    _check_target(report, "model.target", loaded.model.target)
    _check_model_device(report, loaded.device or loaded.model.device)

    for index, stage in enumerate(loaded.stages):
        spec = getattr(stage, "spec", None)
        if spec is None:
            raise RuntimeError(f"stage {stage.name!r} has no typed spec")
        prefix = f"stages.{index}.{stage.name}"
        report.add(
            f"{prefix}.kind",
            True,
            "stage kind configured",
            kind=stage.kind,
        )
        if isinstance(spec, QuantStageSpec):
            _check_quant_config(
                report,
                stage_spec_to_config(
                    spec,
                    QuantConfig,
                    overrides={"enabled": True},
                ),
                prefix=prefix,
                cuda_name="hardware.cuda",
            )
        elif isinstance(spec, PruneStageSpec):
            _check_prune_config(
                report,
                stage_spec_to_config(
                    spec,
                    PruneConfig,
                    overrides={"enabled": True},
                ),
                device=loaded.device or loaded.model.device,
                task_type=loaded.task.type,
                prefix=prefix,
            )
        elif isinstance(spec, OperatorStageSpec):
            _check_operator_targets(
                report,
                spec.targets,
                default_engine=spec.default_engine,
                prefix=prefix,
            )
        elif isinstance(spec, (ExportStageSpec, DeployStageSpec)):
            _check_export_targets(
                report, list(spec.targets), prefix=f"{prefix}.targets"
            )
            if isinstance(spec, DeployStageSpec):
                _check_deploy_runtime_handle(
                    report,
                    spec.runtime_handle,
                    prefix=prefix,
                    targets=list(spec.targets),
                )
        elif isinstance(spec, AnalyzeStageSpec):
            report.add(
                f"{prefix}.analysis.metrics",
                bool(spec.metrics),
                "analysis metrics configured"
                if spec.metrics
                else "analysis metrics missing",
                level="info" if spec.metrics else "error",
                metrics=list(spec.metrics),
            )
        elif isinstance(spec, BenchmarkStageSpec):
            report.add(
                f"{prefix}.benchmark",
                True,
                "benchmark override recorded",
                warmup=spec.warmup,
                iterations=spec.iterations,
                measure_memory=spec.measure_memory,
            )
    return report


__all__ = [
    "PreflightCheck",
    "PreflightReport",
    "preflight_optimization_config",
]
