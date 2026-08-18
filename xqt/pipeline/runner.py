"""Context builders for XQT optimization workflows."""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from xqt.core.artifact import ArtifactManifest, file_sha256
from xqt.core.config import ConfigInput
from xqt.core.schema import (
    AnalysisConfig,
    OperatorOptimizationConfig,
    OutputDiffConfig,
    PruneConfig,
    QuantConfig,
    TaskConfig,
)
from xqt.core.types import XQTContext
from xqt.core.workflow_schema import OptimizationConfig


def _task_to_manifest_dict(task: TaskConfig) -> dict[str, Any]:
    return {
        "type": task.type,
        "class_names": list(task.class_names),
        "detection_postprocess": {
            "format": task.detection_postprocess.format,
            "box_format": task.detection_postprocess.box_format,
            "score_threshold": task.detection_postprocess.score_threshold,
            "iou_threshold": task.detection_postprocess.iou_threshold,
            "max_detections": task.detection_postprocess.max_detections,
            "score_activation": task.detection_postprocess.score_activation,
            "has_objectness": task.detection_postprocess.has_objectness,
            "class_agnostic_nms": task.detection_postprocess.class_agnostic_nms,
            "rescale_to_original": task.detection_postprocess.rescale_to_original,
        },
        "params": dict(task.params),
    }


def _checkpoint_checksum(checkpoint: str | None) -> Optional[str]:
    if not checkpoint:
        return None
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        return None
    return file_sha256(checkpoint_path)


def _create_context_from_optimization_config(
    config: ConfigInput | OptimizationConfig,
    *,
    model: Any = None,
    example_inputs: Any = None,
    calibration_inputs: Any = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    manifest: Optional[ArtifactManifest] = None,
) -> XQTContext:
    from xqt.workflows.optimization import load_optimization_config

    loaded = load_optimization_config(config)
    project = dict(loaded.project)
    project_name = str(project.get("name", "xqt_optimization"))
    artifact_dir = str(project.get("artifact_dir", "artifacts/xqt/optimization"))
    device = loaded.device or loaded.model.device
    compression_axes = list(loaded.compression_axes)
    config_snapshot = asdict(loaded) if is_dataclass(loaded) else dict(loaded)

    return XQTContext(
        model=model,
        reference_model=copy.deepcopy(model) if model is not None else None,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
        artifacts=dict(artifacts or {}),
        metrics=dict(metrics or {}),
        device=device,
        artifact_dir=artifact_dir,
        project_name=project_name,
        task_type=loaded.task.type,
        compression_axes=compression_axes,
        model_target=loaded.model.target,
        model_params=copy.deepcopy(loaded.model.params),
        quant_config=QuantConfig(),
        prune_config=PruneConfig(),
        analysis_config=AnalysisConfig(),
        benchmark_config=copy.deepcopy(loaded.benchmark),
        operator_config=OperatorOptimizationConfig(),
        output_diff_config=OutputDiffConfig(),
        export_targets=[],
        manifest=manifest
        or ArtifactManifest(
            project_name=project_name,
            source_checkpoint=loaded.model.checkpoint,
            source_checksum=_checkpoint_checksum(loaded.model.checkpoint),
            compression_axes=compression_axes,
            task=_task_to_manifest_dict(loaded.task),
            config_snapshot=config_snapshot,
        ),
    )


def create_context(
    config: ConfigInput | OptimizationConfig,
    *,
    model: Any = None,
    example_inputs: Any = None,
    calibration_inputs: Any = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    manifest: Optional[ArtifactManifest] = None,
) -> XQTContext:
    """Build an XQTContext from a stage workflow."""

    return _create_context_from_optimization_config(
        config,
        model=model,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
        artifacts=artifacts,
        metrics=metrics,
        manifest=manifest,
    )


__all__ = [
    "create_context",
]
