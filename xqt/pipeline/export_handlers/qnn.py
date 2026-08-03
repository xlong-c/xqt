"""Qualcomm QNN export handler."""

from __future__ import annotations

from pathlib import Path

from xqt.core.artifact import ArtifactRecord
from xqt.core.errors import XQTBackendError
from xqt.core.schema import ExportTargetConfig
from xqt.core.types import XQTContext

from .. import export_pass as _ep

from ._context import _target_summary


def handle_qnn(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    artifact_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    qnn = target.qnn
    source_path = qnn.source_path or context.artifacts.get("last_onnx")
    if source_path is None:
        context.metrics[f"export_{index}.qnn_readiness"] = _ep.mobile_export_diagnosis(
            target_format="qnn",
            dry_run=qnn.dry_run,
            source_missing=True,
            message="QNN export requires qnn.source_path or a prior ONNX export",
        )
        raise ValueError(
            "QNN export requires qnn.source_path or a prior ONNX export"
        )
    output_path = target.output_path
    if output_path is None:
        output_path = str(artifact_dir / f"model_{index}_qnn")
    try:
        result = _ep.export_qnn_from_onnx(
            source_path,
            output_path,
            converter_path=qnn.converter_path,
            extra_args=qnn.extra_args,
            timeout=qnn.timeout,
            dry_run=qnn.dry_run,
        )
    except XQTBackendError as exc:
        context.metrics[f"export_{index}.qnn_readiness"] = _ep.mobile_export_diagnosis(
            target_format="qnn",
            dry_run=qnn.dry_run,
            converter_missing="executable not found" in str(exc),
            materialized=False,
            message=str(exc),
        )
        raise
    context.artifacts[f"export_{index}"] = result.output_paths[0]
    if context.manifest is not None and result.checksums:
        for artifact_path in result.checksums:
            context.manifest.add_artifact(
                ArtifactRecord(
                    path=artifact_path,
                    format="qnn",
                    runtime="qnn",
                    checksum=result.checksums[artifact_path],
                    metadata={
                        "dry_run": result.dry_run,
                        "command": result.command,
                    },
                )
            )
    readiness = _ep.mobile_export_diagnosis(
        target_format="qnn",
        dry_run=result.dry_run,
        materialized=not result.dry_run,
    )
    exported_entry: dict[str, object] = {
        "path": str(result.output_paths[0]),
        "format": "qnn",
        "dry_run": result.dry_run,
        "artifact_status": "command_only"
        if result.dry_run
        else "materialized",
        "backend_execution": "dry_run"
        if result.dry_run
        else "executed",
        "command": result.command,
        "checksums": result.checksums,
        "export_readiness": readiness,
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
