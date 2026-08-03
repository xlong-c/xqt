"""MNN export handler."""

from __future__ import annotations

from pathlib import Path

from xqt.core.artifact import ArtifactRecord
from xqt.core.schema import ExportTargetConfig
from xqt.core.types import XQTContext

from .. import export_pass as _ep

from ._context import _target_summary


def handle_mnn(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    artifact_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    mnn = target.mnn
    source_path = mnn.source_path or context.artifacts.get("last_onnx")
    if source_path is None:
        context.metrics[f"export_{index}.mnn_readiness"] = _ep.mobile_export_diagnosis(
            target_format="mnn",
            dry_run=mnn.dry_run,
            source_missing=True,
            message="MNN export requires mnn.source_path or a prior ONNX export",
        )
        raise ValueError(
            "MNN export requires mnn.source_path or a prior ONNX export"
        )
    output_path = target.output_path
    if output_path is None:
        output_path = str(artifact_dir / f"model_{index}.mnn")
    try:
        result = _ep.export_mnn_from_onnx(
            source_path,
            output_path,
            converter_path=mnn.converter_path,
            framework=mnn.framework,
            extra_args=mnn.extra_args,
            timeout=mnn.timeout,
            dry_run=mnn.dry_run,
        )
    except Exception as exc:
        context.metrics[f"export_{index}.mnn_readiness"] = _ep.mobile_export_diagnosis(
            target_format="mnn",
            dry_run=mnn.dry_run,
            converter_missing="executable not found" in str(exc),
            materialized=False,
            message=str(exc),
        )
        raise
    context.artifacts[f"export_{index}"] = result.output_paths[0]
    if context.manifest is not None and result.checksums:
        output = result.output_paths[0]
        context.manifest.add_artifact(
            ArtifactRecord(
                path=str(output),
                format="mnn",
                runtime="mnn",
                checksum=result.checksums.get(str(output)),
                metadata={
                    "dry_run": result.dry_run,
                    "command": result.command,
                },
            )
        )
    exported_entry: dict[str, object] = {
        "path": str(result.output_paths[0]),
        "format": "mnn",
        "dry_run": result.dry_run,
        "artifact_status": "command_only"
        if result.dry_run
        else "materialized",
        "backend_execution": "dry_run"
        if result.dry_run
        else "executed",
        "command": result.command,
        "checksums": result.checksums,
        "export_readiness": _ep.mobile_export_diagnosis(
            target_format="mnn",
            dry_run=result.dry_run,
            materialized=not result.dry_run,
        ),
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
