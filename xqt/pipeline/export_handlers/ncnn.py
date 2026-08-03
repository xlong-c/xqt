"""ncnn export handler."""

from __future__ import annotations

from pathlib import Path

from xqt.core.artifact import ArtifactRecord
from xqt.core.schema import ExportTargetConfig
from xqt.core.types import XQTContext

from .. import export_pass as _ep

from ._context import _target_summary


def handle_ncnn(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    artifact_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    ncnn = target.ncnn
    source_path = ncnn.source_path
    if source_path is None:
        if ncnn.converter == "pnnx":
            source_path = context.artifacts.get(
                "last_torchscript"
            ) or context.artifacts.get("last_onnx")
        else:
            source_path = context.artifacts.get("last_onnx")
    if source_path is None:
        raise ValueError(
            "ncnn export requires ncnn.source_path or a compatible prior export"
        )
    param_path = target.output_path
    if param_path is None:
        param_path = str(artifact_dir / f"model_{index}.param")
    bin_path = ncnn.bin_path
    if bin_path is None:
        bin_path = str(Path(param_path).with_suffix(".bin"))
    if ncnn.converter == "pnnx":
        result = _ep.export_ncnn_with_pnnx(
            source_path,
            pnnx_path=ncnn.pnnx_path,
            param_path=param_path,
            bin_path=bin_path,
            extra_args=ncnn.extra_args,
            timeout=ncnn.timeout,
            dry_run=ncnn.dry_run,
        )
    else:
        result = _ep.export_ncnn_from_onnx(
            source_path,
            param_path,
            bin_path,
            onnx2ncnn_path=ncnn.onnx2ncnn_path,
            extra_args=ncnn.extra_args,
            timeout=ncnn.timeout,
            dry_run=ncnn.dry_run,
        )
    context.artifacts[f"export_{index}"] = result.output_paths
    if context.manifest is not None and result.checksums:
        for output in result.output_paths:
            context.manifest.add_artifact(
                ArtifactRecord(
                    path=str(output),
                    format="ncnn",
                    runtime="ncnn",
                    checksum=result.checksums.get(str(output)),
                    metadata={
                        "dry_run": result.dry_run,
                        "command": result.command,
                    },
                )
            )
    exported_entry: dict[str, object] = {
        "paths": [str(path) for path in result.output_paths],
        "format": "ncnn",
        "dry_run": result.dry_run,
        "artifact_status": "command_only"
        if result.dry_run
        else "materialized",
        "backend_execution": "dry_run"
        if result.dry_run
        else "executed",
        "command": result.command,
        "checksums": result.checksums,
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
