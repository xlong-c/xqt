"""ExecuTorch export handler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from xqt.core.artifact import ArtifactRecord
from xqt.core.schema import ExportTargetConfig
from xqt.core.types import XQTContext

from .. import export_pass as _ep

from ._context import _target_summary


def handle_executorch(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    export_model: nn.Module,
    example_input: Any,
    artifact_dir: Path,
    output_diff_config: object = None,
    export_guard: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    output_path = target.output_path
    if output_path is None:
        output_path = str(artifact_dir / f"model_{index}.pte")
    executorch = target.executorch
    try:
        result = _ep.export_executorch_program(
            export_model,
            example_input,
            output_path,
            dry_run=executorch.dry_run,
            metadata={"precision": target.precision},
        )
    except Exception as exc:
        context.metrics[f"export_{index}.executorch_readiness"] = (
            _ep.mobile_export_diagnosis(
                target_format="executorch",
                dry_run=executorch.dry_run,
                converter_missing="executorch is required" in str(exc),
                materialized=False,
                message=str(exc),
            )
        )
        raise
    context.artifacts[f"export_{index}"] = result.pte_path
    if context.manifest is not None and result.checksum is not None:
        context.manifest.add_artifact(
            ArtifactRecord(
                path=str(result.pte_path),
                format="executorch",
                runtime="executorch",
                checksum=result.checksum,
                metadata=result.metadata | {"dry_run": result.dry_run},
            )
        )
    exported_entry: dict[str, object] = {
        "path": str(result.pte_path),
        "format": "executorch",
        "dry_run": result.dry_run,
        "artifact_status": "command_only"
        if result.dry_run
        else "materialized",
        "backend_execution": "dry_run"
        if result.dry_run
        else "executed",
        "checksum": result.checksum,
        "export_readiness": _ep.mobile_export_diagnosis(
            target_format="executorch",
            dry_run=result.dry_run,
            materialized=not result.dry_run,
        ),
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
