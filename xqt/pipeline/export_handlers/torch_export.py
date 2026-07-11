"""torch.export handler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from xqt.core.artifact import ArtifactRecord
from xqt.core.schema import ExportTargetConfig, OutputDiffConfig
from xqt.core.types import XQTContext

from .. import export_pass as _ep

from ._context import _target_summary


def handle_torch_export(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    export_model: nn.Module,
    example_input: Any,
    artifact_dir: Path,
    output_diff_config: OutputDiffConfig,
    export_guard: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    output_path = target.output_path
    if output_path is None:
        output_path = str(artifact_dir / f"model_{index}.pt2")
    torch_export = target.torch_export
    result = _ep.export_torch_program(
        export_model,
        example_input,
        output_path,
        dynamic_shapes=target.dynamic_shapes,
        strict=torch_export.strict,
        validate=torch_export.validate,
        compare_output=torch_export.runtime_diff,
        atol=output_diff_config.atol,
        rtol=output_diff_config.rtol,
    )
    record = ArtifactRecord.from_file(
        result.path,
        format="torch_export",
        runtime="pytorch",
        metadata={
            "checked": result.checked,
            "output_diff": (
                result.output_diff.to_dict()
                if result.output_diff is not None
                else None
            ),
            "export_guard": dict(export_guard),
            **result.metadata,
        },
    )
    context.artifacts[f"export_{index}"] = result.path
    if context.manifest is not None:
        context.manifest.add_artifact(record)
    exported_entry: dict[str, object] = {
        "path": str(result.path),
        "format": "torch_export",
        "checked": result.checked,
        "checksum": result.checksum,
        "output_diff": (
            result.output_diff.to_dict()
            if result.output_diff is not None
            else None
        ),
        "export_guard": dict(export_guard),
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
