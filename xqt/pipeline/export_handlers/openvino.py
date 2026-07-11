"""OpenVINO export handler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from xqt.core.artifact import ArtifactRecord
from xqt.core.schema import ExportTargetConfig, OutputDiffConfig
from xqt.core.types import XQTContext

from .. import export_pass as _ep

from ._context import _target_summary


def handle_openvino(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    export_model: nn.Module | None,
    example_input: Any,
    reference_output: torch.Tensor | None,
    artifact_dir: Path,
    output_diff_config: OutputDiffConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    openvino = target.openvino
    source = openvino.onnx_path or context.artifacts.get("last_onnx")
    if source is None:
        if export_model is None:
            raise ValueError(
                "OpenVINO export requires openvino.onnx_path or a loaded model"
            )
        source = export_model
    output_path = target.output_path
    if output_path is None:
        output_path = str(artifact_dir / f"model_{index}.xml")
    result = _ep.export_openvino_ir(
        source,
        output_path,
        example_input=example_input
        if isinstance(source, nn.Module)
        else None,
        input_shape=openvino.input_shape,
        dry_run=openvino.dry_run,
    )
    diff = None
    if (
        not result.dry_run
        and openvino.runtime_diff
        and result.xml_path.is_file()
    ):
        if reference_output is None or example_input is None:
            raise ValueError(
                "OpenVINO runtime_diff requires a loaded model and example_inputs"
            )
        diff = _ep.compare_openvino_outputs(
            result.xml_path,
            reference_output,
            example_input,
            device=openvino.device,
            atol=output_diff_config.atol,
            rtol=output_diff_config.rtol,
        )
        result.output_diff = diff
    metadata = {
        "precision": target.precision,
        "dry_run": result.dry_run,
        "source_path": (
            str(result.source_path)
            if result.source_path is not None
            else None
        ),
        "input_shape": result.metadata.get("input_shape"),
        "output_diff": diff.to_dict() if diff is not None else None,
        **result.metadata,
    }
    context.artifacts[f"export_{index}"] = result.xml_path
    if context.manifest is not None:
        context.manifest.add_artifact(
            ArtifactRecord(
                path=str(result.xml_path),
                format="openvino",
                runtime="openvino",
                checksum=result.checksum,
                metadata=metadata,
            )
        )
    exported_entry: dict[str, object] = {
        "path": str(result.xml_path),
        "bin_path": (
            str(result.bin_path)
            if result.bin_path is not None
            else None
        ),
        "format": "openvino",
        "dry_run": result.dry_run,
        "artifact_status": "command_only"
        if result.dry_run
        else "materialized",
        "backend_execution": "dry_run"
        if result.dry_run
        else "executed",
        "checksum": result.checksum,
        "precision": target.precision,
        "source_path": metadata.get("source_path"),
        "input_shape": metadata.get("input_shape"),
        "output_diff": diff.to_dict() if diff is not None else None,
        "command": metadata.get("command"),
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
