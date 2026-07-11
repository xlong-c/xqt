"""ONNX export handler."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from xqt.core.artifact import ArtifactRecord
from xqt.core.schema import ExportTargetConfig, OutputDiffConfig
from xqt.core.types import XQTContext
from xqt.export.input_utils import default_input_names

from .. import export_pass as _ep

from ._context import _target_summary


def handle_onnx(
    context: XQTContext,
    target: ExportTargetConfig,
    index: int,
    *,
    export_model: nn.Module,
    example_input: Any,
    reference_output: torch.Tensor,
    artifact_dir: Path,
    output_diff_config: OutputDiffConfig,
    export_guard: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    output_path = target.output_path
    if output_path is None:
        output_path = str(artifact_dir / f"model_{index}.onnx")
    onnx = target.onnx
    result = _ep.export_onnx(
        export_model,
        example_input,
        output_path,
        opset=target.opset,
        dynamic_shapes=target.dynamic_shapes,
        input_names=onnx.input_names or default_input_names(example_input),
        output_names=onnx.output_names,
        dynamo=onnx.dynamo,
        validate=onnx.validate,
        pre_export_fusion=asdict(onnx.pre_export_fusion),
        pre_export_lowering=asdict(onnx.pre_export_lowering),
    )
    optimization = onnx.optimization
    optimized_result = None
    if optimization.enabled:
        optimized_result = _ep.optimize_onnx(
            result.path,
            optimization.output_path,
            backend=optimization.backend,
            level=optimization.level,
            output_suffix=optimization.output_suffix,
            validate=optimization.validate,
            providers=list(optimization.providers),
            native_qdq=optimization.native_qdq,
            metadata={
                "source_export_path": str(result.path),
            },
        )
        result.metadata["onnx_optimization"] = optimized_result.to_dict()
        result.path = optimized_result.path
        result.checksum = optimized_result.checksum
        result.checked = optimized_result.checked
    diff = None
    if onnx.runtime_diff:
        if reference_output is None or example_input is None:
            raise ValueError(
                "onnx runtime_diff requires model reference output"
            )
        diff = _ep.compare_onnxruntime_outputs(
            result.path,
            reference_output,
            example_input,
            input_names=result.metadata.get("input_names"),
            atol=output_diff_config.atol,
            rtol=output_diff_config.rtol,
        )
        result.output_diff = diff
    result_metadata = getattr(result, "metadata", {})
    record = ArtifactRecord.from_file(
        result.path,
        format="onnx",
        runtime="onnxruntime" if diff is not None else None,
        metadata={
            "opset": result.opset,
            "checked": result.checked,
            "output_diff": diff.to_dict() if diff is not None else None,
            "export_guard": dict(export_guard),
            **result_metadata,
        },
    )
    context.artifacts[f"export_{index}"] = result.path
    context.artifacts["last_onnx"] = result.path
    if optimized_result is not None:
        context.artifacts[f"export_{index}_source_onnx"] = (
            optimized_result.source_path
        )
        context.artifacts["last_onnx_source"] = optimized_result.source_path
    if context.manifest is not None:
        context.manifest.add_artifact(record)
    exported_entry: dict[str, object] = {
        "path": str(result.path),
        "format": "onnx",
        "opset": result.opset,
        "checked": result.checked,
        "checksum": result.checksum,
        "input_names": list(result_metadata.get("input_names", [])),
        "output_names": list(result_metadata.get("output_names", [])),
        "dynamic_shapes": dict(result_metadata.get("dynamic_shapes", {})),
        "output_diff": diff.to_dict() if diff is not None else None,
        "pre_export_fusion": result_metadata.get("pre_export_fusion"),
        "pre_export_lowering": result_metadata.get("pre_export_lowering"),
        "onnx_optimization": result_metadata.get("onnx_optimization"),
        "export_guard": dict(export_guard),
    }
    summary = _target_summary(target, exported_entry)
    return exported_entry, summary
