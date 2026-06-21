"""ONNX export and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError
from xqt.eval.compare import TensorDiff, compare_tensors
from xqt.export.fusion import apply_pre_export_fusion
from xqt.export.input_utils import (
    build_onnx_feed,
    default_input_names,
    split_example_input,
)


@dataclass
class ONNXExportResult:
    """ONNX export metadata."""

    path: Path
    opset: Optional[int]
    checksum: str
    checked: bool
    output_diff: Optional[TensorDiff] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def export_onnx(
    model: nn.Module,
    example_input: Any,
    output_path: str | Path,
    *,
    opset: Optional[int] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Mapping[str, Any]] = None,
    dynamo: bool = True,
    validate: bool = True,
    pre_export_fusion: Optional[Mapping[str, Any]] = None,
) -> ONNXExportResult:
    """Export a PyTorch module to ONNX using the modern dynamo exporter by default."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example_spec = split_example_input(example_input)
    resolved_input_names = list(input_names or default_input_names(example_input))
    fusion_result = apply_pre_export_fusion(model, pre_export_fusion)
    export_model = fusion_result.model
    export_model.eval()

    kwargs: dict[str, Any] = {
        "dynamo": dynamo,
        "input_names": resolved_input_names,
        "output_names": list(output_names or ["output"]),
    }
    if opset is not None:
        kwargs["opset_version"] = opset
    if dynamic_shapes:
        kwargs["dynamic_shapes"] = dict(dynamic_shapes)
    if example_spec.kwargs:
        kwargs["kwargs"] = dict(example_spec.kwargs)

    try:
        torch.onnx.export(export_model, example_spec.args, str(output), **kwargs)
    except TypeError:
        fallback_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"dynamo", "dynamic_shapes"}
        }
        torch.onnx.export(export_model, example_spec.args, str(output), **fallback_kwargs)

    checked = validate_onnx(output) if validate else False
    return ONNXExportResult(
        path=output,
        opset=opset,
        checksum=file_sha256(output),
        checked=checked,
        metadata={
            "dynamo": dynamo,
            "input_names": resolved_input_names,
            "output_names": list(output_names or ["output"]),
            "pre_export_fusion": dict(fusion_result.metadata),
        },
    )


def validate_onnx(path: str | Path) -> bool:
    """Run ONNX checker on an exported model."""

    try:
        import onnx
    except ImportError as exc:
        raise XQTBackendError("onnx is required to validate ONNX exports") from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return True


def convert_onnx_to_fp16(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    keep_io_types: bool = False,
    validate: bool = True,
) -> ONNXExportResult:
    """Convert an ONNX model to FP16 using onnxconverter-common."""

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise XQTBackendError(
            "onnx and onnxconverter-common are required for ONNX FP16 conversion"
        ) from exc

    source = Path(onnx_path)
    if not source.is_file():
        raise XQTBackendError(f"ONNX file not found: {source}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
    )
    onnx.save(fp16_model, str(output))
    checked = validate_onnx(output) if validate else False
    return ONNXExportResult(
        path=output,
        opset=None,
        checksum=file_sha256(output),
        checked=checked,
        metadata={
            "precision": "fp16",
            "source_path": str(source),
            "keep_io_types": keep_io_types,
            "converter": "onnxconverter_common.float16",
        },
    )


def compare_onnxruntime_outputs(
    onnx_path: str | Path,
    reference_output: torch.Tensor,
    example_input: Any,
    *,
    input_name: str = "input",
    input_names: Optional[Sequence[str]] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TensorDiff:
    """Run ONNX Runtime and compare its first output with a PyTorch tensor."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise XQTBackendError("onnxruntime is required for ONNX Runtime diff") from exc

    resolved_input_names = list(input_names or [input_name])
    feeds = build_onnx_feed(example_input, input_names=resolved_input_names)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, feeds)[0]
    candidate = torch.from_numpy(np.asarray(ort_output))
    return compare_tensors(reference_output, candidate, atol=atol, rtol=rtol)


__all__ = [
    "ONNXExportResult",
    "compare_onnxruntime_outputs",
    "convert_onnx_to_fp16",
    "export_onnx",
    "validate_onnx",
]
