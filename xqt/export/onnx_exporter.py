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


@dataclass
class ONNXExportResult:
    """ONNX export metadata."""

    path: Path
    opset: Optional[int]
    checksum: str
    checked: bool
    output_diff: Optional[TensorDiff] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_example_args(example_input: Any) -> tuple[Any, ...]:
    if isinstance(example_input, tuple):
        return example_input
    if isinstance(example_input, list):
        return tuple(example_input)
    return (example_input,)


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
) -> ONNXExportResult:
    """Export a PyTorch module to ONNX using the modern dynamo exporter by default."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    args = _as_example_args(example_input)

    kwargs: dict[str, Any] = {
        "dynamo": dynamo,
        "input_names": list(input_names or ["input"]),
        "output_names": list(output_names or ["output"]),
    }
    if opset is not None:
        kwargs["opset_version"] = opset
    if dynamic_shapes:
        kwargs["dynamic_shapes"] = dict(dynamic_shapes)

    try:
        torch.onnx.export(model, args, str(output), **kwargs)
    except TypeError:
        fallback_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"dynamo", "dynamic_shapes"}
        }
        torch.onnx.export(model, args, str(output), **fallback_kwargs)

    checked = validate_onnx(output) if validate else False
    return ONNXExportResult(
        path=output,
        opset=opset,
        checksum=file_sha256(output),
        checked=checked,
        metadata={
            "dynamo": dynamo,
            "input_names": list(input_names or ["input"]),
            "output_names": list(output_names or ["output"]),
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


def compare_onnxruntime_outputs(
    onnx_path: str | Path,
    reference_output: torch.Tensor,
    example_input: Any,
    *,
    input_name: str = "input",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TensorDiff:
    """Run ONNX Runtime and compare its first output with a PyTorch tensor."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise XQTBackendError("onnxruntime is required for ONNX Runtime diff") from exc

    args = _as_example_args(example_input)
    if len(args) != 1 or not isinstance(args[0], torch.Tensor):
        raise ValueError("compare_onnxruntime_outputs currently supports one Tensor input")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {input_name: args[0].detach().cpu().numpy()})[0]
    candidate = torch.from_numpy(np.asarray(ort_output))
    return compare_tensors(reference_output, candidate, atol=atol, rtol=rtol)


__all__ = [
    "ONNXExportResult",
    "compare_onnxruntime_outputs",
    "export_onnx",
    "validate_onnx",
]
