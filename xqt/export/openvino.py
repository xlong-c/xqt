"""OpenVINO export and runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError
from xqt.eval.compare import TensorDiff, compare_tensors


@dataclass
class OpenVINOExportResult:
    """OpenVINO IR export metadata."""

    xml_path: Path
    bin_path: Optional[Path]
    checksum: Optional[str]
    output_diff: Optional[TensorDiff] = None
    dry_run: bool = False
    source_path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _import_openvino() -> Any:
    try:
        import openvino as ov
    except ImportError as exc:
        raise XQTBackendError("openvino is required for OpenVINO export") from exc
    return ov


def export_openvino_ir(
    model: nn.Module | str | Path,
    output_path: str | Path,
    *,
    example_input: Optional[Any] = None,
    input_shape: Optional[list[int]] = None,
    dry_run: bool = False,
) -> OpenVINOExportResult:
    """Convert a PyTorch module or ONNX path to OpenVINO IR."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = None if isinstance(model, nn.Module) else Path(model)

    if dry_run:
        return OpenVINOExportResult(
            xml_path=output,
            bin_path=output.with_suffix(".bin"),
            checksum=None,
            dry_run=True,
            source_path=source_path,
            metadata={
                "input_shape": input_shape,
                "source_path": str(source_path) if source_path is not None else None,
                "command": [
                    "openvino.convert_model",
                    str(source_path) if source_path is not None else "<torch.nn.Module>",
                    f"--output={output}",
                ],
            },
        )

    ov = _import_openvino()

    if isinstance(model, nn.Module):
        if example_input is None:
            raise ValueError("example_input is required when converting a PyTorch module")
        converted = ov.convert_model(model, example_input=example_input)
    else:
        converted = ov.convert_model(str(model), input=input_shape)

    ov.save_model(converted, str(output))
    bin_path = output.with_suffix(".bin")
    return OpenVINOExportResult(
        xml_path=output,
        bin_path=bin_path if bin_path.is_file() else None,
        checksum=file_sha256(output),
        dry_run=False,
        source_path=source_path,
        metadata={"input_shape": input_shape},
    )


def compare_openvino_outputs(
    xml_path: str | Path,
    reference_output: torch.Tensor,
    example_input: torch.Tensor,
    *,
    device: str = "CPU",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TensorDiff:
    """Run OpenVINO runtime and compare its first output with a reference tensor."""

    ov = _import_openvino()
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), device)
    output = compiled([example_input.detach().cpu().numpy()])[0]
    candidate = torch.from_numpy(np.asarray(output))
    return compare_tensors(reference_output, candidate, atol=atol, rtol=rtol)


__all__ = [
    "OpenVINOExportResult",
    "compare_openvino_outputs",
    "export_openvino_ir",
]
