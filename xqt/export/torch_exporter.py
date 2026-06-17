"""PyTorch native export helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.core.artifact import file_sha256
from xqt.eval.compare import TensorDiff, compare_tensors


@dataclass
class TorchExportResult:
    """torch.export artifact metadata."""

    path: Path
    checksum: str
    checked: bool
    output_diff: Optional[TensorDiff] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TorchScriptExportResult:
    """TorchScript artifact metadata."""

    path: Path
    checksum: str
    output_diff: Optional[TensorDiff] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_example_args(example_input: Any) -> tuple[Any, ...]:
    if isinstance(example_input, tuple):
        return example_input
    if isinstance(example_input, list):
        return tuple(example_input)
    return (example_input,)


def _first_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("export diff currently requires a Tensor or tuple/list first Tensor output")


def export_torch_program(
    model: nn.Module,
    example_input: Any,
    output_path: str | Path,
    *,
    dynamic_shapes: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    validate: bool = True,
    compare_output: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TorchExportResult:
    """Export a module with torch.export and save an ExportedProgram."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    args = _as_example_args(example_input)

    with torch.no_grad():
        reference_output = _first_tensor_output(model(*args))
    exported = torch.export.export(
        model,
        args,
        dynamic_shapes=dict(dynamic_shapes) if dynamic_shapes else None,
        strict=strict,
    )
    torch.export.save(exported, output)

    checked = False
    diff = None
    if validate:
        loaded = torch.export.load(output)
        checked = True
        if compare_output:
            loaded_module = loaded.module()
            with torch.no_grad():
                candidate_output = _first_tensor_output(loaded_module(*args))
            diff = compare_tensors(reference_output, candidate_output, atol=atol, rtol=rtol)

    return TorchExportResult(
        path=output,
        checksum=file_sha256(output),
        checked=checked,
        output_diff=diff,
        metadata={
            "strict": strict,
            "dynamic_shapes": dict(dynamic_shapes or {}),
            "validated": validate,
        },
    )


def export_torchscript(
    model: nn.Module,
    example_input: Any,
    output_path: str | Path,
    *,
    method: str = "trace",
    check_trace: bool = True,
    compare_output: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TorchScriptExportResult:
    """Export a module to TorchScript using trace or script."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    args = _as_example_args(example_input)
    with torch.no_grad():
        reference_output = _first_tensor_output(model(*args))

    if method == "trace":
        scripted = torch.jit.trace(model, args, check_trace=check_trace)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError("TorchScript export method must be trace or script")
    torch.jit.save(scripted, str(output))

    diff = None
    if compare_output:
        loaded = torch.jit.load(str(output), map_location="cpu")
        loaded.eval()
        with torch.no_grad():
            candidate_output = _first_tensor_output(loaded(*args))
        diff = compare_tensors(reference_output, candidate_output, atol=atol, rtol=rtol)

    return TorchScriptExportResult(
        path=output,
        checksum=file_sha256(output),
        output_diff=diff,
        metadata={
            "method": method,
            "check_trace": check_trace,
        },
    )


__all__ = [
    "TorchExportResult",
    "TorchScriptExportResult",
    "export_torch_program",
    "export_torchscript",
]
