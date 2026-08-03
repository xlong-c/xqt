"""PyTorch native export helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.core.artifact import file_sha256
from xqt.analysis.compare import TensorDiff, compare_tensors
from xqt.export.input_utils import (
    call_model_with_example_input,
    first_tensor_output,
    split_example_input,
)


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
    example_spec = split_example_input(example_input)

    with torch.no_grad():
        reference_output = first_tensor_output(
            call_model_with_example_input(model, example_input)
        )
    exported = torch.export.export(
        model,
        example_spec.args,
        kwargs=dict(example_spec.kwargs) if example_spec.kwargs else None,
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
                candidate_output = first_tensor_output(
                    call_model_with_example_input(loaded_module, example_input)
                )
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
    example_spec = split_example_input(example_input)
    with torch.no_grad():
        reference_output = first_tensor_output(
            call_model_with_example_input(model, example_input)
        )

    if method == "trace":
        if example_spec.kwargs:
            scripted = torch.jit.trace(
                model,
                example_kwarg_inputs=dict(example_spec.kwargs),
                check_trace=check_trace,
                strict=False,
            )
        else:
            scripted = torch.jit.trace(
                model,
                example_spec.args,
                check_trace=check_trace,
            )
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
            candidate_output = first_tensor_output(
                call_model_with_example_input(loaded, example_input)
            )
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
