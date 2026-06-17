"""Mobile and edge deployment adapters."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import nn

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError


@dataclass
class CommandExportResult:
    """Result for command-line based export adapters."""

    output_paths: list[Path]
    command: list[str]
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    checksums: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecuTorchExportResult:
    """ExecuTorch export metadata."""

    pte_path: Path
    checksum: Optional[str] = None
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _ensure_input_file(path: str | Path, *, description: str) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise XQTBackendError(f"{description} file not found: {file_path}")
    return file_path


def _run_command_export(
    command: list[str],
    output_paths: Sequence[Path],
    *,
    executable_name: str,
    timeout: Optional[float] = None,
    dry_run: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> CommandExportResult:
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return CommandExportResult(
            output_paths=list(output_paths),
            command=command,
            dry_run=True,
            metadata=dict(metadata or {}),
        )

    executable = shutil.which(command[0])
    if executable is None:
        raise XQTBackendError(f"{executable_name} executable not found: {command[0]}")
    command = [executable, *command[1:]]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise XQTBackendError(
            f"{executable_name} failed with return code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    missing = [str(path) for path in output_paths if not path.is_file()]
    if missing:
        raise XQTBackendError(f"{executable_name} did not create outputs: {missing}")
    checksums = {str(path): file_sha256(path) for path in output_paths}
    return CommandExportResult(
        output_paths=list(output_paths),
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        checksums=checksums,
        dry_run=False,
        metadata=dict(metadata or {}),
    )


def build_onnx2ncnn_command(
    onnx_path: str | Path,
    param_path: str | Path,
    bin_path: str | Path,
    *,
    onnx2ncnn_path: str = "onnx2ncnn",
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build an `onnx2ncnn` conversion command."""

    return [
        onnx2ncnn_path,
        str(Path(onnx_path)),
        str(Path(param_path)),
        str(Path(bin_path)),
        *(str(arg) for arg in (extra_args or ())),
    ]


def build_pnnx_command(
    model_path: str | Path,
    *,
    pnnx_path: str = "pnnx",
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build a `pnnx` conversion command for TorchScript or ONNX inputs."""

    return [
        pnnx_path,
        str(Path(model_path)),
        *(str(arg) for arg in (extra_args or ())),
    ]


def export_ncnn_with_pnnx(
    model_path: str | Path,
    *,
    pnnx_path: str = "pnnx",
    param_path: Optional[str | Path] = None,
    bin_path: Optional[str | Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
) -> CommandExportResult:
    """Convert TorchScript or ONNX to ncnn using the current pnnx path."""

    source = _ensure_input_file(model_path, description="Model")
    param = Path(param_path) if param_path is not None else source.with_suffix(".ncnn.param")
    binary = Path(bin_path) if bin_path is not None else source.with_suffix(".ncnn.bin")
    command = build_pnnx_command(
        source,
        pnnx_path=pnnx_path,
        extra_args=extra_args,
    )
    return _run_command_export(
        command,
        [param, binary],
        executable_name="pnnx",
        timeout=timeout,
        dry_run=dry_run,
        metadata={"source": str(source), "converter": "pnnx"},
    )


def export_ncnn_from_onnx(
    onnx_path: str | Path,
    param_path: str | Path,
    bin_path: str | Path,
    *,
    onnx2ncnn_path: str = "onnx2ncnn",
    extra_args: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
) -> CommandExportResult:
    """Convert ONNX to ncnn param/bin files."""

    onnx = _ensure_input_file(onnx_path, description="ONNX")
    param = Path(param_path)
    binary = Path(bin_path)
    command = build_onnx2ncnn_command(
        onnx,
        param,
        binary,
        onnx2ncnn_path=onnx2ncnn_path,
        extra_args=extra_args,
    )
    return _run_command_export(
        command,
        [param, binary],
        executable_name="onnx2ncnn",
        timeout=timeout,
        dry_run=dry_run,
        metadata={"source": str(onnx)},
    )


def build_mnnconvert_command(
    onnx_path: str | Path,
    mnn_path: str | Path,
    *,
    converter_path: str = "MNNConvert",
    framework: str = "ONNX",
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build an `MNNConvert` ONNX conversion command."""

    return [
        converter_path,
        "-f",
        framework,
        "--modelFile",
        str(Path(onnx_path)),
        "--MNNModel",
        str(Path(mnn_path)),
        *(str(arg) for arg in (extra_args or ())),
    ]


def export_mnn_from_onnx(
    onnx_path: str | Path,
    mnn_path: str | Path,
    *,
    converter_path: str = "MNNConvert",
    framework: str = "ONNX",
    extra_args: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
) -> CommandExportResult:
    """Convert ONNX to an MNN model file."""

    onnx = _ensure_input_file(onnx_path, description="ONNX")
    mnn = Path(mnn_path)
    command = build_mnnconvert_command(
        onnx,
        mnn,
        converter_path=converter_path,
        framework=framework,
        extra_args=extra_args,
    )
    return _run_command_export(
        command,
        [mnn],
        executable_name="MNNConvert",
        timeout=timeout,
        dry_run=dry_run,
        metadata={"source": str(onnx), "framework": framework},
    )


def export_executorch_program(
    model: nn.Module,
    example_input: Any,
    pte_path: str | Path,
    *,
    dry_run: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> ExecuTorchExportResult:
    """Export a PyTorch module to an ExecuTorch `.pte` program."""

    output = Path(pte_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return ExecuTorchExportResult(
            pte_path=output,
            dry_run=True,
            metadata=dict(metadata or {}),
        )

    try:
        from executorch.exir import to_edge  # type: ignore[import-untyped]
    except ImportError as exc:
        raise XQTBackendError("executorch is required for ExecuTorch export") from exc

    model.eval()
    args = example_input if isinstance(example_input, tuple) else (example_input,)
    exported = torch.export.export(model, args)
    edge_program = to_edge(exported)
    executorch_program = edge_program.to_executorch()
    with output.open("wb") as handle:
        executorch_program.write_to_file(handle)
    return ExecuTorchExportResult(
        pte_path=output,
        checksum=file_sha256(output),
        dry_run=False,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "CommandExportResult",
    "ExecuTorchExportResult",
    "build_mnnconvert_command",
    "build_onnx2ncnn_command",
    "build_pnnx_command",
    "export_executorch_program",
    "export_mnn_from_onnx",
    "export_ncnn_from_onnx",
    "export_ncnn_with_pnnx",
]
