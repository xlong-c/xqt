"""Runtime gate for the non-production SM89 CUTLASS FP8 capability probe."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from xqt.core.errors import XQTBackendError

from ..fp8 import fp8_format_spec


_SYMBOL = "xqt_fp8_cutlass_probe_sm89_run"


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 FP8 probe artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 FP8 probe artifact: {path}") from exc
    if not hasattr(library, _SYMBOL):
        raise XQTBackendError(f"SM89 FP8 probe artifact lacks {_SYMBOL}")
    function = getattr(library, _SYMBOL)
    function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return library


def sm89_fp8_probe_artifact_available(artifact: str | Path) -> bool:
    """Return whether the probe shared object and symbol are loadable."""

    try:
        _load_library(artifact)
    except XQTBackendError:
        return False
    return True


def run_sm89_fp8_probe(
    artifact: str | Path,
    *,
    format_name: str,
    device: torch.device | str | None = None,
) -> float:
    """Launch one zero-input FP8 MMA probe and return its scalar output."""

    format_spec = fp8_format_spec(format_name)
    if format_name not in {"fp8_e4m3", "fp8_e5m2"}:
        raise ValueError(f"unsupported FP8 probe format: {format_name!r}")
    if not torch.cuda.is_available():
        raise XQTBackendError("SM89 FP8 probe requires CUDA")
    cuda_device = torch.device(device or torch.device("cuda", torch.cuda.current_device()))
    if cuda_device.type != "cuda":
        raise ValueError("SM89 FP8 probe device must be CUDA")
    major, minor = torch.cuda.get_device_capability(cuda_device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 FP8 probe received sm_{major}{minor}")
    library = _load_library(artifact)
    output = torch.empty(1, dtype=torch.float32, device=cuda_device)
    stream = torch.cuda.current_stream(cuda_device).cuda_stream
    error = getattr(library, _SYMBOL)(
        output.data_ptr(),
        0 if format_name == "fp8_e4m3" else 1,
        stream,
    )
    if error != 0:
        raise XQTBackendError(f"SM89 FP8 probe failed with CUDA error {error}")
    output = output + 0.0
    torch.cuda.current_stream(cuda_device).synchronize()
    del format_spec
    return float(output.item())


__all__ = ["run_sm89_fp8_probe", "sm89_fp8_probe_artifact_available"]
