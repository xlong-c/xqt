"""Runtime adapter for the non-production SM89 CUTLASS mixed-input probe."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from xqt.core.errors import XQTBackendError


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 mixed-input probe artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 mixed-input probe: {path}") from exc
    decode = getattr(library, "xqt_sm89_mixed_input_probe", None)
    if decode is not None:
        decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        decode.restype = ctypes.c_int
    size = getattr(library, "xqt_sm89_mixed_input_probe_type_size", None)
    if size is not None:
        size.argtypes = []
        size.restype = ctypes.c_int
    return library


def sm89_mixed_input_probe_artifact_available(artifact: str | Path) -> bool:
    """Return whether the probe exports both its decoder and type evidence."""

    try:
        library = _load_library(artifact)
    except XQTBackendError:
        return False
    return hasattr(library, "xqt_sm89_mixed_input_probe") and hasattr(
        library, "xqt_sm89_mixed_input_probe_type_size"
    )


def run_sm89_mixed_input_probe(
    packed: torch.Tensor,
    *,
    elements: int | None = None,
    artifact: str | Path,
) -> torch.Tensor:
    """Decode canonical low/high signed nibbles through the probe artifact."""

    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise XQTBackendError("mixed-input probe expects a CUDA uint8 packed tensor")
    if not packed.is_cuda or packed.ndim != 1:
        raise XQTBackendError("mixed-input probe expects a CUDA rank-1 packed tensor")
    count = int(elements if elements is not None else packed.numel() * 2)
    if count <= 0 or (count + 1) // 2 > packed.numel():
        raise XQTBackendError("mixed-input probe element count exceeds packed storage")
    major, minor = torch.cuda.get_device_capability(packed.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"mixed-input probe received sm_{major}{minor}")
    library = _load_library(artifact)
    decode = getattr(library, "xqt_sm89_mixed_input_probe", None)
    size = getattr(library, "xqt_sm89_mixed_input_probe_type_size", None)
    if decode is None or size is None or int(size()) <= 0:
        raise XQTBackendError("SM89 mixed-input probe artifact is incomplete")
    output = torch.empty((count,), device=packed.device, dtype=torch.int8)
    stream = torch.cuda.current_stream(packed.device).cuda_stream
    error = decode(packed.contiguous().data_ptr(), output.data_ptr(), count, stream)
    if error != 0:
        raise XQTBackendError(f"SM89 mixed-input probe failed with CUDA error {error}")
    return output


__all__ = [
    "run_sm89_mixed_input_probe",
    "sm89_mixed_input_probe_artifact_available",
]
