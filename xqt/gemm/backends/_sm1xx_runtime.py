"""Shared validation helpers for explicit SM90/SM120 CUTLASS probes."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Callable

import torch

from xqt.core.errors import XQTBackendError


_CUTLASS_SCALE_BLOCK_M = 128
_CUTLASS_SCALE_BLOCK_N = 128
_CUTLASS_SCALE_BLOCK_K = 128
_CUTLASS_NVFP4_SCALE_GROUP = 16
_CUTLASS_NVFP4_SCALE_TILE_BYTES = 1024


def cutlass_blockscale_shape(
    rows: int,
    cols: int,
    *,
    row_block: int = _CUTLASS_SCALE_BLOCK_M,
    block_k: int = _CUTLASS_SCALE_BLOCK_K,
) -> tuple[int, int]:
    """Return the explicit CUTLASS blockscale grid for one operand.

    The trivial SM90/SM120 probe uses ``row_block=128``.  The SM120
    groupwise probe uses ``row_block=1`` for SFA and ``row_block=128`` for
    SFB.  This is deliberately different from assuming one universal XQT
    canonical scale layout.
    """

    if int(rows) <= 0 or int(cols) <= 0:
        raise ValueError("CUTLASS blockscale rows and cols must be positive")
    if int(row_block) <= 0 or int(block_k) <= 0:
        raise ValueError("CUTLASS blockscale row_block and block_k must be positive")
    return (
        (int(rows) + int(row_block) - 1) // int(row_block),
        (int(cols) + int(block_k) - 1) // int(block_k),
    )


def prepare_cutlass_blockscales(
    value: torch.Tensor,
    *,
    rows: int,
    cols: int,
    padded_rows: int | None = None,
    padded_cols: int | None = None,
    row_block: int = _CUTLASS_SCALE_BLOCK_M,
    block_k: int = _CUTLASS_SCALE_BLOCK_K,
    major: str = "mn",
    name: str,
) -> torch.Tensor:
    """Validate and flatten an explicit SM1xx blockscale tensor.

    ``value`` must already use the CUTLASS blockscale semantics.  No
    per-row-to-per-block reduction is performed here.  Extra padded blocks
    are initialized to one because their corresponding input elements are
    zero-padded and therefore do not affect the logical result.  The returned
    flat storage is column-major over the non-zero block grid, matching the
    CUTLASS layout stride ``block_row + row_block_count * block_k`` for
    ``major="mn"``.  ``major="k"`` uses the row-major block grid required by
    the SM90 SFB groupwise layout.
    """

    if not isinstance(value, torch.Tensor):
        raise XQTBackendError(f"{name} must be a CUDA tensor")
    if major not in {"mn", "k"}:
        raise XQTBackendError(f"{name} CUTLASS scale major must be 'mn' or 'k'")
    if value.ndim != 2:
        raise XQTBackendError(
            f"{name} must use CUTLASS blockscale shape "
            f"{cutlass_blockscale_shape(rows, cols)}, got {tuple(value.shape)}"
        )
    expected = cutlass_blockscale_shape(
        rows,
        cols,
        row_block=row_block,
        block_k=block_k,
    )
    if tuple(int(item) for item in value.shape) != expected:
        raise XQTBackendError(
            f"{name} must use explicit CUTLASS blockscale shape {expected}; "
            f"canonical XQT per-row scales are not accepted, got {tuple(value.shape)}"
        )
    if not value.is_cuda or value.dtype != torch.float32:
        raise XQTBackendError(f"{name} must be contiguous CUDA float32")
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise XQTBackendError(f"{name} must contain finite positive scales")
    target_rows = int(padded_rows if padded_rows is not None else rows)
    target_cols = int(padded_cols if padded_cols is not None else cols)
    target_shape = cutlass_blockscale_shape(
        target_rows,
        target_cols,
        row_block=row_block,
        block_k=block_k,
    )
    if target_rows < int(rows) or target_cols < int(cols):
        raise XQTBackendError(f"{name} padded dimensions cannot be smaller than logical dimensions")
    padded = torch.ones(target_shape, device=value.device, dtype=torch.float32)
    padded[: expected[0], : expected[1]].copy_(value)
    return flatten_cutlass_blockscale_grid(padded)


def flatten_cutlass_blockscale_grid(
    value: torch.Tensor,
    *,
    major: str = "mn",
) -> torch.Tensor:
    """Flatten a conceptual ``[row_block, k_block]`` grid for CUTLASS."""

    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("CUTLASS blockscale grid must be a rank-2 tensor")
    if major == "mn":
        return value.transpose(0, 1).contiguous().view(-1)
    if major == "k":
        return value.contiguous().view(-1)
    raise ValueError("CUTLASS blockscale major must be 'mn' or 'k'")


def cutlass_nvfp4_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """Return the conceptual NVFP4 scale grid for an operand."""

    if int(rows) <= 0 or int(cols) <= 0:
        raise ValueError("NVFP4 scale rows and cols must be positive")
    return int(rows), (int(cols) + _CUTLASS_NVFP4_SCALE_GROUP - 1) // _CUTLASS_NVFP4_SCALE_GROUP


def cutlass_nvfp4_scale_storage_size(padded_rows: int, padded_cols: int) -> int:
    """Return the compact SM120 NVFP4 scale storage size in bytes."""

    if int(padded_rows) <= 0 or int(padded_cols) <= 0:
        raise ValueError("NVFP4 padded scale dimensions must be positive")
    if int(padded_rows) % 128 != 0 or int(padded_cols) % 128 != 0:
        raise ValueError("NVFP4 padded scale dimensions must be multiples of 128")
    return (
        (int(padded_rows) // 128)
        * (int(padded_cols) // 128)
        * _CUTLASS_NVFP4_SCALE_TILE_BYTES
    )


def cutlass_nvfp4_scale_storage_offset(
    row: int,
    group: int,
    *,
    padded_rows: int,
    padded_cols: int,
) -> int:
    """Return one compact SFVec16 scale offset for the SM120 layout."""

    if not 0 <= int(row) < int(padded_rows):
        raise ValueError("NVFP4 scale row is outside the padded extent")
    groups = int(padded_cols) // _CUTLASS_NVFP4_SCALE_GROUP
    if not 0 <= int(group) < groups:
        raise ValueError("NVFP4 scale group is outside the padded extent")
    k_tiles = int(padded_cols) // 128
    row_local = int(row) % 128
    return (
        (int(row) // 128) * k_tiles * _CUTLASS_NVFP4_SCALE_TILE_BYTES
        + (int(group) // 4) * 512
        + (row_local % 32) * 16
        + ((row_local // 32) % 4) * 4
        + (int(group) % 4)
    )


def _nvfp4_scale_bytes(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise XQTBackendError(f"{name} must be a rank-2 NVFP4 scale tensor")
    if value.dtype == torch.uint8:
        encoded = value.contiguous()
        decoded = encoded.view(torch.float8_e4m3fn).to(torch.float32)
    elif value.dtype == torch.float8_e4m3fn:
        encoded = value.contiguous().view(torch.uint8)
        decoded = value.to(torch.float32)
    else:
        decoded = value.to(torch.float32)
        encoded = None
    if not bool(torch.isfinite(decoded).all()) or bool((decoded <= 0).any()):
        raise XQTBackendError(f"{name} must contain finite positive scales")
    if encoded is None:
        if bool((decoded > 448.0).any()):
            raise XQTBackendError(f"{name} exceeds the float_ue4m3 range")
        encoded = decoded.to(torch.float8_e4m3fn).view(torch.uint8)
    return encoded


def prepare_cutlass_nvfp4_scales(
    value: torch.Tensor,
    *,
    rows: int,
    cols: int,
    padded_rows: int | None = None,
    padded_cols: int | None = None,
    global_scale: torch.Tensor | None = None,
    name: str,
) -> torch.Tensor:
    """Pack conceptual NVFP4 scales into CUTLASS SM120 interleaved storage.

    CUTLASS uses ``float_ue4m3_t`` scales and a SFVecSize of 16 for the
    cooperative SM120 NVFP4 builder.  The physical layout is one 1024-byte
    tile per 128 rows and 128 K elements.  The input remains an explicit
    ``[rows, ceil(cols / 16)]`` scale grid.
    """

    expected = cutlass_nvfp4_scale_shape(rows, cols)
    if tuple(int(item) for item in value.shape) != expected:
        raise XQTBackendError(
            f"{name} must use NVFP4 scale shape {expected}, got {tuple(value.shape)}"
        )
    target_rows = int(padded_rows if padded_rows is not None else rows)
    target_cols = int(padded_cols if padded_cols is not None else cols)
    if target_rows < int(rows) or target_cols < int(cols):
        raise XQTBackendError(f"{name} padded dimensions cannot be smaller than logical dimensions")
    if target_rows % 128 != 0 or target_cols % 128 != 0:
        raise XQTBackendError(f"{name} padded dimensions must be multiples of 128")
    if not value.is_cuda:
        raise XQTBackendError(f"{name} must be a CUDA tensor")
    encoded = _nvfp4_scale_bytes(value, name=name)
    if global_scale is not None:
        if not isinstance(global_scale, torch.Tensor) or global_scale.numel() != 1:
            raise XQTBackendError(f"{name} global_scale must be a scalar tensor")
        scalar = global_scale.to(device=value.device, dtype=torch.float32).reshape(())
        if not bool(torch.isfinite(scalar)) or bool(scalar <= 0):
            raise XQTBackendError(f"{name} global_scale must be finite and positive")
        decoded = encoded.view(torch.float8_e4m3fn).to(torch.float32) / scalar
        if bool((decoded > 448.0).any()) or bool((decoded <= 0).any()):
            raise XQTBackendError(f"{name} divided by global_scale is outside float_ue4m3")
        encoded = decoded.to(torch.float8_e4m3fn).view(torch.uint8)

    groups = target_cols // _CUTLASS_NVFP4_SCALE_GROUP
    padded = torch.ones(
        (target_rows, groups),
        device=value.device,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn).view(torch.uint8)
    padded[:rows, : expected[1]].copy_(encoded)

    row = torch.arange(target_rows, device=value.device, dtype=torch.int64).view(-1, 1)
    group = torch.arange(groups, device=value.device, dtype=torch.int64).view(1, -1)
    row_tile = row // 128
    row_local = row % 128
    k_tiles = target_cols // 128
    offsets = (
        row_tile * k_tiles * _CUTLASS_NVFP4_SCALE_TILE_BYTES
        + (group // 4) * 512
        + (row_local % 32) * 16
        + ((row_local // 32) % 4) * 4
        + (group % 4)
    )
    result = torch.empty(
        cutlass_nvfp4_scale_storage_size(target_rows, target_cols),
        device=value.device,
        dtype=torch.uint8,
    )
    result[offsets.reshape(-1)] = padded.reshape(-1)
    return result


def pad_matrix(
    value: torch.Tensor,
    *,
    rows: int,
    cols: int,
    row_multiple: int = 128,
    col_multiple: int = 128,
) -> torch.Tensor:
    """Zero-pad a rank-2 CUDA matrix without relying on FP8 ``F.pad`` support."""

    if value.ndim != 2 or tuple(int(item) for item in value.shape) != (rows, cols):
        raise XQTBackendError(
            f"matrix shape must be [{rows},{cols}], got {tuple(value.shape)}"
        )
    if int(row_multiple) <= 0 or int(col_multiple) <= 0:
        raise ValueError("matrix padding multiples must be positive")
    padded_rows = ((int(rows) + row_multiple - 1) // row_multiple) * row_multiple
    padded_cols = ((int(cols) + col_multiple - 1) // col_multiple) * col_multiple
    if (padded_rows, padded_cols) == (int(rows), int(cols)):
        return value.contiguous()
    result = torch.zeros(
        (padded_rows, padded_cols),
        device=value.device,
        dtype=value.dtype,
    )
    result[:rows, :cols].copy_(value)
    return result


def require_target_cuda(
    value: torch.Tensor,
    *,
    capability: tuple[int, int],
    name: str,
) -> torch.device:
    """Validate CUDA placement and exact target SM for an opt-in probe."""

    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise XQTBackendError(f"{name} requires a CUDA tensor")
    actual = torch.cuda.get_device_capability(value.device)
    if actual != capability:
        raise XQTBackendError(
            f"{name} received sm_{actual[0]}{actual[1]}, "
            f"but this artifact targets sm_{capability[0]}{capability[1]}"
        )
    return value.device


def require_same_cuda_device(
    reference: torch.Tensor,
    values: tuple[torch.Tensor, ...],
    *,
    names: tuple[str, ...],
) -> None:
    """Require all runtime buffers to share the reference tensor's device."""

    for value, name in zip(values, names):
        if not isinstance(value, torch.Tensor) or not value.is_cuda:
            raise XQTBackendError(f"{name} must be a CUDA tensor")
        if value.device != reference.device:
            raise XQTBackendError(f"{name} must use device {reference.device}, got {value.device}")


def load_runtime_function(
    artifact: str | Path,
    *,
    symbol: str,
    argtypes: list[object],
) -> Callable[..., int]:
    """Load one explicit runtime symbol and configure its C ABI."""

    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"runtime artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load runtime artifact: {path}") from exc
    if not hasattr(library, symbol):
        raise XQTBackendError(f"runtime artifact lacks {symbol}")
    function = getattr(library, symbol)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def current_cuda_stream(device: torch.device) -> ctypes.c_void_p:
    """Return the current PyTorch stream as a C ABI pointer."""

    return ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)


def raise_runtime_error(symbol: str, error: int) -> None:
    """Convert a non-zero CUTLASS status code into an XQT backend error."""

    if int(error) != 0:
        raise XQTBackendError(f"{symbol} failed with CUTLASS/CUDA status {int(error)}")


__all__ = [
    "cutlass_blockscale_shape",
    "cutlass_nvfp4_scale_shape",
    "cutlass_nvfp4_scale_storage_offset",
    "cutlass_nvfp4_scale_storage_size",
    "current_cuda_stream",
    "flatten_cutlass_blockscale_grid",
    "load_runtime_function",
    "pad_matrix",
    "prepare_cutlass_blockscales",
    "prepare_cutlass_nvfp4_scales",
    "raise_runtime_error",
    "require_same_cuda_device",
    "require_target_cuda",
]
