"""Shared helpers for grouped FP4-style activation quantization kernels."""

from __future__ import annotations

from typing import Final

import torch

from xqt.core.errors import XQTBackendError


NVFP4_E2M1_CODEBOOK: Final[tuple[float, ...]] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# These are the CTA B-tile dimensions of the TileLang NVFP4 fast path.  The
# prepack changes only the persistent memory order; each uint8 still contains
# the same adjacent pair of FP4 nibbles as the generic row-major storage.
TILELANG_NVFP4_WEIGHT_BLOCK_N: Final[int] = 16
TILELANG_NVFP4_WEIGHT_BLOCK_K: Final[int] = 128

_NVFP4_POSITIVE_THRESHOLDS: Final[tuple[float, ...]] = (
    0.25,
    0.75,
    1.25,
    1.75,
    2.5,
    3.5,
    5.0,
)


def _require_2d_inputs(inputs: torch.Tensor) -> tuple[int, int]:
    if inputs.ndim != 2:
        raise XQTBackendError("FP4 quantization kernels currently expect a 2D tensor")
    rows, cols = int(inputs.shape[0]), int(inputs.shape[1])
    if rows <= 0 or cols <= 0:
        raise XQTBackendError("FP4 quantization kernels require positive tensor extents")
    return rows, cols


def _canonical_group_size(group_size: int, *, allowed: tuple[int, ...]) -> int:
    normalized = int(group_size)
    if normalized not in allowed:
        allowed_text = ", ".join(str(item) for item in allowed)
        raise XQTBackendError(f"group_size must be one of {allowed_text}")
    return normalized


def pad_last_dim_for_group(
    inputs: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, int]:
    rows, cols = _require_2d_inputs(inputs)
    normalized_group_size = _canonical_group_size(group_size, allowed=(16, 32))
    padded_cols = ((cols + normalized_group_size - 1) // normalized_group_size) * normalized_group_size
    if padded_cols == cols:
        return inputs, cols
    padded = torch.zeros((rows, padded_cols), device=inputs.device, dtype=inputs.dtype)
    padded[:, :cols] = inputs
    return padded, cols


def grouped_absmax(inputs: torch.Tensor, *, group_size: int) -> torch.Tensor:
    normalized_group_size = _canonical_group_size(group_size, allowed=(16, 32))
    padded, _ = pad_last_dim_for_group(inputs, group_size=normalized_group_size)
    rows = int(padded.shape[0])
    groups = int(padded.shape[1]) // normalized_group_size
    grouped = padded.to(torch.float32).reshape(rows, groups, normalized_group_size)
    return grouped.abs().amax(dim=-1)


def quantize_nvfp4_scale(scale_raw: torch.Tensor) -> torch.Tensor:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        return scale_raw.to(torch.float32)
    try:
        return scale_raw.to(dtype)
    except Exception:
        return scale_raw.to(torch.float32)


def quantize_mxfp_scale(scale_raw: torch.Tensor) -> torch.Tensor:
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is not None:
        try:
            return scale_raw.to(dtype)
        except Exception:
            pass
    scale_fp32 = scale_raw.to(torch.float32)
    positive = scale_fp32 > 0
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=scale_fp32.device,
        dtype=scale_fp32.dtype,
    )
    safe = torch.where(positive, torch.maximum(scale_fp32, tiny), torch.ones_like(scale_fp32))
    quantized = torch.exp2(torch.ceil(torch.log2(safe)))
    return torch.where(positive, quantized, torch.zeros_like(scale_fp32))


def nvfp4_codes_from_scaled(values: torch.Tensor) -> torch.Tensor:
    scaled = values.to(torch.float32)
    abs_values = scaled.abs()
    codes = torch.zeros_like(abs_values, dtype=torch.uint8)
    for threshold in _NVFP4_POSITIVE_THRESHOLDS:
        codes = codes + (abs_values > threshold).to(torch.uint8)
    negative = torch.signbit(scaled)
    return codes + negative.to(torch.uint8) * 8


def pack_nibble_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2:
        raise XQTBackendError("packed FP4 helpers expect a 2D nibble code tensor")
    if codes.shape[1] % 2 != 0:
        raise XQTBackendError("packed FP4 helpers require an even trailing dimension")
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    return (low | (high << 4)).contiguous()


def _normalize_prepack_scale(
    weight_scale: torch.Tensor,
    *,
    rows: int,
    groups: int,
) -> torch.Tensor:
    if weight_scale.ndim == 3 and weight_scale.shape[2] == 1:
        weight_scale = weight_scale.squeeze(-1)
    if weight_scale.ndim != 2 or tuple(weight_scale.shape) != (rows, groups):
        raise XQTBackendError(
            "TileLang NVFP4 weight_scale must be shaped [out_features, groups] "
            "or [out_features, groups, 1]"
        )
    return weight_scale


def prepack_nvfp4_weight_for_tilelang(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    group_size: int = 16,
    block_n: int = TILELANG_NVFP4_WEIGHT_BLOCK_N,
    block_k: int = TILELANG_NVFP4_WEIGHT_BLOCK_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out NVFP4 weight code and scale tiles for TileLang's CTA B loads.

    The generic representation is row-major ``[N, K / 2]``.  TileLang's
    fused NVFP4 GEMM consumes one B tile at a time, so this returns contiguous
    ``[N / block_n, K / block_k, block_n, block_k / 2]`` code tiles and the
    matching ``[N / block_n, K / block_k, block_n, block_k / group_size]``
    scale tiles.  Call this during quantization or model construction, never
    in ``forward``.
    """

    if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
        raise XQTBackendError(
            "TileLang NVFP4 prepack expects a 2D uint8 packed_weight tensor"
        )
    normalized_group_size = _canonical_group_size(group_size, allowed=(16,))
    normalized_block_n = int(block_n)
    normalized_block_k = int(block_k)
    if normalized_block_n <= 0 or normalized_block_k <= 0:
        raise XQTBackendError("TileLang NVFP4 prepack block sizes must be positive")
    if normalized_block_k % 2 != 0:
        raise XQTBackendError("TileLang NVFP4 prepack block_k must be even")
    if normalized_block_k % normalized_group_size != 0:
        raise XQTBackendError(
            "TileLang NVFP4 prepack block_k must be divisible by group_size"
        )

    rows, packed_k = (int(packed_weight.shape[0]), int(packed_weight.shape[1]))
    input_features = packed_k * 2
    if rows % normalized_block_n != 0:
        raise XQTBackendError(
            "TileLang NVFP4 prepack requires out_features divisible by block_n"
        )
    if input_features % normalized_block_k != 0:
        raise XQTBackendError(
            "TileLang NVFP4 prepack requires padded input_features divisible by block_k"
        )
    groups = input_features // normalized_group_size
    scale_2d = _normalize_prepack_scale(weight_scale, rows=rows, groups=groups)

    n_tiles = rows // normalized_block_n
    k_tiles = input_features // normalized_block_k
    packed_block_k = normalized_block_k // 2
    groups_per_tile = normalized_block_k // normalized_group_size
    tiled_weight = (
        packed_weight.reshape(n_tiles, normalized_block_n, k_tiles, packed_block_k)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    tiled_scale = (
        scale_2d.reshape(n_tiles, normalized_block_n, k_tiles, groups_per_tile)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return tiled_weight, tiled_scale


def unpack_nvfp4_weight_from_tilelang(
    tiled_weight: torch.Tensor,
    tiled_scale: torch.Tensor,
    *,
    group_size: int = 16,
    block_n: int = TILELANG_NVFP4_WEIGHT_BLOCK_N,
    block_k: int = TILELANG_NVFP4_WEIGHT_BLOCK_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore TileLang-tiled NVFP4 storage to the generic row-major form."""

    if tiled_weight.ndim != 4 or tiled_weight.dtype != torch.uint8:
        raise XQTBackendError(
            "TileLang NVFP4 tiled_weight must be a 4D uint8 tensor"
        )
    if tiled_scale.ndim != 4:
        raise XQTBackendError("TileLang NVFP4 tiled_scale must be a 4D tensor")
    normalized_group_size = _canonical_group_size(group_size, allowed=(16,))
    normalized_block_n = int(block_n)
    normalized_block_k = int(block_k)
    expected_weight_tail = (normalized_block_n, normalized_block_k // 2)
    expected_scale_tail = (
        normalized_block_n,
        normalized_block_k // normalized_group_size,
    )
    if tuple(tiled_weight.shape[2:]) != expected_weight_tail:
        raise XQTBackendError(
            "TileLang NVFP4 tiled_weight does not match the requested block shape"
        )
    if tuple(tiled_scale.shape[2:]) != expected_scale_tail:
        raise XQTBackendError(
            "TileLang NVFP4 tiled_scale does not match the requested block shape"
        )
    if tuple(tiled_weight.shape[:2]) != tuple(tiled_scale.shape[:2]):
        raise XQTBackendError("TileLang NVFP4 tiled code and scale tile counts must match")

    n_tiles, k_tiles = (int(tiled_weight.shape[0]), int(tiled_weight.shape[1]))
    packed_weight = (
        tiled_weight.permute(0, 2, 1, 3)
        .reshape(n_tiles * normalized_block_n, k_tiles * (normalized_block_k // 2))
        .contiguous()
    )
    weight_scale = (
        tiled_scale.permute(0, 2, 1, 3)
        .reshape(
            n_tiles * normalized_block_n,
            k_tiles * (normalized_block_k // normalized_group_size),
        )
        .contiguous()
    )
    return packed_weight, weight_scale


def unpack_nibble_codes(
    packed: torch.Tensor,
    *,
    input_features: int,
) -> torch.Tensor:
    if packed.ndim != 2:
        raise XQTBackendError("packed FP4 tensors must be 2D")
    if packed.dtype != torch.uint8:
        raise XQTBackendError("packed FP4 tensors must use uint8 storage")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    return unpacked[:, : int(input_features)]


def dequantize_nvfp4_codes(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    input_features: int,
    group_size: int,
    global_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    normalized_group_size = _canonical_group_size(group_size, allowed=(16, 32))
    unpacked = unpack_nibble_codes(packed, input_features=input_features)
    codebook = torch.tensor(
        NVFP4_E2M1_CODEBOOK,
        device=packed.device,
        dtype=torch.float32,
    )
    values = codebook[unpacked.long()]
    scale_fp32 = scale.to(device=packed.device, dtype=torch.float32)
    if scale_fp32.ndim == 3 and scale_fp32.shape[2] == 1:
        scale_fp32 = scale_fp32.squeeze(-1)
    if scale_fp32.ndim != 2:
        raise XQTBackendError("scale must be shaped [rows, groups] or [rows, groups, 1]")
    if global_scale is not None:
        scale_fp32 = scale_fp32 / global_scale.to(device=packed.device, dtype=torch.float32).reshape(())
    expanded = scale_fp32.unsqueeze(-1).expand(-1, -1, normalized_group_size).reshape(
        values.shape[0],
        -1,
    )
    return (values * expanded[:, : int(input_features)]).to(output_dtype)


def scaled_nvfp4_quant_reference(
    inputs: torch.Tensor,
    input_global_scale: torch.Tensor,
    *,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_group_size = _canonical_group_size(group_size, allowed=(16,))
    padded, _ = pad_last_dim_for_group(inputs, group_size=normalized_group_size)
    if input_global_scale.numel() != 1:
        raise XQTBackendError("input_global_scale must be a scalar tensor")
    global_scale = input_global_scale.to(device=inputs.device, dtype=torch.float32).reshape(())
    absmax = grouped_absmax(padded, group_size=normalized_group_size)
    scale_raw = absmax * (global_scale / 6.0)
    scale = quantize_nvfp4_scale(scale_raw)
    scale_fp32 = scale.to(torch.float32)
    grouped = padded.to(torch.float32).reshape(
        int(padded.shape[0]),
        int(padded.shape[1]) // normalized_group_size,
        normalized_group_size,
    )
    denom = scale_fp32.unsqueeze(-1)
    scaled = torch.where(
        denom > 0,
        grouped * (global_scale / denom),
        torch.zeros_like(grouped),
    )
    codes = nvfp4_codes_from_scaled(scaled.reshape(int(padded.shape[0]), -1))
    packed = pack_nibble_codes(codes)
    return packed, scale.contiguous()


def scaled_mxfp4_quant_reference(
    inputs: torch.Tensor,
    *,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_group_size = _canonical_group_size(group_size, allowed=(32,))
    padded, _ = pad_last_dim_for_group(inputs, group_size=normalized_group_size)
    absmax = grouped_absmax(padded, group_size=normalized_group_size)
    scale_raw = absmax / 6.0
    scale = quantize_mxfp_scale(scale_raw)
    scale_fp32 = scale.to(torch.float32)
    grouped = padded.to(torch.float32).reshape(
        int(padded.shape[0]),
        int(padded.shape[1]) // normalized_group_size,
        normalized_group_size,
    )
    denom = scale_fp32.unsqueeze(-1)
    scaled = torch.where(
        denom > 0,
        grouped / denom,
        torch.zeros_like(grouped),
    )
    codes = nvfp4_codes_from_scaled(scaled.reshape(int(padded.shape[0]), -1))
    packed = pack_nibble_codes(codes)
    return packed, scale.contiguous()


__all__ = [
    "NVFP4_E2M1_CODEBOOK",
    "TILELANG_NVFP4_WEIGHT_BLOCK_K",
    "TILELANG_NVFP4_WEIGHT_BLOCK_N",
    "dequantize_nvfp4_codes",
    "grouped_absmax",
    "nvfp4_codes_from_scaled",
    "pack_nibble_codes",
    "pad_last_dim_for_group",
    "prepack_nvfp4_weight_for_tilelang",
    "quantize_mxfp_scale",
    "quantize_nvfp4_scale",
    "scaled_mxfp4_quant_reference",
    "scaled_nvfp4_quant_reference",
    "unpack_nibble_codes",
    "unpack_nvfp4_weight_from_tilelang",
]
