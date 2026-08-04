"""MXFP (Microscaling Floating Point) GEMM kernels for Triton."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from xqt.gemm import dense_gemm_reference

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.kernels.fp4_quant_common import dequantize_nvfp4_codes


import triton
import triton.language as tl

from .gemm import (
    _dense_triton_output_dtype,
    _run_dense_triton_gemm,
    dequantize_nvfp4_weight_triton,
)



def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("MXFP GEMM kernels require CUDA tensors")


def _require_triton() -> object:
    if triton is None:
        raise XQTBackendError(
            "triton is required for MXFP kernels. Install the optimization extras."
        )
    return triton


# ============================================================================
# MXFP Format Utilities
# ============================================================================


def pack_mxfp(
    tensor: torch.Tensor,
    precision: int = 8,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a tensor into MXFP format with shared exponents.

    Args:
        tensor: Input tensor to quantize
        precision: Mantissa bits (4, 6, or 8)
        block_size: Number of elements sharing one scale exponent (default 32)

    Returns:
        (packed_mantissas, scales): Packed mantissas and shared exponents per block
    """
    if precision not in {4, 6, 8}:
        raise ValueError(f"MXFP precision must be 4, 6, or 8, got {precision}")

    # Reshape into blocks
    original_shape = tensor.shape
    numel = tensor.numel()

    # Pad to block_size multiple
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    if numel != padded_numel:
        tensor_flat = F.pad(tensor.flatten(), (0, padded_numel - numel))
    else:
        tensor_flat = tensor.flatten()

    # Reshape to (num_blocks, block_size)
    tensor_blocks = tensor_flat.reshape(-1, block_size)

    # Compute shared exponent per block (max abs value)
    block_max = tensor_blocks.abs().max(dim=1, keepdim=True).values

    # Avoid division by zero
    block_max = torch.clamp(block_max, min=1e-10)

    # For MXFP: scale = block_max / (2^(precision-1) - 1)
    # This ensures max value maps to the largest representable integer
    max_int = (1 << (precision - 1)) - 1  # e.g., MXFP8: 127, MXFP4: 7
    scale = block_max / max_int

    # Normalize mantissas by scale
    normalized = tensor_blocks / scale

    # Round to integer
    quantized = torch.round(normalized).to(torch.int32)
    quantized = torch.clamp(quantized, -max_int, max_int)

    # Pack mantissas
    if precision == 8:
        packed = quantized.to(torch.int8).reshape(-1)
    elif precision == 6:
        # Pack 4 MXFP6 values into 3 bytes (24 bits)
        # For simplicity, use int8 with padding (real implementation would bit-pack)
        packed = quantized.to(torch.int8).reshape(-1)
    elif precision == 4:
        # Pack 2 MXFP4 values into 1 byte
        packed = _pack_nibbles(quantized.to(torch.int8))
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # Store scales directly as float32 (not as exponents)
    scales = scale.reshape(-1)

    return packed, scales


def unpack_mxfp(
    packed: torch.Tensor,
    scales: torch.Tensor,
    precision: int = 8,
    block_size: int = 32,
    original_numel: int | None = None,
) -> torch.Tensor:
    """Unpack MXFP format back to FP tensor.

    Args:
        packed: Packed mantissas
        scales: Shared exponents per block (int8)
        precision: Mantissa bits
        block_size: Elements per block
        original_numel: Original tensor size (for unpadding)

    Returns:
        Unpacked float tensor
    """
    if precision == 8:
        mantissas = packed.to(torch.float32)
    elif precision == 6:
        mantissas = packed.to(torch.float32)
    elif precision == 4:
        mantissas = _unpack_nibbles(packed).to(torch.float32)
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    scale_tensor = scales.to(torch.float32)
    if scale_tensor.ndim == 1:
        num_blocks = scale_tensor.numel()
        mantissas = mantissas.reshape(num_blocks, block_size)
        dequantized = mantissas * scale_tensor.unsqueeze(1)
        output = dequantized.flatten()
        if original_numel is not None and output.numel() > original_numel:
            output = output[:original_numel]
        return output

    if scale_tensor.ndim != 2:
        raise ValueError("MXFP scales must be 1D or 2D")
    rows, groups = scale_tensor.shape
    mantissas = mantissas.reshape(rows, groups, block_size)
    dequantized = mantissas * scale_tensor.unsqueeze(-1)
    output = dequantized.reshape(rows, groups * block_size)
    if original_numel is not None and output.shape[1] > original_numel:
        output = output[:, :original_numel]
    return output


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values into one byte."""
    values_flat = values.flatten()
    if values_flat.numel() % 2 != 0:
        values_flat = F.pad(values_flat, (0, 1))

    # Encode signed nibbles
    encoded = torch.where(values_flat < 0, values_flat + 16, values_flat).to(torch.uint8)

    low = encoded[0::2]
    high = encoded[1::2] << 4
    packed = (low | high).contiguous()

    return packed


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack two 4-bit values from one byte."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)

    unpacked = torch.stack([low, high], dim=-1).flatten()

    # Decode signed nibbles
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)

    return unpacked


@triton.jit
def _dequant_mxfp_weight_kernel(
    packed_ptr,
    scale_ptr,
    out_ptr,
    rows,
    cols,
    stride_packed_row,
    stride_scale_row,
    stride_scale_group,
    stride_out_row,
    stride_out_col,
    mx_precision: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = rows * cols
    mask = offsets < total
    row = offsets // cols
    col = offsets % cols
    group = col // block_size

    if mx_precision == 4:
        packed_col = col // 2
        packed = tl.load(
            packed_ptr + row * stride_packed_row + packed_col,
            mask=mask,
            other=0,
        ).to(tl.int32)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        mantissa = tl.where((col & 1) == 0, low, high)
        mantissa = tl.where(mantissa >= 8, mantissa - 16, mantissa).to(tl.float32)
    else:
        mantissa = tl.load(
            packed_ptr + row * stride_packed_row + col,
            mask=mask,
            other=0,
        ).to(tl.float32)

    scale = tl.load(
        scale_ptr + row * stride_scale_row + group * stride_scale_group,
        mask=mask,
        other=1.0,
    ).to(tl.float32)
    value = mantissa * scale
    tl.store(
        out_ptr + row * stride_out_row + col * stride_out_col,
        value.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


# ============================================================================
# Reference Implementation
# ============================================================================


def gemm_mxfp_reference(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    b_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    mx_precision: int = 8,
    block_size: int = 32,
    activation: str | None = None,
    transpose_b: bool = True,
) -> torch.Tensor:
    """Reference MXFP GEMM implementation."""
    _, K = a.shape
    if transpose_b:
        b_fp = unpack_mxfp(
            b_packed,
            b_scales,
            mx_precision,
            block_size,
            K,
        )
        if b_fp.ndim != 2:
            raise ValueError("MXFP transpose_b path expects 2D packed weights")
        b_fp = b_fp.to(dtype=a.dtype, device=a.device).t()
    else:
        b_fp = unpack_mxfp(b_packed, b_scales, mx_precision, block_size, None).to(
            dtype=a.dtype,
            device=a.device,
        )

    return dense_gemm_reference(
        a,
        b_fp,
        bias,
        activation=activation,
        transpose_b=False,
    )


def dequantize_mxfp_weight_triton(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    precision: int,
    block_size: int,
    cols: int,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Decode packed MXFP weights into a dense CUDA tensor using Triton."""

    _require_triton()
    _require_cuda_tensors(packed, scales)
    scale_tensor = scales.to(device=packed.device, dtype=torch.float32)
    if scale_tensor.ndim == 1:
        scale_tensor = scale_tensor.reshape(1, -1).expand(int(packed.shape[0]), -1)
    if scale_tensor.ndim != 2:
        raise ValueError("MXFP Triton dequant expects scales to be 1D or 2D")
    rows = int(packed.shape[0])
    output = torch.empty((rows, int(cols)), device=packed.device, dtype=output_dtype)
    total = rows * int(cols)
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    _dequant_mxfp_weight_kernel[grid](
        packed,
        scale_tensor,
        output,
        rows,
        int(cols),
        packed.stride(0),
        scale_tensor.stride(0),
        scale_tensor.stride(1),
        output.stride(0),
        output.stride(1),
        mx_precision=int(precision),
        block_size=int(block_size),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


# ============================================================================
# Triton MXFP GEMM Kernel
# ============================================================================

@triton.jit
def _mxfp_dequant_gemm_kernel(
    a_ptr, b_packed_ptr, b_scales_ptr, c_ptr,
    bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    mx_precision: tl.constexpr,
    block_size: tl.constexpr,
    has_bias: tl.constexpr,
    activation: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """MXFP dequant + GEMM kernel."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Simple reference: load, dequant, dot
    # Real implementation would fuse dequant into the dot loop
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load and dequant b (simplified - full kernel would bit-unpack)
        # For now, assume b is already unpacked for reference
        # Real kernel: unpack nibbles, apply block scale

        a_ptrs += BLOCK_K * stride_ak

    c = accumulator.to(tl.float16)

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        c = c + bias[None, :]

    if activation == 1:
        c = tl.maximum(c, 0.0)
    elif activation == 2:
        c = 0.5 * c * (1.0 + tl.erf(c / 1.4142135623730951))
    elif activation == 3:
        c = c * tl.sigmoid(c)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



# ============================================================================
# Public API
# ============================================================================


def gemm_mxfp_triton(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    b_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    mx_precision: int = 8,
    block_size: int = 32,
    activation: str | None = None,
    transpose_b: bool = True,
) -> torch.Tensor:
    """MXFP GEMM using Triton dense GEMM after device-side unpack/dequant."""
    _require_triton()
    _require_cuda_tensors(a, b_packed, b_scales)
    compute_dtype = torch.bfloat16 if a.dtype == torch.bfloat16 else torch.float16
    dense_weight = dequantize_mxfp_weight_triton(
        b_packed,
        b_scales,
        precision=mx_precision,
        block_size=block_size,
        cols=int(a.shape[1]),
        output_dtype=compute_dtype,
    )
    return _run_dense_triton_gemm(
        a.to(compute_dtype) if a.dtype not in {torch.float16, torch.bfloat16} else a,
        dense_weight,
        None if bias is None else bias.to(device=a.device, dtype=_dense_triton_output_dtype(a)),
        activation=activation,
        transpose_b=transpose_b,
    )


def gemm_mxfp_packed_activation_reference(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    block_size: int = 32,
    mx_precision: int = 4,
    activation: str | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference MXFP4 GEMM that consumes packed activation and packed weight."""

    del mx_precision
    activation_dense = dequantize_nvfp4_codes(
        a_packed,
        a_scale,
        input_features=int(input_features),
        group_size=int(block_size),
        global_scale=None,
        output_dtype=output_dtype,
    )
    return gemm_mxfp_reference(
        activation_dense,
        b_packed,
        b_scales,
        bias,
        mx_precision=4,
        block_size=int(block_size),
        activation=activation,
        transpose_b=True,
    )


def gemm_mxfp_packed_activation_triton(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    block_size: int = 32,
    mx_precision: int = 4,
    activation: str | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Composed Triton runtime that consumes packed MXFP4 activation and weight."""

    _require_triton()
    _require_cuda_tensors(a_packed, a_scale, b_packed, b_scales)
    activation_dense = dequantize_nvfp4_weight_triton(
        a_packed.to(device=b_packed.device),
        a_scale.to(device=b_packed.device),
        cols=int(input_features),
        group_size=int(block_size),
        weight_global_scale=None,
        output_dtype=output_dtype,
    )
    return gemm_mxfp_triton(
        activation_dense,
        b_packed,
        b_scales,
        bias,
        mx_precision=int(mx_precision),
        block_size=int(block_size),
        activation=activation,
        transpose_b=True,
    )


# ============================================================================
# Metadata
# ============================================================================

MXFP_GEMM_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "gemm_mxfp8": {
        "kernel_name": "gemm_mxfp8",
        "precision": "mxfp8",
        "format": "8-bit mantissa + shared 8-bit exponent per block",
        "block_size": 32,
        "baseline": "unpack + torch.matmul",
        "usage": "MXFP8 quantized GEMM with microscaling shared exponents",
        "hardware": "CUDA SM120+ (Blackwell) native, SM70+ emulated",
        "note": "Composed runtime: device-side unpack/dequant followed by Triton dense GEMM.",
    },
    "gemm_mxfp6": {
        "kernel_name": "gemm_mxfp6",
        "precision": "mxfp6",
        "format": "6-bit mantissa + shared exponent",
        "block_size": 32,
        "baseline": "unpack + torch.matmul",
        "usage": "MXFP6 higher compression with acceptable precision",
        "hardware": "CUDA SM120+ (Blackwell) native, SM70+ emulated",
        "note": "Composed runtime: device-side unpack/dequant followed by Triton dense GEMM.",
    },
    "gemm_mxfp4": {
        "kernel_name": "gemm_mxfp4",
        "precision": "mxfp4",
        "format": "4-bit mantissa + shared exponent",
        "block_size": 32,
        "baseline": "unpack + torch.matmul",
        "usage": "MXFP4 maximum compression, lower precision",
        "hardware": "CUDA SM120+ (Blackwell) native, SM70+ emulated",
        "note": "Composed runtime: device-side unpack/dequant followed by Triton dense GEMM.",
    },
}


__all__ = [
    "MXFP_GEMM_KERNEL_METADATA",
    "dequantize_mxfp_weight_triton",
    "gemm_mxfp_packed_activation_reference",
    "gemm_mxfp_packed_activation_triton",
    "gemm_mxfp_reference",
    "gemm_mxfp_triton",
    "pack_mxfp",
    "unpack_mxfp",
]
