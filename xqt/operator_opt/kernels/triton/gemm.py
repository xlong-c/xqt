"""Triton multi-precision GEMM kernels for XQT operator optimization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from xqt.gemm import dense_gemm_reference
from xqt.core.errors import XQTBackendError
from xqt.operator_opt.runtime import target_arch_mismatch
from xqt.operator_opt.kernels.fp4_quant_common import dequantize_nvfp4_codes
from xqt.runtime.bridges.nvfp4 import expand_group_scale, unpack_nvfp4e2m1


import triton
import triton.language as tl
from triton.language.extra import libdevice


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("Triton GEMM kernels require CUDA tensors")


def _require_triton() -> object:
    """Return triton module (top-level import guarantees availability)."""
    return triton


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _tl_dtype_from_torch(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise XQTBackendError(f"unsupported Triton dtype mapping: {dtype}")


@triton.jit
def _dequant_int4_weight_kernel(
    packed_ptr,
    scale_ptr,
    zero_ptr,
    out_ptr,
    rows,
    cols,
    stride_packed_row,
    stride_scale_row,
    stride_scale_group,
    stride_zero_row,
    stride_zero_group,
    stride_out_row,
    stride_out_col,
    has_zero: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = rows * cols
    mask = offsets < total
    row = offsets // cols
    col = offsets % cols

    packed_col = col // 2
    packed = tl.load(
        packed_ptr + row * stride_packed_row + packed_col,
        mask=mask,
        other=0,
    ).to(tl.int32)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibble = tl.where((col & 1) == 0, low, high)
    signed = tl.where(nibble > 7, nibble - 16, nibble).to(tl.float32)

    group = col // group_size
    scale = tl.load(
        scale_ptr + row * stride_scale_row + group * stride_scale_group,
        mask=mask,
        other=1.0,
    ).to(tl.float32)
    value = signed * scale
    if has_zero:
        zero = tl.load(
            zero_ptr + row * stride_zero_row + group * stride_zero_group,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value = value - zero
    tl.store(
        out_ptr + row * stride_out_row + col * stride_out_col,
        value.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _dequant_nvfp4_weight_kernel(
    packed_ptr,
    scale_ptr,
    codebook_ptr,
    out_ptr,
    rows,
    cols,
    stride_packed_row,
    stride_scale_row,
    stride_scale_group,
    stride_out_row,
    stride_out_col,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = rows * cols
    mask = offsets < total
    row = offsets // cols
    col = offsets % cols

    packed_col = col // 2
    packed = tl.load(
        packed_ptr + row * stride_packed_row + packed_col,
        mask=mask,
        other=0,
    ).to(tl.int32)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    code_idx = tl.where((col & 1) == 0, low, high)
    code = tl.load(codebook_ptr + code_idx, mask=mask, other=0.0).to(tl.float32)

    group = col // group_size
    scale = tl.load(
        scale_ptr + row * stride_scale_row + group * stride_scale_group,
        mask=mask,
        other=1.0,
    ).to(tl.float32)
    value = code * scale
    tl.store(
        out_ptr + row * stride_out_row + col * stride_out_col,
        value.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


# ============================================================================
# Reference Implementations
# ============================================================================


def gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
) -> torch.Tensor:
    """Compatibility wrapper for the xqt.gemm-owned dense reference path."""

    return dense_gemm_reference(
        a,
        b,
        bias,
        activation=activation,
        transpose_b=transpose_b,
    )


def _dense_triton_output_dtype(a: torch.Tensor) -> torch.dtype:
    if a.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return a.dtype
    return torch.float16


def _run_dense_triton_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    activation: str | None,
    transpose_b: bool,
) -> torch.Tensor:
    output_dtype = _dense_triton_output_dtype(a)
    if output_dtype == torch.bfloat16:
        return gemm_bf16_triton(
            a,
            b,
            bias,
            activation=activation,
            transpose_b=transpose_b,
            output_dtype=output_dtype,
        )
    return gemm_fp16_triton(
        a,
        b,
        bias,
        activation=activation,
        transpose_b=transpose_b,
        output_dtype=output_dtype,
    )


def gemm_int8_reference(
    a: torch.Tensor,
    b_int8: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
) -> torch.Tensor:
    """Reference INT8 GEMM with dequantization."""
    # Dequantize to FP16/FP32 for computation
    b_fp = b_int8.to(a.dtype)
    if b_scale is not None:
        b_fp = b_fp * b_scale

    a_fp = a
    if a_scale is not None:
        a_fp = a * a_scale

    return gemm_reference(a_fp, b_fp, bias, activation=activation, transpose_b=transpose_b)


def gemm_fp8_reference(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
) -> torch.Tensor:
    """Reference FP8 GEMM with scaling."""
    # Convert FP8 to FP16 for computation (torch.matmul doesn't support FP8 directly)
    b_fp = b_fp8.to(torch.float16)
    if b_scale is not None:
        b_fp = b_fp * b_scale

    a_fp = a.to(torch.float16) if a.dtype == torch.float8_e4m3fn or a.dtype == torch.float8_e5m2 else a
    if a_scale is not None:
        a_fp = a_fp * a_scale

    return gemm_reference(a_fp, b_fp, bias, activation=activation, transpose_b=transpose_b)


def gemm_int4_dequant_reference(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    b_zero: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    group_size: int = 128,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference INT4 dequant GEMM (weight-only quantization)."""
    # Unpack INT4 weights (simplified reference)
    # In practice, b_packed contains 2 INT4 values per byte
    n, k_packed = b_packed.shape
    k = k_packed * 2

    # Unpack nibbles
    low = (b_packed & 0x0F).to(torch.int8)
    high = ((b_packed >> 4) & 0x0F).to(torch.int8)
    b_int4 = torch.stack([low, high], dim=-1).reshape(n, k)

    # Convert to signed INT4 range [-8, 7]
    b_int4 = torch.where(b_int4 > 7, b_int4 - 16, b_int4)

    # Dequantize with per-group scaling
    b_fp = b_int4.to(a.dtype)
    scale_tensor = b_scale
    if scale_tensor.ndim == 3 and scale_tensor.shape[2] == 1:
        scale_tensor = scale_tensor.squeeze(-1)
    zero_tensor = b_zero
    if zero_tensor is not None and zero_tensor.ndim == 3 and zero_tensor.shape[2] == 1:
        zero_tensor = zero_tensor.squeeze(-1)
    num_groups = (k + group_size - 1) // group_size

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, k)
        scale = (
            scale_tensor[:, g : g + 1]
            if scale_tensor.dim() > 1
            else scale_tensor[g : g + 1]
        )
        b_fp[:, start:end] = b_fp[:, start:end] * scale

        if zero_tensor is not None:
            zero = (
                zero_tensor[:, g : g + 1]
                if zero_tensor.dim() > 1
                else zero_tensor[g : g + 1]
            )
            b_fp[:, start:end] = b_fp[:, start:end] - zero

    return gemm_reference(a, b_fp, bias, activation=activation, transpose_b=True)


def dequantize_int4_weight_triton(
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    group_size: int,
    cols: int,
    b_zero: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Decode packed INT4 weights into a dense CUDA tensor using Triton."""

    _require_triton()
    _require_cuda_tensors(b_packed)
    scale = b_scale
    if scale.ndim == 3 and scale.shape[2] == 1:
        scale = scale.squeeze(-1)
    if scale.ndim == 1:
        scale = scale.reshape(1, -1).expand(int(b_packed.shape[0]), -1)
    if scale.ndim != 2:
        raise XQTBackendError("INT4 Triton dequant expects scale to be 1D or 2D")
    zero = b_zero
    if zero is not None:
        if zero.ndim == 3 and zero.shape[2] == 1:
            zero = zero.squeeze(-1)
        if zero.ndim == 1:
            zero = zero.reshape(1, -1).expand(int(b_packed.shape[0]), -1)
        if zero.ndim != 2:
            raise XQTBackendError("INT4 Triton dequant expects zero to be 1D or 2D")
        zero = zero.to(device=b_packed.device, dtype=scale.dtype)
    rows = int(b_packed.shape[0])
    output = torch.empty((rows, int(cols)), device=b_packed.device, dtype=output_dtype)
    total = rows * int(cols)
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    _dequant_int4_weight_kernel[grid](
        b_packed,
        scale.to(device=b_packed.device),
        zero if zero is not None else b_packed,
        output,
        rows,
        int(cols),
        b_packed.stride(0),
        scale.stride(0),
        scale.stride(1),
        0 if zero is None else zero.stride(0),
        0 if zero is None else zero.stride(1),
        output.stride(0),
        output.stride(1),
        has_zero=zero is not None,
        group_size=int(group_size),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


def _nvfp4_codebook_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
        dtype=torch.float32,
    )


def dequantize_nvfp4_weight_triton(
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    cols: int,
    group_size: int,
    weight_global_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Decode packed NVFP4 weights into a dense CUDA tensor using Triton."""

    _require_triton()
    _require_cuda_tensors(b_packed)
    scale = b_scale
    if scale.ndim == 3 and scale.shape[2] == 1:
        scale = scale.squeeze(-1)
    if scale.ndim == 1:
        scale = scale.reshape(1, -1).expand(int(b_packed.shape[0]), -1)
    if scale.ndim != 2:
        raise XQTBackendError("NVFP4 Triton dequant expects scale to be 1D or 2D")
    scale = scale.to(device=b_packed.device, dtype=torch.float32)
    if weight_global_scale is not None:
        scale = scale / weight_global_scale.to(device=b_packed.device, dtype=torch.float32).reshape(1, 1)
    rows = int(b_packed.shape[0])
    output = torch.empty((rows, int(cols)), device=b_packed.device, dtype=output_dtype)
    codebook = _nvfp4_codebook_tensor(b_packed.device)
    total = rows * int(cols)
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    _dequant_nvfp4_weight_kernel[grid](
        b_packed,
        scale,
        codebook,
        output,
        rows,
        int(cols),
        b_packed.stride(0),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
        group_size=int(group_size),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


# ============================================================================
# Triton GEMM Kernels
# ============================================================================

@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    has_bias: tl.constexpr,
    activation: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Generic Triton GEMM kernel with optional bias and activation."""
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
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float32)

    # Apply bias
    if has_bias:
        bias_offs = offs_n
        bias_mask = offs_n < N
        bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0).to(tl.float32)
        c = c + bias[None, :]

    # Apply activation
    if activation == 1:  # relu
        c = tl.maximum(c, 0.0)
    elif activation == 2:  # gelu (approximation using erf)
        c = 0.5 * c * (1.0 + tl.erf(c / 1.4142135623730951))
    elif activation == 3:  # silu
        c = c * tl.sigmoid(c)

    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _quantize_int8_rowwise_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    rows,
    cols,
    stride_xm,
    stride_xk,
    stride_qm,
    stride_qk,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    values = tl.load(
        x_ptr + row * stride_xm + offsets * stride_xk,
        mask=mask,
        other=0.0,
    )
    max_abs = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(max_abs / 127.0, 1e-30)
    quantized = libdevice.rint(values / scale)
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    tl.store(
        q_ptr + row * stride_qm + offsets * stride_qk,
        quantized.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + row, scale.to(tl.float32))


def quantize_int8_rowwise_triton(
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a CUDA matrix to signed INT8 with one scale per input row."""

    _require_triton()
    _require_cuda_tensors(inputs)
    if inputs.ndim != 2:
        raise ValueError("rowwise INT8 quantization expects a 2D tensor")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("rowwise INT8 quantization expects FP16, BF16, or FP32 input")
    rows, cols = (int(inputs.shape[0]), int(inputs.shape[1]))
    if rows < 1 or cols < 1:
        raise ValueError("rowwise INT8 quantization expects non-empty dimensions")
    quantized = torch.empty_like(inputs, dtype=torch.int8)
    scales = torch.empty((rows,), device=inputs.device, dtype=torch.float32)
    block_size = _next_power_of_2(cols)
    _quantize_int8_rowwise_kernel[(rows,)](
        inputs,
        quantized,
        scales,
        rows,
        cols,
        inputs.stride(0),
        inputs.stride(1),
        quantized.stride(0),
        quantized.stride(1),
        BLOCK_SIZE=block_size,
    )
    return quantized, scales


@triton.jit
def _gemm_int8_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    has_a_scale: tl.constexpr,
    has_b_scale: tl.constexpr,
    per_row_a_scale: tl.constexpr,
    has_bias: tl.constexpr,
    activation: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """INT8 GEMM kernel with dequantization."""
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
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a_int8 = tl.load(a_ptrs, mask=a_mask, other=0)
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)

        accumulator = tl.dot(
            a_int8,
            b_int8,
            accumulator,
            out_dtype=tl.int32,
        )

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float32)

    if has_a_scale:
        if per_row_a_scale:
            a_scale = tl.load(
                a_scale_ptr + offs_m,
                mask=offs_m < M,
                other=1.0,
            )
        else:
            a_scale = tl.load(a_scale_ptr)
        c = c * a_scale[:, None].to(tl.float32)

    if has_b_scale:
        b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0)
        c = c * b_scale[None, :].to(tl.float32)

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


@dataclass(frozen=True)
class TritonGemmSchedule:
    """Resolved Triton GEMM launch schedule."""

    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int
    target_arch: str | None
    preset: str

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "group_m": self.group_m,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "target_arch": self.target_arch,
            "preset": self.preset,
        }


_TRITON_FP16_DEFAULT_SCHEDULE = (128, 128, 32, 8, 4, 3)
_TRITON_FP16_SM89_PRESETS: dict[
    tuple[int, int, int, bool, str | None, bool],
    tuple[str, tuple[int, int, int, int, int, int]],
] = {
    (1, 4096, 4096, False, None, True): (
        "sm89_fp16_decode_m1",
        (16, 64, 64, 4, 4, 3),
    ),
    (1, 4096, 4096, False, None, False): (
        "sm89_fp16_decode_m1",
        (16, 64, 64, 4, 4, 3),
    ),
    (4, 4096, 4096, True, None, True): (
        "sm89_fp16_decode_m4_bias",
        (16, 64, 64, 4, 4, 3),
    ),
    (4, 4096, 4096, True, None, False): (
        "sm89_fp16_decode_m4_bias",
        (16, 64, 64, 4, 4, 3),
    ),
    (8, 11008, 4096, True, "silu", True): (
        "sm89_fp16_decode_m8_silu",
        (32, 128, 32, 4, 4, 3),
    ),
    (8, 11008, 4096, True, "silu", False): (
        "sm89_fp16_decode_m8_silu",
        (32, 128, 32, 4, 4, 3),
    ),
    (64, 1024, 1024, True, None, True): (
        "sm89_fp16_small_prefill_bias",
        (16, 128, 32, 4, 4, 3),
    ),
    (64, 1024, 1024, True, None, False): (
        "sm89_fp16_small_prefill_bias",
        (16, 128, 32, 4, 4, 3),
    ),
    (256, 4096, 4096, True, "gelu", False): (
        "sm89_fp16_medium_prefill_gelu_kn",
        (64, 64, 32, 8, 4, 3),
    ),
}


@lru_cache(maxsize=256)
def resolve_triton_fp16_gemm_schedule(
    *,
    m: int,
    n: int,
    k: int,
    has_bias: bool,
    activation: str | None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    group_m: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    target_arch: str | None = None,
    transpose_b: bool = True,
) -> TritonGemmSchedule:
    """Resolve evidence-backed SM89 FP16 defaults and explicit overrides."""

    preset = "default"
    defaults = _TRITON_FP16_DEFAULT_SCHEDULE
    if target_arch == "sm_89":
        resolved = _TRITON_FP16_SM89_PRESETS.get(
            (
                int(m),
                int(n),
                int(k),
                bool(has_bias),
                activation,
                bool(transpose_b),
            )
        )
        if resolved is not None:
            preset, defaults = resolved

    return TritonGemmSchedule(
        block_m=defaults[0] if block_m is None else int(block_m),
        block_n=defaults[1] if block_n is None else int(block_n),
        block_k=defaults[2] if block_k is None else int(block_k),
        group_m=defaults[3] if group_m is None else int(group_m),
        num_warps=defaults[4] if num_warps is None else int(num_warps),
        num_stages=defaults[5] if num_stages is None else int(num_stages),
        target_arch=target_arch,
        preset=preset,
    )


_TRITON_BF16_DEFAULT_SCHEDULE = (128, 128, 32, 8, 4, 3)
_TRITON_BF16_SM89_PRESETS: dict[
    tuple[int, int, int, bool, str | None],
    tuple[str, tuple[int, int, int, int, int, int]],
] = {
    (1, 4096, 4096, False, None): (
        "sm89_bf16_decode_m1",
        (16, 64, 64, 4, 4, 3),
    ),
    (4, 4096, 4096, True, None): (
        "sm89_bf16_decode_m4_bias",
        (16, 64, 64, 4, 4, 3),
    ),
    (8, 11008, 4096, True, "silu"): (
        "sm89_bf16_decode_m8_silu",
        (32, 128, 32, 4, 4, 3),
    ),
    (64, 1024, 1024, True, None): (
        "sm89_bf16_small_prefill_bias",
        (16, 64, 32, 4, 4, 3),
    ),
    (256, 4096, 4096, True, "gelu"): (
        "sm89_bf16_medium_prefill_gelu",
        (64, 64, 32, 8, 4, 3),
    ),
}


@lru_cache(maxsize=256)
def resolve_triton_bf16_gemm_schedule(
    *,
    m: int,
    n: int,
    k: int,
    has_bias: bool,
    activation: str | None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    group_m: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    target_arch: str | None = None,
) -> TritonGemmSchedule:
    """Resolve evidence-backed SM89 BF16 defaults and explicit overrides."""

    preset = "default"
    defaults = _TRITON_BF16_DEFAULT_SCHEDULE
    if target_arch == "sm_89":
        resolved = _TRITON_BF16_SM89_PRESETS.get(
            (int(m), int(n), int(k), bool(has_bias), activation)
        )
        if resolved is not None:
            preset, defaults = resolved

    return TritonGemmSchedule(
        block_m=defaults[0] if block_m is None else int(block_m),
        block_n=defaults[1] if block_n is None else int(block_n),
        block_k=defaults[2] if block_k is None else int(block_k),
        group_m=defaults[3] if group_m is None else int(group_m),
        num_warps=defaults[4] if num_warps is None else int(num_warps),
        num_stages=defaults[5] if num_stages is None else int(num_stages),
        target_arch=target_arch,
        preset=preset,
    )


@lru_cache(maxsize=16)
def _cuda_target_arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def gemm_fp16_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
    accum_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float16,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    group_m: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    target_arch: str | None = None,
) -> torch.Tensor:
    """FP16 GEMM using Triton with evidence-backed SM89 schedules."""
    _require_triton()
    _require_cuda_tensors(a, b)

    if a.dtype not in {torch.float16, torch.bfloat16}:
        a = a.to(torch.float16)
    if b.dtype not in {torch.float16, torch.bfloat16}:
        b = b.to(torch.float16)
    if bias is not None and bias.dtype != output_dtype:
        bias = bias.to(output_dtype)

    # Resolve the logical B layout through strides without materializing a transpose.
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    if transpose_b:
        N, K_b = b.shape
        assert K == K_b, f"Inner dimensions must match: {K} vs {K_b}"
        stride_bk, stride_bn = b.stride(1), b.stride(0)
    else:
        K_b, N = b.shape
        assert K == K_b
        stride_bk, stride_bn = b.stride(0), b.stride(1)

    resolved_target_arch = target_arch
    if resolved_target_arch is None and a.is_cuda:
        resolved_target_arch = _cuda_target_arch(a.device)
    mismatch = target_arch_mismatch(target_arch, a)
    if mismatch is not None:
        raise XQTBackendError(
            f"Triton GEMM target architecture is not executable: {mismatch}"
        )
    schedule = resolve_triton_fp16_gemm_schedule(
        m=int(M),
        n=int(N),
        k=int(K),
        has_bias=bias is not None,
        activation=activation,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
        target_arch=resolved_target_arch,
        transpose_b=transpose_b,
    )

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    # Encode activation
    act_code = 0
    if activation == "relu":
        act_code = 1
    elif activation == "gelu":
        act_code = 2
    elif activation == "silu":
        act_code = 3

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _gemm_kernel[grid](
        a, b, c,
        bias if bias is not None else a,  # dummy pointer if no bias
        M, N, K,
        a.stride(0), a.stride(1),
        stride_bk, stride_bn,
        c.stride(0), c.stride(1),
        has_bias=bias is not None,
        activation=act_code,
        ACC_TYPE=_tl_dtype_from_torch(accum_dtype),
        BLOCK_M=schedule.block_m,
        BLOCK_N=schedule.block_n,
        BLOCK_K=schedule.block_k,
        GROUP_M=schedule.group_m,
        num_warps=schedule.num_warps,  # type: ignore[call-arg]
        num_stages=schedule.num_stages,  # type: ignore[call-arg]
    )

    return c


def gemm_bf16_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
    accum_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    group_m: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    target_arch: str | None = None,
) -> torch.Tensor:
    """BF16 GEMM using Triton with evidence-backed SM89 schedules."""
    _require_triton()
    _require_cuda_tensors(a, b)

    if a.dtype != torch.bfloat16:
        a = a.to(torch.bfloat16)
    if b.dtype != torch.bfloat16:
        b = b.to(torch.bfloat16)
    if bias is not None and bias.dtype != output_dtype:
        bias = bias.to(output_dtype)

    resolved_target_arch = target_arch
    if resolved_target_arch is None and a.is_cuda:
        resolved_target_arch = _cuda_target_arch(a.device)
    mismatch = target_arch_mismatch(target_arch, a)
    if mismatch is not None:
        raise XQTBackendError(
            f"Triton GEMM target architecture is not executable: {mismatch}"
        )
    if a.dim() == 2 and b.dim() == 2:
        m, k = int(a.shape[0]), int(a.shape[1])
        n = int(b.shape[0]) if transpose_b else int(b.shape[1])
    else:
        m = n = k = 0
    schedule = resolve_triton_bf16_gemm_schedule(
        m=m,
        n=n,
        k=k,
        has_bias=bias is not None,
        activation=activation,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
        target_arch=resolved_target_arch,
    )

    result = gemm_fp16_triton(
        a, b, bias,
        activation=activation,
        transpose_b=transpose_b,
        accum_dtype=accum_dtype,
        output_dtype=output_dtype,
        block_m=schedule.block_m,
        block_n=schedule.block_n,
        block_k=schedule.block_k,
        group_m=schedule.group_m,
        num_warps=schedule.num_warps,
        num_stages=schedule.num_stages,
        target_arch=resolved_target_arch,
    )

    return result


def gemm_int8_triton(
    a: torch.Tensor,
    b_int8: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    group_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """INT8 GEMM with a true W8A8 Tensor Core fast path on CUDA.

    Floating-point activations keep the existing reference behavior because
    they represent W8A16-style callers. The executable Triton path is only
    selected when both operands are signed INT8 and scales are supplied.
    """
    _require_triton()
    _require_cuda_tensors(a, b_int8)

    if b_int8.dtype != torch.int8:
        raise ValueError(f"b_int8 must be INT8, got {b_int8.dtype}")
    if a.dtype != torch.int8:
        return gemm_int8_reference(
            a, b_int8, a_scale, b_scale, bias,
            activation=activation,
            transpose_b=transpose_b,
        )
    if output_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("output_dtype must be float16, bfloat16, or float32")

    if transpose_b:
        n, k_b = b_int8.shape
        if int(a.shape[1]) != int(k_b):
            raise ValueError(f"Inner dimensions must match: {a.shape[1]} vs {k_b}")
        b = b_int8.t().contiguous()
    else:
        k_b, n = b_int8.shape
        if int(a.shape[1]) != int(k_b):
            raise ValueError(f"Inner dimensions must match: {a.shape[1]} vs {k_b}")
        b = b_int8.contiguous()
    m, k = (int(a.shape[0]), int(a.shape[1]))
    if a_scale is None or b_scale is None:
        raise ValueError("true W8A8 Triton GEMM requires activation and weight scales")
    if a_scale.numel() not in {1, m}:
        raise ValueError(
            "Triton W8A8 activation scale must be scalar or one value per input row"
        )
    if b_scale.numel() != int(n):
        raise ValueError("Triton W8A8 weight_scale must contain one value per output")
    if bias is not None and bias.numel() != int(n):
        raise ValueError("bias must contain one value per output")

    output = torch.empty((m, int(n)), device=a.device, dtype=output_dtype)
    per_row_a_scale = int(a_scale.numel()) == m
    activation_scale = a_scale.to(device=a.device, dtype=torch.float32).reshape(-1)
    weight_scale = b_scale.to(device=a.device, dtype=torch.float32).reshape(int(n))
    bias_tensor = (
        bias.to(device=a.device, dtype=torch.float32).reshape(int(n))
        if bias is not None
        else a
    )
    act_code = 0
    if activation == "relu":
        act_code = 1
    elif activation == "gelu":
        act_code = 2
    elif activation == "silu":
        act_code = 3
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(int(n), meta["BLOCK_N"]),
    )
    _gemm_int8_kernel[grid](
        a,
        b,
        output,
        activation_scale,
        weight_scale,
        bias_tensor,
        m,
        int(n),
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        has_a_scale=True,
        has_b_scale=True,
        per_row_a_scale=per_row_a_scale,
        has_bias=bias is not None,
        activation=act_code,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def gemm_fp8_triton(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
    fp8_format: str = "e4m3",
) -> torch.Tensor:
    """FP8 GEMM with scaling using Triton (requires SM89+)."""
    _require_triton()
    _require_cuda_tensors(a, b_fp8)

    # Check FP8 dtype
    if fp8_format == "e4m3":
        expected_dtype = torch.float8_e4m3fn
    elif fp8_format == "e5m2":
        expected_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"unsupported fp8_format: {fp8_format}")

    if b_fp8.dtype != expected_dtype:
        raise ValueError(f"b_fp8 must be {expected_dtype}, got {b_fp8.dtype}")

    # Fallback to reference for now
    return gemm_fp8_reference(
        a, b_fp8, a_scale, b_scale, bias,
        activation=activation,
        transpose_b=transpose_b,
    )


def gemm_int4_dequant_triton(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    b_zero: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    group_size: int = 128,
    activation: str | None = None,
) -> torch.Tensor:
    """INT4 weight-only dequant GEMM using Triton."""
    _require_triton()
    _require_cuda_tensors(a, b_packed)
    if a.ndim != 2 or b_packed.ndim != 2:
        raise XQTBackendError("gemm_int4_dequant expects 2D GEMM inputs")

    packed = b_packed.to(torch.uint8)
    padded_k = int(packed.shape[1]) * 2
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    signed = torch.where(
        unpacked > 7,
        unpacked.to(torch.int16) - 16,
        unpacked.to(torch.int16),
    ).to(torch.float32)
    groups = int(padded_k) // int(group_size)
    scale = b_scale.to(device=a.device, dtype=torch.float32)
    if scale.ndim == 3 and scale.shape[2] == 1:
        scale = scale.squeeze(-1)
    if scale.ndim == 1:
        scale = scale.reshape(1, -1).expand(signed.shape[0], -1)
    if scale.shape != (signed.shape[0], groups):
        raise XQTBackendError(
            "gemm_int4_dequant expects b_scale to match [out_features, k/group_size]"
        )
    compute_dtype = torch.bfloat16 if a.dtype == torch.bfloat16 else torch.float16
    dense_weight = dequantize_int4_weight_triton(
        packed,
        scale,
        group_size=int(group_size),
        cols=int(a.shape[1]),
        b_zero=b_zero,
        output_dtype=compute_dtype,
    )
    return _run_dense_triton_gemm(
        a.to(compute_dtype) if a.dtype not in {torch.float16, torch.bfloat16} else a,
        dense_weight,
        None if bias is None else bias.to(device=a.device, dtype=_dense_triton_output_dtype(a)),
        activation=activation,
        transpose_b=True,
    )


def gemm_nvfp4_packed_dequant_reference(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference NVFP4 packed dequant GEMM."""

    weight_codes = unpack_nvfp4e2m1(
        b_packed.to(device=a.device),
        input_features=int(input_features),
    )
    scale = b_scale.to(device=a.device, dtype=torch.float32)
    if weight_global_scale is not None:
        denom = weight_global_scale.to(device=a.device, dtype=torch.float32)
        scale = scale / (
            denom.reshape(1, 1)
            if scale.ndim == 2
            else denom.reshape(1, 1, 1)
        )
    expanded_scale = expand_group_scale(
        scale,
        group_size=int(group_size),
        input_features=int(input_features),
    ).to(device=a.device, dtype=weight_codes.dtype)
    weight = weight_codes * expanded_scale
    return gemm_reference(
        a,
        weight.to(dtype=a.dtype, device=a.device),
        bias,
        activation=activation,
        transpose_b=True,
    )


def gemm_nvfp4_packed_dequant_triton(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Packed NVFP4 dequant GEMM composed with Triton dense GEMM."""

    _require_triton()
    _require_cuda_tensors(a, b_packed, b_scale)
    compute_dtype = torch.bfloat16 if a.dtype == torch.bfloat16 else torch.float16
    dense_weight = dequantize_nvfp4_weight_triton(
        b_packed.to(device=a.device),
        b_scale.to(device=a.device),
        cols=int(input_features),
        group_size=int(group_size),
        weight_global_scale=weight_global_scale,
        output_dtype=compute_dtype,
    )
    return _run_dense_triton_gemm(
        a.to(compute_dtype) if a.dtype not in {torch.float16, torch.bfloat16} else a,
        dense_weight,
        None if bias is None else bias.to(device=a.device, dtype=_dense_triton_output_dtype(a)),
        activation=activation,
        transpose_b=True,
    )


def gemm_nvfp4_packed_activation_reference(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation_global_scale: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference NVFP4 GEMM that consumes packed activation and packed weight."""

    activation_dense = dequantize_nvfp4_codes(
        a_packed,
        a_scale,
        input_features=int(input_features),
        group_size=int(group_size),
        global_scale=activation_global_scale,
        output_dtype=output_dtype,
    )
    return gemm_nvfp4_packed_dequant_reference(
        activation_dense,
        b_packed,
        b_scale,
        bias,
        input_features=int(input_features),
        group_size=int(group_size),
        weight_global_scale=weight_global_scale,
        activation=activation,
    )


def gemm_nvfp4_packed_activation_triton(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation_global_scale: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Composed Triton runtime that consumes packed NVFP4 activation and weight."""

    _require_triton()
    _require_cuda_tensors(a_packed, a_scale, b_packed, b_scale)
    activation_dense = dequantize_nvfp4_weight_triton(
        a_packed.to(device=b_packed.device),
        a_scale.to(device=b_packed.device),
        cols=int(input_features),
        group_size=int(group_size),
        weight_global_scale=activation_global_scale,
        output_dtype=output_dtype,
    )
    return gemm_nvfp4_packed_dequant_triton(
        activation_dense,
        b_packed,
        b_scale,
        bias,
        input_features=int(input_features),
        group_size=int(group_size),
        weight_global_scale=weight_global_scale,
        activation=activation,
    )


# ============================================================================
# Metadata
# ============================================================================

TRITON_GEMM_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "gemm_fp16": {
        "kernel_name": "gemm_fp16",
        "precision": "fp16",
        "block_shape": [128, 128, 32],
        "group_m": 8,
        "num_warps": 4,
        "num_stages": 3,
        "schedule_policy": (
            "exact SM89 shape/bias/activation/layout presets with explicit override"
        ),
        "schedule_presets": [
            "sm89_fp16_decode_m1",
            "sm89_fp16_decode_m4_bias",
            "sm89_fp16_decode_m8_silu",
            "sm89_fp16_small_prefill_bias",
            "sm89_fp16_medium_prefill_gelu_kn",
        ],
        "supports_bias": True,
        "supports_activation": True,
        "activations": ["relu", "gelu", "silu"],
        "baseline": "torch.matmul (fp16)",
        "usage": "Standard FP16 GEMM with optional bias and activation fusion.",
        "hardware": "CUDA SM70+",
    },
    "gemm_bf16": {
        "kernel_name": "gemm_bf16",
        "precision": "bf16",
        "block_shape": [128, 128, 32],
        "group_m": 8,
        "num_warps": 4,
        "num_stages": 3,
        "schedule_policy": "exact SM89 shape/bias/activation presets with explicit override",
        "schedule_presets": [
            "sm89_bf16_decode_m1",
            "sm89_bf16_decode_m4_bias",
            "sm89_bf16_decode_m8_silu",
            "sm89_bf16_small_prefill_bias",
            "sm89_bf16_medium_prefill_gelu",
        ],
        "supports_bias": True,
        "supports_activation": True,
        "activations": ["relu", "gelu", "silu"],
        "baseline": "torch.matmul (bf16)",
        "usage": "BF16 GEMM with wider dynamic range than FP16.",
        "hardware": "CUDA SM80+ (Ampere)",
    },
    "gemm_int8": {
        "kernel_name": "gemm_int8",
        "precision": "int8",
        "quantization_mode": "W8A8 or W8A16",
        "supports_scaling": True,
        "supports_bias": True,
        "supports_activation": True,
        "baseline": "torch.matmul + dequant",
        "usage": "INT8 quantized GEMM with per-tensor or per-channel scaling.",
        "hardware": "CUDA SM75+ (Turing Tensor Core)",
        "note": "True W8A8 Tensor Core fast path for signed INT8 activation and weight operands; W8A16 callers retain reference fallback.",
    },
    "gemm_fp8": {
        "kernel_name": "gemm_fp8",
        "precision": "fp8",
        "fp8_formats": ["e4m3", "e5m2"],
        "supports_scaling": True,
        "supports_bias": True,
        "supports_activation": True,
        "baseline": "torch.matmul + scale",
        "usage": "FP8 GEMM for H100 Transformer Engine compatibility.",
        "hardware": "CUDA SM89+ (Hopper H100)",
        "note": "Currently uses reference fallback, full kernel TBD",
    },
    "gemm_int4_dequant": {
        "kernel_name": "gemm_int4_dequant",
        "precision": "int4",
        "quantization_mode": "W4A16 weight-only",
        "supports_group_quant": True,
        "group_size": 128,
        "supports_bias": True,
        "supports_activation": True,
        "baseline": "unpack + dequant + torch.matmul",
        "usage": "INT4 weight-only quantization with per-group scaling.",
        "hardware": "CUDA SM70+",
        "note": "Composed runtime: device-side unpack/dequant followed by Triton dense GEMM.",
    },
    "gemm_nvfp4_packed_dequant": {
        "kernel_name": "gemm_nvfp4_packed_dequant",
        "precision": "nvfp4",
        "quantization_mode": "NVFP4 packed weight-only",
        "supports_group_quant": True,
        "group_size": 16,
        "supports_bias": True,
        "supports_activation": True,
        "baseline": "unpack_nvfp4e2m1 + expand_group_scale + torch.matmul",
        "usage": "Packed NVFP4 weight-only quantization with per-group FP8 scale.",
        "hardware": "CUDA SM70+",
        "note": "Composed runtime: device-side NVFP4 decode/dequant followed by Triton dense GEMM.",
    },
}


__all__ = [
    "TRITON_GEMM_KERNEL_METADATA",
    "TritonGemmSchedule",
    "dequantize_int4_weight_triton",
    "dequantize_nvfp4_weight_triton",
    "gemm_reference",
    "gemm_int8_reference",
    "gemm_fp8_reference",
    "gemm_int4_dequant_reference",
    "gemm_fp16_triton",
    "gemm_bf16_triton",
    "gemm_int8_triton",
    "gemm_fp8_triton",
    "gemm_int4_dequant_triton",
    "gemm_nvfp4_packed_activation_reference",
    "gemm_nvfp4_packed_activation_triton",
    "gemm_nvfp4_packed_dequant_reference",
    "gemm_nvfp4_packed_dequant_triton",
    "resolve_triton_fp16_gemm_schedule",
    "resolve_triton_bf16_gemm_schedule",
]
