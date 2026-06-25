"""Triton multi-precision GEMM kernels for XQT operator optimization."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError


import triton
import triton.language as tl


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
    """Reference GEMM implementation using torch.matmul."""
    if transpose_b:
        output = torch.matmul(a, b.t())
    else:
        output = torch.matmul(a, b)

    if bias is not None:
        output = output + bias

    if activation == "relu":
        return F.relu(output)
    elif activation == "gelu":
        return F.gelu(output)
    elif activation == "silu":
        return F.silu(output)
    elif activation is None:
        return output
    else:
        raise ValueError(f"unsupported activation: {activation}")


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
    num_groups = (k + group_size - 1) // group_size

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, k)
        scale = b_scale[:, g:g+1] if b_scale.dim() > 1 else b_scale[g:g+1]
        b_fp[:, start:end] = b_fp[:, start:end] * scale

        if b_zero is not None:
            zero = b_zero[:, g:g+1] if b_zero.dim() > 1 else b_zero[g:g+1]
            b_fp[:, start:end] = b_fp[:, start:end] - zero

    return gemm_reference(a, b_fp, bias, activation=activation, transpose_b=True)


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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    # Apply bias
    if has_bias:
        bias_offs = offs_n
        bias_mask = offs_n < N
        bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
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
    tl.store(c_ptrs, c, mask=c_mask)


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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a_int8 = tl.load(a_ptrs, mask=a_mask, other=0)
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)

        # Convert INT8 to FP32 and apply scaling
        a_fp = a_int8.to(tl.float32)
        b_fp = b_int8.to(tl.float32)

        if has_a_scale:
            a_scale = tl.load(a_scale_ptr)
            a_fp = a_fp * a_scale

        if has_b_scale:
            b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0)
            b_fp = b_fp * b_scale[None, :]

        accumulator += tl.dot(a_fp, b_fp)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

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


def gemm_fp16_triton(
    a: torch.Tensor,
    b: torch.Tensor,
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
) -> torch.Tensor:
    """FP16 GEMM using Triton."""
    _require_triton()
    _require_cuda_tensors(a, b)

    if a.dtype != torch.float16:
        a = a.to(torch.float16)
    if b.dtype != torch.float16:
        b = b.to(torch.float16)

    # Prepare dimensions
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    if transpose_b:
        N, K_b = b.shape
        assert K == K_b, f"Inner dimensions must match: {K} vs {K_b}"
        b = b.t().contiguous()
    else:
        K_b, N = b.shape
        assert K == K_b

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

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
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        has_bias=bias is not None,
        activation=act_code,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,  # type: ignore[call-arg]
        num_stages=num_stages,  # type: ignore[call-arg]
    )

    return c


def gemm_bf16_triton(
    a: torch.Tensor,
    b: torch.Tensor,
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
) -> torch.Tensor:
    """BF16 GEMM using Triton (converts to FP16 internally for Triton compatibility)."""
    _require_triton()
    _require_cuda_tensors(a, b)

    # Convert BF16 to FP16 for Triton (Triton kernel uses FP16 accumulation)
    a_fp16 = a.to(torch.float16) if a.dtype == torch.bfloat16 else a
    b_fp16 = b.to(torch.float16) if b.dtype == torch.bfloat16 else b

    result = gemm_fp16_triton(
        a_fp16, b_fp16, bias,
        activation=activation,
        transpose_b=transpose_b,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        num_warps=num_warps,  # type: ignore[call-arg]
        num_stages=num_stages,  # type: ignore[call-arg]
    )

    # Convert back to BF16 if needed
    if a.dtype == torch.bfloat16:
        result = result.to(torch.bfloat16)

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
) -> torch.Tensor:
    """INT8 GEMM with dequantization using Triton."""
    _require_triton()
    _require_cuda_tensors(a, b_int8)

    if b_int8.dtype != torch.int8:
        raise ValueError(f"b_int8 must be INT8, got {b_int8.dtype}")

    # For now, fallback to reference (full kernel implementation needs more work)
    return gemm_int8_reference(
        a, b_int8, a_scale, b_scale, bias,
        activation=activation,
        transpose_b=transpose_b,
    )


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

    # Fallback to reference
    return gemm_int4_dequant_reference(
        a, b_packed, b_scale, b_zero, bias,
        group_size=group_size,
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
        "note": "Currently uses reference fallback, full kernel TBD",
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
        "note": "Currently uses reference fallback, fused kernel TBD",
    },
}


__all__ = [
    "TRITON_GEMM_KERNEL_METADATA",
    "gemm_reference",
    "gemm_int8_reference",
    "gemm_fp8_reference",
    "gemm_int4_dequant_reference",
    "gemm_fp16_triton",
    "gemm_bf16_triton",
    "gemm_int8_triton",
    "gemm_fp8_triton",
    "gemm_int4_dequant_triton",
]
