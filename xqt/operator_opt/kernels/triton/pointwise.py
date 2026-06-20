"""Triton operator optimization references and CUDA entry points."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError


try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None  # type: ignore[assignment]


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("Triton kernels require CUDA tensors")


def _require_triton() -> object:
    if triton is None:
        raise XQTBackendError(
            "triton is required for Triton operator kernels. Install the optimization extras."
        )
    return triton


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _pointwise_block_size(numel: int, configured: int) -> int:
    if configured <= 0:
        raise ValueError("block_size must be positive")
    return min(configured, _next_power_of_2(numel))


if triton is not None and tl is not None:

    @triton.jit
    def _bias_gelu_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        bias_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        bias_offsets = offsets % bias_size
        bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
        z = x + bias
        cdf = 0.5 * (1.0 + tl.erf(z * 0.7071067811865476))
        out = z * cdf
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _swiglu_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        out = gate * sigmoid * up
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _rmsnorm_residual_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        hidden_dim: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        base = row_id * hidden_dim + offsets
        merged = tl.load(x_ptr + base, mask=mask, other=0.0) + tl.load(
            residual_ptr + base,
            mask=mask,
            other=0.0,
        )
        square_sum = tl.sum(merged * merged, axis=0)
        rms = tl.rsqrt(square_sum / hidden_dim + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        out = merged * rms * weight
        tl.store(out_ptr + base, out, mask=mask)

    @triton.jit
    def _rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_pairs: tl.constexpr,
        half_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        pair_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = pair_offsets < n_pairs
        pair_in_row = pair_offsets % half_dim
        even_offsets = pair_offsets * 2
        odd_offsets = even_offsets + 1
        even = tl.load(x_ptr + even_offsets, mask=mask, other=0.0)
        odd = tl.load(x_ptr + odd_offsets, mask=mask, other=0.0)
        cos = tl.load(cos_ptr + pair_in_row, mask=mask, other=1.0)
        sin = tl.load(sin_ptr + pair_in_row, mask=mask, other=0.0)
        tl.store(out_ptr + even_offsets, even * cos - odd * sin, mask=mask)
        tl.store(out_ptr + odd_offsets, even * sin + odd * cos, mask=mask)

else:
    _bias_gelu_kernel = None
    _swiglu_kernel = None
    _rmsnorm_residual_kernel = None
    _rope_kernel = None


def fused_bias_gelu_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Reference implementation for bias + GELU fusion."""

    return F.gelu(x + bias)


def fused_bias_gelu_triton(
    x: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """CUDA-only Triton entry point for bias + GELU."""

    _require_triton()
    _require_cuda_tensors(x, bias)
    if bias.numel() != x.shape[-1]:
        raise ValueError("bias must match the last dimension of x")
    x_flat = x.contiguous().flatten()
    bias_flat = bias.contiguous().flatten()
    out = torch.empty_like(x_flat)
    launch_block = _pointwise_block_size(x_flat.numel(), block_size)
    grid = (_require_triton().cdiv(x_flat.numel(), launch_block),)
    assert _bias_gelu_kernel is not None
    _bias_gelu_kernel[grid](
        x_flat,
        bias_flat,
        out,
        x_flat.numel(),
        bias_flat.numel(),
        BLOCK_SIZE=launch_block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape_as(x)


def fused_swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference implementation for SwiGLU fusion."""

    return F.silu(gate) * up


def fused_swiglu_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """CUDA-only Triton entry point for SwiGLU."""

    _require_triton()
    _require_cuda_tensors(gate, up)
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must share shape, got {tuple(gate.shape)} and {tuple(up.shape)}"
        )
    gate_flat = gate.contiguous().flatten()
    up_flat = up.contiguous().flatten()
    out = torch.empty_like(gate_flat)
    launch_block = _pointwise_block_size(gate_flat.numel(), block_size)
    grid = (_require_triton().cdiv(gate_flat.numel(), launch_block),)
    assert _swiglu_kernel is not None
    _swiglu_kernel[grid](
        gate_flat,
        up_flat,
        out,
        gate_flat.numel(),
        BLOCK_SIZE=launch_block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape_as(gate)


def fused_rmsnorm_residual_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation for residual add + RMSNorm."""

    merged = x + residual
    variance = merged.pow(2).mean(dim=-1, keepdim=True)
    return merged * torch.rsqrt(variance + eps) * weight


def fused_rmsnorm_residual_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """CUDA-only Triton entry point for residual RMSNorm."""

    _require_triton()
    _require_cuda_tensors(x, residual, weight)
    if x.shape != residual.shape:
        raise ValueError(
            f"x and residual must share shape, got {tuple(x.shape)} and {tuple(residual.shape)}"
        )
    if weight.numel() != x.shape[-1]:
        raise ValueError("weight must match the last dimension of x")
    x_2d = x.contiguous().reshape(-1, x.shape[-1])
    residual_2d = residual.contiguous().reshape_as(x_2d)
    weight_flat = weight.contiguous().flatten()
    out = torch.empty_like(x_2d)
    launch_block = _pointwise_block_size(x_2d.shape[-1], block_size)
    grid = (x_2d.shape[0],)
    assert _rmsnorm_residual_kernel is not None
    _rmsnorm_residual_kernel[grid](
        x_2d,
        residual_2d,
        weight_flat,
        out,
        x_2d.shape[0],
        x_2d.shape[1],
        eps,
        BLOCK_SIZE=launch_block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape_as(x)


def fused_rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation for rotary position embedding."""

    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even hidden dimension")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos.to(dtype=x.dtype, device=x.device)
    sin = sin.to(dtype=x.dtype, device=x.device)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def fused_rope_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """CUDA-only Triton entry point for RoPE."""

    _require_triton()
    _require_cuda_tensors(x, cos, sin)
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even hidden dimension")
    half_dim = x.shape[-1] // 2
    if cos.numel() != half_dim or sin.numel() != half_dim:
        raise ValueError("cos and sin must contain hidden_dim / 2 values")
    x_flat = x.contiguous().flatten()
    cos_flat = cos.contiguous().flatten()
    sin_flat = sin.contiguous().flatten()
    out = torch.empty_like(x_flat)
    n_pairs = x_flat.numel() // 2
    launch_block = _pointwise_block_size(n_pairs, block_size)
    grid = (_require_triton().cdiv(n_pairs, launch_block),)
    assert _rope_kernel is not None
    _rope_kernel[grid](
        x_flat,
        cos_flat,
        sin_flat,
        out,
        n_pairs,
        half_dim,
        BLOCK_SIZE=launch_block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape_as(x)


TRITON_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "bias_gelu": {
        "kernel_name": "fused_bias_gelu",
        "block_size": 1024,
        "num_warps": 4,
        "num_stages": 4,
        "autotune_key": ["numel"],
        "usage": "Linear or matmul epilogue with bias + GELU.",
    },
    "swiglu": {
        "kernel_name": "fused_swiglu",
        "block_size": 1024,
        "num_warps": 4,
        "num_stages": 4,
        "autotune_key": ["numel"],
        "usage": "Transformer MLP gate/up projection epilogue.",
    },
    "rmsnorm_residual": {
        "kernel_name": "fused_rmsnorm_residual",
        "block_size": 1024,
        "num_warps": 4,
        "num_stages": 4,
        "autotune_key": ["hidden_dim"],
        "usage": "Decoder residual add followed by RMSNorm.",
    },
    "rope": {
        "kernel_name": "fused_rope",
        "block_size": 1024,
        "num_warps": 4,
        "num_stages": 4,
        "autotune_key": ["seq_len", "hidden_dim"],
        "usage": "Rotary position embedding over even/odd hidden pairs.",
    },
}


__all__ = [
    "TRITON_KERNEL_METADATA",
    "fused_bias_gelu_reference",
    "fused_bias_gelu_triton",
    "fused_rope_reference",
    "fused_rope_triton",
    "fused_rmsnorm_residual_reference",
    "fused_rmsnorm_residual_triton",
    "fused_swiglu_reference",
    "fused_swiglu_triton",
]
