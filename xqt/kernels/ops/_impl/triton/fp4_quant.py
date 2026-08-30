"""Triton helpers for grouped NVFP4 and MXFP4 activation quantization."""

from __future__ import annotations

from typing import Final

import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops._impl.fp4_quant_common import (
    pad_last_dim_for_group,
    quantize_mxfp_scale,
    quantize_nvfp4_scale,
    scaled_mxfp4_quant_reference,
    scaled_nvfp4_quant_reference,
)

import triton
import triton.language as tl


_FP4_NUM_WARPS: Final[int] = 4


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("Triton FP4 quantization kernels require CUDA tensors")


def _require_triton() -> object:
    return triton


@triton.jit
def _fp4_code_from_value(value):
    abs_value = tl.abs(value)
    code = tl.where(abs_value > 0.25, 1, 0)
    code += tl.where(abs_value > 0.75, 1, 0)
    code += tl.where(abs_value > 1.25, 1, 0)
    code += tl.where(abs_value > 1.75, 1, 0)
    code += tl.where(abs_value > 2.5, 1, 0)
    code += tl.where(abs_value > 3.5, 1, 0)
    code += tl.where(abs_value > 5.0, 1, 0)
    return code + tl.where(value < 0, 8, 0)


@triton.jit
def _pack_grouped_fp4_kernel(
    x_ptr,
    scale_ptr,
    global_scale_ptr,
    out_ptr,
    rows,
    cols,
    num_groups,
    stride_x_row,
    stride_scale_row,
    stride_out_row,
    use_global_scale: tl.constexpr,
    group_size: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups
    pair_offsets = tl.arange(0, group_size // 2)
    col0 = group * group_size + pair_offsets * 2
    col1 = col0 + 1
    valid_row = row < rows
    mask0 = valid_row & (col0 < cols)
    mask1 = valid_row & (col1 < cols)

    scale = tl.load(
        scale_ptr + row * stride_scale_row + group,
        mask=valid_row,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr, mask=use_global_scale, other=1.0).to(tl.float32)
    inv_scale = tl.where(scale > 0, 1.0 / scale, 0.0)
    multiplier = tl.where(use_global_scale, global_scale * inv_scale, inv_scale)

    x0 = tl.load(x_ptr + row * stride_x_row + col0, mask=mask0, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + row * stride_x_row + col1, mask=mask1, other=0.0).to(tl.float32)
    code0 = _fp4_code_from_value(x0 * multiplier)
    code1 = _fp4_code_from_value(x1 * multiplier)
    packed = code0 + code1 * 16
    out_col = group * (group_size // 2) + pair_offsets
    out_mask = valid_row & (out_col < (cols // 2))
    tl.store(out_ptr + row * stride_out_row + out_col, packed.to(tl.uint8), mask=out_mask)


def _launch_grouped_fp4_pack(
    inputs: torch.Tensor,
    scale_fp32: torch.Tensor,
    *,
    group_size: int,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_triton()
    tensors = (inputs, scale_fp32) if global_scale is None else (inputs, scale_fp32, global_scale)
    _require_cuda_tensors(*tensors)
    if inputs.ndim != 2:
        raise XQTBackendError("Triton FP4 quantization kernels expect a 2D tensor")
    if scale_fp32.ndim != 2:
        raise XQTBackendError("scale must be shaped [rows, groups]")
    rows, cols = int(inputs.shape[0]), int(inputs.shape[1])
    num_groups = cols // int(group_size)
    if scale_fp32.shape != (rows, num_groups):
        raise XQTBackendError("scale shape must match [rows, padded_cols/group_size]")
    out = torch.empty((rows, cols // 2), device=inputs.device, dtype=torch.uint8)
    grid = (rows * num_groups,)
    _pack_grouped_fp4_kernel[grid](
        inputs,
        scale_fp32,
        inputs if global_scale is None else global_scale.reshape(1).to(device=inputs.device, dtype=torch.float32),
        out,
        rows,
        cols,
        num_groups,
        inputs.stride(0),
        scale_fp32.stride(0),
        out.stride(0),
        use_global_scale=global_scale is not None,
        group_size=int(group_size),
        num_warps=_FP4_NUM_WARPS,
    )
    return out


def scaled_nvfp4_quant_triton(
    inputs: torch.Tensor,
    input_global_scale: torch.Tensor,
    *,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize dense activations into packed NVFP4 codes with Triton packing."""

    if input_global_scale.numel() != 1:
        raise XQTBackendError("input_global_scale must be a scalar tensor")
    padded, _ = pad_last_dim_for_group(inputs, group_size=group_size)
    absmax = padded.to(torch.float32).reshape(
        int(padded.shape[0]),
        int(padded.shape[1]) // int(group_size),
        int(group_size),
    ).abs().amax(dim=-1)
    global_scale = input_global_scale.to(device=inputs.device, dtype=torch.float32).reshape(())
    scale_raw = absmax * (global_scale / 6.0)
    scale = quantize_nvfp4_scale(scale_raw)
    packed = _launch_grouped_fp4_pack(
        padded.contiguous(),
        scale.to(device=inputs.device, dtype=torch.float32).contiguous(),
        group_size=int(group_size),
        global_scale=global_scale,
    )
    return packed, scale.contiguous()


def scaled_mxfp4_quant_triton(
    inputs: torch.Tensor,
    *,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize dense activations into packed MXFP4 codes with Triton packing."""

    padded, _ = pad_last_dim_for_group(inputs, group_size=group_size)
    absmax = padded.to(torch.float32).reshape(
        int(padded.shape[0]),
        int(padded.shape[1]) // int(group_size),
        int(group_size),
    ).abs().amax(dim=-1)
    scale_raw = absmax / 6.0
    scale = quantize_mxfp_scale(scale_raw)
    packed = _launch_grouped_fp4_pack(
        padded.contiguous(),
        scale.to(device=inputs.device, dtype=torch.float32).contiguous(),
        group_size=int(group_size),
        global_scale=None,
    )
    return packed, scale.contiguous()


__all__ = [
    "scaled_mxfp4_quant_reference",
    "scaled_mxfp4_quant_triton",
    "scaled_nvfp4_quant_reference",
    "scaled_nvfp4_quant_triton",
]
