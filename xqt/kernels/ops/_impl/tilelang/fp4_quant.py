"""TileLang helpers for grouped NVFP4 and MXFP4 activation quantization."""

from functools import lru_cache
from typing import Any

import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops._impl.fp4_quant_common import (
    pad_last_dim_for_group,
    quantize_mxfp_scale,
    quantize_nvfp4_scale,
    scaled_mxfp4_quant_reference,
    scaled_nvfp4_quant_reference,
)
from xqt.kernels.ops._impl.tilelang._common import (
    require_cuda_tensors,
    require_tilelang,
)


def _input_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise XQTBackendError(f"unsupported FP4 activation dtype: {dtype}")


@lru_cache(maxsize=32)
def build_tilelang_grouped_fp4_pack_kernel(
    rows: int,
    cols: int,
    *,
    input_dtype: str,
    group_size: int,
    use_global_scale: bool,
    block_size: int = 128,
    target_arch: str | None = None,
) -> Any:
    """Build a TileLang kernel that packs dense activations into FP4 nibbles."""

    tilelang = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if cols % 2 != 0:
        raise ValueError("cols must be even after padding")
    if input_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("input_dtype must be float16, bfloat16, or float32")
    if group_size not in {16, 32}:
        raise ValueError("group_size must be 16 or 32")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    x_dtype = (
        T.float16
        if input_dtype == "float16"
        else T.bfloat16
        if input_dtype == "bfloat16"
        else T.float32
    )
    groups = cols // group_size
    x_shape = [rows, cols]
    scale_shape = [rows, groups]
    global_scale_shape = [1]
    out_shape = [rows, cols // 2]
    total_pairs = rows * (cols // 2)
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

    def _abs_value(value):
        return T.if_then_else(value < 0.0, -value, value)

    def _threshold_increment(abs_value, threshold):
        return T.if_then_else(abs_value > threshold, 1, 0)

    def _fp4_code(value):
        abs_value = _abs_value(value.astype(T.float32))
        magnitude = (
            _threshold_increment(abs_value, 0.25)
            + _threshold_increment(abs_value, 0.75)
            + _threshold_increment(abs_value, 1.25)
            + _threshold_increment(abs_value, 1.75)
            + _threshold_increment(abs_value, 2.5)
            + _threshold_increment(abs_value, 3.5)
            + _threshold_increment(abs_value, 5.0)
        )
        sign = T.if_then_else(value < 0.0, 8, 0)
        return (magnitude + sign).astype(T.uint8)

    if use_global_scale:

        def grouped_fp4_pack_main(
            x: T.Tensor(x_shape, x_dtype),
            scale: T.Tensor(scale_shape, T.float32),
            global_scale: T.Tensor(global_scale_shape, T.float32),
            out: T.Tensor(out_shape, T.uint8),
        ):
            with T.Kernel(T.ceildiv(total_pairs, block_size), threads=block_size) as block:
                for offset in T.Parallel(block_size):
                    index = block * block_size + offset
                    if index < total_pairs:
                        row = index // (cols // 2)
                        pair_col = index - row * (cols // 2)
                        feature0 = pair_col * 2
                        feature1 = feature0 + 1
                        group = feature0 // group_size
                        scale_value = scale[row, group]
                        inv_scale = T.if_then_else(
                            scale_value > 0.0,
                            global_scale[0] / scale_value,
                            0.0,
                        )
                        code0 = _fp4_code(x[row, feature0] * inv_scale)
                        code1 = _fp4_code(x[row, feature1] * inv_scale)
                        out[row, pair_col] = code0 + code1 * 16

        grouped_fp4_pack_main.__name__ = (
            f"grouped_fp4_pack_main_rows{rows}_cols{cols}_{input_dtype}_"
            f"g{group_size}_gs1_b{block_size}"
        )
        prim = T.prim_func(grouped_fp4_pack_main)
    else:

        def grouped_fp4_pack_main(
            x: T.Tensor(x_shape, x_dtype),
            scale: T.Tensor(scale_shape, T.float32),
            out: T.Tensor(out_shape, T.uint8),
        ):
            with T.Kernel(T.ceildiv(total_pairs, block_size), threads=block_size) as block:
                for offset in T.Parallel(block_size):
                    index = block * block_size + offset
                    if index < total_pairs:
                        row = index // (cols // 2)
                        pair_col = index - row * (cols // 2)
                        feature0 = pair_col * 2
                        feature1 = feature0 + 1
                        group = feature0 // group_size
                        scale_value = scale[row, group]
                        inv_scale = T.if_then_else(scale_value > 0.0, 1.0 / scale_value, 0.0)
                        code0 = _fp4_code(x[row, feature0] * inv_scale)
                        code1 = _fp4_code(x[row, feature1] * inv_scale)
                        out[row, pair_col] = code0 + code1 * 16

        grouped_fp4_pack_main.__name__ = (
            f"grouped_fp4_pack_main_rows{rows}_cols{cols}_{input_dtype}_"
            f"g{group_size}_gs0_b{block_size}"
        )
        prim = T.prim_func(grouped_fp4_pack_main)

    def builder():
        return prim

    builder.__name__ = (
        f"grouped_fp4_pack_builder_rows{rows}_cols{cols}_{input_dtype}_"
        f"g{group_size}_gs{int(use_global_scale)}"
    )
    return tilelang.jit(
        out_idx=[],
        target=target,
        pass_configs=pass_configs,
    )(builder)()


def scaled_nvfp4_quant_tilelang(
    inputs: torch.Tensor,
    input_global_scale: torch.Tensor,
    *,
    group_size: int = 16,
    block_size: int = 128,
    target_arch: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize dense activations into packed NVFP4 codes with TileLang packing."""

    require_cuda_tensors(inputs, input_global_scale)
    if input_global_scale.numel() != 1:
        raise XQTBackendError("input_global_scale must be a scalar tensor")
    padded, _ = pad_last_dim_for_group(inputs, group_size=group_size)
    rows, cols = int(padded.shape[0]), int(padded.shape[1])
    groups = cols // int(group_size)
    absmax = padded.to(torch.float32).reshape(rows, groups, int(group_size)).abs().amax(dim=-1)
    global_scale = input_global_scale.to(device=inputs.device, dtype=torch.float32).reshape(())
    scale_raw = absmax * (global_scale / 6.0)
    scale = quantize_nvfp4_scale(scale_raw)
    kernel = build_tilelang_grouped_fp4_pack_kernel(
        rows,
        cols,
        input_dtype=_input_dtype_name(inputs.dtype),
        group_size=int(group_size),
        use_global_scale=True,
        block_size=int(block_size),
        target_arch=target_arch,
    )
    packed = torch.empty((rows, cols // 2), device=inputs.device, dtype=torch.uint8)
    kernel(
        padded.contiguous(),
        scale.to(device=inputs.device, dtype=torch.float32).contiguous(),
        global_scale.reshape(1),
        packed,
    )
    return packed, scale.contiguous()


def scaled_mxfp4_quant_tilelang(
    inputs: torch.Tensor,
    *,
    group_size: int = 32,
    block_size: int = 128,
    target_arch: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize dense activations into packed MXFP4 codes with TileLang packing."""

    require_cuda_tensors(inputs)
    padded, _ = pad_last_dim_for_group(inputs, group_size=group_size)
    rows, cols = int(padded.shape[0]), int(padded.shape[1])
    groups = cols // int(group_size)
    absmax = padded.to(torch.float32).reshape(rows, groups, int(group_size)).abs().amax(dim=-1)
    scale_raw = absmax / 6.0
    scale = quantize_mxfp_scale(scale_raw)
    kernel = build_tilelang_grouped_fp4_pack_kernel(
        rows,
        cols,
        input_dtype=_input_dtype_name(inputs.dtype),
        group_size=int(group_size),
        use_global_scale=False,
        block_size=int(block_size),
        target_arch=target_arch,
    )
    packed = torch.empty((rows, cols // 2), device=inputs.device, dtype=torch.uint8)
    kernel(
        padded.contiguous(),
        scale.to(device=inputs.device, dtype=torch.float32).contiguous(),
        packed,
    )
    return packed, scale.contiguous()


__all__ = [
    "build_tilelang_grouped_fp4_pack_kernel",
    "scaled_mxfp4_quant_reference",
    "scaled_mxfp4_quant_tilelang",
    "scaled_nvfp4_quant_reference",
    "scaled_nvfp4_quant_tilelang",
]
