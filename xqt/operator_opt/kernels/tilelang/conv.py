"""TileLang Conv operator references and guarded entry points."""

from functools import lru_cache

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)
from xqt.operator_opt.kernels.tilelang.linear import dense_linear_epilogue_tilelang


@lru_cache(maxsize=32)
def build_tilelang_conv1x1_nchw_kernel(
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
    has_bias: bool = False,
):
    import tilelang
    import tilelang.language as T

    if min(batch, in_channels, out_channels, height, width) <= 0:
        raise ValueError("conv1x1 dimensions must be positive")
    if batch * height * width % block_m != 0 or out_channels % block_n != 0:
        raise ValueError(
            "minimal TileLang 1x1 conv builder requires flattened M and out_channels to be block aligned"
        )
    if in_channels % block_k != 0:
        raise ValueError(
            "minimal TileLang 1x1 conv builder requires in_channels to be a multiple of block_k"
        )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    x_shape = [batch, in_channels, height, width]
    weight_shape = [out_channels, in_channels]
    bias_shape = [out_channels]
    out_shape = [batch, out_channels, height, width]
    dtype = T.float16
    accum_dtype = T.float32
    flattened_spatial = batch * height * width

    if has_bias:
        out_idx = [3]
    else:
        out_idx = [2]

    @tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )
    def conv1x1_with_bias():
        @T.prim_func
        def main(
            x: T.Tensor(x_shape, dtype),
            weight: T.Tensor(weight_shape, dtype),
            bias: T.Tensor(bias_shape, dtype),
            out: T.Tensor(out_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(flattened_spatial, block_m),
                T.ceildiv(out_channels, block_n),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared([block_m, block_k], dtype)
                b_shared = T.alloc_shared([block_n, block_k], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.fill(acc_o, 0)
                for k_tile in T.Pipelined(
                    T.ceildiv(in_channels, block_k), num_stages=num_stages
                ):
                    for row_offset, feature_offset in T.Parallel(block_m, block_k):
                        linear_idx = bx * block_m + row_offset
                        batch_idx = linear_idx // (height * width)
                        spatial_idx = linear_idx % (height * width)
                        h_idx = spatial_idx // width
                        w_idx = spatial_idx % width
                        channel_idx = k_tile * block_k + feature_offset
                        a_shared[row_offset, feature_offset] = x[
                            batch_idx, channel_idx, h_idx, w_idx
                        ]
                    T.copy(
                        weight[
                            by * block_n : (by + 1) * block_n,
                            k_tile * block_k : (k_tile + 1) * block_k,
                        ],
                        b_shared,
                    )
                    T.gemm(
                        a_shared,
                        b_shared,
                        acc_o,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                for row_offset, column_offset in T.Parallel(block_m, block_n):
                    linear_idx = bx * block_m + row_offset
                    batch_idx = linear_idx // (height * width)
                    spatial_idx = linear_idx % (height * width)
                    h_idx = spatial_idx // width
                    w_idx = spatial_idx % width
                    out[batch_idx, by * block_n + column_offset, h_idx, w_idx] = (
                        acc_o[row_offset, column_offset]
                        + bias[by * block_n + column_offset]
                    )

        return main

    @tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )
    def conv1x1_without_bias():
        @T.prim_func
        def main(
            x: T.Tensor(x_shape, dtype),
            weight: T.Tensor(weight_shape, dtype),
            out: T.Tensor(out_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(flattened_spatial, block_m),
                T.ceildiv(out_channels, block_n),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared([block_m, block_k], dtype)
                b_shared = T.alloc_shared([block_n, block_k], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.fill(acc_o, 0)
                for k_tile in T.Pipelined(
                    T.ceildiv(in_channels, block_k), num_stages=num_stages
                ):
                    for row_offset, feature_offset in T.Parallel(block_m, block_k):
                        linear_idx = bx * block_m + row_offset
                        batch_idx = linear_idx // (height * width)
                        spatial_idx = linear_idx % (height * width)
                        h_idx = spatial_idx // width
                        w_idx = spatial_idx % width
                        channel_idx = k_tile * block_k + feature_offset
                        a_shared[row_offset, feature_offset] = x[
                            batch_idx, channel_idx, h_idx, w_idx
                        ]
                    T.copy(
                        weight[
                            by * block_n : (by + 1) * block_n,
                            k_tile * block_k : (k_tile + 1) * block_k,
                        ],
                        b_shared,
                    )
                    T.gemm(
                        a_shared,
                        b_shared,
                        acc_o,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                for row_offset, column_offset in T.Parallel(block_m, block_n):
                    linear_idx = bx * block_m + row_offset
                    batch_idx = linear_idx // (height * width)
                    spatial_idx = linear_idx % (height * width)
                    h_idx = spatial_idx // width
                    w_idx = spatial_idx % width
                    out[batch_idx, by * block_n + column_offset, h_idx, w_idx] = acc_o[
                        row_offset, column_offset
                    ]

        return main

    if has_bias:
        return conv1x1_with_bias()
    return conv1x1_without_bias()


def _effective_tile_block(configured: int | None, extent: int, name: str) -> int:
    if extent <= 0:
        raise XQTBackendError(f"{name} extent must be positive")
    if configured is None or int(configured) <= 0:
        configured = 64
    configured = int(configured)
    if extent % configured == 0:
        return configured
    limit = min(configured, extent)
    for candidate in range(limit, 0, -1):
        if extent % candidate == 0:
            return candidate
    raise XQTBackendError(f"could not resolve a valid block size for {name}")


def conv2d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Reference Conv2d path used by operator-family runtime splitting."""

    return F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv2d_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only half Conv2d path lowered through unfold + TileLang GEMM."""

    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    if x.ndim != 4 or weight.ndim != 4:
        raise XQTBackendError("TileLang Conv2d path expects 4D x and weight tensors")
    if groups != 1:
        raise XQTBackendError(
            "TileLang Conv2d GEMM lowering currently supports only groups=1"
        )
    if x.shape[1] != weight.shape[1]:
        raise XQTBackendError(
            "TileLang Conv2d path requires x.shape[1] == weight.shape[1]"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise XQTBackendError("TileLang Conv2d path expects bias shaped [out_channels]")
    stride = (int(stride[0]), int(stride[1]))
    padding = (int(padding[0]), int(padding[1]))
    dilation = (int(dilation[0]), int(dilation[1]))
    kernel_size = (int(weight.shape[2]), int(weight.shape[3]))

    # 1x1 conv with unit stride and no padding can avoid both im2col and layout
    # materialization by reading NCHW directly inside the TileLang kernel.
    if (
        kernel_size == (1, 1)
        and stride == (1, 1)
        and padding == (0, 0)
        and dilation == (1, 1)
    ):
        flat_weight = weight.reshape(
            int(weight.shape[0]), int(weight.shape[1])
        ).contiguous()
        flattened_spatial = int(x.shape[0]) * int(x.shape[2]) * int(x.shape[3])
        resolved_block_m = _effective_tile_block(
            block_m, flattened_spatial, "conv batch-spatial"
        )
        resolved_block_n = _effective_tile_block(
            block_n, int(flat_weight.shape[0]), "conv out_channels"
        )
        resolved_block_k = _effective_tile_block(
            block_k, int(x.shape[1]), "conv reduction"
        )
        require_tilelang()
        kernel = build_tilelang_conv1x1_nchw_kernel(
            batch=int(x.shape[0]),
            in_channels=int(x.shape[1]),
            out_channels=int(weight.shape[0]),
            height=int(x.shape[2]),
            width=int(x.shape[3]),
            block_m=resolved_block_m,
            block_n=resolved_block_n,
            block_k=resolved_block_k,
            threads=int(threads),
            num_stages=int(num_stages),
            target_arch=target_arch,
            has_bias=bias is not None,
        )
        if bias is not None:
            return kernel(
                x.contiguous(),
                flat_weight,
                bias.to(dtype=x.dtype, device=x.device),
            )
        return kernel(
            x.contiguous(),
            flat_weight,
        )

    out_h = (
        int(x.shape[2]) + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    out_w = (
        int(x.shape[3]) + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise XQTBackendError(
            "TileLang Conv2d path computed a non-positive output spatial shape"
        )

    columns = F.unfold(
        x,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    flat_input = columns.transpose(1, 2).contiguous().reshape(-1, int(columns.shape[1]))
    flat_weight = weight.reshape(int(weight.shape[0]), -1).contiguous()
    resolved_block_m = _effective_tile_block(
        block_m, int(flat_input.shape[0]), "conv batch-spatial"
    )
    resolved_block_n = _effective_tile_block(
        block_n, int(flat_weight.shape[0]), "conv out_channels"
    )
    resolved_block_k = _effective_tile_block(
        block_k, int(flat_input.shape[1]), "conv reduction"
    )
    flat_output = dense_linear_epilogue_tilelang(
        flat_input,
        flat_weight,
        bias,
        activation=None,
        block_m=resolved_block_m,
        block_n=resolved_block_n,
        block_k=resolved_block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )
    return (
        flat_output.reshape(int(x.shape[0]), out_h * out_w, int(weight.shape[0]))
        .transpose(1, 2)
        .contiguous()
        .reshape(int(x.shape[0]), int(weight.shape[0]), out_h, out_w)
    )


def conv3d_1x1x1_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Reference Conv3d path used by operator-family runtime splitting."""

    return F.conv3d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv3d_1x1x1_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only half Conv3d 1x1x1 path lowered through TileLang dense GEMM."""

    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    if x.ndim != 5 or weight.ndim != 5:
        raise XQTBackendError("TileLang Conv3d 1x1x1 path expects 5D x and weight tensors")
    if groups != 1:
        raise XQTBackendError(
            "TileLang Conv3d 1x1x1 path currently supports only groups=1"
        )
    if (
        tuple(int(value) for value in weight.shape[2:]) != (1, 1, 1)
        or tuple(int(value) for value in stride) != (1, 1, 1)
        or tuple(int(value) for value in padding) != (0, 0, 0)
        or tuple(int(value) for value in dilation) != (1, 1, 1)
    ):
        raise XQTBackendError(
            "TileLang Conv3d fastpath requires kernel_size=stride=dilation=(1,1,1) and padding=(0,0,0)"
        )
    if x.shape[1] != weight.shape[1]:
        raise XQTBackendError(
            "TileLang Conv3d 1x1x1 path requires x.shape[1] == weight.shape[1]"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise XQTBackendError("TileLang Conv3d 1x1x1 path expects bias shaped [out_channels]")
    flat_input = (
        x.permute(0, 2, 3, 4, 1)
        .contiguous()
        .reshape(-1, int(x.shape[1]))
    )
    flat_weight = weight.reshape(int(weight.shape[0]), int(weight.shape[1])).contiguous()
    padded_in_channels = ((int(flat_input.shape[1]) + 15) // 16) * 16
    channel_padding = padded_in_channels - int(flat_input.shape[1])
    if channel_padding:
        flat_input = F.pad(flat_input, (0, channel_padding))
        flat_weight = F.pad(flat_weight, (0, channel_padding))
    resolved_block_m = _effective_tile_block(
        block_m, int(flat_input.shape[0]), "conv3d batch-spatiotemporal"
    )
    resolved_block_n = _effective_tile_block(
        block_n, int(flat_weight.shape[0]), "conv3d out_channels"
    )
    resolved_block_k = _effective_tile_block(
        block_k, padded_in_channels, "conv3d reduction"
    )
    flat_output = dense_linear_epilogue_tilelang(
        flat_input,
        flat_weight,
        bias,
        activation=None,
        block_m=resolved_block_m,
        block_n=resolved_block_n,
        block_k=resolved_block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )
    return (
        flat_output.reshape(
            int(x.shape[0]),
            int(x.shape[2]),
            int(x.shape[3]),
            int(x.shape[4]),
            int(weight.shape[0]),
        )
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


TILELANG_CONV_KERNEL_METADATA = {
    "conv": {
        "kernel_name": "conv2d_tilelang_half_gemm",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.conv2d",
        "usage": (
            "Standalone half Conv2d path that uses a direct NCHW 1x1 TileLang fastpath "
            "and otherwise falls back to torch unfold/im2col plus TileLang GEMM."
        ),
        "weight_encoding": "dense_fp16",
        "fusion_status": "tilelang_conv_half_gemm",
        "fastpath": "tilelang_conv1x1_nchw_direct",
        "fallback": "torch_unfold_plus_tilelang_half_gemm",
        "supports_grouped_conv": False,
        "epilogue_stage": None,
    },
    "conv3d_1x1x1": {
        "kernel_name": "conv3d_1x1x1_tilelang_half_gemm",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.conv3d",
        "usage": (
            "Standalone half Conv3d 1x1x1 path that flattens BxTxHxW into GEMM M "
            "and routes through the TileLang dense half epilogue kernel."
        ),
        "weight_encoding": "dense_fp16",
        "fusion_status": "tilelang_conv3d_1x1x1_half_gemm",
        "fastpath": "tilelang_conv3d_1x1x1_bthwc_gemm",
        "fallback": "eager_torch_conv3d",
        "supports_grouped_conv": False,
        "requires_kernel_size": [1, 1, 1],
        "requires_stride": [1, 1, 1],
        "requires_padding": [0, 0, 0],
        "requires_dilation": [1, 1, 1],
        "epilogue_stage": None,
    },
}

__all__ = [
    "TILELANG_CONV_KERNEL_METADATA",
    "build_tilelang_conv1x1_nchw_kernel",
    "conv2d_reference",
    "conv2d_tilelang",
    "conv3d_1x1x1_reference",
    "conv3d_1x1x1_tilelang",
]
