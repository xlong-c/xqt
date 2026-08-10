"""TileLang SVDQuant fused kernel (FUSE_DOWN / FUSE_UP).

单个 TileLang kernel 完成 SVDQuant 双分支数据流:

  y = dequant_gemm(x, packed_int4_residual, groupwise_scale)
      + up_proj(down_proj(x)) + bias

- FUSE_DOWN: activation tile 只从 global memory 读入 shared memory 一次,
  同一份 ``a_shared`` 同时喂给主 dequant GEMM 和 down projection GEMM.
- FUSE_UP: up projection GEMM 直接累加进主 dequant GEMM 的 fp32
  accumulator, bias 在同一 epilogue 加.

数值契约: fp16 输入与权重, packed uint8 signed INT4 residual,
fp32 groupwise scale 在 kernel 内转 fp16 参与反量化, fp32 累加,
fp16 输出. rank 不足 16 倍数时在 host 侧零填充到 16 的倍数
(Tensor Core mma 片段对齐), 零填充对结果无贡献.
"""

# TileLang DSL patterns (T.Tensor, T.ceildiv, .astype(), PassConfigKey enum keys
# in @tilelang.jit pass_configs) trigger Pylance false positives; these rules are
# not applicable inside DSL-decorated builder functions.
# pyright: reportInvalidTypeForm=false
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

from dataclasses import asdict, dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)

# mma 片段对齐: 实测 TileLang 0.1.12 T.gemm 在 N/K 非 16 倍数时拒绝编译
# (rank=4) 或数值错误 (rank=8), 因此 rank 统一零填充到 16 的倍数.
_RANK_ALIGNMENT = 16


@dataclass(frozen=True)
class SVDQuantFusedSchedule:
    """Launch parameters for the promoted SVDQuant CUDA path.

    The schedule is deliberately a small immutable value object so runtime
    dispatch can cache/inspect it without retaining tensors or compiled
    TileLang handles.
    """

    block_m: int = 64
    block_n: int = 64
    block_k: int = 64
    threads: int = 128
    num_stages: int = 2

    def to_dict(self) -> dict[str, int]:
        """Return the launch parameters as a plain mapping."""

        return {key: int(value) for key, value in asdict(self).items()}


def resolve_svd_fused_schedule(
    m: int,
    n: int,
    input_features: int,
    rank: int,
    *,
    target_arch: str | None = None,
) -> tuple[SVDQuantFusedSchedule | None, str]:
    """Resolve the measured CUDA promotion gate for one SVD shape.

    On the local Ada ``sm_89`` target, the fused dequant/low-rank kernel wins
    for decode and short-prefill batches, while large prefill batches are
    faster with the cached-dequant cuBLAS reference path.  Keep that choice
    explicit instead of silently promoting a slower single kernel.  The
    direct :func:`svd_fused_dequant_gemm_low_rank_tilelang` entry remains
    available for explicit experiments on other shapes/architectures.
    """

    m_value = int(m)
    n_value = int(n)
    k_value = int(input_features)
    rank_value = int(rank)
    if min(m_value, n_value, k_value, rank_value) <= 0:
        raise ValueError("m, n, input_features, and rank must be positive")

    arch = None if target_arch is None else str(target_arch).lower()
    if arch in {"sm_89", "sm89"}:
        # R-008 evidence on RTX 4070 Ti SUPER: 64x64x64/128 threads is the
        # winning short-prefill schedule; M=1024,N=2048,K=2048 regresses by
        # about 31% because each CTA carries the fused low-rank accumulator.
        if m_value > 256:
            return None, "sm_89 promotion gate keeps M>256 on cached-dequant reference"
        if n_value > 2048 and m_value > 128:
            return None, (
                "sm_89 promotion gate keeps wide N with M>128 on cached-dequant reference"
            )

    del rank_value
    return SVDQuantFusedSchedule(), "promoted_svd_fused_schedule"

TILELANG_SVD_FUSED_KERNEL_METADATA = {
    "svd_fused_dequant_gemm_low_rank": {
        "kernel_name": "svd_fused_dequant_gemm_low_rank",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "SVDQuantLinear.forward (dequant + fp16 matmul + 两个低秩 GEMM)",
        "usage": (
            "SVDQuantLinear 单 kernel 融合: packed INT4 residual groupwise "
            "dequant GEMM + down/up 低秩分支 + bias epilogue."
        ),
        "fuse_down": "activation tile 单次读入 shared memory, 同时喂主 GEMM 与 down GEMM",
        "fuse_up": "up GEMM 与 bias 共享主 dequant GEMM 的 fp32 accumulator",
    },
}


def _pad_rank_to_alignment(rank: int) -> int:
    """把 rank 向上对齐到 mma 片段要求的 16 的倍数."""

    return ((int(rank) + _RANK_ALIGNMENT - 1) // _RANK_ALIGNMENT) * _RANK_ALIGNMENT


@lru_cache(maxsize=32)
def build_tilelang_svd_fused_kernel(
    m: int,
    n: int,
    input_features: int,
    group_size: int,
    rank: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
    has_bias: bool = False,
):
    """构建 SVDQuant FUSE_DOWN/FUSE_UP 单 kernel.

    ``rank`` 必须是 16 的倍数 (调用方负责零填充). 要求 ``m % block_m == 0``,
    ``n % block_n == 0``, ``input_features % block_k == 0``.
    """

    import tilelang
    import tilelang.language as T

    if m <= 0 or n <= 0 or input_features <= 0 or group_size <= 0 or rank <= 0:
        raise ValueError("m, n, input_features, group_size, and rank must be positive")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            "minimal fused SVD TileLang kernel requires m,n to be multiples of block sizes"
        )
    if input_features % block_k != 0:
        raise ValueError(
            "minimal fused SVD TileLang kernel requires input_features to be a multiple of block_k"
        )
    if rank % _RANK_ALIGNMENT != 0:
        raise ValueError(
            "fused SVD TileLang kernel requires rank padded to a multiple of 16"
        )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    padded_input_features = (
        (input_features + group_size - 1) // group_size
    ) * group_size
    packed_k = (padded_input_features + 1) // 2
    groups = padded_input_features // group_size
    a_shape = [m, input_features]
    packed_shape = [n, packed_k]
    scale_shape = [n, groups]
    down_shape = [rank, input_features]
    up_shape = [n, rank]
    bias_shape = [n]
    c_shape = [m, n]
    dtype = T.float16
    accum_dtype = T.float32

    if has_bias:
        out_idx = [6]
    else:
        out_idx = [5]
    shape_suffix = (
        f"m{m}_n{n}_k{input_features}_g{group_size}_r{rank}_bm{block_m}_"
        f"bn{block_n}_bk{block_k}_t{threads}_s{num_stages}_{target_arch or 'auto'}"
    )

    def tilelang_svd_fused_with_bias_main(
        a: T.Tensor(a_shape, dtype),
        packed_weight: T.Tensor(packed_shape, T.uint8),
        scale: T.Tensor(scale_shape, dtype),
        down_weight: T.Tensor(down_shape, dtype),
        up_weight: T.Tensor(up_shape, dtype),
        bias: T.Tensor(bias_shape, dtype),
        out: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], dtype)
            b_shared = T.alloc_shared([block_n, block_k], dtype)
            d_shared = T.alloc_shared([rank, block_k], dtype)
            h_shared = T.alloc_shared([block_m, rank], dtype)
            u_shared = T.alloc_shared([block_n, rank], dtype)
            o_shared = T.alloc_shared([block_m, block_n], dtype)
            acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)
            h_acc = T.alloc_fragment([block_m, rank], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(h_acc, 0)
            for k_tile in T.Pipelined(
                T.ceildiv(input_features, block_k), num_stages=num_stages
            ):
                # FUSE_DOWN: 同一 k tile 的 activation 只从 global memory 读一次,
                # a_shared 同时服务主 dequant GEMM 与 down projection GEMM.
                T.copy(
                    a[
                        bx * block_m : (bx + 1) * block_m,
                        k_tile * block_k : (k_tile + 1) * block_k,
                    ],
                    a_shared,
                )
                T.copy(
                    down_weight[:, k_tile * block_k : (k_tile + 1) * block_k],
                    d_shared,
                )
                for row_offset, feature_offset in T.Parallel(block_n, block_k):
                    row = by * block_n + row_offset
                    feature = k_tile * block_k + feature_offset
                    byte_u16 = packed_weight[row, feature // 2].astype(T.uint16)
                    low = T.bitwise_and(byte_u16, 15)
                    high = T.bitwise_and(T.shift_right(byte_u16, 4), 15)
                    nibble = T.if_then_else(feature % 2 == 0, low, high)
                    signed = T.if_then_else(
                        nibble >= 8,
                        nibble.astype(T.int16) - 16,
                        nibble.astype(T.int16),
                    )
                    b_shared[row_offset, feature_offset] = (
                        signed.astype(dtype) * scale[row, feature // group_size]
                    )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    a_shared,
                    d_shared,
                    h_acc,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            # FUSE_UP: h = down(x) 不再写回 global memory, up GEMM 直接累加进
            # 主 dequant GEMM 的 fp32 accumulator, bias 在同一 epilogue 加.
            T.copy(h_acc, h_shared)
            T.copy(up_weight[by * block_n : (by + 1) * block_n, :], u_shared)
            T.gemm(
                h_shared,
                u_shared,
                acc_o,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for row_offset, column_offset in T.Parallel(block_m, block_n):
                acc_o[row_offset, column_offset] = (
                    acc_o[row_offset, column_offset]
                    + bias[by * block_n + column_offset].astype(accum_dtype)
                )
            T.copy(acc_o, o_shared)
            T.copy(
                o_shared,
                out[
                    bx * block_m : (bx + 1) * block_m,
                    by * block_n : (by + 1) * block_n,
                ],
            )

    tilelang_svd_fused_with_bias_main.__name__ = (
        f"tilelang_svd_fused_with_bias_main_{shape_suffix}"
    )
    tilelang_svd_fused_with_bias_prim = T.prim_func(tilelang_svd_fused_with_bias_main)

    def fused_with_bias():
        return tilelang_svd_fused_with_bias_prim

    fused_with_bias.__name__ = f"tilelang_svd_fused_with_bias_builder_{shape_suffix}"
    fused_with_bias_jit = tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )(fused_with_bias)

    def tilelang_svd_fused_without_bias_main(
        a: T.Tensor(a_shape, dtype),
        packed_weight: T.Tensor(packed_shape, T.uint8),
        scale: T.Tensor(scale_shape, dtype),
        down_weight: T.Tensor(down_shape, dtype),
        up_weight: T.Tensor(up_shape, dtype),
        out: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], dtype)
            b_shared = T.alloc_shared([block_n, block_k], dtype)
            d_shared = T.alloc_shared([rank, block_k], dtype)
            h_shared = T.alloc_shared([block_m, rank], dtype)
            u_shared = T.alloc_shared([block_n, rank], dtype)
            o_shared = T.alloc_shared([block_m, block_n], dtype)
            acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)
            h_acc = T.alloc_fragment([block_m, rank], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(h_acc, 0)
            for k_tile in T.Pipelined(
                T.ceildiv(input_features, block_k), num_stages=num_stages
            ):
                T.copy(
                    a[
                        bx * block_m : (bx + 1) * block_m,
                        k_tile * block_k : (k_tile + 1) * block_k,
                    ],
                    a_shared,
                )
                T.copy(
                    down_weight[:, k_tile * block_k : (k_tile + 1) * block_k],
                    d_shared,
                )
                for row_offset, feature_offset in T.Parallel(block_n, block_k):
                    row = by * block_n + row_offset
                    feature = k_tile * block_k + feature_offset
                    byte_u16 = packed_weight[row, feature // 2].astype(T.uint16)
                    low = T.bitwise_and(byte_u16, 15)
                    high = T.bitwise_and(T.shift_right(byte_u16, 4), 15)
                    nibble = T.if_then_else(feature % 2 == 0, low, high)
                    signed = T.if_then_else(
                        nibble >= 8,
                        nibble.astype(T.int16) - 16,
                        nibble.astype(T.int16),
                    )
                    b_shared[row_offset, feature_offset] = (
                        signed.astype(dtype) * scale[row, feature // group_size]
                    )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    a_shared,
                    d_shared,
                    h_acc,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            T.copy(h_acc, h_shared)
            T.copy(up_weight[by * block_n : (by + 1) * block_n, :], u_shared)
            T.gemm(
                h_shared,
                u_shared,
                acc_o,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(acc_o, o_shared)
            T.copy(
                o_shared,
                out[
                    bx * block_m : (bx + 1) * block_m,
                    by * block_n : (by + 1) * block_n,
                ],
            )

    tilelang_svd_fused_without_bias_main.__name__ = (
        f"tilelang_svd_fused_without_bias_main_{shape_suffix}"
    )
    tilelang_svd_fused_without_bias_prim = T.prim_func(
        tilelang_svd_fused_without_bias_main
    )

    def fused_without_bias():
        return tilelang_svd_fused_without_bias_prim

    fused_without_bias.__name__ = (
        f"tilelang_svd_fused_without_bias_builder_{shape_suffix}"
    )
    fused_without_bias_jit = tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )(fused_without_bias)

    if has_bias:
        return fused_with_bias_jit()
    return fused_without_bias_jit()


def _resolve_cuda_target_arch(tensor: torch.Tensor, target_arch: str | None) -> str | None:
    if target_arch is not None:
        return str(target_arch)
    major, minor = torch.cuda.get_device_capability(tensor.device)
    return f"sm_{major}{minor}"


def svd_fused_dequant_gemm_low_rank_tilelang(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only SVDQuant fused entry: dequant GEMM + low-rank + bias 单 kernel.

    维度不满足 block 对齐时显式抛 ``XQTBackendError``, 不做静默 fallback.
    """

    tensors = (x, packed_weight, scale, down_weight, up_weight)
    if bias is not None:
        tensors = (*tensors, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(x, down_weight, up_weight)
    if packed_weight.dtype != torch.uint8:
        raise XQTBackendError("fused SVD TileLang path expects uint8 packed_weight")
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise XQTBackendError("fused SVD TileLang path expects 2D x and packed_weight")
    if x.shape[1] != int(input_features):
        raise XQTBackendError(
            "fused SVD TileLang path requires x.shape[1] == input_features"
        )
    rank = int(down_weight.shape[0])
    if down_weight.shape != (rank, int(input_features)):
        raise XQTBackendError(
            "fused SVD TileLang path expects down_weight shaped [rank, input_features]"
        )
    if up_weight.shape != (packed_weight.shape[0], rank):
        raise XQTBackendError(
            "fused SVD TileLang path expects up_weight shaped [out_features, rank]"
        )
    if int(group_size) <= 0:
        raise XQTBackendError("group_size must be positive")
    padded_input_features = (
        (int(input_features) + int(group_size) - 1) // int(group_size)
    ) * int(group_size)
    if int(packed_weight.shape[1]) * 2 != padded_input_features:
        raise XQTBackendError(
            "packed_weight must exactly match the group-padded input_features extent"
        )
    scale_2d = scale
    if scale_2d.ndim == 3 and scale_2d.shape[2] == 1:
        scale_2d = scale_2d.squeeze(-1)
    if (
        scale_2d.ndim != 2
        or scale_2d.shape[0] != packed_weight.shape[0]
        or scale_2d.shape[1] * int(group_size) != padded_input_features
    ):
        raise XQTBackendError(
            "fused SVD TileLang path expects scale shaped [out_features, groups] "
            "with groups exactly covering the packed padded input features"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != packed_weight.shape[0]):
        raise XQTBackendError("bias must be 1D and match packed_weight out_features")
    if x.shape[0] % int(block_m) != 0 or packed_weight.shape[0] % int(block_n) != 0:
        raise XQTBackendError(
            "minimal fused SVD TileLang path requires batch and out_features to be multiples of block sizes"
        )
    if int(input_features) % int(block_k) != 0:
        raise XQTBackendError(
            "minimal fused SVD TileLang path requires input_features to be a multiple of block_k"
        )
    require_tilelang()

    padded_rank = _pad_rank_to_alignment(rank)
    if padded_rank != rank:
        # 零填充 rank 到 mma 片段对齐; 零行/列对累加结果无贡献.
        down_padded = F.pad(down_weight, (0, 0, 0, padded_rank - rank))
        up_padded = F.pad(up_weight, (0, padded_rank - rank))
    else:
        down_padded = down_weight.contiguous()
        up_padded = up_weight.contiguous()

    kernel = build_tilelang_svd_fused_kernel(
        m=int(x.shape[0]),
        n=int(packed_weight.shape[0]),
        input_features=int(input_features),
        group_size=int(group_size),
        rank=padded_rank,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=_resolve_cuda_target_arch(x, target_arch),
        has_bias=bias is not None,
    )
    tilelang_scale = scale_2d.to(device=x.device, dtype=torch.float16).contiguous()
    if bias is not None:
        tilelang_bias = bias.to(device=x.device, dtype=torch.float16).contiguous()
        return kernel(
            x, packed_weight, tilelang_scale, down_padded, up_padded, tilelang_bias
        )
    return kernel(x, packed_weight, tilelang_scale, down_padded, up_padded)


__all__ = [
    "TILELANG_SVD_FUSED_KERNEL_METADATA",
    "SVDQuantFusedSchedule",
    "build_tilelang_svd_fused_kernel",
    "resolve_svd_fused_schedule",
    "svd_fused_dequant_gemm_low_rank_tilelang",
]
