"""TileLang KV-int8 fused attention kernel.

该模块提供真正消费 int8 存储 K/V 的融合 attention CUDA kernel:
K/V 以 int8 tensor 加 per-tensor scale 传入, kernel 在把 tile 搬进
shared memory 后立刻做 int8 -> fp32 * scale -> fp16 的 dequant, 再进入
flash-attention 风格的 online softmax mainloop. dequant 发生在 kernel
内部, 不需要在 host 侧先物化 fp16 K/V.

量化语义与 `xqt.runtime.modules.kv_attention.KvScaleAttention` 的
`_quantize_kv` 完全一致: round(x / scale).clamp(-qmax, qmax).to(int8),
dequant 为 int8 * scale. 本模块只做 attention 本身, 不涉及 page table,
KV cache 管理或 serving 调度.

注意: 本文件不能使用 `from __future__ import annotations`, TileLang eager
builder 通过 get_type_hints 求值 prim_func 注解, 字符串化注解会丢失闭包内的
shape/dtype 变量.
"""

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)

#: 已在本机 (sm_89) 验证的 fused kernel 名, report/contract 引用此常量.
KV_INT8_FUSED_KERNEL_NAME = "tilelang_kv_int8_fused_attention"

_REFERENCE_KERNEL_NAME = "torch_sdpa_kv_int8_dequant_reference"


def _validate_kv_int8_inputs(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
) -> None:
    """校验 fused kernel 的输入形状与 dtype 契约."""

    if q.ndim != 4 or k_int8.ndim != 4 or v_int8.ndim != 4:
        raise XQTBackendError(
            "KV-int8 attention expects 4D tensors shaped [batch, heads, seq, head_dim]"
        )
    if k_int8.dtype != torch.int8 or v_int8.dtype != torch.int8:
        raise XQTBackendError("KV-int8 attention requires int8 K/V storage tensors")
    if q.shape[0] != k_int8.shape[0] or q.shape[0] != v_int8.shape[0]:
        raise XQTBackendError("KV-int8 attention requires matching batch size for q, k, v")
    if q.shape[1] != k_int8.shape[1] or q.shape[1] != v_int8.shape[1]:
        raise XQTBackendError("KV-int8 attention requires matching head count for q, k, v")
    if q.shape[3] != k_int8.shape[3] or q.shape[3] != v_int8.shape[3]:
        raise XQTBackendError("KV-int8 attention requires matching head_dim for q, k, v")
    if k_int8.shape[2] != v_int8.shape[2]:
        raise XQTBackendError("KV-int8 attention requires matching key/value sequence length")
    if k_int8.shape[2] < q.shape[2]:
        raise XQTBackendError("KV-int8 attention currently requires seq_kv >= seq_q")
    if q.shape[3] % 16 != 0:
        raise XQTBackendError(
            "KV-int8 attention requires head_dim to be a multiple of 16 for fp16 tensor-core GEMM"
        )


@lru_cache(maxsize=32)
def build_tilelang_kv_int8_attention_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 2,
    threads: int = 128,
) -> Any:
    """编译 (并按形状缓存) KV-int8 fused attention TileLang kernel.

    mainloop 结构沿用 `learn/tilelang/flashatt.py` 的 flash-attention 设计,
    差异在于 K/V 输入为 int8, 每个 kv tile 先经 `T.copy` 进入 int8 shared,
    再在 kernel 内 dequant 为 fp16 shared 后参与 GEMM.
    """

    require_tilelang()
    import tilelang
    import tilelang.language as T

    scale = (1.0 / head_dim) ** 0.5 * 1.44269504
    q_shape = [batch, heads, seq_q, head_dim]
    kv_shape = [batch, heads, seq_kv, head_dim]
    dtype = T.float16
    int8_dtype = "int8"
    accum_dtype = T.float32
    past_len = seq_kv - seq_q

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

    @tilelang.jit(out_idx=[5], pass_configs=pass_configs)
    def kernel() -> Any:
        @T.prim_func
        def main(
            q: T.Tensor(q_shape, dtype),
            k_int8: T.Tensor(kv_shape, int8_dtype),
            v_int8: T.Tensor(kv_shape, int8_dtype),
            k_scale: T.float32,
            v_scale: T.float32,
            out: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_q, block_m),
                heads,
                batch,
                threads=threads,
            ) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, head_dim], dtype)
                k_shared_int8 = T.alloc_shared([block_n, head_dim], int8_dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                v_shared_int8 = T.alloc_shared([block_n, head_dim], int8_dtype)
                v_shared = T.alloc_shared([block_n, head_dim], dtype)

                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, head_dim], accum_dtype)

                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.copy(q[bz, by, bx * block_m : (bx + 1) * block_m, :], q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(
                        T.ceildiv(seq_kv, block_n),
                        T.ceildiv((bx + 1) * block_m + past_len, block_n),
                    )
                    if causal
                    else T.ceildiv(seq_kv, block_n)
                )

                for kv_tile in T.Pipelined(loop_range, num_stages=num_stages):
                    # int8 K tile 进 shared, kernel 内 dequant 为 fp16.
                    T.copy(
                        k_int8[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        k_shared_int8,
                    )
                    for i, j in T.Parallel(block_n, head_dim):
                        k_shared[i, j] = T.cast(
                            T.cast(k_shared_int8[i, j], accum_dtype) * k_scale,
                            dtype,
                        )

                    if causal:
                        for i, j in T.Parallel(block_m, block_n):
                            q_idx = bx * block_m + i + past_len
                            k_idx = kv_tile * block_n + j
                            acc_s[i, j] = T.if_then_else(
                                q_idx >= k_idx,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(
                                kv_tile * block_n + j >= seq_kv,
                                -T.infinity(acc_s.dtype),
                                0,
                            )

                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale
                        )

                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_m, head_dim):
                        acc_o[i, j] *= scores_scale[i]

                    # int8 V tile 进 shared, kernel 内 dequant 为 fp16.
                    T.copy(
                        v_int8[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        v_shared_int8,
                    )
                    for i, j in T.Parallel(block_n, head_dim):
                        v_shared[i, j] = T.cast(
                            T.cast(v_shared_int8[i, j], accum_dtype) * v_scale,
                            dtype,
                        )
                    T.gemm(
                        acc_s_cast,
                        v_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, head_dim):
                    acc_o[i, j] /= logsum[i]

                T.copy(acc_o, out[bz, by, bx * block_m : (bx + 1) * block_m, :])

        return main

    return kernel()


def kv_int8_attention_dequant_reference(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: float,
    v_scale: float,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """参考路径: int8 K/V 先 dequant 为 q 的 dtype, 再走 torch SDPA.

    dequant 语义为 int8 -> dtype * scale, 与 fused kernel 内的
    int8 -> fp32 * scale -> fp16 在 fp16 舍入级别一致.
    """

    k = k_int8.to(dtype=q.dtype) * float(k_scale)
    v = v_int8.to(dtype=q.dtype) * float(v_scale)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def fused_kv_int8_attention_forward_tilelang(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: float,
    v_scale: float,
    *,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
) -> torch.Tensor:
    """CUDA-only KV-int8 fused attention 入口.

    输入契约: q 为 fp16 CUDA tensor [batch, heads, seq_q, head_dim];
    k_int8/v_int8 为 int8 CUDA tensor [batch, heads, seq_kv, head_dim];
    k_scale/v_scale 为 per-tensor fp32 scale (host 标量). dequant 在
    kernel 内完成, 不物化完整 fp16 K/V.
    """

    require_cuda_tensors(q, k_int8, v_int8)
    require_fp16_tensors(q)
    _validate_kv_int8_inputs(q, k_int8, v_int8)
    kernel = build_tilelang_kv_int8_attention_kernel(
        batch=int(q.shape[0]),
        heads=int(q.shape[1]),
        seq_q=int(q.shape[2]),
        seq_kv=int(k_int8.shape[2]),
        head_dim=int(q.shape[3]),
        causal=bool(causal),
        block_m=int(block_m),
        block_n=int(block_n),
        num_stages=int(num_stages),
        threads=int(threads),
    )
    return kernel(q, k_int8, v_int8, float(k_scale), float(v_scale))


TILELANG_KV_INT8_ATTENTION_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "kv_int8_attention": {
        "kernel_name": KV_INT8_FUSED_KERNEL_NAME,
        "block_m": 64,
        "block_n": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": _REFERENCE_KERNEL_NAME,
        "kv_storage": "int8 per-tensor scale, in-kernel dequant to fp16",
        "source_mainloop": "learn/tilelang/flashatt.py",
    },
}

__all__ = [
    "KV_INT8_FUSED_KERNEL_NAME",
    "TILELANG_KV_INT8_ATTENTION_KERNEL_METADATA",
    "build_tilelang_kv_int8_attention_kernel",
    "fused_kv_int8_attention_forward_tilelang",
    "kv_int8_attention_dequant_reference",
]
