"""TileLang FlashAttention forward kernel builder.

Vendored from the learning script ``learn/tilelang/flashatt.py`` so that XQT
owns its production kernel source and no longer imports from a learning
directory at runtime.  Only the kernel builder is carried over; the teaching
harness (benchmarking, SDPA comparison, CLI entry) stays in the learning
script.
"""

# TileLang DSL patterns (T.Tensor, T.ceildiv, .astype(), PassConfigKey enum keys
# in @tilelang.jit pass_configs) trigger Pylance false positives; these rules are
# not applicable inside DSL-decorated builder functions.
# pyright: reportInvalidTypeForm=false
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


def build_tilelang_flashatt(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    input_dtype: str = "float16",
) -> Callable[..., torch.Tensor]:
    """Build a fixed-shape FlashAttention forward kernel."""

    if seq_kv < seq_q:
        raise ValueError("seq_kv >= seq_q is required")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("input_dtype must be float16 or bfloat16")

    scale = (1.0 / head_dim) ** 0.5 * 1.44269504
    q_shape = [batch, heads, seq_q, head_dim]
    kv_shape = [batch, heads, seq_kv, head_dim]
    dtype = T.float16 if input_dtype == "float16" else T.bfloat16
    accum_dtype = T.float32
    past_len = seq_kv - seq_q

    @tilelang.jit(
        out_idx=[3],
        pass_configs=PASS_CONFIGS,
    )
    def flashatt():
        @T.prim_func
        def main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            out: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_q, block_m),
                heads,
                batch,
                threads=threads,
            ) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
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
                    T.copy(
                        k[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        k_shared,
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

                    T.copy(
                        v[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        v_shared,
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

    return flashatt()


__all__ = ["PASS_CONFIGS", "build_tilelang_flashatt"]
