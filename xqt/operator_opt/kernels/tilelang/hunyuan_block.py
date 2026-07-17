# pyright: reportInvalidTypeForm=false
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
"""TileLang kernels used by the HunyuanOCR decoder block decode pipeline."""

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_tilelang,
)


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise XQTBackendError("HunyuanOCR TileLang kernels require float16 or bfloat16")


def _validate_same_low_precision_dtype(*tensors: torch.Tensor) -> None:
    require_cuda_tensors(*tensors)
    dtypes = {tensor.dtype for tensor in tensors}
    if len(dtypes) != 1 or next(iter(dtypes)) not in _SUPPORTED_DTYPES:
        raise XQTBackendError(
            "HunyuanOCR TileLang kernels require matching float16 or bfloat16 tensors"
        )


def rmsnorm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Reference last-dimension RMSNorm for Hunyuan decoder blocks."""

    variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
    return (
        hidden_states
        * torch.rsqrt(variance + float(eps)).to(dtype=hidden_states.dtype)
        * weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
    )


def residual_rmsnorm_reference(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference residual add plus RMSNorm with both outputs preserved."""

    residual_out = hidden_states + residual
    return residual_out, rmsnorm_reference(residual_out, weight, eps=eps)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU pointwise activation for Hunyuan decoder blocks."""

    return F.silu(gate) * up


def residual_add_reference(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Reference decoder residual add."""

    return hidden_states + residual


@lru_cache(maxsize=64)
def _build_rmsnorm_kernel(
    rows: int,
    hidden_size: int,
    *,
    input_dtype: str,
    eps: float,
    threads: int,
    target_arch: str | None,
) -> Any:
    tilelang = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or hidden_size <= 0:
        raise ValueError("rows and hidden_size must be positive")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("input_dtype must be float16 or bfloat16")
    dtype = T.float16 if input_dtype == "float16" else T.bfloat16
    target = {"kind": "cuda", "arch": target_arch} if target_arch else None
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    shape = [rows, hidden_size]
    weight_shape = [hidden_size]

    @tilelang.jit(out_idx=[2], target=target, pass_configs=pass_configs)
    def rmsnorm():
        @T.prim_func
        def main(
            hidden_states: T.Tensor(shape, dtype),
            weight: T.Tensor(weight_shape, dtype),
            out: T.Tensor(shape, dtype),
        ):
            with T.Kernel(rows, threads=threads) as row:
                values = T.alloc_fragment([1, hidden_size], T.float32)
                squares = T.alloc_fragment([1, hidden_size], T.float32)
                variance = T.alloc_fragment([1], T.float32)

                for column in T.Parallel(hidden_size):
                    values[0, column] = hidden_states[row, column].astype(T.float32)
                    squares[0, column] = values[0, column] * values[0, column]
                T.reduce_sum(squares, variance, dim=1)

                for column in T.Parallel(hidden_size):
                    out[row, column] = (
                        values[0, column]
                        * T.rsqrt(variance[0] / hidden_size + eps)
                        * weight[column].astype(T.float32)
                    )

        return main

    return rmsnorm()


@lru_cache(maxsize=64)
def _build_residual_rmsnorm_kernel(
    rows: int,
    hidden_size: int,
    *,
    input_dtype: str,
    eps: float,
    threads: int,
    target_arch: str | None,
) -> Any:
    tilelang = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or hidden_size <= 0:
        raise ValueError("rows and hidden_size must be positive")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("input_dtype must be float16 or bfloat16")
    dtype = T.float16 if input_dtype == "float16" else T.bfloat16
    target = {"kind": "cuda", "arch": target_arch} if target_arch else None
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    shape = [rows, hidden_size]
    weight_shape = [hidden_size]

    @tilelang.jit(out_idx=[3, 4], target=target, pass_configs=pass_configs)
    def residual_rmsnorm():
        @T.prim_func
        def main(
            hidden_states: T.Tensor(shape, dtype),
            residual: T.Tensor(shape, dtype),
            weight: T.Tensor(weight_shape, dtype),
            residual_out: T.Tensor(shape, dtype),
            normed_out: T.Tensor(shape, dtype),
        ):
            with T.Kernel(rows, threads=threads) as row:
                values = T.alloc_fragment([1, hidden_size], T.float32)
                squares = T.alloc_fragment([1, hidden_size], T.float32)
                variance = T.alloc_fragment([1], T.float32)

                for column in T.Parallel(hidden_size):
                    values[0, column] = hidden_states[row, column].astype(
                        T.float32
                    ) + residual[row, column].astype(T.float32)
                    squares[0, column] = values[0, column] * values[0, column]
                T.reduce_sum(squares, variance, dim=1)

                for column in T.Parallel(hidden_size):
                    residual_out[row, column] = values[0, column]
                    normed_out[row, column] = (
                        values[0, column]
                        * T.rsqrt(variance[0] / hidden_size + eps)
                        * weight[column].astype(T.float32)
                    )

        return main

    return residual_rmsnorm()


@lru_cache(maxsize=64)
def _build_swiglu_kernel(
    rows: int,
    intermediate_size: int,
    *,
    input_dtype: str,
    threads: int,
    target_arch: str | None,
) -> Any:
    tilelang = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or intermediate_size <= 0:
        raise ValueError("rows and intermediate_size must be positive")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("input_dtype must be float16 or bfloat16")
    dtype = T.float16 if input_dtype == "float16" else T.bfloat16
    target = {"kind": "cuda", "arch": target_arch} if target_arch else None
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    shape = [rows, intermediate_size]
    total = rows * intermediate_size

    @tilelang.jit(out_idx=[2], target=target, pass_configs=pass_configs)
    def swiglu():
        @T.prim_func
        def main(
            gate: T.Tensor(shape, dtype),
            up: T.Tensor(shape, dtype),
            out: T.Tensor(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as block:
                for offset in T.Parallel(threads):
                    index = block * threads + offset
                    if index < total:
                        row = index // intermediate_size
                        column = index - row * intermediate_size
                        gate_value = gate[row, column].astype(T.float32)
                        up_value = up[row, column].astype(T.float32)
                        out[row, column] = gate_value * T.sigmoid(gate_value) * up_value

        return main

    return swiglu()


@lru_cache(maxsize=64)
def _build_residual_add_kernel(
    rows: int,
    hidden_size: int,
    *,
    input_dtype: str,
    threads: int,
    target_arch: str | None,
) -> Any:
    tilelang = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or hidden_size <= 0:
        raise ValueError("rows and hidden_size must be positive")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("input_dtype must be float16 or bfloat16")
    dtype = T.float16 if input_dtype == "float16" else T.bfloat16
    target = {"kind": "cuda", "arch": target_arch} if target_arch else None
    shape = [rows, hidden_size]
    total = rows * hidden_size

    @tilelang.jit(out_idx=[2], target=target)
    def residual_add():
        @T.prim_func
        def main(
            hidden_states: T.Tensor(shape, dtype),
            residual: T.Tensor(shape, dtype),
            out: T.Tensor(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as block:
                for offset in T.Parallel(threads):
                    index = block * threads + offset
                    if index < total:
                        row = index // hidden_size
                        column = index - row * hidden_size
                        out[row, column] = hidden_states[row, column].astype(
                            T.float32
                        ) + residual[row, column].astype(T.float32)

        return main

    return residual_add()


@lru_cache(maxsize=64)
def _build_gqa_decode_kernel(
    batch_size: int,
    query_heads: int,
    key_value_heads: int,
    kv_bucket_size: int,
    head_dim: int,
    *,
    input_dtype: str,
    block_n: int,
    query_tile_rows: int,
    threads: int,
    num_stages: int,
    target_arch: str | None,
) -> Any:
    tilelang = require_tilelang()
    import tilelang.language as T

    if min(batch_size, query_heads, key_value_heads, kv_bucket_size, head_dim) <= 0:
        raise ValueError("GQA decode dimensions must be positive")
    if query_heads % key_value_heads != 0:
        raise ValueError("query_heads must be divisible by key_value_heads")
    if kv_bucket_size % block_n != 0:
        raise ValueError("kv_bucket_size must be divisible by block_n")
    if query_tile_rows <= 0:
        raise ValueError("query_tile_rows must be positive")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("input_dtype must be float16 or bfloat16")
    dtype = T.float16 if input_dtype == "float16" else T.bfloat16
    target = {"kind": "cuda", "arch": target_arch} if target_arch else None
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    q_shape = [batch_size, query_heads, query_tile_rows, head_dim]
    kv_shape = [batch_size, key_value_heads, kv_bucket_size, head_dim]
    group_size = query_heads // key_value_heads
    scale = (1.0 / head_dim) ** 0.5 * 1.4426950408889634

    @tilelang.jit(out_idx=[3], target=target, pass_configs=pass_configs)
    def gqa_decode():
        @T.prim_func
        def main(
            query: T.Tensor(q_shape, dtype),
            key_cache: T.Tensor(kv_shape, dtype),
            value_cache: T.Tensor(kv_shape, dtype),
            out: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(query_heads, batch_size, threads=threads) as (
                query_head,
                batch,
            ):
                key_value_head = query_head // group_size
                query_shared = T.alloc_shared([query_tile_rows, head_dim], dtype)
                key_shared = T.alloc_shared([block_n, head_dim], dtype)
                value_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([query_tile_rows, block_n], T.float32)
                scores_cast = T.alloc_fragment([query_tile_rows, block_n], dtype)
                output = T.alloc_fragment([query_tile_rows, head_dim], T.float32)
                max_score = T.alloc_fragment([query_tile_rows], T.float32)
                previous_max = T.alloc_fragment([query_tile_rows], T.float32)
                scale_previous = T.alloc_fragment([query_tile_rows], T.float32)
                score_sum = T.alloc_fragment([query_tile_rows], T.float32)
                softmax_sum = T.alloc_fragment([query_tile_rows], T.float32)

                T.copy(query[batch, query_head, :, :], query_shared)
                T.fill(output, 0)
                T.fill(softmax_sum, 0)
                T.fill(max_score, -T.infinity(T.float32))

                for kv_tile in T.Pipelined(
                    T.ceildiv(kv_bucket_size, block_n), num_stages=num_stages
                ):
                    T.copy(
                        key_cache[
                            batch,
                            key_value_head,
                            kv_tile * block_n : (kv_tile + 1) * block_n,
                            :,
                        ],
                        key_shared,
                    )
                    T.fill(scores, 0)
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(max_score, previous_max)
                    T.fill(max_score, -T.infinity(T.float32))
                    T.reduce_max(scores, max_score, dim=1, clear=False)
                    for row in T.Parallel(query_tile_rows):
                        max_score[row] = T.max(max_score[row], previous_max[row])
                        scale_previous[row] = T.exp2(
                            previous_max[row] * scale - max_score[row] * scale
                        )
                    for row, column in T.Parallel(query_tile_rows, block_n):
                        scores[row, column] = T.exp2(
                            scores[row, column] * scale - max_score[row] * scale
                        )
                    T.reduce_sum(scores, score_sum, dim=1)
                    for row in T.Parallel(query_tile_rows):
                        softmax_sum[row] = (
                            softmax_sum[row] * scale_previous[row] + score_sum[row]
                        )
                    T.copy(scores, scores_cast)
                    for row, column in T.Parallel(query_tile_rows, head_dim):
                        output[row, column] *= scale_previous[row]
                    T.copy(
                        value_cache[
                            batch,
                            key_value_head,
                            kv_tile * block_n : (kv_tile + 1) * block_n,
                            :,
                        ],
                        value_shared,
                    )
                    T.gemm(
                        scores_cast,
                        value_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for row, column in T.Parallel(query_tile_rows, head_dim):
                    out[batch, query_head, row, column] = (
                        output[row, column] / softmax_sum[row]
                    )

        return main

    return gqa_decode()


def rmsnorm_tilelang(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    threads: int = 256,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run Hunyuan RMSNorm with a TileLang last-dimension reduction kernel."""

    _validate_same_low_precision_dtype(hidden_states, weight)
    if hidden_states.ndim < 2 or weight.ndim != 1:
        raise XQTBackendError(
            "RMSNorm expects hidden_states[..., hidden] and weight[hidden]"
        )
    if hidden_states.shape[-1] != weight.shape[0]:
        raise XQTBackendError("RMSNorm hidden size must match weight size")
    hidden_size = int(weight.shape[0])
    flat = hidden_states.contiguous().reshape(-1, hidden_size)
    kernel = _build_rmsnorm_kernel(
        int(flat.shape[0]),
        hidden_size,
        input_dtype=_dtype_name(flat.dtype),
        eps=float(eps),
        threads=int(threads),
        target_arch=target_arch,
    )
    return kernel(flat, weight.contiguous()).reshape_as(hidden_states)


def residual_rmsnorm_tilelang(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    threads: int = 256,
    target_arch: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a decoder residual add with its following RMSNorm in TileLang."""

    _validate_same_low_precision_dtype(hidden_states, residual, weight)
    if hidden_states.shape != residual.shape or hidden_states.ndim < 2:
        raise XQTBackendError("residual RMSNorm expects matching hidden-state tensors")
    if hidden_states.shape[-1] != weight.shape[0]:
        raise XQTBackendError("residual RMSNorm hidden size must match weight size")
    hidden_size = int(weight.shape[0])
    flat_hidden = hidden_states.contiguous().reshape(-1, hidden_size)
    flat_residual = residual.contiguous().reshape(-1, hidden_size)
    kernel = _build_residual_rmsnorm_kernel(
        int(flat_hidden.shape[0]),
        hidden_size,
        input_dtype=_dtype_name(flat_hidden.dtype),
        eps=float(eps),
        threads=int(threads),
        target_arch=target_arch,
    )
    residual_out, normed_out = kernel(
        flat_hidden,
        flat_residual,
        weight.contiguous(),
    )
    return residual_out.reshape_as(hidden_states), normed_out.reshape_as(hidden_states)


def swiglu_tilelang(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    threads: int = 256,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run Hunyuan SwiGLU pointwise fusion with a TileLang kernel."""

    _validate_same_low_precision_dtype(gate, up)
    if gate.shape != up.shape or gate.ndim < 2:
        raise XQTBackendError(
            "SwiGLU expects matching tensors shaped [..., intermediate]"
        )
    intermediate_size = int(gate.shape[-1])
    flat_gate = gate.contiguous().reshape(-1, intermediate_size)
    flat_up = up.contiguous().reshape(-1, intermediate_size)
    kernel = _build_swiglu_kernel(
        int(flat_gate.shape[0]),
        intermediate_size,
        input_dtype=_dtype_name(flat_gate.dtype),
        threads=int(threads),
        target_arch=target_arch,
    )
    return kernel(flat_gate, flat_up).reshape_as(gate)


def residual_add_tilelang(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    *,
    threads: int = 256,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run the final Hunyuan decoder residual add with a TileLang kernel."""

    _validate_same_low_precision_dtype(hidden_states, residual)
    if hidden_states.shape != residual.shape or hidden_states.ndim < 2:
        raise XQTBackendError("residual add expects matching hidden-state tensors")
    hidden_size = int(hidden_states.shape[-1])
    flat_hidden = hidden_states.contiguous().reshape(-1, hidden_size)
    flat_residual = residual.contiguous().reshape(-1, hidden_size)
    kernel = _build_residual_add_kernel(
        int(flat_hidden.shape[0]),
        hidden_size,
        input_dtype=_dtype_name(flat_hidden.dtype),
        threads=int(threads),
        target_arch=target_arch,
    )
    return kernel(flat_hidden, flat_residual).reshape_as(hidden_states)


def gqa_decode_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    """Reference GQA decode attention for one-token queries and static KV buckets."""

    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise XQTBackendError("GQA decode expects query[B,Hq,D] and cache[B,Hkv,K,D]")
    if key_cache.shape != value_cache.shape:
        raise XQTBackendError("GQA decode key and value caches must share shape")
    batch, query_heads, head_dim = query.shape
    cache_batch, key_value_heads, bucket_size, cache_dim = key_cache.shape
    if batch != cache_batch or head_dim != cache_dim:
        raise XQTBackendError("GQA decode query/cache dimensions do not match")
    if query_heads % key_value_heads != 0:
        raise XQTBackendError("query heads must be divisible by key/value heads")
    groups = query_heads // key_value_heads
    key = key_cache.repeat_interleave(groups, dim=1)
    value = value_cache.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query.unsqueeze(2), key.transpose(-1, -2))
    scores = scores * (head_dim**-0.5)
    probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    return torch.matmul(probabilities, value).squeeze(2)


def gqa_decode_attention_tilelang(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_n: int = 64,
    query_tile_rows: int = 1,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run one-token GQA attention over an exact static KV cache.

    ``query_tile_rows=1`` uses the exact single-token reference path (no query
    padding). ``query_tile_rows=64`` uses the legacy TileLang MMA schedule.
    Intermediate tiles (16/32) are unsupported by the current TileLang layout.
    """

    _validate_same_low_precision_dtype(query, key_cache, value_cache)
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise XQTBackendError("GQA decode expects query[B,Hq,D] and cache[B,Hkv,K,D]")
    if key_cache.shape != value_cache.shape:
        raise XQTBackendError("GQA decode key and value caches must share shape")
    batch, query_heads, head_dim = (int(dim) for dim in query.shape)
    cache_batch, key_value_heads, bucket_size, cache_head_dim = (
        int(dim) for dim in key_cache.shape
    )
    if batch != cache_batch or head_dim != cache_head_dim:
        raise XQTBackendError("GQA decode query/cache dimensions do not match")
    if query_heads % key_value_heads != 0:
        raise XQTBackendError("query heads must be divisible by key/value heads")
    if bucket_size % int(block_n) != 0:
        raise XQTBackendError("KV cache length must align with block_n")
    tile_rows = int(query_tile_rows)
    if tile_rows <= 0:
        raise XQTBackendError("query_tile_rows must be positive")
    if tile_rows == 1:
        return gqa_decode_attention_reference(query, key_cache, value_cache)
    if tile_rows != 64:
        raise XQTBackendError(
            "GQA TileLang schedule currently supports query_tile_rows in {1, 64}"
        )
    kernel = _build_gqa_decode_kernel(
        batch,
        query_heads,
        key_value_heads,
        bucket_size,
        head_dim,
        input_dtype=_dtype_name(query.dtype),
        block_n=int(block_n),
        query_tile_rows=tile_rows,
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
    )
    query_padded = torch.zeros(
        (batch, query_heads, tile_rows, head_dim),
        device=query.device,
        dtype=query.dtype,
    )
    query_padded[:, :, 0, :].copy_(query)
    output = kernel(
        query_padded,
        key_cache.contiguous(),
        value_cache.contiguous(),
    )
    return output[:, :, 0, :]


__all__ = [
    "gqa_decode_attention_reference",
    "gqa_decode_attention_tilelang",
    "residual_add_reference",
    "residual_add_tilelang",
    "residual_rmsnorm_reference",
    "residual_rmsnorm_tilelang",
    "rmsnorm_reference",
    "rmsnorm_tilelang",
    "swiglu_reference",
    "swiglu_tilelang",
]
