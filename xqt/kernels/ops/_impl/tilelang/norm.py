# pyright: reportInvalidTypeForm=false
"""TileLang Norm operator references and guarded entry points."""

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)


def layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference LayerNorm path used by the TileLang backend."""

    normalized_shape = (int(weight.shape[0]),)
    ref_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return F.layer_norm(
        x,
        normalized_shape,
        weight.to(dtype=x.dtype, device=x.device),
        ref_bias,
        eps=float(eps),
    )


@lru_cache(maxsize=64)
def _build_tilelang_layer_norm_kernel(
    rows: int,
    hidden_dim: int,
    eps: float,
    threads: int,
) -> Any:
    tilelang: Any = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or hidden_dim <= 0:
        raise ValueError("rows and hidden_dim must be positive")
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    x_shape = [rows, hidden_dim]
    w_shape = [hidden_dim]
    dtype = T.float16
    accum_dtype = T.float32

    @tilelang.jit(
        out_idx=[3],
        pass_configs=pass_configs,
    )
    def layer_norm():
        @T.prim_func
        def main(
            x: T.Tensor(x_shape, dtype),
            weight: T.Tensor(w_shape, dtype),
            bias: T.Tensor(w_shape, dtype),
            out: T.Tensor(x_shape, dtype),
        ):
            with T.Kernel(rows, threads=threads) as row:
                x_frag = T.alloc_fragment([1, hidden_dim], accum_dtype)
                squared = T.alloc_fragment([1, hidden_dim], accum_dtype)
                mean = T.alloc_fragment([1], accum_dtype)
                variance = T.alloc_fragment([1], accum_dtype)

                for col in T.Parallel(hidden_dim):
                    x_frag[0, col] = x[row, col]

                T.reduce_sum(x_frag, mean, dim=1)

                for col in T.Parallel(hidden_dim):
                    x_frag[0, col] = x_frag[0, col] - mean[0] / hidden_dim
                    squared[0, col] = x_frag[0, col] * x_frag[0, col]

                T.reduce_sum(squared, variance, dim=1)

                for col in T.Parallel(hidden_dim):
                    out[row, col] = (
                        x_frag[0, col]
                        * T.rsqrt(variance[0] / hidden_dim + eps)
                        * weight[col]
                        + bias[col]
                    )

        return main

    return layer_norm()


def layer_norm_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
    threads: int = 64,
) -> torch.Tensor:
    """CUDA-only half LayerNorm path backed by a TileLang reduction kernel."""

    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    if x.shape[-1] != weight.shape[0]:
        raise XQTBackendError("TileLang LayerNorm path requires x.shape[-1] == weight.shape[0]")
    if bias is not None and bias.shape != weight.shape:
        raise XQTBackendError("TileLang LayerNorm path requires bias and weight to share shape")
    hidden_dim = int(weight.shape[0])
    x_2d = x.contiguous().reshape(-1, hidden_dim)
    ref_bias = bias if bias is not None else torch.zeros_like(weight)
    kernel = _build_tilelang_layer_norm_kernel(
        rows=int(x_2d.shape[0]),
        hidden_dim=hidden_dim,
        eps=float(eps),
        threads=int(threads),
    )
    out = kernel(
        x_2d,
        weight.contiguous(),
        ref_bias.contiguous(),
    )
    return out.reshape_as(x)



TILELANG_NORM_KERNEL_METADATA = {
    "norm": {
        "kernel_name": "layer_norm_half_reduce_sum",
        "threads": 64,
        "baseline": "torch.nn.functional.layer_norm",
        "usage": "Standalone half LayerNorm path for direct TileLang operator benchmarking.",
        "weight_encoding": "dense_fp16",
        "fusion_status": "tilelang_reduce_sum_layer_norm",
        "normalized_last_dim_only": True,
        "epilogue_stage": None,
    },
}

__all__ = [
    "TILELANG_NORM_KERNEL_METADATA",
    "layer_norm_reference",
    "layer_norm_tilelang",
]
