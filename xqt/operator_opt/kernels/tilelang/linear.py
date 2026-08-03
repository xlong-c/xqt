"""TileLang Linear operator references and guarded entry points."""

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)
from xqt.operator_opt.kernels.tilelang.gemm_builder import build_tilelang_gemm_kernel


def dense_linear_epilogue_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference dense Linear with optional bias and activation epilogue."""

    output = F.linear(
        x,
        weight.to(dtype=x.dtype, device=x.device),
        None if bias is None else bias.to(dtype=x.dtype, device=x.device),
    )
    if activation is None:
        return output
    if activation == "gelu":
        return F.gelu(output)
    if activation == "silu":
        return F.silu(output)
    if activation == "relu":
        return F.relu(output)
    raise ValueError(f"unsupported activation: {activation}")


def dense_linear_epilogue_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only dense Linear path that uses TileLang half GEMM on static weights."""

    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    if x.ndim != 2 or weight.ndim != 2:
        raise XQTBackendError("dense Linear TileLang path expects 2D x and weight")
    if x.shape[1] != weight.shape[1]:
        raise XQTBackendError("dense Linear TileLang path requires x.shape[1] == weight.shape[1]")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise XQTBackendError("dense Linear TileLang path expects bias shaped [out_features]")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    if x.shape[0] % int(block_m) != 0 or weight.shape[0] % int(block_n) != 0:
        raise XQTBackendError(
            "dense Linear TileLang path requires batch and out_features to be multiples of block sizes"
        )
    if x.shape[1] % int(block_k) != 0:
        raise XQTBackendError("dense Linear TileLang path requires in_features to be a multiple of block_k")
    require_tilelang()
    kernel = build_tilelang_gemm_kernel(
        m=int(x.shape[0]),
        n=int(weight.shape[0]),
        k=int(x.shape[1]),
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
        has_bias=bias is not None,
        activation=activation,
    )
    if bias is not None:
        return kernel(x, weight, bias.to(dtype=x.dtype, device=x.device))
    return kernel(x, weight)


def half_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference half Linear path used by the TileLang backend."""

    return F.linear(
        x,
        weight.to(dtype=x.dtype, device=x.device),
        None if bias is None else bias.to(dtype=x.dtype, device=x.device),
    )


def half_linear_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only standalone half Linear path backed by TileLang GEMM."""

    return dense_linear_epilogue_tilelang(
        x,
        weight,
        bias,
        activation=None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )



TILELANG_LINEAR_KERNEL_METADATA = {
    "dense_linear_epilogue": {
        "kernel_name": "dense_linear_epilogue",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.linear + epilogue",
        "usage": "Static-weight dense Linear path for Ada/Hopper half GEMM fastpaths after one-time dequant.",
        "weight_encoding": "dense_fp16",
        "unpack_stage": "one_time_eager_dequant_cache",
        "fusion_status": "tilelang_dense_half_gemm_epilogue",
        "epilogue_stage": "torch_bias_activation",
    },
    "linear": {
        "kernel_name": "half_linear",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.linear",
        "usage": "Standalone half Linear path for direct TileLang operator benchmarking.",
        "weight_encoding": "dense_fp16",
        "fusion_status": "tilelang_half_gemm",
        "epilogue_stage": None,
    },
}

__all__ = [
    "TILELANG_LINEAR_KERNEL_METADATA",
    "dense_linear_epilogue_reference",
    "dense_linear_epilogue_tilelang",
    "half_linear_reference",
    "half_linear_tilelang",
]
