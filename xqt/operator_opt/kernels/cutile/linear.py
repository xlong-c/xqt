"""CuTile Linear operator references and guarded entry points."""

from typing import Any

import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import dense_gemm_reference

from ._common import require_cuda_tensors, require_cutile, require_fp16_tensors


def dense_linear_epilogue_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference dense Linear with optional bias and activation epilogue."""

    runtime_weight = weight.to(dtype=x.dtype, device=x.device)
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        runtime_weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


def dense_linear_epilogue_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only CuTile guarded dense Linear path."""

    del block_m, block_n, block_k, threads, target_arch
    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    if x.ndim != 2 or weight.ndim != 2:
        raise XQTBackendError("dense Linear CuTile path expects 2D x and weight")
    if x.shape[1] != weight.shape[1]:
        raise XQTBackendError(
            "dense Linear CuTile path requires x.shape[1] == weight.shape[1]"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise XQTBackendError(
            "dense Linear CuTile path expects bias shaped [out_features]"
        )
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    require_cutile()
    return dense_linear_epilogue_reference(x, weight, bias, activation=activation)


def half_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference half Linear path used by the CuTile backend."""

    return dense_gemm_reference(
        x,
        weight.to(dtype=x.dtype, device=x.device),
        None if bias is None else bias.to(dtype=x.dtype, device=x.device),
        transpose_b=True,
    )


def half_linear_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only standalone half Linear path backed by CuTile metadata."""

    return dense_linear_epilogue_cutile(
        x,
        weight,
        bias,
        activation=None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        target_arch=target_arch,
    )


CUTILE_LINEAR_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "dense_linear_epilogue": {
        "kernel_name": "dense_linear_epilogue",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "baseline": "torch.nn.functional.linear + epilogue",
        "usage": "Static-weight dense Linear path aligned with TileLang dense half GEMM epilogue coverage.",
        "weight_encoding": "dense_fp16",
        "unpack_stage": "one_time_eager_dequant_cache",
        "fusion_status": "cutile_dense_half_gemm_epilogue_reference_guarded",
        "epilogue_stage": "torch_bias_activation",
        "production_status": "reference_guarded",
    },
    "linear": {
        "kernel_name": "half_linear",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "baseline": "torch.nn.functional.linear",
        "usage": "Standalone half Linear path for direct CuTile operator planning.",
        "weight_encoding": "dense_fp16",
        "fusion_status": "cutile_half_gemm_reference_guarded",
        "epilogue_stage": None,
        "production_status": "reference_guarded",
    },
}

__all__ = [
    "CUTILE_LINEAR_KERNEL_METADATA",
    "dense_linear_epilogue_cutile",
    "dense_linear_epilogue_reference",
    "half_linear_cutile",
    "half_linear_reference",
]
