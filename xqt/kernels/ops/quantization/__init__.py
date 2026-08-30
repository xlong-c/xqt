"""quantization kernels."""
from __future__ import annotations

from typing import Any

import torch

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend, KernelSpec
from xqt.kernels.ops._legacy_api import load_legacy
from xqt.kernels.ops.quantization.nvfp4 import (
    expand_group_scale,
    normalize_group_scale,
    unpack_nvfp4e2m1,
)


_CUDA = frozenset({CapabilityRequirement.CUDA})

def _per_token_quant_fp8_torch(x):
    import torch
    scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 448.0
    q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16)
    return q, scale

register_kernel(KernelSpec(op="quantization.per_token_quant_fp8", backend=KernelBackend.TORCH, target="xqt.kernels.ops.quantization:_per_token_quant_fp8_torch", format_signature=FormatSignature(description="per token quant fp8")))
register_kernel(
    KernelSpec(
        op="quantization.svd_fused_dequant_gemm_low_rank",
        backend=KernelBackend.TILELANG,
        target="xqt.kernels.ops._impl.tilelang.svd_fused:svd_fused_dequant_gemm_low_rank_tilelang",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("float16",),
            description="SVDQuant fused dequant GEMM with low-rank branch",
        ),
    )
)

__all__ = [
    "_per_token_quant_fp8_torch",
    "expand_group_scale",
    "normalize_group_scale",
    "per_token_quant_fp8",
    "svd_fused_dequant_gemm_low_rank_tilelang",
    "unpack_nvfp4e2m1",
]


def __getattr__(name: str) -> Any:
    return load_legacy("quantization", name)

def per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from xqt.kernels.selector import get_kernel
    return get_kernel("quantization.per_token_quant_fp8", KernelBackend.TORCH)(x)


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
    """Run the SVDQuant fused TileLang kernel through the unified registry."""

    from xqt.kernels.selector import get_kernel

    return get_kernel(
        "quantization.svd_fused_dequant_gemm_low_rank",
        KernelBackend.TILELANG,
    )(
        x,
        packed_weight,
        scale,
        down_weight,
        up_weight,
        bias,
        input_features=input_features,
        group_size=group_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )
