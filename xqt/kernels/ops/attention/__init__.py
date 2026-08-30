"""attention kernels."""
from __future__ import annotations

from typing import Any

import torch

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend, KernelSpec
from xqt.kernels.ops._legacy_api import load_legacy

_CUDA = frozenset({CapabilityRequirement.CUDA})


def _fused_attention_torch(q, k, v, is_causal=False):  # type: ignore[no-untyped-def]
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


register_kernel(
    KernelSpec(
        op="attention.fused_attention",
        backend=KernelBackend.TORCH,
        target="xqt.kernels.ops.attention:_fused_attention_torch",
        format_signature=FormatSignature(description="fused attention torch reference"),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_attention",
        backend=KernelBackend.TRITON,
        target="xqt.kernels.ops._impl.triton.attention:fused_attention_forward_triton",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="fused attention triton"),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_attention",
        backend=KernelBackend.TILELANG,
        target="xqt.kernels.ops._impl.tilelang.attention:fused_attention_forward_tilelang",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="fused attention tilelang"),
    )
)

__all__ = [
    "_fused_attention_torch",
    "fused_attention",
    "fused_attention_tilelang",
    "fused_attention_triton",
]


def __getattr__(name: str) -> Any:
    return load_legacy("attention", name)


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    from xqt.kernels.selector import get_kernel

    return get_kernel("attention.fused_attention", KernelBackend.TORCH)(q, k, v, is_causal=is_causal)


def fused_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    from xqt.kernels.selector import get_kernel

    return get_kernel("attention.fused_attention", KernelBackend.TRITON)(q, k, v, is_causal=is_causal)


def fused_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run the TileLang attention kernel through the unified registry."""

    from xqt.kernels.selector import get_kernel

    return get_kernel("attention.fused_attention", KernelBackend.TILELANG)(
        q,
        k,
        v,
        causal=causal,
        dropout_p=dropout_p,
        block_m=block_m,
        block_n=block_n,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )
