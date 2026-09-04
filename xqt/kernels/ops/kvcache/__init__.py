"""kvcache kernels."""
from __future__ import annotations

import torch

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import FormatSignature, KernelBackend, KernelSpec


def _reshape_and_cache_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            key_cache[slot].copy_(key[i])
            value_cache[slot].copy_(value[i])
    return key_cache, value_cache


register_kernel(
    KernelSpec(
        op="kvcache.reshape_and_cache",
        backend=KernelBackend.TORCH,
        target="xqt.kernels.ops.kvcache:_reshape_and_cache_torch",
        format_signature=FormatSignature(in_place=True, description="reshape and cache"),
    )
)

__all__ = ["reshape_and_cache", "_reshape_and_cache_torch"]


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from xqt.kernels.selector import get_kernel

    return get_kernel("kvcache.reshape_and_cache", KernelBackend.TORCH)(
        key, value, key_cache, value_cache, slot_mapping
    )
