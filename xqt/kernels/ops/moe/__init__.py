"""moe kernels."""
from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import FormatSignature, KernelBackend, KernelSpec

def _moe_align_block_size_torch(topk_ids, num_experts, block_size):
    import torch
    sorted_ids = topk_ids.flatten().argsort()
    return sorted_ids

register_kernel(KernelSpec(op="moe.moe_align_block_size", backend=KernelBackend.TORCH, target="xqt.kernels.ops.moe:_moe_align_block_size_torch", format_signature=FormatSignature(description="moe align")))

__all__ = ["moe_align_block_size", "_moe_align_block_size_torch"]

def moe_align_block_size(topk_ids, num_experts, block_size):
    from xqt.kernels.selector import get_kernel
    return get_kernel("moe.moe_align_block_size", KernelBackend.TORCH)(topk_ids, num_experts, block_size)
