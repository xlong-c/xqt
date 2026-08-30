"""layernorm kernels (BaseFusedOp style, spec 4.2)."""
from __future__ import annotations

import torch

from xqt.kernels.fused_op import BaseFusedOp, register_fused_op
from xqt.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend

_CUDA = frozenset({CapabilityRequirement.CUDA})


def _rmsnorm_torch(x, weight, eps: float = 1e-6):  # type: ignore[no-untyped-def]
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return (x_norm * weight).to(x.dtype)


class RMSNormOp(BaseFusedOp):
    op = "layernorm.rmsnorm"
    priority = (KernelBackend.TRITON, KernelBackend.TORCH)
    capabilities = {
        KernelBackend.TORCH: frozenset(),
        KernelBackend.TRITON: _CUDA,
    }
    format_signature = FormatSignature(supported_dtypes=("float16", "bfloat16"), description="rmsnorm")

    def forward_native(self, x, weight, eps: float = 1e-6):  # type: ignore[override]
        return _rmsnorm_torch(x, weight, eps=eps)

    def forward_triton(self, x, weight, eps: float = 1e-6):  # type: ignore[override]
        from xqt.kernels.ops._impl.tilelang.hunyuan_block import rmsnorm_tilelang

        return rmsnorm_tilelang(x, weight, eps=eps)


_RMSNORM = register_fused_op(RMSNormOp(), __name__, "_RMSNORM")

__all__ = ["rmsnorm", "_RMSNORM", "RMSNormOp", "_rmsnorm_torch"]


def rmsnorm(x, weight, eps: float = 1e-6):  # type: ignore[no-untyped-def]
    return _RMSNORM(x, weight, eps=eps)  # type: ignore[operator]
