"""activation kernels (BaseFusedOp style, spec 4.2)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from xqt.kernels.fused_op import BaseFusedOp, register_fused_op
from xqt.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend

_CUDA = frozenset({CapabilityRequirement.CUDA})


def _silu_and_mul_torch(x):  # type: ignore[no-untyped-def]
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return F.silu(gate) * up


class SiluAndMulOp(BaseFusedOp):
    op = "activation.silu_and_mul"
    priority = (KernelBackend.TRITON, KernelBackend.TORCH)
    capabilities = {
        KernelBackend.TORCH: frozenset(),
        KernelBackend.TRITON: _CUDA,
    }
    format_signature = FormatSignature(description="silu and mul")

    def forward_native(self, x):  # type: ignore[override]
        return _silu_and_mul_torch(x)

    def forward_triton(self, x):  # type: ignore[override]
        from xqt.kernels.ops._impl.triton.pointwise import fused_swiglu_triton

        d = x.shape[-1] // 2
        return fused_swiglu_triton(x[..., :d], x[..., d:])


_SILU = register_fused_op(SiluAndMulOp(), __name__, "_SILU")

__all__ = ["silu_and_mul", "_SILU", "SiluAndMulOp", "_silu_and_mul_torch"]


def silu_and_mul(x):  # type: ignore[no-untyped-def]
    return _SILU(x)  # type: ignore[operator]
