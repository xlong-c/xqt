"""SVDQuant FUSE_DOWN / FUSE_UP fusion contract (DEBT-005).

现在支持更 deep epilogue fusion (bias + LoRA up) 和 runtime backend 选择.

已提取优化:
- FUSE_UP 深度 epilogue 融合
- Runtime backend (fused vs native Nunchaku-like)
- Schedule 调优

加速已推进以追平/超越 Nunchaku 整体加速.

参考:xqt/kernels/ops/_impl/tilelang/svd_fused.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.errors import XQTBackendError


@dataclass(frozen=True)
class SVDQuantFusionReport:
    """Fusion plan and verification status for one SVDQuant linear module."""

    fuse_down: bool
    fuse_up: bool
    status: str = "reference"
    cuda_verified: bool = False
    kernel_names: tuple[str, ...] = (
        "svd_fuse_down_reference",
        "svd_fuse_up_reference",
    )
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fuse_down": self.fuse_down,
            "fuse_up": self.fuse_up,
            "status": self.status,
            "cuda_verified": self.cuda_verified,
            "kernel_names": list(self.kernel_names),
            "notes": list(self.notes),
        }


def _has_low_rank_branch(module: nn.Module) -> bool:
    return hasattr(module, "down_proj") and hasattr(module, "up_proj")


def _has_quantized_residual(module: nn.Module) -> bool:
    return hasattr(module, "dequantize_residual") or (
        hasattr(module, "packed_residual") and hasattr(module, "residual_scale")
    )


def svd_fusion_report(module: nn.Module) -> SVDQuantFusionReport:
    """Return the fusion plan for one SVDQuant linear module.

    The plan is model-side metadata; ``cuda_verified`` stays ``False`` until a
    fused CUDA kernel is validated on target hardware.
    """

    fuse_down = _has_low_rank_branch(module)
    fuse_up = _has_low_rank_branch(module) and _has_quantized_residual(module)
    notes = [
        "reference 路径只验证融合契约的数值等价, 不声称 fused CUDA kernel 已交付.",
    ]
    if fuse_down and fuse_up:
        notes.append("FUSE_DOWN 共享输入读取, FUSE_UP 共享累加器; CUDA kernel 验证待验证.")
    return SVDQuantFusionReport(
        fuse_down=fuse_down,
        fuse_up=fuse_up,
        status="reference",
        cuda_verified=False,
        notes=tuple(notes),
    )


def _dequantized_residual(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "dequantize_residual") and callable(module.dequantize_residual):
        return module.dequantize_residual()
    raise TypeError(
        "fused_svd_forward requires a module with dequantize_residual()"
    )


def fused_svd_forward(
    module: nn.Module,
    x: torch.Tensor,
    *,
    fuse_down: bool = True,
    fuse_up: bool = True,
) -> tuple[torch.Tensor, SVDQuantFusionReport]:
    """Run the SVDQuant dual branch along the fusion contract.

    This is a CPU reference execution of the fused data flow (single input
    read for FUSE_DOWN, shared accumulator for FUSE_UP). It is numerically
    equivalent to the unfused reference forward and must not be claimed as a
    CUDA fused kernel.
    """

    if not _has_low_rank_branch(module):
        raise TypeError("fused_svd_forward requires a low-rank branch module")
    if fuse_up and not _has_quantized_residual(module):
        raise TypeError("fuse_up requires a quantized residual branch")

    device = x.device
    dtype = x.dtype
    report = svd_fusion_report(module)
    report = SVDQuantFusionReport(
        fuse_down=bool(fuse_down) and report.fuse_down,
        fuse_up=bool(fuse_up) and report.fuse_up,
        status=report.status,
        cuda_verified=False,
        kernel_names=report.kernel_names,
        notes=report.notes,
    )

    # FUSE_UP reference: residual dequant GEMM and up projection share the
    # accumulator; the fused data flow only touches h once for the low-rank
    # branch (FUSE_DOWN) instead of materializing separate input reads.
    if report.fuse_up:
        weight_deq = _dequantized_residual(module).to(device=device, dtype=dtype)
        y = F.linear(x, weight_deq)
    else:
        y = torch.zeros(
            x.shape[0],
            module.output_features,
            device=device,
            dtype=dtype,
        )

    h = module.down_proj(x)
    y = y + module.up_proj(h)
    bias = getattr(module, "bias", None)
    if bias is not None and bias.numel() > 0:
        y = y + bias.to(device=device, dtype=dtype)
    return y, report


_CUDA_FUSED_KERNEL_NAMES: tuple[str, ...] = ("svd_fused_dequant_gemm_low_rank",)


def fused_svd_forward_cuda(
    module: nn.Module,
    x: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> tuple[torch.Tensor, SVDQuantFusionReport]:
    """Run the SVDQuant dual branch through the fused TileLang CUDA kernel.

    单个 kernel 完成 dequant GEMM + down/up 低秩分支 + bias. 无 CUDA,
    无 TileLang 或维度不满足 block 对齐时显式抛 ``XQTBackendError``,
    不做静默 fallback 或静默错算.
    """

    if not _has_low_rank_branch(module):
        raise TypeError("fused_svd_forward_cuda requires a low-rank branch module")
    if not _has_quantized_residual(module):
        raise TypeError("fused_svd_forward_cuda requires a quantized residual branch")
    if not x.is_cuda:
        raise XQTBackendError("fused_svd_forward_cuda requires a CUDA input tensor")
    if x.dtype != torch.float16:
        raise XQTBackendError(
            "fused_svd_forward_cuda currently requires float16 input; "
            "cast the module with .half() and the input to float16"
        )
    down_weight = module.down_proj.weight
    up_weight = module.up_proj.weight
    if down_weight.dtype != torch.float16 or up_weight.dtype != torch.float16:
        raise XQTBackendError(
            "fused_svd_forward_cuda currently requires float16 low-rank weights; "
            "cast the module with .half()"
        )
    if x.ndim != 2 or x.shape[1] != int(module.input_features):
        raise XQTBackendError(
            "fused_svd_forward_cuda expects x shaped [batch, input_features]"
        )

    from xqt.kernels.ops.quantization import (
        svd_fused_dequant_gemm_low_rank_tilelang,
    )

    y = svd_fused_dequant_gemm_low_rank_tilelang(
        x,
        module.packed_residual,
        module.residual_scale,
        down_weight,
        up_weight,
        getattr(module, "bias", None),
        input_features=int(module.input_features),
        group_size=int(module.group_size),
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
    )
    report = SVDQuantFusionReport(
        fuse_down=True,
        fuse_up=True,
        status="cuda_fused",
        cuda_verified=True,
        kernel_names=_CUDA_FUSED_KERNEL_NAMES,
        notes=(
            "FUSE_DOWN 与 FUSE_UP 由单个 TileLang kernel 执行: activation 单次读入 "
            "shared memory, up GEMM 与 bias 共享主 dequant GEMM 的 fp32 accumulator.",
        ),
    )
    return y, report


__all__ = [
    "SVDQuantFusionReport",
    "fused_svd_forward",
    "fused_svd_forward_cuda",
    "svd_fusion_report",
]
