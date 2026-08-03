"""TileLang Conv2d wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..backends.tilelang import run_tilelang_kernel
from ._common import _resolved_target_arch


class _TileLangConvWrapper(nn.Module):
    """Conv2d operator-family wrapper with architecture-aware native runtime routing."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = conv
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "conv"
        self.last_fastpath = "none"

    def _prefer_native_conv_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("conv_fastpath", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        return _resolved_target_arch(self.settings, x) == "sm_89"

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                "conv",
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                block_m=int(self.settings.get("block_m", 64)),
                block_n=int(self.settings.get("block_n", 64)),
                block_k=int(self.settings.get("block_k", 64)),
                threads=int(self.settings.get("threads", 128)),
                num_stages=int(self.settings.get("num_stages", 2)),
                target_arch=self.settings.get("target_arch"),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang conv runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_conv_fastpath(x)
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_cudnn_conv2d"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_half_conv2d_im2col_gemm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang conv fastpath requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_native_fastpath":
            return F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )
        if self.last_execution_mode == "cuda_tilelang_entry":
            return self._run_tilelang_or_reference(x)
        return F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": ["conv"],
                "operator_families": ["conv"],
                "supports_grouped_conv": False,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }
