"""TileLang Conv3d wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from xqt.kernels.ops._impl.engines.tilelang import run_tilelang_kernel


class _TileLangConv3dWrapper(nn.Module):
    """Conv3d 1x1x1 operator-family wrapper with TileLang fastpath routing."""

    def __init__(
        self,
        conv: nn.Conv3d,
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

    def _is_supported_fastpath(self) -> bool:
        return (
            tuple(int(value) for value in self.conv.kernel_size) == (1, 1, 1)
            and tuple(int(value) for value in self.conv.stride) == (1, 1, 1)
            and tuple(int(value) for value in self.conv.padding) == (0, 0, 0)
            and tuple(int(value) for value in self.conv.dilation) == (1, 1, 1)
            and int(self.conv.groups) == 1
        )

    def _run_eager(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                "conv3d_1x1x1",
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
            self.last_execution_reason = f"TileLang conv3d runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self._run_eager(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_supported_fastpath():
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                "TileLang Conv3d fastpath requires kernel_size=stride=dilation=(1,1,1), "
                "padding=(0,0,0), and groups=1."
            )
            self.last_fastpath = "eager_reference_fallback"
            return self._run_eager(x)
        self.last_execution_mode = (
            "cuda_tilelang_entry" if x.is_cuda else "reference_fallback"
        )
        self.last_fastpath = (
            "tilelang_half_conv3d_1x1x1_gemm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "TileLang conv3d fastpath requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_tilelang_entry":
            return self._run_tilelang_or_reference(x)
        return self._run_eager(x)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "kernel_pattern": "conv3d_1x1x1",
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": ["conv3d_1x1x1"],
                "operator_families": ["conv"],
                "supports_grouped_conv": False,
                "requires_kernel_size": [1, 1, 1],
                "requires_stride": [1, 1, 1],
                "requires_padding": [0, 0, 0],
                "requires_dilation": [1, 1, 1],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }
