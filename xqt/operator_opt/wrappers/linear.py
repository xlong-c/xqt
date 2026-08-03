"""TileLang Linear wrapper and eager dense linear fallback module."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ..backends.tilelang import run_tilelang_kernel
from ._common import _resolved_target_arch


class _TileLangLinearWrapper(nn.Module):
    """Standalone half Linear wrapper for direct TileLang operator targets."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.linear = linear
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "linear"
        self.last_fastpath = "none"

    def _prefer_native_linear_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("linear_runtime", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        return _resolved_target_arch(self.settings, x) == "sm_89"

    def _selected_linear_pattern(self) -> str:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list) and "linear_marlin" in patterns:
            return "linear_marlin"
        return "linear"

    @staticmethod
    def _flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 0:
            raise XQTBackendError("TileLang linear target requires at least 1D input")
        if x.shape[-1] <= 0:
            raise XQTBackendError(
                "TileLang linear target requires a non-empty trailing feature dimension"
            )
        return x.reshape(-1, int(x.shape[-1])), tuple(int(dim) for dim in x.shape[:-1])

    @staticmethod
    def _restore_output(output: torch.Tensor, prefix_shape: tuple[int, ...]) -> torch.Tensor:
        return output.reshape(*prefix_shape, int(output.shape[-1]))

    def _run_tilelang_or_reference(
        self,
        x: torch.Tensor,
        flat_input: torch.Tensor,
    ) -> torch.Tensor:
        pattern = self._selected_linear_pattern()
        try:
            if pattern == "linear_marlin":
                return run_tilelang_kernel(
                    pattern,
                    flat_input,
                    self.linear.weight,
                    bias=self.linear.bias,
                    precision=str(self.settings.get("precision", "auto")),
                    block_m=int(self.settings.get("block_m", 64)),
                    block_n=int(self.settings.get("block_n", 64)),
                    block_k=int(self.settings.get("block_k", 64)),
                    threads=int(self.settings.get("threads", 128)),
                    num_stages=int(self.settings.get("num_stages", 2)),
                    target_arch=self.settings.get("target_arch"),
                    fallback=self.fallback,
                )
            return run_tilelang_kernel(
                pattern,
                flat_input,
                self.linear.weight,
                self.linear.bias,
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
            self.last_execution_reason = f"TileLang linear runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self.linear(x).reshape(-1, int(self.linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix_shape = tuple(int(dim) for dim in x.shape[:-1])
        flat_input, _ = self._flatten_input(x)
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_linear_fastpath(x)
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_torch_linear"
            if self.last_execution_mode == "cuda_native_fastpath"
            else (
                "tilelang_marlin_linear_kernel"
                if self._selected_linear_pattern() == "linear_marlin"
                else "tilelang_half_linear_kernel"
            )
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang linear kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
            return self.linear(x)
        output = self._run_tilelang_or_reference(x, flat_input)
        return self._restore_output(output, prefix_shape)

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
                "supported_precisions": ["fp16", "bf16", "int8", "int4"],
                "supported_patterns": ["linear", "linear_marlin"],
                "operator_families": ["linear"],
                "supports_rank_gte_1_via_batch_flatten": True,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TileLangEagerDenseLinearModule(nn.Module):
    """One-time dequantized dense Linear replacement for sm_89 native runtime parity."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.linear = linear
        self._execution_metadata = dict(metadata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self._execution_metadata)
