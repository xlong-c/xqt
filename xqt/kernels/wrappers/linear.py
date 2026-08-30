"""TileLang Linear wrapper and eager dense linear fallback module."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.engines.tilelang import get_tilelang_kernel_spec
from xqt.kernels.ops._impl.tilelang.linear import resolve_tilelang_linear_schedule
from ._common import _matching_tensor_dtype_name, _resolved_target_arch


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
        self.last_kernel_dtype: str | None = None
        self.last_kernel_schedule: dict[str, int | str | None] | None = None
        self._linear_runtime = str(self.settings.get("linear_runtime", "auto"))
        patterns = self.settings.get("preferred_patterns")
        self._kernel_pattern = (
            "linear_marlin"
            if isinstance(patterns, list) and "linear_marlin" in patterns
            else "linear"
        )
        self._kernel = get_tilelang_kernel_spec(self._kernel_pattern).kernel
        self._precision = str(self.settings.get("precision", "auto"))
        self._block_m = (
            int(self.settings["block_m"]) if "block_m" in self.settings else None
        )
        self._block_n = (
            int(self.settings["block_n"]) if "block_n" in self.settings else None
        )
        self._block_k = (
            int(self.settings["block_k"]) if "block_k" in self.settings else None
        )
        self._threads = int(self.settings.get("threads", 128))
        self._num_stages = int(self.settings.get("num_stages", 2))
        self._target_arch = self.settings.get("target_arch")

    def _prefer_native_linear_fastpath(self, x: torch.Tensor) -> bool:
        if self._linear_runtime == "native":
            return True
        if self._linear_runtime == "tilelang":
            return False
        return _resolved_target_arch(self.settings, x) == "sm_89"

    def _selected_linear_pattern(self) -> str:
        return self._kernel_pattern

    @staticmethod
    def _flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 0:
            raise XQTBackendError("TileLang linear target requires at least 1D input")
        if x.shape[-1] <= 0:
            raise XQTBackendError(
                "TileLang linear target requires a non-empty trailing feature dimension"
            )
        if x.ndim == 2:
            return x, (int(x.shape[0]),)
        return x.reshape(-1, int(x.shape[-1])), tuple(int(dim) for dim in x.shape[:-1])

    @staticmethod
    def _restore_output(output: torch.Tensor, prefix_shape: tuple[int, ...]) -> torch.Tensor:
        if len(prefix_shape) == 1 and prefix_shape[0] == int(output.shape[0]):
            return output
        return output.reshape(*prefix_shape, int(output.shape[-1]))

    def _run_tilelang_or_reference(
        self,
        x: torch.Tensor,
        flat_input: torch.Tensor,
    ) -> torch.Tensor:
        pattern = self._selected_linear_pattern()
        try:
            if pattern == "linear_marlin":
                block_m = 64 if self._block_m is None else self._block_m
                block_n = 64 if self._block_n is None else self._block_n
                block_k = 64 if self._block_k is None else self._block_k
                self.last_kernel_schedule = {
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "threads": self._threads,
                    "num_stages": self._num_stages,
                    "target_arch": self._target_arch,
                    "preset": "linear_marlin_default_or_explicit",
                }
                return self._kernel(
                    flat_input,
                    self.linear.weight,
                    bias=self.linear.bias,
                    precision=self._precision,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    threads=self._threads,
                    num_stages=self._num_stages,
                    target_arch=self._target_arch,
                )
            schedule = resolve_tilelang_linear_schedule(
                flat_input,
                block_m=self._block_m,
                block_n=self._block_n,
                block_k=self._block_k,
                threads=self._threads,
                num_stages=self._num_stages,
                target_arch=self._target_arch,
            )
            self.last_kernel_schedule = schedule.to_dict()
            return self._kernel(
                flat_input,
                self.linear.weight,
                self.linear.bias,
                block_m=schedule.block_m,
                block_n=schedule.block_n,
                block_k=schedule.block_k,
                threads=schedule.threads,
                num_stages=schedule.num_stages,
                target_arch=schedule.target_arch,
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
        dtype_tensors = (
            (flat_input, self.linear.weight)
            if self.linear.bias is None
            else (flat_input, self.linear.weight, self.linear.bias)
        )
        self.last_kernel_dtype = _matching_tensor_dtype_name(*dtype_tensors)
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
            "kernel_schedule": (
                None
                if self.last_kernel_schedule is None
                else dict(self.last_kernel_schedule)
            ),
            "kernel_constraints": {
                "dtype": self.last_kernel_dtype or "unknown",
                "supported_dtypes": ["float16", "bfloat16"],
                "bfloat16_block_k_multiple": 16,
                "requires_k_multiple_of_block_k": True,
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
