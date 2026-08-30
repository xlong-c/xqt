"""TileLang LayerNorm wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.kernels.ops._impl.engines.tilelang import run_tilelang_kernel
from .runtime import (
    DEFAULT_CUDA_GRAPH_WARMUP,
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)
from ._common import _resolved_target_arch


class _TileLangNormWrapper(nn.Module):
    """Standalone half LayerNorm wrapper for direct TileLang operator targets."""

    def __init__(
        self,
        norm: nn.LayerNorm,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.norm = norm
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "norm"
        self.last_fastpath = "none"
        self.last_graph_state = "disabled"
        self.last_graph_reason: str | None = None
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _prefer_native_norm_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("norm_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return _resolved_target_arch(self.settings, x) == "sm_89"

    def _prefer_graph_norm_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("norm_fastpath", "auto"))
        if mode in {"graph", "tilelang_graph"}:
            return True
        return False

    def _norm_weight_bias(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized_dim = int(self.norm.normalized_shape[-1])
        weight = self.norm.weight
        bias = self.norm.bias
        if weight is None:
            weight = torch.ones(normalized_dim, device=x.device, dtype=x.dtype)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return weight.to(device=x.device, dtype=x.dtype), bias

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self._norm_weight_bias(x)
        try:
            return run_tilelang_kernel(
                "norm",
                x,
                weight,
                bias,
                eps=float(self.norm.eps),
                threads=int(self.settings.get("threads", 64)),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang norm runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self.norm(x)

    def _norm_graph_cache_key(
        self,
        x: torch.Tensor,
    ) -> tuple[Any, ...]:
        normalized_dim = int(self.norm.normalized_shape[-1])
        return (
            cuda_graph_tensor_signature(x),
            normalized_dim,
            float(self.norm.eps),
            int(self.settings.get("threads", 64)),
        )

    def _run_norm_with_optional_graph(self, x: torch.Tensor) -> torch.Tensor:
        if not self._prefer_graph_norm_fastpath(x):
            self.last_graph_state = "disabled"
            self.last_graph_reason = "norm_fastpath is not set to graph mode"
            return self._run_tilelang_or_reference(x)
        weight, bias = self._norm_weight_bias(x)
        graph_bias = bias if bias is not None else torch.zeros_like(weight)
        cache_key = self._norm_graph_cache_key(x)
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = capture_cuda_graph_with_static_state(
                    (x,),
                    body=lambda x_arg: run_tilelang_kernel(
                        "norm",
                        x_arg,
                        weight,
                        graph_bias,
                        eps=float(self.norm.eps),
                        threads=int(self.settings.get("threads", 64)),
                        fallback=self.fallback,
                    ),
                    warmup=int(
                        self.settings.get("cuda_graph_warmup", DEFAULT_CUDA_GRAPH_WARMUP)
                    ),
                )
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = f"CUDA Graph capture failed: {exc}"
                return self._run_tilelang_or_reference(x)
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
            return replay_cuda_graph_tensor_callable(state, (x,))
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return replay_cuda_graph_tensor_callable(state, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_graph_tilelang = x.is_cuda and self._prefer_graph_norm_fastpath(x)
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_norm_fastpath(x)
            else "cuda_graph_tilelang_entry"
            if use_graph_tilelang
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_torch_layer_norm"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_half_layer_norm_cuda_graph"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "tilelang_half_layer_norm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {
                "cuda_native_fastpath",
                "cuda_graph_tilelang_entry",
                "cuda_tilelang_entry",
            }
            else "TileLang norm kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode != "cuda_graph_tilelang_entry":
            self.last_graph_state = "disabled"
            self.last_graph_reason = (
                None
                if self.last_execution_mode == "cuda_native_fastpath"
                else "graph fastpath was not selected"
            )
        if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
            return self.norm(x)
        if self.last_execution_mode == "cuda_graph_tilelang_entry":
            return self._run_norm_with_optional_graph(x)
        return self._run_tilelang_or_reference(x)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "cuda_graph_replay"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
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
                "supported_patterns": ["norm"],
                "operator_families": ["norm"],
                "normalized_last_dim_only": True,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
            "cuda_graph": {
                "state": self.last_graph_state,
                "reason": self.last_graph_reason,
                "cache_size": len(self._graph_cache),
            },
        }
