"""TileLang xqt.nn.Attention wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ..backends.tilelang import run_tilelang_kernel
from ..runtime import (
    DEFAULT_CUDA_GRAPH_WARMUP,
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)
from ._common import _matching_tensor_dtype_name, _resolved_target_arch

if TYPE_CHECKING:
    from xqt import nn as xqt_nn


class _TileLangXqtAttentionWrapper(nn.Module):
    """Executable wrapper for xqt.nn.Attention with TileLang kernel fastpath."""

    def __init__(
        self,
        attention: "xqt_nn.Attention",
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.attention = attention
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "attention"
        self.last_fastpath = "none"
        self.last_graph_state = "disabled"
        self.last_graph_reason: str | None = None
        self.last_kernel_dtype: str | None = None
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = (
            self.attention._reshape_qkv(self.attention.q_proj(x)),
            self.attention._reshape_qkv(self.attention.k_proj(x)),
            self.attention._reshape_qkv(self.attention.v_proj(x)),
        )
        self.last_kernel_dtype = _matching_tensor_dtype_name(q, k, v)
        return q, k, v

    def _run_tilelang_attention_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        return run_tilelang_kernel(
            "attention",
            q,
            k,
            v,
            causal=is_causal,
            dropout_p=float(self.attention.dropout_p),
            block_m=int(self.settings.get("block_m", 64)),
            block_n=int(self.settings.get("block_n", 64)),
            threads=int(self.settings.get("threads", 128)),
            num_stages=int(self.settings.get("num_stages", 2)),
            fallback=self.fallback,
        )

    def _run_tilelang_xqt_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        attn_output = self._run_tilelang_attention_kernel(
            q, k, v, is_causal=bool(self.attention.causal)
        )
        merged = self.attention._merge_heads(attn_output)
        return self.attention.out_proj(merged)

    def _run_sdpa_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=float(self.attention.dropout_p),
            is_causal=bool(self.attention.causal),
        )
        merged = self.attention._merge_heads(attn_output)
        return self.attention.out_proj(merged)

    def _forward_prefer_tilelang(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self._run_tilelang_xqt_attention_forward(x)
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang attention runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self._run_sdpa_attention_forward(x)

    def _prefer_native_attention_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return _resolved_target_arch(self.settings, x) == "sm_89"

    def _prefer_graph_attention_fastpath(self) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        return mode in {"graph", "tilelang_graph"}

    def _attention_graph_cache_key(self, x: torch.Tensor) -> tuple[Any, ...]:
        return (
            cuda_graph_tensor_signature(x),
            bool(self.attention.causal),
            float(self.attention.dropout_p),
            int(self.settings.get("block_m", 64)),
            int(self.settings.get("block_n", 64)),
            int(self.settings.get("threads", 128)),
            int(self.settings.get("num_stages", 2)),
            str(self.settings.get("target_arch") or ""),
        )

    def _graph_capture_attention(self, x: torch.Tensor) -> dict[str, Any]:
        state = capture_cuda_graph_with_static_state(
            (x,),
            body=self._run_tilelang_xqt_attention_forward,
            warmup=int(self.settings.get("cuda_graph_warmup", DEFAULT_CUDA_GRAPH_WARMUP)),
        )
        state["kind"] = "tilelang_xqt_attention_full_forward"
        return state

    def _run_attention_with_optional_graph(self, x: torch.Tensor) -> torch.Tensor:
        if not self._prefer_graph_attention_fastpath():
            self.last_graph_state = "disabled"
            self.last_graph_reason = "attention_fastpath is not set to graph mode"
            return self._forward_prefer_tilelang(x)
        cache_key = self._attention_graph_cache_key(x)
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = self._graph_capture_attention(x)
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = f"CUDA Graph capture failed: {exc}"
                return self._forward_prefer_tilelang(x)
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
            return replay_cuda_graph_tensor_callable(state, (x,))
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return replay_cuda_graph_tensor_callable(state, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_is_cuda = x.is_cuda
        use_graph_tilelang = input_is_cuda and self._prefer_graph_attention_fastpath()
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if input_is_cuda and self._prefer_native_attention_fastpath(x)
            else "cuda_graph_tilelang_entry"
            if use_graph_tilelang
            else "cuda_tilelang_entry"
            if input_is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_sdpa"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_xqt_attention_cuda_graph"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "tilelang_xqt_attention_kernel"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode
            in {
                "cuda_native_fastpath",
                "cuda_graph_tilelang_entry",
                "cuda_tilelang_entry",
            }
            else "TileLang attention kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode != "cuda_graph_tilelang_entry":
            self.last_graph_state = "disabled"
            self.last_graph_reason = (
                None
                if self.last_execution_mode == "cuda_native_fastpath"
                else "graph fastpath was not selected"
            )
        if self.last_execution_mode == "cuda_native_fastpath":
            return self._run_sdpa_attention_forward(x)
        if self.last_execution_mode == "cuda_graph_tilelang_entry":
            return self._run_attention_with_optional_graph(x)
        if self.last_execution_mode == "cuda_tilelang_entry":
            return self._forward_prefer_tilelang(x)
        return self._run_sdpa_attention_forward(x)

    def runtime_config(self) -> dict[str, Any]:
        base = (
            self.attention.runtime_config()
            if hasattr(self.attention, "runtime_config")
            else {"engine": "tilelang"}
        )
        return {
            **base,
            "execution": self.execution_metadata(),
        }

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
                "dtype": self.last_kernel_dtype or "unknown",
                "supported_dtypes": ["float16", "bfloat16"],
                "bfloat16_head_dim_multiple": 16,
                "dropout_p": 0.0,
                "requires_seq_kv_gte_seq_q": True,
                "supported_patterns": ["attention"],
                "operator_families": ["attention"],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
            "cuda_graph": {
                "state": self.last_graph_state,
                "reason": self.last_graph_reason,
                "cache_size": len(self._graph_cache),
            },
        }
