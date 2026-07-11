"""TileLang xqt.nn.Attention wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ..backends.tilelang import run_tilelang_kernel
from ._common import _resolved_target_arch


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

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.attention._reshape_qkv(self.attention.q_proj(x)),
            self.attention._reshape_qkv(self.attention.k_proj(x)),
            self.attention._reshape_qkv(self.attention.v_proj(x)),
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_is_cuda = x.is_cuda
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if input_is_cuda and self._prefer_native_attention_fastpath(x)
            else "cuda_tilelang_entry"
            if input_is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_sdpa"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_xqt_attention_kernel"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang attention kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_native_fastpath":
            return self._run_sdpa_attention_forward(x)
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
                "dropout_p": 0.0,
                "requires_seq_kv_gte_seq_q": True,
                "supported_patterns": ["attention"],
                "operator_families": ["attention"],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }
