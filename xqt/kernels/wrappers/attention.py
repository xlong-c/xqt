"""TileLang MHA attention wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.engines.tilelang import run_tilelang_kernel
from .runtime import (
    DEFAULT_CUDA_GRAPH_WARMUP,
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)
from ._common import (
    _matching_tensor_dtype_name,
    _resolved_target_arch,
    _scaled_dot_product_attention_with_causal_semantics,
    _target_arch_mismatch_reason,
)


class _TileLangAttentionWrapper(nn.Module):
    """Minimal executable wrapper for attention-pattern TileLang targets."""

    def __init__(
        self,
        attention: nn.MultiheadAttention,
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
        self.last_qkv_projection = "not_run"
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _prefer_native_attention_fastpath(self, q: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return _resolved_target_arch(self.settings, q) == "sm_89"

    def _prefer_graph_attention_fastpath(self, q: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode in {"graph", "tilelang_graph"}:
            return True
        return False

    def _canonicalize_attention_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_input = query if self.attention.batch_first else query.transpose(0, 1)
        source_key = query if key is None else key
        if source_key is query:
            k_input = q_input
        else:
            k_input = source_key if self.attention.batch_first else source_key.transpose(0, 1)
        source_value = source_key if value is None else value
        if source_value is source_key:
            v_input = k_input
        elif source_value is query:
            v_input = q_input
        else:
            v_input = source_value if self.attention.batch_first else source_value.transpose(0, 1)
        return q_input, k_input, v_input

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not getattr(self.attention, "_qkv_same_embed_dim", True):
            raise XQTBackendError(
                "TileLang attention wrapper currently supports only qkv_same_embed_dim=True"
            )
        if self.attention.in_proj_weight is None:
            raise XQTBackendError("TileLang attention wrapper requires packed in_proj_weight")
        embed_dim = int(self.attention.embed_dim)
        if query is key and key is value:
            packed_qkv = F.linear(
                query,
                self.attention.in_proj_weight,
                self.attention.in_proj_bias,
            )
            q_proj, k_proj, v_proj = torch.split(
                packed_qkv,
                embed_dim,
                dim=-1,
            )
            self.last_qkv_projection = "packed_self_attention"
            return q_proj, k_proj, v_proj
        self.last_qkv_projection = "separate_qkv"
        q_proj = F.linear(
            query,
            self.attention.in_proj_weight[:embed_dim],
            None if self.attention.in_proj_bias is None else self.attention.in_proj_bias[:embed_dim],
        )
        k_proj = F.linear(
            key,
            self.attention.in_proj_weight[embed_dim : 2 * embed_dim],
            None
            if self.attention.in_proj_bias is None
            else self.attention.in_proj_bias[embed_dim : 2 * embed_dim],
        )
        v_proj = F.linear(
            value,
            self.attention.in_proj_weight[2 * embed_dim :],
            None
            if self.attention.in_proj_bias is None
            else self.attention.in_proj_bias[2 * embed_dim :],
        )
        return q_proj, k_proj, v_proj

    def _reshape_for_tilelang(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = tensor.shape
        num_heads = int(self.attention.num_heads)
        head_dim = embed_dim // num_heads
        return tensor.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _merge_from_tilelang(tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim).contiguous()

    def _project_qkv_for_tilelang(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_proj, k_proj, v_proj = self._project_qkv(query, key, value)
        q, k, v = (
            self._reshape_for_tilelang(q_proj),
            self._reshape_for_tilelang(k_proj),
            self._reshape_for_tilelang(v_proj),
        )
        self.last_kernel_dtype = _matching_tensor_dtype_name(q, k, v)
        return q, k, v

    def _finalize_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        merged = self._merge_from_tilelang(attn_output)
        return self.attention.out_proj(merged)

    @staticmethod
    def _attention_graph_runtime_args(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[tuple[int, int, int], tuple[torch.Tensor, ...]]:
        unique_args: list[torch.Tensor] = []
        arg_mapping: list[int] = []
        tensor_index: dict[int, int] = {}
        for tensor in (query, key, value):
            index = tensor_index.get(id(tensor))
            if index is None:
                index = len(unique_args)
                unique_args.append(tensor)
                tensor_index[id(tensor)] = index
            arg_mapping.append(index)
        return (arg_mapping[0], arg_mapping[1], arg_mapping[2]), tuple(unique_args)

    @staticmethod
    def _resolve_attention_graph_args(
        dynamic_args: tuple[torch.Tensor, ...],
        arg_mapping: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            dynamic_args[arg_mapping[0]],
            dynamic_args[arg_mapping[1]],
            dynamic_args[arg_mapping[2]],
        )

    def _attention_graph_cache_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> tuple[Any, ...]:
        arg_mapping, runtime_args = self._attention_graph_runtime_args(query, key, value)
        return (
            tuple(cuda_graph_tensor_signature(tensor) for tensor in runtime_args),
            arg_mapping,
            bool(is_causal),
            float(self.attention.dropout),
            int(self.settings.get("block_m", 64)),
            int(self.settings.get("block_n", 64)),
            int(self.settings.get("threads", 128)),
            int(self.settings.get("num_stages", 2)),
            str(self.settings.get("target_arch") or ""),
        )

    def _run_tilelang_attention_body(
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
            dropout_p=float(self.attention.dropout),
            block_m=int(self.settings.get("block_m", 64)),
            block_n=int(self.settings.get("block_n", 64)),
            threads=int(self.settings.get("threads", 128)),
            num_stages=int(self.settings.get("num_stages", 2)),
            target_arch=self.settings.get("target_arch"),
            fallback=self.fallback,
        )

    def _run_tilelang_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv_for_tilelang(query, key, value)
        return self._finalize_attention_output(
            self._run_tilelang_attention_body(q, k, v, is_causal=is_causal)
        )

    def _run_native_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv_for_tilelang(query, key, value)
        attn_output = _scaled_dot_product_attention_with_causal_semantics(
            q,
            k,
            v,
            dropout_p=float(self.attention.dropout),
            causal=is_causal,
        )
        return self._finalize_attention_output(attn_output)

    def _run_reference_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        *,
        need_weights: bool,
        average_attn_weights: bool,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        reason: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Use the complete PyTorch MHA contract for unsupported fastpath inputs."""

        self.last_execution_mode = "reference_fallback"
        self.last_fastpath = "eager_reference_fallback"
        self.last_execution_reason = reason
        self.last_graph_state = "disabled"
        self.last_graph_reason = reason
        self.last_kernel_dtype = str(query.dtype).removeprefix("torch.")
        self.last_qkv_projection = "reference_fallback"
        reference_key = query if key is None else key
        reference_value = reference_key if value is None else value
        output, weights = self.attention(
            query,
            reference_key,
            reference_value,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return output, weights

    def _graph_capture_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> dict[str, Any]:
        arg_mapping, runtime_args = self._attention_graph_runtime_args(query, key, value)
        state = capture_cuda_graph_with_static_state(
            runtime_args,
            body=lambda *dynamic_args: self._run_tilelang_attention_forward(
                *self._resolve_attention_graph_args(
                    tuple(dynamic_args),
                    arg_mapping,
                ),
                is_causal=is_causal,
            ),
            warmup=int(self.settings.get("cuda_graph_warmup", DEFAULT_CUDA_GRAPH_WARMUP)),
        )
        state["kind"] = "tilelang_attention_full_forward"
        state["arg_mapping"] = arg_mapping
        return state

    def _run_attention_with_optional_graph(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        if not self._prefer_graph_attention_fastpath(query):
            self.last_graph_state = "disabled"
            self.last_graph_reason = "attention_fastpath is not set to graph mode"
            return self._run_tilelang_attention_forward(query, key, value, is_causal=is_causal)
        _, runtime_args = self._attention_graph_runtime_args(query, key, value)
        cache_key = self._attention_graph_cache_key(query, key, value, is_causal=is_causal)
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = self._graph_capture_attention(query, key, value, is_causal=is_causal)
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = f"CUDA Graph capture failed: {exc}"
                return self._run_tilelang_attention_forward(query, key, value, is_causal=is_causal)
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
            return replay_cuda_graph_tensor_callable(state, runtime_args)
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return replay_cuda_graph_tensor_callable(state, runtime_args)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.last_qkv_projection = "not_run"
        if attn_mask is not None or key_padding_mask is not None:
            return self._run_reference_attention_forward(
                query,
                key,
                value,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                reason="attention_mask_requires_reference",
            )
        q_input, k_input, v_input = self._canonicalize_attention_inputs(query, key, value)
        self.last_qkv_projection = (
            "packed_self_attention"
            if q_input is k_input and k_input is v_input
            else "separate_qkv"
        )
        mismatch = _target_arch_mismatch_reason(self.settings, q_input)
        if mismatch is not None:
            return self._run_reference_attention_forward(
                query,
                key,
                value,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                attn_mask=None,
                key_padding_mask=None,
                is_causal=is_causal,
                reason=mismatch,
            )
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q_input, k_input, v_input)
        ):
            return self._run_reference_attention_forward(
                query,
                key,
                value,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                attn_mask=None,
                key_padding_mask=None,
                is_causal=is_causal,
                reason="autograd_unsupported",
            )
        input_is_cuda = q_input.is_cuda and k_input.is_cuda and v_input.is_cuda
        use_graph_tilelang = (
            input_is_cuda and self._prefer_graph_attention_fastpath(q_input)
        )
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if input_is_cuda and self._prefer_native_attention_fastpath(q_input)
            else "cuda_graph_tilelang_entry"
            if use_graph_tilelang
            else "cuda_tilelang_entry"
            if input_is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_sdpa"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_attention_cuda_graph"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "tilelang_attention_kernel"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {
                "cuda_tilelang_entry",
                "cuda_graph_tilelang_entry",
                "cuda_native_fastpath",
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
            projected = self._run_native_attention_forward(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        elif self.last_execution_mode == "cuda_graph_tilelang_entry":
            projected = self._run_attention_with_optional_graph(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        else:
            projected = self._run_tilelang_attention_forward(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        output = projected if self.attention.batch_first else projected.transpose(0, 1)
        weights = None
        if need_weights:
            batch_size = int(q_input.shape[0]) if q_input.ndim == 3 else 0
            target_len = int(q_input.shape[1]) if q_input.ndim == 3 else 0
            source_len = int(k_input.shape[1]) if k_input.ndim == 3 else 0
            weights = output.new_zeros((batch_size, target_len, source_len))
        return output, weights

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "cuda_graph_replay"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "qkv_projection": self.last_qkv_projection,
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
