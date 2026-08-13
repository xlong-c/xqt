"""Inference-only FLUX attention over fused SVDQuant W4A4 QKV projections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .svd_composite import SVDQuantLinear

_HEAD_DIM = 128


@dataclass(frozen=True)
class SVDQuantFluxRotaryEmb:
    """Disambiguated packed RoPE tensors for FLUX hidden/context streams."""

    hidden: torch.Tensor
    context: torch.Tensor | None = None


class SVDQuantFluxAttention(nn.Module):
    """Run Diffusers FLUX attention with native fused SVDQuant QKV epilogues.

    The wrapper consumes already-joint ``to_qkv`` and optional
    ``add_qkv_proj`` SVDQuant artifacts. It deliberately does not concatenate
    three independently decomposed Q/K/V modules because that changes the
    low-rank contract. The native path is forward-only, targets ``sm_89``, and
    requires head dimension 128.
    """

    def __init__(
        self,
        attention: nn.Module,
        to_qkv: SVDQuantLinear,
        *,
        add_qkv_proj: SVDQuantLinear | None = None,
        output_projection: nn.Module | None = None,
        native_fusion: bool = True,
        attention_processor: str = "flashattn2",
    ) -> None:
        super().__init__()
        if attention_processor not in {"flashattn2", "nunchaku-fp16"}:
            raise ValueError(
                "attention_processor must be 'flashattn2' or 'nunchaku-fp16'"
            )
        attention_type = type(attention)
        if attention_type.__name__ != "FluxAttention" or not attention_type.__module__.startswith(
            "diffusers."
        ):
            raise TypeError("attention must be a Diffusers FluxAttention module")
        if not isinstance(to_qkv, SVDQuantLinear):
            raise TypeError("to_qkv must be an SVDQuantLinear module")

        self.head_dim = int(attention.head_dim)
        self.inner_dim = int(attention.inner_dim)
        self.query_dim = int(attention.query_dim)
        self.out_dim = int(attention.out_dim)
        self.heads = int(attention.heads)
        self.added_kv_proj_dim = attention.added_kv_proj_dim
        self.pre_only = bool(attention.pre_only)
        if self.head_dim != _HEAD_DIM:
            raise ValueError("native SVDQuant FLUX attention requires head_dim=128")
        if self.inner_dim != self.heads * self.head_dim:
            raise ValueError("FluxAttention inner_dim must equal heads * head_dim")
        if to_qkv.input_features != self.query_dim:
            raise ValueError("to_qkv input_features must match attention query_dim")
        if to_qkv.output_features != 3 * self.inner_dim:
            raise ValueError("to_qkv output_features must equal 3 * attention inner_dim")

        has_added_projection = self.added_kv_proj_dim is not None
        if has_added_projection != (add_qkv_proj is not None):
            raise ValueError(
                "add_qkv_proj must be provided exactly when attention has added_kv_proj_dim"
            )
        if add_qkv_proj is not None:
            if not isinstance(add_qkv_proj, SVDQuantLinear):
                raise TypeError("add_qkv_proj must be an SVDQuantLinear module")
            if add_qkv_proj.input_features != int(self.added_kv_proj_dim):
                raise ValueError(
                    "add_qkv_proj input_features must match added_kv_proj_dim"
                )
            if add_qkv_proj.output_features != 3 * self.inner_dim:
                raise ValueError(
                    "add_qkv_proj output_features must equal 3 * attention inner_dim"
                )

        self.to_qkv = to_qkv
        self.add_qkv_proj = add_qkv_proj
        self.norm_q = attention.norm_q
        self.norm_k = attention.norm_k
        self.norm_added_q = (
            attention.norm_added_q if has_added_projection else None
        )
        self.norm_added_k = (
            attention.norm_added_k if has_added_projection else None
        )
        if output_projection is not None and not self.pre_only:
            raise ValueError(
                "output_projection is only valid for pre_only FLUX attention"
            )
        if isinstance(output_projection, SVDQuantLinear):
            if output_projection.input_features != self.inner_dim:
                raise ValueError(
                    "output_projection input_features must match attention inner_dim"
                )
            if output_projection.output_features != self.out_dim:
                raise ValueError(
                    "output_projection output_features must match attention out_dim"
                )
        self.to_out = (
            output_projection
            if output_projection is not None
            else getattr(attention, "to_out", None)
        )
        self.to_add_out = getattr(attention, "to_add_out", None)
        self._native_fusion_enabled = bool(native_fusion)
        self.attention_processor = attention_processor
        self._native_hot_cache: dict[
            tuple[Any, ...],
            tuple[
                tuple[Any, ...],
                Any,
                Any,
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            ],
        ] = {}
        self._last_native_qkv_used = False
        self._last_fallback_reason: str | None = None

        if self._native_fusion_enabled:
            for projection in self._output_projections():
                if isinstance(projection, SVDQuantLinear):
                    projection.enable_fusion()

    def _output_projections(self) -> tuple[nn.Module, ...]:
        projections: list[nn.Module] = []
        if isinstance(self.to_out, (nn.ModuleList, nn.Sequential)) and self.to_out:
            projections.append(self.to_out[0])
        elif isinstance(self.to_out, nn.Module):
            projections.append(self.to_out)
        if isinstance(self.to_add_out, nn.Module):
            projections.append(self.to_add_out)
        return tuple(projections)

    def _clear_runtime_cache(self) -> None:
        self._native_hot_cache.clear()

    def _apply(self, fn: Any) -> "SVDQuantFluxAttention":
        super()._apply(fn)
        self._clear_runtime_cache()
        self._last_native_qkv_used = False
        self._last_fallback_reason = None
        return self

    def enable_fusion(self) -> bool:
        """Enable the native QKV path and report current target availability."""

        self._native_fusion_enabled = True
        if not torch.cuda.is_available():
            return False
        try:
            from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
                native_w4a4_available,
            )

            return bool(native_w4a4_available(build=False))
        except Exception:
            return False

    def disable_fusion(self) -> None:
        """Disable the native path; calls then fail instead of changing semantics."""

        self._native_fusion_enabled = False
        self._last_native_qkv_used = False
        self._last_fallback_reason = "native fused FLUX attention is disabled"

    @staticmethod
    def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            tensor.device.index,
            tensor.dtype,
            int(tensor.data_ptr()),
            int(getattr(tensor, "_version", 0)),
            tuple(int(dim) for dim in tensor.shape),
        )

    def _branch_signature(
        self,
        projection: SVDQuantLinear,
        norm_q: nn.Module,
        norm_k: nn.Module,
    ) -> tuple[Any, ...]:
        norm_q_weight = getattr(norm_q, "weight", None)
        norm_k_weight = getattr(norm_k, "weight", None)
        if not isinstance(norm_q_weight, torch.Tensor) or not isinstance(
            norm_k_weight, torch.Tensor
        ):
            raise RuntimeError("FLUX Q/K RMSNorm must have affine weights")
        return (
            projection._native_w4a4_state_signature(),
            self._tensor_signature(norm_q_weight),
            self._tensor_signature(norm_k_weight),
            float(norm_q.eps),
            float(norm_k.eps),
        )

    @staticmethod
    def _hot_key(
        branch: str,
        inputs: torch.Tensor,
    ) -> tuple[Any, ...]:
        raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        if callable(raw_stream):
            stream_id = int(raw_stream(inputs.device.index))
        else:
            stream_id = int(torch.cuda.current_stream(inputs.device).cuda_stream)
        return (
            branch,
            inputs.device.index,
            inputs.dtype,
            int(inputs.shape[0]),
            stream_id,
        )

    def _native_gate(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[bool, str]:
        if not self._native_fusion_enabled:
            return False, "native fused FLUX attention is disabled"
        if self.training:
            return False, "native fused FLUX attention requires eval mode"
        if torch.is_grad_enabled():
            return False, "native fused FLUX attention requires no_grad or inference_mode"
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            return False, "native fused FLUX attention currently requires [1, S, C] inputs"
        if not hidden_states.is_cuda:
            return False, "native fused FLUX attention requires CUDA inputs"
        if hidden_states.dtype not in {torch.float16, torch.bfloat16}:
            return False, "native fused FLUX attention requires float16 or bfloat16"
        if self.attention_processor == "nunchaku-fp16" and hidden_states.dtype != torch.float16:
            return False, "nunchaku-fp16 attention requires float16 hidden states"
        if int(hidden_states.shape[-1]) != self.query_dim:
            return False, "hidden_states width must match attention query_dim"
        if encoder_hidden_states is not None:
            if self.add_qkv_proj is None:
                return False, "encoder_hidden_states require add_qkv_proj"
            if encoder_hidden_states.ndim != 3 or int(encoder_hidden_states.shape[0]) != 1:
                return False, "context inputs must have shape [1, S_context, C]"
            if encoder_hidden_states.device != hidden_states.device:
                return False, "hidden and context inputs must share a device"
            if encoder_hidden_states.dtype != hidden_states.dtype:
                return False, "hidden and context inputs must share a dtype"
            if int(encoder_hidden_states.shape[-1]) != int(self.added_kv_proj_dim):
                return False, "context width must match added_kv_proj_dim"
        major, minor = torch.cuda.get_device_capability(hidden_states.device)
        if (major, minor) != (8, 9):
            return False, f"native fused FLUX attention targets sm_89, got sm_{major}{minor}"
        try:
            from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
                native_w4a4_available,
            )

            if not native_w4a4_available(build=False):
                return False, "native W4A4 backend is unavailable"
        except Exception as exc:
            return False, f"native W4A4 capability check failed: {exc}"
        return True, "native fused FLUX attention is available"

    @staticmethod
    def _padded_rows(rows: int) -> int:
        return ((int(rows) + 255) // 256) * 256

    @staticmethod
    def _validate_rotary(
        rotary_emb: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        padded_rows = ((int(inputs.shape[1]) + 255) // 256) * 256
        if not rotary_emb.is_cuda or rotary_emb.device != inputs.device:
            raise ValueError("packed rotary_emb must share the input CUDA device")
        if rotary_emb.dtype != torch.float32:
            raise ValueError("packed rotary_emb must be float32")
        if rotary_emb.ndim < 2 or int(rotary_emb.shape[-1]) != _HEAD_DIM:
            raise ValueError("packed rotary_emb trailing dimension must be 128")
        if int(rotary_emb.numel()) != padded_rows * _HEAD_DIM:
            raise ValueError("packed rotary_emb must cover the padded sequence extent")
        return rotary_emb.contiguous()

    def _run_qkv_branch(
        self,
        branch: str,
        projection: SVDQuantLinear,
        norm_q: nn.Module,
        norm_k: nn.Module,
        inputs: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
            allocate_w4a4_workspace,
            bind_svdq_w4a4_qkv_rmsnorm_rope,
        )

        if float(norm_q.eps) != float(norm_k.eps):
            raise ValueError("Q and K RMSNorm eps values must match")
        flat = inputs.reshape(-1, projection.input_features).contiguous()
        packed_rotary = self._validate_rotary(rotary_emb, inputs)
        key = self._hot_key(branch, flat)
        signature = self._branch_signature(projection, norm_q, norm_k)
        cached = self._native_hot_cache.get(key)
        if cached is not None and cached[0] == signature:
            output = cached[3](flat, packed_rotary)
        else:
            if cached is not None:
                self._native_hot_cache.pop(key, None)
            packed = projection._native_w4a4_packed(flat, smalln=False)
            workspace = allocate_w4a4_workspace(
                int(flat.shape[0]),
                packed,
                with_lora_rank=packed.padded_rank,
            )
            norm_q_weight = norm_q.weight.to(
                device=flat.device,
                dtype=flat.dtype,
            ).contiguous()
            norm_k_weight = norm_k.weight.to(
                device=flat.device,
                dtype=flat.dtype,
            ).contiguous()
            native_forward = bind_svdq_w4a4_qkv_rmsnorm_rope(
                packed,
                workspace,
                norm_q_weight,
                norm_k_weight,
                rows=int(flat.shape[0]),
                eps=float(norm_q.eps),
            )
            output = native_forward(flat, packed_rotary)
            if len(self._native_hot_cache) >= 8:
                self._native_hot_cache.clear()
            self._native_hot_cache[key] = (
                signature,
                packed,
                workspace,
                native_forward,
            )
        return output.reshape(1, int(inputs.shape[1]), 3 * self.inner_dim)

    def _run_qkv_branch_packed(
        self,
        branch: str,
        projection: SVDQuantLinear,
        norm_q: nn.Module,
        norm_k: nn.Module,
        inputs: torch.Tensor,
        rotary_emb: torch.Tensor,
        output_q: torch.Tensor,
        output_k: torch.Tensor,
        output_v: torch.Tensor,
        row_offset: int,
    ) -> None:
        from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
            allocate_w4a4_workspace,
            bind_svdq_w4a4_qkv_rmsnorm_rope,
        )

        if float(norm_q.eps) != float(norm_k.eps):
            raise ValueError("Q and K RMSNorm eps values must match")
        flat = inputs.reshape(-1, projection.input_features).contiguous()
        packed_rotary = self._validate_rotary(rotary_emb, inputs)
        key = self._hot_key(f"packed:{branch}", flat)
        signature = self._branch_signature(projection, norm_q, norm_k)
        cached = self._native_hot_cache.get(key)
        if cached is not None and cached[0] == signature:
            native_forward = cached[3]
        else:
            if cached is not None:
                self._native_hot_cache.pop(key, None)
            packed = projection._native_w4a4_packed(flat, smalln=False)
            workspace = allocate_w4a4_workspace(
                int(flat.shape[0]),
                packed,
                with_lora_rank=packed.padded_rank,
            )
            norm_q_weight = norm_q.weight.to(
                device=flat.device,
                dtype=flat.dtype,
            ).contiguous()
            norm_k_weight = norm_k.weight.to(
                device=flat.device,
                dtype=flat.dtype,
            ).contiguous()
            native_forward = bind_svdq_w4a4_qkv_rmsnorm_rope(
                packed,
                workspace,
                norm_q_weight,
                norm_k_weight,
                rows=int(flat.shape[0]),
                eps=float(norm_q.eps),
            )
            if len(self._native_hot_cache) >= 8:
                self._native_hot_cache.clear()
            self._native_hot_cache[key] = (
                signature,
                packed,
                workspace,
                native_forward,
            )
        native_forward.run_packed(
            flat,
            packed_rotary,
            output_q,
            output_k,
            output_v,
            int(row_offset),
        )

    def _project_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.to_out, (nn.ModuleList, nn.Sequential)):
            hidden_states = self.to_out[0](hidden_states.contiguous())
            for module in self.to_out[1:]:
                hidden_states = module(hidden_states)
            return hidden_states
        if isinstance(self.to_out, nn.Module):
            return self.to_out(hidden_states.contiguous())
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: SVDQuantFluxRotaryEmb | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run fused QKV, PyTorch SDPA, and the existing FLUX output projections."""

        if kwargs:
            raise NotImplementedError("joint_attention_kwargs are not supported")
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported")
        allowed, reason = self._native_gate(hidden_states, encoder_hidden_states)
        if not allowed:
            self._last_native_qkv_used = False
            self._last_fallback_reason = reason
            raise RuntimeError(reason)

        if encoder_hidden_states is None:
            if not isinstance(image_rotary_emb, torch.Tensor):
                raise ValueError("single-stream attention requires packed rotary tensor")
            hidden_rotary = image_rotary_emb
            context_rotary = None
        else:
            if not isinstance(image_rotary_emb, SVDQuantFluxRotaryEmb):
                raise ValueError(
                    "joint attention requires SVDQuantFluxRotaryEmb to disambiguate streams"
                )
            hidden_rotary = image_rotary_emb.hidden
            context_rotary = image_rotary_emb.context
            if context_rotary is None:
                raise ValueError("joint attention requires packed context rotary tensor")

        if self.attention_processor == "nunchaku-fp16":
            from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
                svdq_w4a4_attention_fp16,
            )

            hidden_rows = int(hidden_states.shape[1])
            hidden_padded = self._padded_rows(hidden_rows)
            context_rows = (
                int(encoder_hidden_states.shape[1])
                if encoder_hidden_states is not None
                else 0
            )
            context_padded = self._padded_rows(context_rows) if context_rows else 0
            total_padded = context_padded + hidden_padded
            packed_shape = (1, self.heads, total_padded, self.head_dim)
            query = torch.empty(packed_shape, device=hidden_states.device, dtype=torch.float16)
            key = torch.empty_like(query)
            value = torch.empty_like(query)
            if encoder_hidden_states is not None:
                if (
                    context_rotary is None
                    or self.add_qkv_proj is None
                    or self.norm_added_q is None
                    or self.norm_added_k is None
                ):
                    raise RuntimeError("joint attention projection state is incomplete")
                self._run_qkv_branch_packed(
                    "context",
                    self.add_qkv_proj,
                    self.norm_added_q,
                    self.norm_added_k,
                    encoder_hidden_states,
                    context_rotary,
                    query,
                    key,
                    value,
                    0,
                )
            self._run_qkv_branch_packed(
                "hidden",
                self.to_qkv,
                self.norm_q,
                self.norm_k,
                hidden_states,
                hidden_rotary,
                query,
                key,
                value,
                context_padded,
            )
            attention_output = svdq_w4a4_attention_fp16(
                query,
                key,
                value,
                scale=self.head_dim**-0.5,
            )
            if encoder_hidden_states is None:
                context_attention_output = None
                attention_output = attention_output[:, :hidden_rows]
            else:
                context_attention_output = attention_output[:, :context_rows]
                attention_output = attention_output[
                    :, context_padded : context_padded + hidden_rows
                ]
        else:
            qkv = self._run_qkv_branch(
                "hidden",
                self.to_qkv,
                self.norm_q,
                self.norm_k,
                hidden_states,
                hidden_rotary,
            )
            if encoder_hidden_states is not None:
                if self.add_qkv_proj is None or self.norm_added_q is None or self.norm_added_k is None:
                    raise RuntimeError("joint attention projection state is incomplete")
                qkv_context = self._run_qkv_branch(
                    "context",
                    self.add_qkv_proj,
                    self.norm_added_q,
                    self.norm_added_k,
                    encoder_hidden_states,
                    context_rotary,
                )
                qkv = torch.cat((qkv_context, qkv), dim=1)

            query, key, value = qkv.chunk(3, dim=-1)
            query = query.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
            key = key.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
            value = value.view(1, -1, self.heads, self.head_dim).transpose(1, 2)
            attention_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            attention_output = attention_output.transpose(1, 2).reshape(
                1,
                -1,
                self.inner_dim,
            )
            attention_output = attention_output.to(query.dtype)
            context_attention_output = None

        self._last_native_qkv_used = True
        self._last_fallback_reason = None
        if encoder_hidden_states is None:
            return self._project_output(attention_output)

        context_rows = int(encoder_hidden_states.shape[1])
        hidden_rows = int(hidden_states.shape[1])
        if context_attention_output is None:
            context_attention_output = attention_output[:, :context_rows]
            # The SDPA path returns the full context-then-hidden sequence.
            hidden_output = attention_output[
                :, context_rows : context_rows + hidden_rows
            ]
        else:
            # The packed FP16 path slices the hidden stream before reaching
            # this shared projection section.
            hidden_output = attention_output
        context_output = context_attention_output
        hidden_output = self._project_output(hidden_output)
        if not isinstance(self.to_add_out, nn.Module):
            raise RuntimeError("joint attention requires to_add_out")
        context_output = self.to_add_out(context_output.contiguous())
        return hidden_output, context_output

    def execution_metadata(self) -> dict[str, Any]:
        """Describe the last runtime path without claiming Transformer parity."""

        return {
            "implementation": (
                "native_svdq_flux_attention_nunchaku_fp16"
                if self._last_native_qkv_used and self.attention_processor == "nunchaku-fp16"
                else "native_svdq_flux_attention_sdpa"
                if self._last_native_qkv_used
                else "unavailable_svdq_flux_attention"
            ),
            "compute_contract": (
                "svdq_int4_qkv_rmsnorm_rope_fp16_attention"
                if self.attention_processor == "nunchaku-fp16"
                else "svdq_int4_qkv_rmsnorm_rope_sdpa"
            ),
            "attention_processor": self.attention_processor,
            "native_fusion_enabled": bool(self._native_fusion_enabled),
            "native_qkv_used": bool(self._last_native_qkv_used),
            "fallback_reason": self._last_fallback_reason,
            "head_dim": self.head_dim,
            "heads": self.heads,
            "joint_attention": self.added_kv_proj_dim is not None,
            "batch_limit": 1,
        }


__all__ = ["SVDQuantFluxAttention", "SVDQuantFluxRotaryEmb"]
