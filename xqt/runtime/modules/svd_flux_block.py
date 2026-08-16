"""Inference-only FLUX double-stream block over XQT SVDQuant modules."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .awq_w4a16_linear import AWQW4A16Linear
from xqt.contracts.composite import CompositeAddModule
from .svd_flux_attention import SVDQuantFluxAttention, SVDQuantFluxRotaryEmb
from .svd_gelu_mlp import SVDQuantGeluMLP


class SVDQuantAdaLayerNormZero(nn.Module):
    """FLUX AdaLayerNormZero with an inference-only AWQ modulation projection.

    Nunchaku checkpoints store the six modulation fields interleaved by hidden
    channel. Therefore the projection output is split with
    ``view(B, D, 6).permute(2, 0, 1)`` rather than a contiguous six-way chunk.
    ``scale_shift=0`` matches the official FLUX model loader, which has already
    fused the conventional ``+1`` scale into the modulation projection.
    """

    def __init__(
        self,
        norm: nn.Module,
        linear: AWQW4A16Linear,
        *,
        silu: nn.Module | None = None,
        embedding: nn.Module | None = None,
        scale_shift: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(norm, nn.Module):
            raise TypeError("norm must be an nn.Module")
        if not isinstance(linear, AWQW4A16Linear):
            raise TypeError("linear must be an AWQW4A16Linear")
        if linear.output_features != 6 * linear.input_features:
            raise ValueError(
                "AdaLayerNormZero AWQ projection output_features must equal "
                "6 * input_features"
            )
        self.dim = linear.input_features
        self.norm = norm
        self.linear = linear
        self.silu = nn.SiLU() if silu is None else silu
        self.emb = embedding
        self.scale_shift = float(scale_shift)

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.emb is not None:
            emb = self.emb(
                timestep,
                class_labels,
                hidden_dtype=hidden_dtype,
            )
        if emb is None:
            raise ValueError("AdaLayerNormZero requires a precomputed embedding")
        if emb.ndim != 2 or int(emb.shape[-1]) != self.dim:
            raise ValueError("AdaLayerNormZero embedding must have shape [B, D]")
        modulation = self.linear(self.silu(emb))
        modulation = modulation.view(modulation.shape[0], self.dim, 6).permute(
            2,
            0,
            1,
        )
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.unbind(0)
        normalized = self.norm(inputs)
        if self.scale_shift != 0.0:
            scale_msa = scale_msa + self.scale_shift
            scale_mlp = scale_mlp + self.scale_shift
        normalized = normalized * scale_msa[:, None] + shift_msa[:, None]
        return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def execution_metadata(self) -> dict[str, Any]:
        """Return the active AWQ projection contract."""

        return {
            "implementation": "sm89_awq_adaln_zero",
            "dim": self.dim,
            "scale_shift": self.scale_shift,
            "linear": self.linear.execution_metadata(),
        }


class SVDQuantAdaLayerNormZeroSingle(nn.Module):
    """FLUX single-stream AdaLN-Zero with an AWQ modulation projection."""

    def __init__(
        self,
        norm: nn.Module,
        linear: AWQW4A16Linear,
        *,
        silu: nn.Module | None = None,
        scale_shift: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(norm, nn.Module):
            raise TypeError("norm must be an nn.Module")
        if not isinstance(linear, AWQW4A16Linear):
            raise TypeError("linear must be an AWQW4A16Linear")
        if linear.output_features != 3 * linear.input_features:
            raise ValueError(
                "AdaLayerNormZeroSingle AWQ projection output_features must equal "
                "3 * input_features"
            )
        self.dim = linear.input_features
        self.norm = norm
        self.linear = linear
        self.silu = nn.SiLU() if silu is None else silu
        self.scale_shift = float(scale_shift)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if emb.ndim != 2 or int(emb.shape[-1]) != self.dim:
            raise ValueError(
                "AdaLayerNormZeroSingle embedding must have shape [B, D]"
            )
        modulation = self.linear(self.silu(emb))
        modulation = modulation.view(modulation.shape[0], self.dim, 3).permute(
            2,
            0,
            1,
        )
        shift_msa, scale_msa, gate_msa = modulation.unbind(0)
        if self.scale_shift != 0.0:
            scale_msa = scale_msa + self.scale_shift
        normalized = self.norm(inputs)
        normalized = normalized * scale_msa[:, None] + shift_msa[:, None]
        return normalized, gate_msa

    def execution_metadata(self) -> dict[str, Any]:
        """Return the active single-stream AWQ modulation contract."""

        return {
            "implementation": "sm89_awq_adaln_zero_single",
            "dim": self.dim,
            "scale_shift": self.scale_shift,
            "linear": self.linear.execution_metadata(),
        }


class SVDQuantFluxTransformerBlock(nn.Module):
    """Materialized FLUX double-stream block with native XQT low-bit modules."""

    def __init__(
        self,
        *,
        norm1: SVDQuantAdaLayerNormZero,
        norm1_context: SVDQuantAdaLayerNormZero,
        attention: SVDQuantFluxAttention,
        norm2: nn.Module,
        norm2_context: nn.Module,
        feed_forward: SVDQuantGeluMLP,
        feed_forward_context: SVDQuantGeluMLP,
    ) -> None:
        super().__init__()
        if not isinstance(norm1, SVDQuantAdaLayerNormZero):
            raise TypeError("norm1 must be SVDQuantAdaLayerNormZero")
        if not isinstance(norm1_context, SVDQuantAdaLayerNormZero):
            raise TypeError("norm1_context must be SVDQuantAdaLayerNormZero")
        if not isinstance(attention, SVDQuantFluxAttention):
            raise TypeError("attention must be SVDQuantFluxAttention")
        if not isinstance(feed_forward, SVDQuantGeluMLP):
            raise TypeError("feed_forward must be SVDQuantGeluMLP")
        if not isinstance(feed_forward_context, SVDQuantGeluMLP):
            raise TypeError("feed_forward_context must be SVDQuantGeluMLP")
        dim = norm1.dim
        dimensions = (
            norm1_context.dim,
            attention.query_dim,
            attention.out_dim,
            feed_forward.input_features,
            feed_forward.output_features,
            feed_forward_context.input_features,
            feed_forward_context.output_features,
        )
        if any(value != dim for value in dimensions):
            raise ValueError("all FLUX block modules must share the same hidden dimension")
        self.dim = dim
        self.norm1 = norm1
        self.norm1_context = norm1_context
        self.attn = attention
        self.norm2 = norm2
        self.norm2_context = norm2_context
        self.ff = feed_forward
        self.ff_context = feed_forward_context
        self._last_native_used = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: SVDQuantFluxRotaryEmb,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError("SVDQuant FLUX block requires eval mode")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "SVDQuant FLUX block requires no_grad or inference_mode"
            )
        if joint_attention_kwargs:
            raise NotImplementedError("joint_attention_kwargs are not supported")

        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.norm1(hidden_states, emb=temb)
        (
            norm_encoder_hidden_states,
            context_gate_msa,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, emb=temb)

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if not isinstance(attention_outputs, tuple) or len(attention_outputs) != 2:
            raise RuntimeError("joint FLUX attention must return hidden and context outputs")
        attention_output, context_attention_output = attention_outputs

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attention_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * scale_mlp[:, None] + shift_mlp[:, None]
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(
            norm_hidden_states
        )

        encoder_hidden_states = (
            encoder_hidden_states
            + context_gate_msa.unsqueeze(1) * context_attention_output
        )
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * context_scale_mlp[:, None]
            + context_shift_mlp[:, None]
        )
        encoder_hidden_states = (
            encoder_hidden_states
            + context_gate_mlp.unsqueeze(1)
            * self.ff_context(norm_encoder_hidden_states)
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        self._last_native_used = True
        return encoder_hidden_states, hidden_states

    def execution_metadata(self) -> dict[str, Any]:
        """Describe the full block path without extending the claim to a model."""

        return {
            "implementation": (
                "native_svdq_flux_transformer_block"
                if self._last_native_used
                else "not_run_svdq_flux_transformer_block"
            ),
            "scope": "flux_double_stream_transformer_block",
            "inference_only": True,
            "native_block_used": self._last_native_used,
            "norm1": self.norm1.execution_metadata(),
            "norm1_context": self.norm1_context.execution_metadata(),
            "attention": self.attn.execution_metadata(),
            "feed_forward": self.ff.execution_metadata(),
            "feed_forward_context": self.ff_context.execution_metadata(),
        }


class SVDQuantFluxSingleTransformerBlock(nn.Module):
    """Materialized FLUX single-stream block with native XQT low-bit modules."""

    def __init__(
        self,
        *,
        norm: SVDQuantAdaLayerNormZeroSingle,
        attention: SVDQuantFluxAttention,
        feed_forward: SVDQuantGeluMLP,
    ) -> None:
        super().__init__()
        if not isinstance(norm, SVDQuantAdaLayerNormZeroSingle):
            raise TypeError("norm must be SVDQuantAdaLayerNormZeroSingle")
        if not isinstance(attention, SVDQuantFluxAttention):
            raise TypeError("attention must be SVDQuantFluxAttention")
        if attention.added_kv_proj_dim is not None or not attention.pre_only:
            raise ValueError(
                "single-stream FLUX attention must be pre_only without added KV"
            )
        if not isinstance(attention.to_out, CompositeAddModule):
            raise TypeError(
                "single-stream FLUX attention requires a composite output projection"
            )
        if not isinstance(feed_forward, SVDQuantGeluMLP):
            raise TypeError("feed_forward must be SVDQuantGeluMLP")
        dim = norm.dim
        dimensions = (
            attention.query_dim,
            attention.out_dim,
            attention.to_out.input_features,
            attention.to_out.output_features,
            feed_forward.input_features,
            feed_forward.output_features,
        )
        if any(value != dim for value in dimensions):
            raise ValueError(
                "all single-stream FLUX block modules must share the hidden dimension"
            )
        self.dim = dim
        self.norm = norm
        self.attn = attention
        self.ff = feed_forward
        self._last_native_used = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("SVDQuant FLUX single block requires eval mode")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "SVDQuant FLUX single block requires no_grad or inference_mode"
            )
        if joint_attention_kwargs:
            raise NotImplementedError("joint_attention_kwargs are not supported")

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_output = self.ff(norm_hidden_states)
        attention_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if not isinstance(attention_output, torch.Tensor):
            raise RuntimeError("single-stream FLUX attention must return one tensor")
        hidden_states = residual + gate.unsqueeze(1) * (
            attention_output + mlp_output
        )
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        self._last_native_used = True
        return hidden_states

    def execution_metadata(self) -> dict[str, Any]:
        """Describe the single-block path without extending the claim to a model."""

        return {
            "implementation": (
                "native_svdq_flux_single_transformer_block"
                if self._last_native_used
                else "not_run_svdq_flux_single_transformer_block"
            ),
            "scope": "flux_single_stream_transformer_block",
            "inference_only": True,
            "native_block_used": self._last_native_used,
            "norm": self.norm.execution_metadata(),
            "attention": self.attn.execution_metadata(),
            "feed_forward": self.ff.execution_metadata(),
        }


__all__ = [
    "SVDQuantAdaLayerNormZero",
    "SVDQuantAdaLayerNormZeroSingle",
    "SVDQuantFluxSingleTransformerBlock",
    "SVDQuantFluxTransformerBlock",
]
