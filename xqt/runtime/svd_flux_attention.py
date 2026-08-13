"""Explicit materialization helpers for inference-only SVDQuant FLUX attention."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
    pack_svdq_w4a4_rotary_emb,
)
from xqt.runtime.modules.svd_composite import SVDQuantLinear
from xqt.runtime.modules.svd_flux_attention import (
    SVDQuantFluxAttention,
    SVDQuantFluxRotaryEmb,
)

_HEAD_DIM = 128


def materialize_svd_flux_attention(
    attention: nn.Module,
    *,
    to_qkv: SVDQuantLinear,
    add_qkv_proj: SVDQuantLinear | None = None,
    output_projection: nn.Module | None = None,
    native_fusion: bool = True,
    attention_processor: str = "flashattn2",
) -> SVDQuantFluxAttention:
    """Wrap one Diffusers FluxAttention with explicit fused-QKV artifacts.

    ``flashattn2`` is the default SDPA-compatible path. ``nunchaku-fp16`` is
    an explicit SM89-only path that uses packed QKV outputs and the online
    FP16 attention kernel; it is intentionally not selected implicitly.
    """

    return SVDQuantFluxAttention(
        attention,
        to_qkv,
        add_qkv_proj=add_qkv_proj,
        output_projection=output_projection,
        native_fusion=native_fusion,
        attention_processor=attention_processor,
    )


def pack_diffusers_flux_rotary_emb(
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    *,
    hidden_rows: int,
    context_rows: int = 0,
) -> SVDQuantFluxRotaryEmb:
    """Convert Diffusers FLUX ``(cos, sin)`` tensors to native packed RoPE.

    Diffusers orders a joint sequence as context followed by image/hidden
    tokens. This helper is intended to run once per model forward and its
    result can be reused by every materialized block.
    """

    if not isinstance(image_rotary_emb, tuple) or len(image_rotary_emb) != 2:
        raise TypeError("image_rotary_emb must be a (cos, sin) tuple")
    cos, sin = image_rotary_emb
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Diffusers FLUX rotary cos/sin tensors must be float32")
    if cos.device != sin.device or cos.shape != sin.shape:
        raise ValueError("rotary cos and sin tensors must share shape and device")
    if cos.ndim != 2 or int(cos.shape[1]) != _HEAD_DIM:
        raise ValueError("rotary cos/sin tensors must have shape [S, 128]")
    hidden_extent = int(hidden_rows)
    context_extent = int(context_rows)
    if hidden_extent <= 0 or context_extent < 0:
        raise ValueError("hidden_rows must be positive and context_rows non-negative")
    if int(cos.shape[0]) != hidden_extent + context_extent:
        raise ValueError("rotary sequence length must equal context_rows + hidden_rows")

    raw = torch.stack((sin[:, 0::2], cos[:, 0::2]), dim=-1).unsqueeze(-2)
    hidden_raw = raw[context_extent:].contiguous()
    hidden = pack_svdq_w4a4_rotary_emb(
        hidden_raw,
        rows=hidden_extent,
    )
    context = None
    if context_extent > 0:
        context_raw = raw[:context_extent].contiguous()
        context = pack_svdq_w4a4_rotary_emb(
            context_raw,
            rows=context_extent,
        )
    return SVDQuantFluxRotaryEmb(hidden=hidden, context=context)


def svd_flux_attention_metadata(module: nn.Module) -> dict[str, Any]:
    """Return explicit runtime metadata for a materialized FLUX attention."""

    if not isinstance(module, SVDQuantFluxAttention):
        raise TypeError("module must be SVDQuantFluxAttention")
    return module.execution_metadata()


__all__ = [
    "materialize_svd_flux_attention",
    "pack_diffusers_flux_rotary_emb",
    "svd_flux_attention_metadata",
]
