"""Attention operator kernel entry points grouped across backends."""

from __future__ import annotations

from .tilelang.attention import (
    TileLangAttentionDesign,
    build_tilelang_attention_design,
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)
from .triton.attention import (
    TritonAttentionSchedule,
    fused_attention_forward_reference as triton_attention_forward_reference,
    fused_attention_forward_triton,
    resolve_triton_attention_schedule,
)

__all__ = [
    "TritonAttentionSchedule",
    "TileLangAttentionDesign",
    "build_tilelang_attention_design",
    "fused_attention_forward_reference",
    "fused_attention_forward_tilelang",
    "fused_attention_forward_triton",
    "resolve_triton_attention_schedule",
    "triton_attention_forward_reference",
]
