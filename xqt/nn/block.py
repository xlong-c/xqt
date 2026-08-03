"""Pre-norm transformer block composed of Attention and FeedForward facades."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .attention import Attention
from .feedforward import FeedForward
from .linear import LayerNorm
from .norm import RMSNorm
from ._mixin import _SemanticModuleMixin
from ._precision import (
    ActivationKind,
    EngineKind,
    NormKind,
    _resolve_engine_alias,
)


class TransformerBlock(nn.Module, _SemanticModuleMixin):
    """Pre-norm transformer block composed of Attention and FeedForward facades."""

    def __init__(
        self,
        dim: int,
        *,
        dim_out: int | None = None,
        heads: int = 8,
        head_dim: int | None = None,
        ffn_mult: int = 4,
        ffn_activation: ActivationKind = "gelu",
        norm: NormKind | None = "layernorm",
        eps: float = 1e-5,
        qkv_bias: bool = False,
        out_bias: bool = False,
        ffn_bias: bool = True,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        causal: bool = False,
        engine: str | None = None,
        ffn_engine: EngineKind | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.dim = int(dim)
        self.dim_out = int(dim if dim_out is None else dim_out)
        self.heads = int(heads)
        self.norm_kind = norm
        self._init_runtime_intent(engine=engine)
        if self.engine not in {"torch", "tilelang"}:
            raise ValueError(f"unsupported TransformerBlock engine: {self.engine}")
        resolved_ffn_engine: EngineKind = (
            "torch"
            if ffn_engine is None and self.engine == "tilelang"
            else (ffn_engine or "torch")
        )
        if norm is None:
            self.norm1: nn.Module | None = None
            self.norm2: nn.Module | None = None
        elif norm == "layernorm":
            self.norm1 = LayerNorm(self.dim, eps=eps, engine="torch")
            self.norm2 = LayerNorm(self.dim, eps=eps, engine="torch")
        elif norm == "rmsnorm":
            self.norm1 = RMSNorm(self.dim, eps=eps)
            self.norm2 = RMSNorm(self.dim, eps=eps)
        else:
            raise ValueError(f"unsupported norm: {norm}")
        self.attn = Attention(
            self.dim,
            dim_out=self.dim,
            heads=heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            dropout=dropout,
            causal=causal,
            engine=self.engine,
        )
        self.ffn = FeedForward(
            self.dim,
            dim_out=self.dim_out,
            mult=ffn_mult,
            activation=ffn_activation,
            dropout=ffn_dropout,
            bias=ffn_bias,
            engine=resolved_ffn_engine,
        )

    def configure_runtime(
        self,
        *,
        engine: str | None = None,
        activation_dtype: str | None = None,
        weight_dtype: str | None = None,
        bias_dtype: str | None = None,
        mma_dtype: str | None = None,
        accum_dtype: str | None = None,
        output_dtype: str | None = None,
    ) -> None:
        if engine is not None:
            resolved = _resolve_engine_alias(
                engine=engine,
                context="TransformerBlock runtime",
            )
            if resolved not in {"torch", "tilelang"}:
                raise ValueError(f"unsupported TransformerBlock engine: {resolved}")
            self.engine = resolved
        _SemanticModuleMixin.configure_runtime(
            self,
            engine=None,
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            mma_dtype=mma_dtype,
            accum_dtype=accum_dtype,
            output_dtype=output_dtype,
        )
        self.attn.configure_runtime(
            engine=self.engine,
            activation_dtype=self.runtime_precision["activation"],
            weight_dtype=self.runtime_precision["weight"],
            bias_dtype=self.runtime_precision["bias"],
            mma_dtype=self.runtime_precision["mma"],
            accum_dtype=self.runtime_precision["accum"],
            output_dtype=self.runtime_precision["output"],
        )
        ffn_engine = "torch" if self.engine == "tilelang" else self.engine
        if ffn_engine in {"torch", "triton"}:
            self.ffn.configure_runtime(
                engine=ffn_engine,
                activation_dtype=self.runtime_precision["activation"],
                weight_dtype=self.runtime_precision["weight"],
                bias_dtype=self.runtime_precision["bias"],
                mma_dtype=self.runtime_precision["mma"],
                accum_dtype=self.runtime_precision["accum"],
                output_dtype=self.runtime_precision["output"],
            )

    def runtime_config(self) -> dict[str, Any]:
        return {
            **_SemanticModuleMixin.runtime_config(self),
            "norm": self.norm_kind,
            "attention": self.attn.runtime_config(),
            "feedforward": self.ffn.runtime_config(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = x if self.norm1 is None else self.norm1(x)
        hidden = residual + self.attn(hidden)
        residual = hidden
        hidden = hidden if self.norm2 is None else self.norm2(hidden)
        return residual + self.ffn(hidden)
