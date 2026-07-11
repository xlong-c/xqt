"""Self-attention facade with torch SDPA and TileLang engine intent."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .linear import Linear
from ._mixin import _SemanticModuleMixin
from ._precision import _resolve_engine_alias


class Attention(nn.Module, _SemanticModuleMixin):
    """Self-attention facade with torch SDPA and TileLang engine intent."""

    def __init__(
        self,
        dim: int,
        *,
        dim_out: int | None = None,
        heads: int = 8,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
        engine: str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if dim <= 0:
            raise ValueError("dim must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
        resolved_head_dim = int(dim // heads if head_dim is None else head_dim)
        if resolved_head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim is None and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when head_dim is omitted")
        self.dim = int(dim)
        self.dim_out = int(dim if dim_out is None else dim_out)
        self.heads = int(heads)
        self.head_dim = resolved_head_dim
        self.inner_dim = self.heads * self.head_dim
        self.dropout_p = float(dropout)
        self.causal = bool(causal)
        self._init_runtime_intent(engine=engine)
        if self.engine not in {"torch", "tilelang"}:
            raise ValueError(f"unsupported Attention engine: {self.engine}")
        proj_engine = self.engine if self.engine in {"torch", "triton"} else "torch"
        self.q_proj = Linear(self.dim, self.inner_dim, bias=qkv_bias, engine=proj_engine)
        self.k_proj = Linear(self.dim, self.inner_dim, bias=qkv_bias, engine=proj_engine)
        self.v_proj = Linear(self.dim, self.inner_dim, bias=qkv_bias, engine=proj_engine)
        self.out_proj = Linear(
            self.inner_dim,
            self.dim_out,
            bias=out_bias,
            engine=proj_engine,
        )
        self.runtime_fallback: dict[str, Any] | None = None
        self.runtime_fallback_count = 0

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
            resolved = _resolve_engine_alias(engine=engine, context="Attention runtime")
            if resolved not in {"torch", "tilelang"}:
                raise ValueError(f"unsupported Attention engine: {resolved}")
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
        proj_engine = self.engine if self.engine in {"torch", "triton"} else "torch"
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            projection.configure_runtime(
                engine=proj_engine,
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
            "heads": self.heads,
            "head_dim": self.head_dim,
            "causal": self.causal,
            "dropout": self.dropout_p,
            "fallback": (
                None if self.runtime_fallback is None else dict(self.runtime_fallback)
            ),
            "fallback_count": self.runtime_fallback_count,
        }

    def _reshape_qkv(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = tensor.shape
        return (
            tensor.reshape(batch, seq, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _heads, seq, head_dim = tensor.shape
        return (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, seq, self.heads * head_dim)
        )

    def _record_runtime_fallback(self, *, stage: str, reason: Exception) -> None:
        self.runtime_fallback_count += 1
        self.runtime_fallback = {
            "engine": self.engine,
            "stage": stage,
            "reason": str(reason),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Attention expects 3D input shaped [batch, seq, dim]")
        if int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"Attention input last dim {int(x.shape[-1])} does not match dim {self.dim}"
            )
        q = self._reshape_qkv(self.q_proj(x))
        k = self._reshape_qkv(self.k_proj(x))
        v = self._reshape_qkv(self.v_proj(x))
        if self.engine == "tilelang" and self.dropout_p == 0.0:
            try:
                from xqt.operator_opt.kernels.attention import (
                    fused_attention_forward_tilelang,
                )

                attn = fused_attention_forward_tilelang(
                    q.to(dtype=torch.float16),
                    k.to(dtype=torch.float16),
                    v.to(dtype=torch.float16),
                    causal=self.causal,
                    dropout_p=0.0,
                ).to(dtype=x.dtype)
            except Exception as exc:
                self._record_runtime_fallback(stage="attention", reason=exc)
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout_p,
                    is_causal=self.causal,
                )
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p,
                is_causal=self.causal,
            )
        return self.out_proj(self._merge_heads(attn))
