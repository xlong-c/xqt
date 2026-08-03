"""Smoke-only deterministic small LLM module for model-family recipes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SmokeLLMBlock(nn.Module):
    """Tiny decoder block with separate q/k/v projections for KV calibration."""

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        num_heads: int = 2,
        dim_feedforward: int = 32,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp_gate = nn.Linear(hidden_dim, dim_feedforward)
        self.mlp_up = nn.Linear(hidden_dim, dim_feedforward)
        self.mlp_down = nn.Linear(dim_feedforward, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.norm1(x)
        batch, seq, _ = hidden.shape
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.view(batch, seq, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        attn_output = F.scaled_dot_product_attention(
            reshape(query),
            reshape(key),
            reshape(value),
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, self.hidden_dim)
        hidden = residual + self.out_proj(attn_output)
        residual = hidden
        hidden = self.norm2(hidden)
        hidden = residual + self.mlp_down(F.silu(self.mlp_gate(hidden)) * self.mlp_up(hidden))
        return hidden


class SmokeLLM(nn.Module):
    """Tiny decoder-only LLM-style module for CPU smoke tests."""

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        hidden_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 32,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                SmokeLLMBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)


def build_smoke_llm(
    *,
    vocab_size: int = 64,
    hidden_dim: int = 16,
    num_heads: int = 2,
    num_layers: int = 2,
    dim_feedforward: int = 32,
) -> SmokeLLM:
    """Build a deterministic small LLM smoke model."""

    return SmokeLLM(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )


__all__ = [
    "SmokeLLM",
    "SmokeLLMBlock",
    "build_smoke_llm",
]
