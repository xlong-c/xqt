"""Attention module pruning rewrites."""

from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .linear_conv import prune_linear_in_features, prune_linear_out_features


def _dropout_probability(module_or_probability: Any) -> float:
    if isinstance(module_or_probability, nn.Dropout):
        return float(module_or_probability.p)
    return float(module_or_probability)


def infer_attention_role(module: nn.Module) -> str:
    for attribute_name in ("attention_role", "attn_role"):
        role = getattr(module, attribute_name, None)
        if isinstance(role, str) and role in {"self", "cross"}:
            return role
    for attribute_name in ("is_cross_attention", "cross_attention"):
        value = getattr(module, attribute_name, None)
        if isinstance(value, bool):
            return "cross" if value else "self"
    try:
        parameter_names = list(inspect.signature(module.forward).parameters)
    except (TypeError, ValueError):
        return "self"
    cross_names = {
        "context",
        "encoder_hidden_states",
        "memory",
        "key_value_states",
        "kv",
    }
    return "cross" if any(name in cross_names for name in parameter_names[1:]) else "self"


def attention_variant(*, num_heads: int, num_kv_heads: int) -> str:
    if num_kv_heads == num_heads:
        return "mha"
    if num_kv_heads == 1:
        return "mqa"
    return "gqa"


class PrunedMultiHeadAttention(nn.Module):
    """Attention module with fewer heads but unchanged input/output embed dim."""

    def __init__(self, source: nn.Module, keep_head_indices: list[int]) -> None:
        super().__init__()
        qkv = getattr(source, "qkv")
        proj = getattr(source, "proj")
        attn_dropout = getattr(source, "attn_dropout")
        proj_dropout = getattr(source, "proj_dropout")
        embed_dim = int(getattr(source, "embed_dim"))
        head_dim = int(getattr(source, "head_dim"))
        keep_features = [
            head * head_dim + offset
            for head in keep_head_indices
            for offset in range(head_dim)
        ]
        qkv_keep = (
            keep_features
            + [embed_dim + index for index in keep_features]
            + [2 * embed_dim + index for index in keep_features]
        )

        self.embed_dim = embed_dim
        self.num_heads = len(keep_head_indices)
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv = prune_linear_out_features(qkv, qkv_keep)
        self.attn_dropout = nn.Dropout(attn_dropout.p)
        self.proj = prune_linear_in_features(proj, keep_features)
        self.proj_dropout = nn.Dropout(proj_dropout.p)
        self.train(source.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        x = self.proj(x)
        return self.proj_dropout(x)


class PrunedSplitProjectionAttention(nn.Module):
    """Attention module rewritten from split q/k/v/out projections."""

    def __init__(self, source: nn.Module, keep_head_indices: list[int]) -> None:
        super().__init__()
        q_proj = getattr(source, "q_proj")
        k_proj = getattr(source, "k_proj")
        v_proj = getattr(source, "v_proj")
        out_proj = getattr(source, "out_proj")
        num_heads = int(getattr(source, "num_heads"))
        head_dim = int(getattr(source, "head_dim"))
        embed_dim = int(getattr(source, "embed_dim"))
        dropout = float(getattr(source, "dropout", 0.0))
        if not all(isinstance(module, nn.Linear) for module in (q_proj, k_proj, v_proj, out_proj)):
            raise TypeError("split attention pruning requires q_proj/k_proj/v_proj/out_proj")
        if len(keep_head_indices) >= num_heads:
            raise ValueError("keep_head_indices must prune at least one attention head")

        keep_features = [
            head * head_dim + offset
            for head in keep_head_indices
            for offset in range(head_dim)
        ]
        self.embed_dim = embed_dim
        self.num_heads = len(keep_head_indices)
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.attention_role = infer_attention_role(source)
        self.q_proj = prune_linear_out_features(q_proj, keep_features)
        self.k_proj = prune_linear_out_features(k_proj, keep_features)
        self.v_proj = prune_linear_out_features(v_proj, keep_features)
        self.out_proj = prune_linear_in_features(out_proj, keep_features)
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(
            _dropout_probability(getattr(source, "proj_dropout", dropout))
        )
        self.train(source.training)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        if self.attention_role == "cross":
            if context is None:
                raise ValueError("cross-attention pruning rewrite requires context input")
            kv_source = context
        else:
            kv_source = x if context is None else context
        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        kv_batch_size, kv_token_count, _ = kv_source.shape
        if kv_batch_size != batch_size:
            raise ValueError("cross-attention context batch size must match query batch size")
        k = self.k_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_heads,
            self.head_dim,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        return self.proj_dropout(self.out_proj(x))


class PrunedGroupedQueryAttention(nn.Module):
    """Attention rewrite for GQA/MQA split q/k/v/out projections."""

    def __init__(self, source: nn.Module, keep_kv_head_indices: list[int]) -> None:
        super().__init__()
        q_proj = getattr(source, "q_proj")
        k_proj = getattr(source, "k_proj")
        v_proj = getattr(source, "v_proj")
        out_proj = getattr(source, "out_proj")
        num_heads = int(getattr(source, "num_heads"))
        num_kv_heads = int(getattr(source, "num_kv_heads"))
        head_dim = int(getattr(source, "head_dim"))
        embed_dim = int(getattr(source, "embed_dim"))
        dropout = float(getattr(source, "dropout", 0.0))
        if not all(isinstance(module, nn.Linear) for module in (q_proj, k_proj, v_proj, out_proj)):
            raise TypeError("grouped-query attention pruning requires q_proj/k_proj/v_proj/out_proj")
        if num_kv_heads <= 0 or num_heads <= 0:
            raise ValueError("grouped-query attention requires positive num_heads and num_kv_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("grouped-query attention requires num_heads divisible by num_kv_heads")
        if len(keep_kv_head_indices) >= num_kv_heads:
            raise ValueError("keep_kv_head_indices must prune at least one kv head")

        query_heads_per_kv_head = num_heads // num_kv_heads
        keep_q_head_indices = [
            kv_head_index * query_heads_per_kv_head + query_head_offset
            for kv_head_index in keep_kv_head_indices
            for query_head_offset in range(query_heads_per_kv_head)
        ]
        q_keep_features = [
            head_index * head_dim + offset
            for head_index in keep_q_head_indices
            for offset in range(head_dim)
        ]
        kv_keep_features = [
            head_index * head_dim + offset
            for head_index in keep_kv_head_indices
            for offset in range(head_dim)
        ]

        self.embed_dim = embed_dim
        self.num_heads = len(keep_q_head_indices)
        self.num_kv_heads = len(keep_kv_head_indices)
        self.query_heads_per_kv_head = query_heads_per_kv_head
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.attention_role = infer_attention_role(source)
        self.attention_variant = attention_variant(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.q_proj = prune_linear_out_features(q_proj, q_keep_features)
        self.k_proj = prune_linear_out_features(k_proj, kv_keep_features)
        self.v_proj = prune_linear_out_features(v_proj, kv_keep_features)
        self.out_proj = prune_linear_in_features(out_proj, q_keep_features)
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(
            _dropout_probability(getattr(source, "proj_dropout", dropout))
        )
        self.train(source.training)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        if self.attention_role == "cross":
            if context is None:
                raise ValueError("cross-attention pruning rewrite requires context input")
            kv_source = context
        else:
            kv_source = x if context is None else context
        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        kv_batch_size, kv_token_count, _ = kv_source.shape
        if kv_batch_size != batch_size:
            raise ValueError("cross-attention context batch size must match query batch size")
        k = self.k_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_kv_heads,
            self.head_dim,
        )
        v = self.v_proj(kv_source).reshape(
            kv_batch_size,
            kv_token_count,
            self.num_kv_heads,
            self.head_dim,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        attn = self.attn_dropout(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        return self.proj_dropout(self.out_proj(x))
