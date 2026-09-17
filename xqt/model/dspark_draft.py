"""Native implementation of the official ``openbmb/MiniCPM5-2B-DSpark`` draft.

The released checkpoint is a *DSpark* speculative-decoding draft: a 5-layer
Qwen3-style decoder that (a) consumes hidden states from five layers of the
target model, (b) attends over a KV cache built directly from those target
hidden states, and (c) chains its block predictions through a rank-256 Markov
head. It is not the same architecture as :class:`xqt.runtime.spec_decode.DSparkDrafter`,
which was written before the official checkpoint existed.

Semantics here were ported from SGLang's reference implementation
(``python/sglang/srt/models/dspark.py`` and ``.../models/dflash.py``) so the
released state dict loads tensor-for-tensor. SGLang is a *semantic reference
only* -- nothing in this module imports it.

Two conventions from the reference are easy to get wrong and are load-bearing:

* ``target_layer_ids`` are **HF-style post-layer indices**. The reference is
  explicit that "DFlash uses hidden states *after* each selected target layer
  (HF-style)" while SGLang captures "before layer i", so the capture point for
  id ``L`` is ``hidden_states[L + 1]`` in HF's ``output_hidden_states`` tuple.
* The block's causal window is **bottom-right aligned**, not top-left: row ``i``
  sits at absolute position ``base + i`` and sees ctx ``[0, base)`` plus block
  rows ``[0, i]``. ``F.scaled_dot_product_attention(is_causal=True)`` aligns to
  the top left when ``q_len < kv_len`` and is therefore unusable here.

Shape chain, per draft round::

    target hidden (5 layers)  [N, 5*2048]
      -> fc                   [N, 2048]
      -> hidden_norm
      -> per draft layer: k/v projection, k_norm, RoPE -> draft KV cache
    block embeds [7, 2048] (row 0 is the anchor token, rows 1..6 are mask tokens)
      -> 5 decoder layers attending over ctx KV + the block itself
      -> norm                 [7, 2048]
      -> target lm_head        [7, vocab]        (base logits)
      -> Markov chain          [7]               (draft tokens)

The Markov step is ``logits_k = base_k + W2 @ W1[prev_k]`` with
``prev_0 = anchor`` and ``prev_k = argmax(logits_{k-1})``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.errors import XQTBackendError

_DEFAULT_CHECKPOINT = "downloads/MiniCPM5-2B-DSpark"


@dataclass(frozen=True)
class DSparkDraftConfig:
    """Fields read from the checkpoint's ``config.json``."""

    hidden_size: int = 2048
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 131072
    vocab_size: int = 130560
    block_size: int = 7
    mask_token_id: int = 75982
    markov_rank: int = 256
    target_layer_ids: tuple[int, ...] = (1, 10, 20, 30, 39)
    sample_from_anchor: bool = True
    attention_value_scale: float | None = None
    confidence_head_with_markov: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> DSparkDraftConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in known:
                continue
            if key == "target_layer_ids":
                value = tuple(int(item) for item in value)
            kwargs[key] = value
        # The released checkpoint has no flat ``rope_theta``: it carries the
        # value only under ``rope_parameters``, which is what a 5.x
        # ``Qwen3RotaryEmbedding`` reads. A 4.x runtime reads the flat key
        # instead and would silently fall back to ``rope_theta=10000`` -- a 500x
        # error that still produces fluent-looking tokens. Lift the nested value
        # so the field means the same thing either way; an explicit flat key
        # always wins.
        if "rope_theta" not in kwargs:
            params = payload.get("rope_parameters")
            if isinstance(params, dict) and "rope_theta" in params:
                kwargs["rope_theta"] = float(params["rope_theta"])
        config = cls(**kwargs)
        if config.markov_rank <= 0:
            raise XQTBackendError(
                "the DSpark draft requires markov_rank > 0; the Markov head is "
                f"the semi-autoregressive core, got {config.markov_rank}"
            )
        return config

    @property
    def num_target_features(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def query_token_num(self) -> int:
        """Block length fed to the draft (equals ``block_size`` when anchoring on
        the bonus token, one longer when the anchor row is dropped)."""

        return self.block_size if self.sample_from_anchor else self.block_size + 1


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with the reference's *exact* rounding.

    ``Qwen3RMSNorm`` reduces in float32, casts the normalized value back to the
    activation dtype, and only then scales by the weight in that same dtype.
    Folding the weight multiply into the float32 stage is mathematically
    equivalent but rounds differently, which is visible after five layers.
    """

    dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normalised = (x32 * torch.rsqrt(variance + eps)).to(dtype)
    return weight * normalised


class DSparkRMSNorm(nn.Module):
    """RMSNorm with the reference's optional fused residual.

    ``forward(x, residual)`` returns ``(norm(residual + x), residual + x)``,
    which is how the draft layers carry the residual stream.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
        else:
            residual = residual + x
        return _rms_norm(residual, self.weight, self.variance_epsilon), residual


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """HF/Qwen3 rotary embedding (``rotate_half`` convention).

    ``cos``/``sin`` are ``[N, head_dim]`` gathered per position, so the head
    dimension is the last axis of both operands. The tables are built in
    float32 but the rotation itself runs in the activation dtype, matching the
    reference.
    """

    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class DSparkAttention(nn.Module):
    """Qwen3-style GQA attention with per-head Q/K RMSNorm."""

    def __init__(self, config: DSparkDraftConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.v_scale = config.attention_value_scale

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)
        self.q_norm = DSparkRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = DSparkRMSNorm(self.head_dim, config.rms_norm_eps)

    def _norm_heads(self, x: torch.Tensor, norm: DSparkRMSNorm) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.head_dim)
        flat, _ = norm(flat)
        return flat.view(shape)

    def ctx_kv(
        self,
        ctx_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build K/V for context positions from projected target hidden states.

        Only K/V are needed: Q for the cached positions is never consumed.
        """

        tokens = ctx_hidden.shape[0]
        k = self.k_proj(ctx_hidden).view(tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(ctx_hidden)
        k = self._norm_heads(k, self.k_norm)
        k = _apply_rope(k, cos.unsqueeze(1), sin.unsqueeze(1)).reshape(tokens, -1)
        if self.v_scale is not None:
            v = v * self.v_scale
        return k, v

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        base: int,
    ) -> torch.Tensor:
        """Block attention; writes this block's K/V at ``base..base+rows-1``."""

        rows = hidden.shape[0]
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        q = self._norm_heads(
            q.view(rows, self.num_heads, self.head_dim), self.q_norm
        )
        k = self._norm_heads(
            k.view(rows, self.num_kv_heads, self.head_dim), self.k_norm
        )
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q = _apply_rope(q, cos_q, sin_q).reshape(rows, self.q_size)
        k = _apply_rope(k, cos_q, sin_q).reshape(rows, self.kv_size)
        if self.v_scale is not None:
            v = v * self.v_scale

        k_cache[:, :, base : base + rows] = k.view(
            1, rows, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v_cache[:, :, base : base + rows] = v.view(
            1, rows, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        q_4d = q.view(1, rows, self.num_heads, self.head_dim).transpose(1, 2)
        valid = base + rows
        # No mask, by design. The draft is a *parallel* block decoder: every row
        # is produced in one forward pass and the training mask
        # (``dspark_mask_mod``) only restricts a query to its own block -- it is
        # NOT causal inside the block. The reference inference path calls
        # ``_forward_backbone(..., attention_mask=None, is_causal=False)`` for
        # exactly this reason. Making the block causal here would hide the later
        # mask rows from row 0 and silently break every slot.
        out = F.scaled_dot_product_attention(
            q_4d,
            k_cache[:, :, :valid],
            v_cache[:, :, :valid],
            attn_mask=None,
            enable_gqa=True,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).reshape(rows, self.q_size)
        return self.o_proj(out)


class DSparkMLP(nn.Module):
    def __init__(self, config: DSparkDraftConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config: DSparkDraftConfig) -> None:
        super().__init__()
        self.input_layernorm = DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DSparkAttention(config)
        self.post_attention_layernorm = DSparkRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = DSparkMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        base: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, residual = self.input_layernorm(hidden, residual)
        attn_out = self.self_attn(normed, cos, sin, k_cache, v_cache, base)
        normed, residual = self.post_attention_layernorm(attn_out, residual)
        return self.mlp(normed), residual


class DSparkMarkovHead(nn.Module):
    """Rank-``r`` factorisation of a full-vocabulary token transition.

    ``W = markov_w2 @ markov_w1.T`` is ``[vocab, vocab]`` at rank ``r``; the
    step bias only ever needs ``W2 @ W1[prev]``, which is a ``[r] -> [vocab]``
    projection rather than a dense ``vocab x vocab`` matrix.
    """

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(rank)
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def step_logits(
        self, base_logits: torch.Tensor, prev_tokens: torch.Tensor
    ) -> torch.Tensor:
        return base_logits + self.markov_w2(self.prev_embeddings(prev_tokens))

    @torch.inference_mode()
    def sample_block_greedy(
        self, base_logits: torch.Tensor, anchor_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Greedy chain over the block; returns ``[B, query_token_num]`` tokens."""

        proposal_len = base_logits.shape[1]
        prev = anchor_tokens.long()
        sampled: list[torch.Tensor] = []
        for step in range(proposal_len):
            prev = self.step_logits(base_logits[:, step, :], prev).argmax(dim=-1)
            sampled.append(prev)
        return torch.stack(sampled, dim=1)


class DSparkConfidenceHead(nn.Module):
    """Predicts whether a drafted token will survive verification.

    Input is the draft hidden concatenated with the Markov embedding of the
    preceding token, matching ``confidence_head_with_markov``.
    """

    def __init__(
        self, hidden_size: int, markov_rank: int, *, with_markov: bool = True
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = hidden_size + (markov_rank if self.with_markov else 0)
        self.proj = nn.Linear(input_dim, 1, bias=True)
        self.register_buffer(
            "sts_temperatures", torch.ones(()), persistent=False
        )

    def forward(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed is None:
                raise XQTBackendError(
                    "the DSpark confidence head was trained with the Markov "
                    "feature; pass markov_embed"
                )
            features = torch.cat([hidden, markov_embed.to(hidden.dtype)], dim=-1)
        else:
            features = hidden
        return self.proj(features.to(self.proj.weight.dtype)).squeeze(-1)

    def probability(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(hidden, markov_embed).float() / self.sts_temperatures
        )


class DSparkDraftModel(nn.Module):
    """The released MiniCPM5-2B DSpark draft, in plain PyTorch."""

    def __init__(self, config: DSparkDraftConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DSparkDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(config.num_target_features, config.hidden_size, bias=False)
        self.hidden_norm = DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(config.vocab_size, config.markov_rank)
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size,
            config.markov_rank,
            with_markov=config.confidence_head_with_markov,
        )

    # ------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(
        cls,
        path: str | Path = _DEFAULT_CHECKPOINT,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> DSparkDraftModel:
        directory = Path(path)
        config = DSparkDraftConfig.from_file(directory / "config.json")
        model = cls(config).to(device=device, dtype=dtype)
        state = _load_safetensors(directory / "model.safetensors", device=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise XQTBackendError(
                "the DSpark checkpoint does not match this implementation; "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        model.eval()
        return model

    # --------------------------------------------------------------- ctx KV
    @torch.inference_mode()
    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Concatenated target-layer features ``[N, 5*2048]`` -> ``[N, 2048]``."""

        expected = self.config.num_target_features
        if target_hidden.shape[-1] != expected:
            raise XQTBackendError(
                f"target hidden features must be {expected} "
                f"({len(self.config.target_layer_ids)} layers x "
                f"{self.config.hidden_size}); got {target_hidden.shape[-1]}"
            )
        projected = self.fc(target_hidden.to(self.fc.weight.dtype))
        normed, _ = self.hidden_norm(projected)
        return normed

    @torch.inference_mode()
    def ctx_kv(
        self, ctx_hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-layer K/V for context positions, from projected target hidden."""

        cos, sin = self._rope_tables(positions)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for layer in self.layers:
            k, v = layer.self_attn.ctx_kv(ctx_hidden, cos, sin)
            keys.append(k)
            values.append(v)
        return keys, values

    def _rope_tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE cos/sin for ``positions`` using the checkpoint's theta."""

        table = _rope_cache(self.config, positions.device)
        return table[0][positions], table[1][positions]

    # ---------------------------------------------------------------- block
    @torch.inference_mode()
    def block_forward(
        self,
        block_embeds: torch.Tensor,
        positions: torch.Tensor,
        k_caches: Sequence[torch.Tensor],
        v_caches: Sequence[torch.Tensor],
        base: int,
    ) -> torch.Tensor:
        """Run the draft over one block; returns the final-normalized hidden."""

        cos, sin = self._rope_tables(positions)
        hidden = block_embeds
        residual: torch.Tensor | None = None
        for index, layer in enumerate(self.layers):
            hidden, residual = layer(
                hidden, residual, cos, sin, k_caches[index], v_caches[index], base
            )
        if residual is None:
            return self.norm(hidden)[0]
        normed, _ = self.norm(hidden, residual)
        return normed

    @torch.inference_mode()
    def propose(
        self,
        block_embeds: torch.Tensor,
        positions: torch.Tensor,
        k_caches: Sequence[torch.Tensor],
        v_caches: Sequence[torch.Tensor],
        base: int,
        anchor_token: int,
        lm_head: nn.Module,
    ) -> tuple[list[int], torch.Tensor]:
        """One draft round: returns ``(draft_tokens, confidence)``.

        ``lm_head`` is the *target* model's head, applied to the draft's own
        hidden states -- the DSpark checkpoint carries no head of its own.
        """

        hidden = self.block_forward(block_embeds, positions, k_caches, v_caches, base)
        # A dense head must see its own weight dtype; a quantized head stores no
        # ``weight`` at all (packed int32 buffers + fp32 scales) and insists on
        # fp16/bf16 activations, so the draft's own dtype is the right answer.
        head_dtype = getattr(getattr(lm_head, "weight", None), "dtype", hidden.dtype)
        base_logits = lm_head(hidden.to(head_dtype)).unsqueeze(0)
        anchor = torch.tensor([anchor_token], dtype=torch.long, device=hidden.device)
        tokens = self.markov_head.sample_block_greedy(base_logits, anchor)

        prev_ids = torch.cat([anchor, tokens[0, :-1]])
        markov_embed = self.markov_head.prev_embeddings(prev_ids)
        confidence = self.confidence_head.probability(hidden, markov_embed)
        return [int(token) for token in tokens[0].tolist()], confidence


def _load_safetensors(path: Path, *, device: torch.device | str) -> dict[str, Any]:
    from safetensors.torch import load_file

    if not path.exists():
        raise XQTBackendError(
            f"DSpark checkpoint not found at {path}; download "
            "openbmb/MiniCPM5-2B-DSpark (config.json + model.safetensors)"
        )
    return load_file(str(path), device=str(device))


_ROPE_CACHE: dict[tuple[int, int, torch.dtype, str], torch.Tensor] = {}


def _rope_cache(config: DSparkDraftConfig, device: torch.device) -> torch.Tensor:
    """``[2, max_position, head_dim]`` table: index 0 is cos, index 1 is sin."""

    key = (config.head_dim, config.max_position_embeddings, torch.float32, str(device))
    cached = _ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    inv_freq = 1.0 / (
        config.rope_theta
        ** (
            torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device)
            / config.head_dim
        )
    )
    positions = torch.arange(
        config.max_position_embeddings, dtype=torch.float32, device=device
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    table = torch.stack((emb.cos(), emb.sin()), dim=0).contiguous()
    _ROPE_CACHE[key] = table
    return table
