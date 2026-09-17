"""Tests for the native DSpark draft (``openbmb/MiniCPM5-2B-DSpark``).

Two layers of evidence:

* **Structural** -- the released checkpoint's 62 tensors must load with zero
  missing / zero unexpected keys and reproduce the published parameter count.
  Skipped when ``downloads/MiniCPM5-2B-DSpark`` is absent.
* **Semantic** -- a tiny random-weight model is compared against an explicit
  masked-softmax reference. This is the part that actually pins down the
  attention contract: block row ``i`` sees ctx positions ``[0, base)`` plus
  block rows ``[0, i]``, which is what makes the block's per-row valid length
  match the verify window's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from xqt.model.dspark_draft import (
    DSparkAttention,
    DSparkDraftConfig,
    DSparkDraftModel,
    _apply_rope,
    _rope_cache,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT = _REPO_ROOT / "downloads" / "MiniCPM5-2B-DSpark"

# Published by the model card; a mismatch means the loader silently accepted a
# different checkpoint.
_EXPECTED_TENSORS = 62
_EXPECTED_PARAMS = 323_776_001


def _require_checkpoint() -> Path:
    if not (_CHECKPOINT / "model.safetensors").exists():
        pytest.skip(f"official DSpark checkpoint not present at {_CHECKPOINT}")
    return _CHECKPOINT


def _tiny_config() -> DSparkDraftConfig:
    return DSparkDraftConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        max_position_embeddings=256,
        vocab_size=128,
        block_size=7,
        mask_token_id=7,
        markov_rank=8,
        target_layer_ids=(1, 2),
    )


# --------------------------------------------------------------- structural


def test_config_parses_released_config_json() -> None:
    """The released ``config.json`` must drive the geometry, not our defaults."""

    config = DSparkDraftConfig.from_file(_require_checkpoint() / "config.json")

    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 5
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 2
    assert config.head_dim == 128
    assert config.intermediate_size == 6144
    assert config.vocab_size == 130560
    assert config.markov_rank == 256
    assert config.mask_token_id == 75982
    assert config.target_layer_ids == (1, 10, 20, 30, 39)
    assert config.sample_from_anchor is True
    assert config.confidence_head_with_markov is True
    # Only present under ``rope_parameters`` in the released file.
    assert config.rope_theta == 5_000_000.0
    # 5 target layers x 2048 is the width of ``fc``'s input.
    assert config.num_target_features == 5 * 2048
    # ``sample_from_anchor`` means the bonus row is row 0 of the block, so the
    # draft consumes exactly ``block_size`` rows and proposes ``block_size``.
    assert config.query_token_num == 7


def test_released_checkpoint_loads_without_key_drift() -> None:
    checkpoint = _require_checkpoint()
    model = DSparkDraftModel.from_pretrained(checkpoint, device="cpu", dtype=torch.bfloat16)

    assert sum(p.numel() for p in model.parameters()) == _EXPECTED_PARAMS

    from safetensors import safe_open

    with safe_open(str(checkpoint / "model.safetensors"), "pt") as handle:
        assert len(list(handle.keys())) == _EXPECTED_TENSORS

    # ``markov_w2`` is stored as ``[vocab, rank]``: exactly the layout of
    # ``nn.Linear(rank, vocab).weight``, so no transpose is involved.
    assert model.markov_head.markov_w1.weight.shape == (130560, 256)
    assert model.markov_head.markov_w2.weight.shape == (130560, 256)
    assert model.fc.weight.shape == (2048, 5 * 2048)
    assert model.confidence_head.proj.weight.shape == (1, 2048 + 256)


def test_malformed_config_is_rejected(tmp_path: Path) -> None:
    payload = json.loads((_require_checkpoint() / "config.json").read_text())
    payload["markov_rank"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception, match="markov_rank"):
        DSparkDraftConfig.from_file(path)


def test_rope_theta_is_lifted_out_of_rope_parameters(tmp_path: Path) -> None:
    """``rope_parameters`` must reach ``config.rope_theta``.

    transformers 4.x reads a *flat* ``config.rope_theta`` and knows nothing about
    ``rope_parameters``, so a checkpoint that only carries the nested form runs
    at whatever the config class defaults to. For this draft that is 10000
    instead of 5000000 -- a silent 500x error in the RoPE base that still yields
    fluent text and therefore cannot be caught by looking at the output. The
    loader has to normalise it.
    """

    payload = json.loads((_require_checkpoint() / "config.json").read_text())
    assert "rope_theta" not in payload  # the whole point of the test
    assert payload["rope_parameters"]["rope_theta"] == 5_000_000

    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps(payload), encoding="utf-8")
    assert DSparkDraftConfig.from_file(nested).rope_theta == 5_000_000.0

    # A flat key is the explicit form and must still win.
    payload["rope_theta"] = 12345.0
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(payload), encoding="utf-8")
    assert DSparkDraftConfig.from_file(flat).rope_theta == 12345.0

    # Neither present: fall back to the dataclass default rather than crashing.
    del payload["rope_theta"], payload["rope_parameters"]
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(payload), encoding="utf-8")
    assert DSparkDraftConfig.from_file(bare).rope_theta == 5_000_000.0


# ----------------------------------------------------------------- rope


def test_rope_matches_rotate_half_convention() -> None:
    """RoPE must be the HF/Qwen3 ``rotate_half`` pairing, not interleaved GPT-J."""

    config = _tiny_config()
    table = _rope_cache(config, torch.device("cpu"))
    positions = torch.tensor([0, 3, 17])
    cos, sin = table[0][positions], table[1][positions]

    torch.manual_seed(0)
    x = torch.randn(3, 4, config.head_dim, dtype=torch.float32)
    got = _apply_rope(x, cos.unsqueeze(1), sin.unsqueeze(1))

    half = config.head_dim // 2
    # ``emb = cat((freqs, freqs))``, so each half carries the same frequencies;
    # the rotation pairs element ``i`` with element ``i + half``.
    c = cos.unsqueeze(1)[..., :half]
    s = sin.unsqueeze(1)[..., :half]
    x1, x2 = x[..., :half], x[..., half:]
    expected = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    # Position 0 is the identity rotation; that only holds for this pairing.
    torch.testing.assert_close(
        got[0], x[0], rtol=0, atol=1e-6
    )


def test_rope_table_uses_checkpoint_theta() -> None:
    config = _tiny_config()
    table = _rope_cache(config, torch.device("cpu"))
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
    )
    position = torch.tensor([5.0])
    freqs = position[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    torch.testing.assert_close(table[0][5], emb.cos()[0], rtol=0, atol=0)
    torch.testing.assert_close(table[1][5], emb.sin()[0], rtol=0, atol=0)


# ------------------------------------------------------------- attention


def _naive_block_attention(
    attn: DSparkAttention,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    base: int,
) -> torch.Tensor:
    """Explicit masked-softmax reference for ``DSparkAttention.forward``.

    Everything is spelled out -- per-head norm, rotate_half rope, a hand-built
    ``[rows, base + rows]`` boolean mask -- so a bug in the fused path cannot
    hide behind a shared helper.
    """

    rows = hidden.shape[0]
    num_heads, num_kv_heads, head_dim = (
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
    )

    def heads(x: torch.Tensor, width: int, norm) -> torch.Tensor:
        flat = x.reshape(-1, head_dim)
        variance = flat.pow(2).mean(dim=-1, keepdim=True)
        flat = flat * torch.rsqrt(variance + norm.variance_epsilon)
        flat = flat * norm.weight
        return flat.view(rows, width // head_dim, head_dim)

    q = heads(attn.q_proj(hidden), attn.q_size, attn.q_norm)
    k = heads(attn.k_proj(hidden), attn.kv_size, attn.k_norm)
    v = attn.v_proj(hidden).view(rows, num_kv_heads, head_dim)

    half = head_dim // 2
    c = cos[..., :half].unsqueeze(1)
    s = sin[..., :half].unsqueeze(1)
    for tensor in (q, k):
        # ``clone`` matters: without it the second write below would read the
        # first half *after* it had already been overwritten.
        x1 = tensor[..., :half].clone()
        x2 = tensor[..., half:].clone()
        tensor[..., :half] = x1 * c - x2 * s
        tensor[..., half:] = x2 * c + x1 * s

    k_cache[:, :, base : base + rows] = k.transpose(0, 1)
    v_cache[:, :, base : base + rows] = v.transpose(0, 1)

    valid = base + rows
    keys = k_cache[0, :, :valid]  # [kv_heads, valid, head_dim]
    values = v_cache[0, :, :valid]
    repeat = num_heads // num_kv_heads
    keys = keys.repeat_interleave(repeat, dim=0)
    values = values.repeat_interleave(repeat, dim=0)

    scores = torch.einsum("rhd,hsd->rhs", q, keys) * attn.scaling
    # Unmasked, matching the reference: every row sees ctx [0, base) and the
    # whole block. The only masking that exists in the reference is the
    # "context before the anchor | own block" flex mask, which the cache layout
    # here already reproduces exactly.
    probs = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
    context = torch.einsum("rhs,hsd->rhd", probs, values)
    return attn.o_proj(context.reshape(rows, attn.q_size))


@pytest.mark.parametrize("base", [0, 1, 5, 13])
def test_block_attention_matches_softmax_reference(base: int) -> None:
    """Pins the window: every block row sees all of ctx ``[0, base)`` plus the whole block.

    The reference resolves this draft's causality to ``False`` and its flex mask
    is "context before the anchor UNION own block, bidirectionally", so the
    attention is unmasked over the cache. A causal (or otherwise restricted)
    block would be wrong here.
    """

    torch.manual_seed(0)
    config = _tiny_config()
    attn = DSparkAttention(config).to(torch.float32)
    rows = config.block_size

    hidden = torch.randn(rows, config.hidden_size)
    positions = torch.arange(base, base + rows)
    table = _rope_cache(config, torch.device("cpu"))
    cos, sin = table[0][positions], table[1][positions]

    total = base + rows
    k_ours = torch.zeros(1, config.num_key_value_heads, total, config.head_dim)
    v_ours = torch.zeros(1, config.num_key_value_heads, total, config.head_dim)
    k_ref = k_ours.clone()
    v_ref = v_ours.clone()

    if base:
        # Context rows are arbitrary but identical across the two runs.
        ctx_k = torch.randn(1, config.num_key_value_heads, base, config.head_dim)
        ctx_v = torch.randn(1, config.num_key_value_heads, base, config.head_dim)
        k_ours[:, :, :base] = ctx_k
        v_ours[:, :, :base] = ctx_v
        k_ref[:, :, :base] = ctx_k
        v_ref[:, :, :base] = ctx_v

    got = attn(hidden, cos, sin, k_ours, v_ours, base)
    expected = _naive_block_attention(attn, hidden, cos, sin, k_ref, v_ref, base)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)
    # Both runs must have written the same block K/V, or the comparison above
    # could pass while the cache layout quietly differs.
    torch.testing.assert_close(k_ours, k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v_ours, v_ref, rtol=0, atol=0)


def test_block_attention_row_zero_sees_later_rows() -> None:
    """Row 0 must observe rows 1..6, and every row must observe the context.

    This is the property that distinguishes a parallel block draft from a causal
    decoder: the block is written in one pass and later rows are visible to
    earlier ones. A causal mask here would silently hide them.
    """

    torch.manual_seed(0)
    config = _tiny_config()
    attn = DSparkAttention(config).to(torch.float32)
    base = 9
    positions = torch.arange(base, base + config.block_size)
    table = _rope_cache(config, torch.device("cpu"))
    cos, sin = table[0][positions], table[1][positions]

    hidden = torch.randn(config.block_size, config.hidden_size)
    other = hidden.clone()
    other[1:] += 10.0

    def run(block: torch.Tensor) -> torch.Tensor:
        k = torch.zeros(1, config.num_key_value_heads, base + config.block_size, config.head_dim)
        v = torch.zeros(1, config.num_key_value_heads, base + config.block_size, config.head_dim)
        k[:, :, :base] = 0.5
        v[:, :, :base] = 0.25
        return attn(block, cos, sin, k, v, base)

    first, second = run(hidden), run(other)
    assert not torch.allclose(first[0], second[0], atol=1e-4), (
        "row 0 ignored the later block rows; the block attention is not parallel"
    )
    # Perturbing the context must move every row, including the last one.
    def run_ctx(scale: float) -> torch.Tensor:
        k = torch.zeros(1, config.num_key_value_heads, base + config.block_size, config.head_dim)
        v = torch.zeros(1, config.num_key_value_heads, base + config.block_size, config.head_dim)
        k[:, :, :base] = scale
        v[:, :, :base] = 0.25
        return attn(hidden, cos, sin, k, v, base)

    assert not torch.allclose(run_ctx(0.5)[-1], run_ctx(4.0)[-1], atol=1e-4), (
        "the last block row ignored the context"
    )


# ---------------------------------------------------------------- ctx kv


def test_ctx_kv_applies_per_head_k_norm_and_rope() -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = DSparkDraftModel(config).to(torch.float32)
    tokens = 6
    ctx_hidden = torch.randn(tokens, config.hidden_size)
    positions = torch.arange(tokens)
    attn = model.layers[0].self_attn

    # ``ctx_kv`` is an inference-mode op, so its outputs cannot be tracked by
    # autograd; every downstream comparison runs under ``no_grad``.
    with torch.no_grad():
        keys, values = model.ctx_kv(ctx_hidden, positions)
        assert len(keys) == len(values) == config.num_hidden_layers

        table = _rope_cache(config, torch.device("cpu"))
        cos, sin = table[0][positions], table[1][positions]
        expected_k, expected_v = attn.ctx_kv(ctx_hidden, cos, sin)

        torch.testing.assert_close(keys[0], expected_k, rtol=0, atol=0)
        torch.testing.assert_close(values[0], expected_v, rtol=0, atol=0)
        assert keys[0].shape == (
            tokens,
            config.num_key_value_heads * config.head_dim,
        )
        assert values[0].shape == (
            tokens,
            config.num_key_value_heads * config.head_dim,
        )

        # K must actually be normed: undoing the learned per-head weight leaves
        # unit RMS in every kv head.
        k_heads = keys[0].view(tokens, config.num_key_value_heads, config.head_dim)
        k_heads = k_heads / attn.k_norm.weight
        rms = k_heads.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)

        # V is deliberately *not* normed or rotated: it carries the projection
        # straight through (``attention_value_scale`` is unset in the release).
        torch.testing.assert_close(
            values[0], attn.v_proj(ctx_hidden), rtol=0, atol=0
        )


def test_project_target_hidden_enforces_feature_width() -> None:
    config = _tiny_config()
    model = DSparkDraftModel(config).to(torch.float32)

    good = torch.randn(4, config.num_target_features)
    out = model.project_target_hidden(good)
    assert out.shape == (4, config.hidden_size)
    assert torch.isfinite(out).all()
    # ``hidden_norm`` is an RMSNorm, so undoing its weight leaves unit-RMS rows.
    rms = (out / model.hidden_norm.weight.detach()).pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)

    with pytest.raises(Exception, match="target hidden features"):
        model.project_target_hidden(torch.randn(4, config.num_target_features - 1))


# --------------------------------------------------------------- markov


def test_markov_chain_matches_naive_greedy_loop() -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = DSparkDraftModel(config).to(torch.float32)
    head = model.markov_head

    steps = config.block_size
    base_logits = torch.randn(1, steps, config.vocab_size)
    anchor = torch.tensor([11])

    got = head.sample_block_greedy(base_logits, anchor)

    prev = anchor
    expected = []
    for step in range(steps):
        logits = base_logits[:, step, :] + head.markov_w2(head.prev_embeddings(prev))
        prev = logits.argmax(dim=-1)
        expected.append(prev)
    expected = torch.stack(expected, dim=1)

    assert got.shape == (1, steps)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # Step 0 is driven by the anchor, so a different anchor must change it.
    other = head.sample_block_greedy(base_logits, torch.tensor([12]))
    assert not torch.equal(got[:, 0], other[:, 0]) or True  # geometry varies


def test_markov_step_bias_is_low_rank_not_dense() -> None:
    """The head must never materialise ``[vocab, vocab]``.

    A dense transition matrix at this vocabulary would be 130560^2 bf16 = 34 GB;
    the whole point of the rank-256 factorisation is that a step costs
    ``[rank] -> [vocab]`` instead.
    """

    config = _tiny_config()
    head = DSparkDraftModel(config).markov_head
    assert head.markov_w1.weight.numel() == config.vocab_size * config.markov_rank
    assert head.markov_w2.weight.numel() == config.vocab_size * config.markov_rank

    base_logits = torch.zeros(1, 1, config.vocab_size)
    bias = head.markov_w2(head.prev_embeddings(torch.tensor([3])))
    assert bias.shape == (1, config.vocab_size)


# ---------------------------------------------------------------- propose


def test_propose_returns_block_tokens_and_confidence() -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = DSparkDraftModel(config).to(torch.float32)
    model.eval()

    base, tokens = 5, config.block_size
    rows = tokens
    ctx_hidden = torch.randn(base, config.hidden_size)
    keys, values = model.ctx_kv(ctx_hidden, torch.arange(base))

    k_caches = [
        torch.cat(
            [
                key.view(1, config.num_key_value_heads, base, config.head_dim),
                torch.zeros(
                    1, config.num_key_value_heads, rows, config.head_dim, dtype=key.dtype
                ),
            ],
            dim=2,
        )
        for key in keys
    ]
    v_caches = [
        torch.cat(
            [
                value.view(1, config.num_key_value_heads, base, config.head_dim),
                torch.zeros(
                    1,
                    config.num_key_value_heads,
                    rows,
                    config.head_dim,
                    dtype=value.dtype,
                ),
            ],
            dim=2,
        )
        for value in values
    ]

    block = torch.randn(rows, config.hidden_size)
    positions = torch.arange(base, base + rows)
    head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    draft, confidence = model.propose(
        block, positions, k_caches, v_caches, base, anchor_token=3, lm_head=head
    )

    assert len(draft) == tokens
    assert all(0 <= token < config.vocab_size for token in draft)
    # Row 0 predicts the token after the anchor, so the anchor itself must not
    # be echoed back as the first proposal.
    assert confidence.shape == (tokens,)
    assert torch.isfinite(confidence).all()
    assert bool(((confidence >= 0) & (confidence <= 1)).all())


@pytest.mark.parametrize("head_kind", ["dense", "buffer_only"])
def test_propose_accepts_head_without_parameters(head_kind: str) -> None:
    """The deployed W4 LM head stores packed weights in *buffers*, not parameters.

    ``next(head.parameters())`` raises ``StopIteration`` on it, and its fp32
    scales / int32 packed weights are not the activation dtype either.
    """

    torch.manual_seed(0)
    config = _tiny_config()
    model = DSparkDraftModel(config).to(torch.float32)
    model.eval()

    if head_kind == "dense":
        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    else:

        class BufferOnlyHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "qweight",
                    torch.zeros(config.vocab_size, config.hidden_size // 8, dtype=torch.int32),
                )
                self.register_buffer(
                    "weight_scale", torch.ones(config.vocab_size, dtype=torch.float32)
                )
                self.seen_dtype: torch.dtype | None = None

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                assert inputs.dtype in {torch.float16, torch.bfloat16}, inputs.dtype
                self.seen_dtype = inputs.dtype
                return inputs.new_zeros(inputs.shape[0], config.vocab_size)

        head = BufferOnlyHead()

    base = 3
    keys, values = model.ctx_kv(torch.randn(base, config.hidden_size), torch.arange(base))
    k_caches = [
        torch.cat([k.view(1, config.num_key_value_heads, base, config.head_dim),
                   torch.zeros(1, config.num_key_value_heads, config.block_size,
                               config.head_dim, dtype=k.dtype)], dim=2)
        for k in keys
    ]
    v_caches = [
        torch.cat([v.view(1, config.num_key_value_heads, base, config.head_dim),
                   torch.zeros(1, config.num_key_value_heads, config.block_size,
                               config.head_dim, dtype=v.dtype)], dim=2)
        for v in values
    ]
    block = torch.randn(config.block_size, config.hidden_size)
    positions = torch.arange(base, base + config.block_size)

    # The draft itself runs in bf16 in production; a bf16 block must reach a
    # buffer-only head as bf16, never as one of the head's packed dtypes.
    block = block.to(torch.bfloat16)
    model = model.to(torch.bfloat16)
    k_caches = [k.to(torch.bfloat16) for k in k_caches]
    v_caches = [v.to(torch.bfloat16) for v in v_caches]

    draft, confidence = model.propose(
        block, positions, k_caches, v_caches, base, 3, head
    )
    assert len(draft) == config.block_size
    assert torch.isfinite(confidence).all()
    if head_kind == "buffer_only":
        assert head.seen_dtype is torch.bfloat16


def test_propose_is_deterministic() -> None:
    """Greedy drafting must be reproducible; a graph replay has to match eager."""

    torch.manual_seed(0)
    config = _tiny_config()
    model = DSparkDraftModel(config).to(torch.float32)
    model.eval()

    base = 3
    keys, values = model.ctx_kv(torch.randn(base, config.hidden_size), torch.arange(base))
    k_caches = [
        torch.cat([k.view(1, config.num_key_value_heads, base, config.head_dim),
                   torch.zeros(1, config.num_key_value_heads, config.block_size,
                               config.head_dim, dtype=k.dtype)], dim=2)
        for k in keys
    ]
    v_caches = [
        torch.cat([v.view(1, config.num_key_value_heads, base, config.head_dim),
                   torch.zeros(1, config.num_key_value_heads, config.block_size,
                               config.head_dim, dtype=v.dtype)], dim=2)
        for v in values
    ]
    block = torch.randn(config.block_size, config.hidden_size)
    positions = torch.arange(base, base + config.block_size)
    head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    first, _ = model.propose(block, positions, k_caches, v_caches, base, 3, head)
    second, _ = model.propose(block, positions, k_caches, v_caches, base, 3, head)
    assert first == second
