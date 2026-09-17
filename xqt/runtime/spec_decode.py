"""Training-free speculative decoding on the graph decode runtime (R-055+).

The W4A8 runtime is DRAM-bound: every decode step must read 1.27 GB of INT4
weights regardless of activation precision, so a single-token step cannot
exploit INT8 compute. The route to more tokens per weight read is speculative
decoding: draft ``k`` tokens for free (prompt-lookup n-grams), then verify
them against the target model in **one** forward pass whose projections use
the Batch (M <= 8) native GEMV and whose attention reads K/V once for all
``k`` queries.

The accept rule is exact greedy verification: the target's argmax token
sequence decides which drafted prefix survives, so accepted output is bit-wise
the greedy sequence the non-speculative runtime would produce.

Layout notes (why ``ROWS = k + 1``): the verify pass runs the drafted window
plus one corrector row. If all ``k`` drafted tokens match, the last row's
argmax provides the next fresh token, so a fully accepted step still advances
``k + 1`` tokens. On a mismatch at position ``i`` the greedy continuation is:
accept ``i`` drafted tokens plus the target token at mismatch ``i``; the
remaining drafted cache slots are simply overwritten by the next verify pass.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.runtime.graph_decode import CudaGraphDecodeSession

from xqt.kernels.ops._impl.triton.spec_decode_kernels import (
    decode_attention_rows_forward_triton,
    rope_write_qkv_rows_triton,
)

MAX_DRAFT = 8


class PromptLookupDrafter:
    """Draft token continuations by n-gram lookup into the prompt/history.

    For each draft position, find the longest n-gram (n up to ``max_ngram``)
    ending at the current tail that also occurs in the combined prompt +
    generated history, and continue with the tokens that followed that
    occurrence. Falls back to the shortest usable n-gram, then to no draft.
    """

    def __init__(self, prompt_tokens: list[int], *, max_ngram: int = 4) -> None:
        if max_ngram < 1 or max_ngram > 8:
            raise ValueError("max_ngram must be in [1, 8]")
        self._max_ngram = int(max_ngram)
        self._history: list[int] = list(prompt_tokens)
        # occurrence starts per ngram length: for speed, index 2..max_ngram
        self._ngram_index = {n: {} for n in range(2, max_ngram + 1)}
        for start in range(0, len(prompt_tokens)):
            for n in range(2, self._max_ngram + 1):
                if start + n < len(prompt_tokens):
                    key = tuple(prompt_tokens[start : start + n])
                    # keep the LAST occurrence so continuations favor recent
                    # context (the tail nearest the generation point).
                    self._ngram_index[n][key] = start + n

    def observe(self, tokens: list[int]) -> None:
        """Append accepted tokens to the observable history."""

        if not tokens:
            return
        begin = len(self._history)
        self._history.extend(tokens)
        # Incremental re-index: only ngrams that START within the newly
        # observed window gained a continuation, so scanning the whole history
        # here would make observe() quadratic in the generated length. An
        # ngram ending at the very tail still has no continuation and must not
        # clobber an older mid-history match, hence the ``pos + n < end``
        # guard (the same rule the constructor uses).
        end = len(self._history)
        for pos in range(max(0, begin - self._max_ngram), end - 1):
            for n in range(2, self._max_ngram + 1):
                if pos + n < end:
                    self._ngram_index[n][tuple(self._history[pos : pos + n])] = pos + n

    def draft(self, k: int) -> list[int]:
        """Return up to ``k`` drafted tokens (may be shorter or empty)."""

        out: list[int] = []
        tail = deque(self._history[-self._max_ngram :])
        while len(out) < k:
            matched = False
            for n in range(min(len(tail), self._max_ngram), 1, -1):
                key = tuple(list(tail)[-n:])
                nxt = self._ngram_index[n].get(key)
                if nxt is not None and nxt < len(self._history):
                    token = self._history[nxt]
                    out.append(token)
                    tail.append(token)
                    if len(tail) > self._max_ngram:
                        tail.popleft()
                    matched = True
                    break
            if not matched:
                break
        return out[:k]


class NullDrafter:
    """Drafter that never proposes anything.

    Passing this to :class:`SpecDecodeSession` turns the verify path into a
    plain greedy decoder with a one-token window, which is how the runtime
    proves that verifying scores exactly the model the single-step route
    deploys.
    """

    def draft(self, k: int) -> list[int]:
        return []

    def observe(self, tokens: list[int]) -> None:
        return None


class DSparkDrafter(nn.Module):
    """Ultra-lightweight neural drafter (DSpark) for speculative decoding.

    Architecture:
    - 1-layer TransformerEncoderLayer (d_model=2048, nhead=16, dim_feedforward=3072)
    - K residual MTP MLP heads (modulating base layer features)
    - 1 confidence scoring head for dynamic speculation length scheduling

    Inference drafts are projected through the target model's LM head,
    leveraging frozen target vocabulary representations.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 6,
        num_layers: int = 2,
        ffn_hidden_size: int = 3072,
        *,
        conf_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.conf_threshold = float(conf_threshold)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=16,
                    dim_feedforward=ffn_hidden_size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.mtp_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(num_heads)
            ]
        )
        self.conf_head = nn.Linear(hidden_size, num_heads)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass over [B, S, H] or [1, H] hidden states."""

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        drafter_dtype = next(self.layers[0].parameters()).dtype
        if hidden_states.dtype != drafter_dtype:
            hidden_states = hidden_states.to(dtype=drafter_dtype)
        z = hidden_states
        for layer in self.layers:
            z = layer(z)
        head_features: list[torch.Tensor] = [z + head(z) for head in self.mtp_heads]
        conf_logits = self.conf_head(z)
        return head_features, conf_logits

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        conf_threshold: float = 0.05,
    ) -> DSparkDrafter:
        path = Path(path)
        config_path = path.parent / "config.json"
        num_heads = 6
        num_layers = 2
        hidden_size = 2048
        ffn_hidden_size = 3072
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                num_heads = cfg.get("num_draft_tokens", num_heads)
                num_layers = cfg.get("num_layers", num_layers)
                hidden_size = cfg.get("hidden_size", hidden_size)
                ffn_hidden_size = cfg.get("ffn_hidden_size", ffn_hidden_size)
        drafter = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_hidden_size=ffn_hidden_size,
            conf_threshold=conf_threshold,
        ).to(device=device, dtype=dtype)
        state = torch.load(path, map_location=device, weights_only=True)
        drafter.load_state_dict(state)
        drafter.eval()
        return drafter

    @torch.inference_mode()
    def draft(
        self,
        hidden: torch.Tensor,
        head_fn: Callable[[torch.Tensor], torch.Tensor],
        k: int,
    ) -> list[int]:
        """Draft up to ``k`` candidate tokens using confidence scheduling.

        ``hidden`` is the normalized hidden state of the token immediately
        preceding ``tokens[-1]``. Head 1 predicts the token after ``tokens[-1]``,
        Head 2 predicts the next, etc.
        """

        feats, conf_logits = self(hidden)
        probs = torch.sigmoid(conf_logits.squeeze(0).squeeze(0))
        selected = []
        for idx in range(1, min(k + 1, self.num_heads)):
            if probs[idx].item() >= self.conf_threshold:
                selected.append(feats[idx].squeeze(0))
            else:
                break
        if not selected:
            return []
        stacked = torch.cat(selected, dim=0)
        logits = head_fn(stacked)
        return logits.argmax(-1).tolist()


class SpecDecodeSession:
    """Greedy generation with prompt-lookup drafting and exact verification.

    Wraps a :class:`CudaGraphDecodeSession` target. The verify forward runs
    eagerly first and is then captured into a second CUDA graph whose input
    window (draft tokens, positions, valid lens, KV cache) lives in stable
    device buffers, so replays cost one weight sweep for ``ROWS`` tokens.
    """

    def __init__(
        self,
        session: CudaGraphDecodeSession,
        *,
        draft_tokens: int = 4,
        drafter: PromptLookupDrafter | DSparkDrafter | NullDrafter | None = None,
    ) -> None:
        if not isinstance(session, CudaGraphDecodeSession):
            raise XQTBackendError("SpecDecodeSession wraps a CudaGraphDecodeSession")
        if not 1 <= draft_tokens <= MAX_DRAFT - 1:
            raise ValueError(f"draft_tokens must be in [1, {MAX_DRAFT - 1}]")
        self._session = session
        self._k = int(draft_tokens)
        self._drafter = drafter
        self._final_hidden: torch.Tensor | None = None
        self._prompt_tokens = 0
        if session._kv_quant != "none":
            # ``decode_attention_rows_forward_triton`` has no k_scale/v_scale
            # parameters and requires the cache's last dim to equal head_dim,
            # so an int4 cache fails a shape check and an int8 cache fails to
            # compile in ``tl.dot``. Surface that here instead of inside Triton.
            raise XQTBackendError(
                "SpecDecodeSession verify requires kv_quant='none' (bf16 KV); "
                f"got kv_quant={session._kv_quant!r}. The rows attention kernel "
                "cannot read a quantized cache - build the "
                "CudaGraphDecodeSession with kv_quant='none'."
            )
        # ``rope_write_qkv_rows_triton`` writes slot ``pos + row`` for every
        # row with no mask, and reads the RoPE tables at the same offset, so an
        # over-long window corrupts device memory silently rather than raising.
        # Every bound check in ``generate`` is expressed against this.
        self._max_cache_len = int(session.max_cache_len)
        # Triton's tl.arange requires a power-of-two extent, so the verify
        # window rounds up; only the first ``_active`` rows carry drafts and
        # the padded rows reuse the last real row's valid length (their logits
        # are never read).
        self._rows = 1 << (self._k + 1 - 1).bit_length()
        self._model = session.model
        self._device = next(self._model.parameters()).device
        self._layers = session.num_layers
        self._heads = session.num_q_heads
        self._kv_heads = session.num_kv_heads
        self._head_dim = session.head_dim
        self._hidden = session.hidden_size
        self._attention_splits = int(session._attention_splits)
        self._valid_lens = torch.zeros(
            self._rows, dtype=torch.int32, device=self._device
        )
        # Token window: row r predicts what comes AFTER draft slot r-1.
        # window[0] is the last confirmed token; rows 1..k are drafts.
        self._window = torch.zeros(
            (self._rows, 1), dtype=torch.long, device=self._device
        )
        self._draft_pad = session.token  # unused; placeholder for symmetry
        self._graphs: list[torch.cuda.CUDAGraph] = []
        self._dspark_graphs: list[torch.cuda.CUDAGraph] = []
        self._dspark_in_hidden = torch.zeros(
            (1, 1, self._hidden), dtype=torch.bfloat16, device=self._device
        )
        self._dspark_tokens_out: torch.Tensor | None = None
        self._dspark_confs_out: torch.Tensor | None = None
        self._window_cpu = torch.zeros(
            (self._rows, 1), dtype=torch.long, pin_memory=True
        )
        self._lens_offsets = torch.arange(
            1, self._rows + 1, dtype=torch.int32, device=self._device
        )
        self._pos = torch.zeros(1, dtype=torch.long, device=self._device)
        self._logits_out = None  # filled by _verify_body
        # Row-shaped true-INT8 staging buffers. ``CudaGraphDecodeSession`` owns
        # single-row equivalents that cannot be resized, so the verify path
        # allocates its own set. These must exist before the graph is captured
        # so replays always address the same memory.
        self._int8_rows = bool(session._int8_activations)
        self._norm_int8_buf: torch.Tensor | None = None
        self._norm_scale_buf: torch.Tensor | None = None
        self._mlp_norm_int8_buf: torch.Tensor | None = None
        self._mlp_norm_scale_buf: torch.Tensor | None = None
        self._attn_int8_buf: torch.Tensor | None = None
        self._attn_scale_buf: torch.Tensor | None = None
        self._swiglu_int8_buf: torch.Tensor | None = None
        self._swiglu_scale_buf: torch.Tensor | None = None
        if self._int8_rows:
            cfg = getattr(self._model, "config", None)
            intermediate = int(
                getattr(cfg, "intermediate_size", self._hidden * 3)
            )
            activate = torch.zeros(
                (self._rows, self._hidden), dtype=torch.int8, device=self._device
            )
            scale = torch.zeros(
                (self._rows,), dtype=torch.float32, device=self._device
            )
            self._norm_int8_buf = activate
            self._mlp_norm_int8_buf = torch.zeros_like(activate)
            self._attn_int8_buf = torch.zeros_like(activate)
            self._norm_scale_buf = scale
            self._mlp_norm_scale_buf = torch.zeros_like(scale)
            self._attn_scale_buf = torch.zeros_like(scale)
            self._swiglu_int8_buf = torch.zeros(
                (self._rows, intermediate), dtype=torch.int8, device=self._device
            )
            self._swiglu_scale_buf = torch.zeros_like(scale)

    # ------------------------------------------------------------- drafting
    def _drafter_for(self, input_ids: list[int]) -> PromptLookupDrafter:
        return PromptLookupDrafter(input_ids, max_ngram=4)

    def _propose_tokens(self, k: int, tokens: list[int]) -> list[int] | None:
        """Draft hook; subclasses override to supply their own proposals.

        Returning ``None`` means "no opinion" and selects the built-in
        prompt-lookup / DSpark paths, so the default implementation keeps the
        existing behaviour for every caller that does not override it.
        """

        return None

    def _after_verify(self, *, base: int, accepted: int, rows: int) -> None:
        """Post-round hook for drafters that maintain their own state.

        ``base`` is the cache slot of the window's first row, ``accepted`` the
        number of drafts that survived verification, and ``rows`` the window
        height. Called after the accept decision and before the next round.
        """

        return None

    # --------------------------------------------------------- verify body
    def _verify_body(self) -> None:
        """One verify pass: ROWS rows through the target network.

        Reads ``self._window`` (device), ``self._pos`` (device int64 scalar
        holding the cache slot of window[0]) and ``self._valid_lens``; writes
        per-row logits into ``self._logits_out``.
        """

        model = self._model
        # The verify body keeps every activation 2-D [ROWS, hidden]:
        # ``_MiniCPM5W4A16HybridLinear`` returns 2-D for flattened inputs while
        # ``_FusedProjection`` preserves the input's leading dims, and mixing
        # [ROWS, 1, H] with [ROWS, H] silently broadcasts to [ROWS, ROWS, H].
        hidden = model.model.embed_tokens(self._window).reshape(
            self._rows, self._hidden
        )
        quantize = self._int8_rows
        rows = self._rows
        k_caches = self._session.k_cache
        v_caches = self._session.v_cache
        session = self._session
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            attention = layer.self_attn
            fused = session._fused_qkv[index]
            if quantize:
                # Mirror ``CudaGraphDecodeSession._decode_body`` exactly: the
                # same true-INT8 kernel set, so the verifier scores the model
                # that the single-step route actually deploys.
                session._rmsnorm_true_int8(
                    hidden,
                    layer.input_layernorm.weight,
                    eps=float(layer.input_layernorm.variance_epsilon),
                    out_q=self._norm_int8_buf,
                    out_scale=self._norm_scale_buf,
                )
                q, k, v = session._project_qkv_w4a8(
                    index, layer, self._norm_int8_buf, self._norm_scale_buf
                )
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                normed = session._norm(layer.input_layernorm, hidden, quantize)
                if fused is not None:
                    qkv = fused(normed)
                else:
                    qkv = (
                        attention.q_proj(normed),
                        attention.k_proj(normed),
                        attention.v_proj(normed),
                    )
                # Contiguous [1, heads, ROWS, head_dim] buffers: the Triton
                # wrappers reshape into (heads, ROWS, head_dim) views, and
                # non-contiguous inputs would make those reshapes silently copy
                # into scrap tensors that the kernels never write back.
                q = (
                    qkv[0]
                    .reshape(1, rows, self._heads, self._head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )
                k = (
                    qkv[1]
                    .reshape(1, rows, self._kv_heads, self._head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )
                v = (
                    qkv[2]
                    .reshape(1, rows, self._kv_heads, self._head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )
            q_out = torch.empty_like(q)
            rope_write_qkv_rows_triton(
                q,
                k,
                v,
                session.rope_cos,
                session.rope_sin,
                q_out,
                k_caches[index],
                v_caches[index],
                self._pos,
            )
            out = decode_attention_rows_forward_triton(
                q_out,
                k_caches[index],
                v_caches[index],
                self._valid_lens,
                splits=self._attention_splits,
            )
            # [heads, ROWS, head_dim] -> [ROWS, heads * head_dim]
            out = out.permute(1, 0, 2).reshape(rows, -1)
            if quantize:
                session._quantize_row_true_int8(
                    out,
                    out_q=self._attn_int8_buf,
                    out_scale=self._attn_scale_buf,
                )
                hidden = session._residual_project_w4a8(
                    attention.o_proj,
                    self._attn_int8_buf,
                    self._attn_scale_buf,
                    residual,
                )
                residual = hidden
                session._rmsnorm_true_int8(
                    hidden,
                    layer.post_attention_layernorm.weight,
                    eps=float(layer.post_attention_layernorm.variance_epsilon),
                    out_q=self._mlp_norm_int8_buf,
                    out_scale=self._mlp_norm_scale_buf,
                )
                hidden = session._mlp_forward_w4a8(
                    index,
                    layer,
                    self._mlp_norm_int8_buf,
                    self._mlp_norm_scale_buf,
                    residual=residual,
                    swiglu_out=self._swiglu_int8_buf,
                    swiglu_scale=self._swiglu_scale_buf,
                )
            else:
                # The AWQ residual epilogue (decode_bias_out) takes a single [N]
                # bias row, so multi-row verifies must not route through
                # bind_residual; the GEMV plus an elementwise add is exact.
                hidden = residual + attention.o_proj(out)
                residual = hidden
                hidden = session._mlp_forward(
                    index,
                    layer,
                    session._norm(layer.post_attention_layernorm, hidden, quantize),
                    fused_ops=True,
                    quantize=quantize,
                    residual=None,
                )
                hidden = residual + hidden
            session._capture_layer_rows(index, hidden, rows)
        # Match ``CudaGraphDecodeSession._decode_body``'s final norm exactly
        # (same fused/non-fused choice), because this tensor is both the
        # verifier's logits source and the next round's drafter input.
        final = session._norm(model.model.norm, hidden, False)
        head = self._session._lm_head_override
        if head is not None:
            logits = head(final)
        else:
            logits = model.lm_head(final)
        self._logits_out = logits
        self._final_hidden = final

    def _draft_body_dspark(self) -> None:
        """Draft pass: DSpark drafter + LM head on static device buffers."""

        assert isinstance(self._drafter, DSparkDrafter)
        feats, conf_logits = self._drafter(self._dspark_in_hidden)
        n_draft = min(self._k, self._drafter.num_heads - 1)
        selected = [feats[i].squeeze(1) for i in range(1, n_draft + 1)]
        stacked = torch.cat(selected, dim=0)  # [n_draft, hidden]
        head_fn = (
            self._session._lm_head_override
            if self._session._lm_head_override is not None
            else self._model.lm_head
        )
        logits = head_fn(stacked)
        self._dspark_tokens_out = logits.argmax(-1)
        self._dspark_confs_out = conf_logits.squeeze(0).squeeze(0)

    def _capture_dspark(self) -> None:
        """Capture the DSpark draft graph once."""

        if self._dspark_graphs or not isinstance(self._drafter, DSparkDrafter):
            return
        with torch.inference_mode():
            for _ in range(3):
                self._draft_body_dspark()
            torch.cuda.synchronize()
            self._dspark_graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(self._dspark_graphs[0]):
                self._draft_body_dspark()
            torch.cuda.synchronize()

    def _capture(self) -> None:
        """Capture the verify graph once (requires prefill done)."""

        if self._graphs:
            return
        with torch.inference_mode():
            self._verify_body()  # warm: build all kernels
            torch.cuda.synchronize()
            self._graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(self._graphs[0]):
                self._verify_body()
            torch.cuda.synchronize()

    # ---------------------------------------------------------- public API
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
        on_cache_full: str = "raise",
    ) -> tuple[list[int], dict[str, Any]]:
        """Greedy generation with exact speculative verification.

        The verify window writes ``rows`` cache slots starting at the last
        confirmed token, so a run needs ``prompt + max_new_tokens + rows <=
        max_cache_len``. ``on_cache_full`` decides what to do when that does
        not hold: ``"raise"`` (default) refuses before touching the cache,
        ``"truncate"`` shrinks ``max_new_tokens`` to the largest value that
        fits. Truncation is applied to ``max_new_tokens`` and never to ``k``:
        ``rows`` and the captured graph are fixed at construction, so a
        smaller ``k`` would not narrow the write window.
        """

        session = self._session
        prompt_tokens = int(input_ids.shape[1])
        self._prompt_tokens = prompt_tokens
        if on_cache_full not in {"raise", "truncate"}:
            raise ValueError(
                f"on_cache_full must be 'raise' or 'truncate', got {on_cache_full!r}"
            )
        budget = self._max_cache_len - prompt_tokens - self._rows
        if max_new_tokens > budget:
            if on_cache_full == "raise":
                raise ValueError(
                    f"prompt {prompt_tokens} + max_new_tokens {max_new_tokens} + "
                    f"rows {self._rows} exceeds max_cache_len "
                    f"{self._max_cache_len}; the rows kernel writes "
                    "position + row with no mask"
                )
            if budget < 1:
                raise ValueError(
                    f"prompt {prompt_tokens} leaves no room to generate within "
                    f"max_cache_len {self._max_cache_len} (rows={self._rows})"
                )
            max_new_tokens = budget
        first = session.prefill(input_ids)
        tokens: list[int] = [first]
        hit_eos = first in eos_token_ids
        steps = 0
        total_draft = 0
        total_accepted = 0

        is_dspark = isinstance(self._drafter, DSparkDrafter)
        if is_dspark:
            pld_drafter = None
            dspark_drafter: DSparkDrafter = self._drafter
            lm_head_fn = (
                session._lm_head_override
                if session._lm_head_override is not None
                else self._model.lm_head
            )
            last_hidden = session.last_hidden
            if not self._dspark_graphs:
                self._capture_dspark()
        else:
            # Any explicitly supplied non-DSpark drafter is used as-is; only a
            # missing one falls back to prompt lookup.
            pld_drafter = (
                self._drafter_for(input_ids[0].tolist() + [first])
                if self._drafter is None
                else self._drafter
            )
            dspark_drafter = None

        while len(tokens) < max_new_tokens and not hit_eos:
            k = min(self._k, max_new_tokens - len(tokens) - 1)
            draft = self._propose_tokens(k, tokens)
            if draft is not None:
                pass
            elif is_dspark:
                if last_hidden is not None and k == self._k and self._dspark_graphs:
                    self._dspark_in_hidden.copy_(
                        last_hidden.reshape(1, 1, self._hidden)
                    )
                    self._dspark_graphs[0].replay()
                    assert self._dspark_tokens_out is not None
                    assert self._dspark_confs_out is not None
                    tokens_list = self._dspark_tokens_out.tolist()
                    probs_list = torch.sigmoid(self._dspark_confs_out).tolist()
                    draft = []
                    for idx in range(1, len(tokens_list) + 1):
                        if probs_list[idx] >= self._drafter.conf_threshold:
                            draft.append(tokens_list[idx - 1])
                        else:
                            break
                elif last_hidden is not None and k > 0:
                    assert dspark_drafter is not None
                    draft = dspark_drafter.draft(last_hidden, lm_head_fn, k)
                else:
                    draft = []
            elif pld_drafter is not None and k > 0:
                draft = pld_drafter.draft(k)
            else:
                draft = []

            # Padded positions repeat the last real token: they are written to
            # the cache by the fixed graph and must not contain garbage.
            filled = [tokens[-1]] + draft
            n_filled = len(filled)
            last_val = filled[-1]
            for r in range(n_filled):
                self._window_cpu[r, 0] = filled[r]
            for r in range(n_filled, self._rows):
                self._window_cpu[r, 0] = last_val
            total_draft += len(draft)
            base = prompt_tokens + len(tokens) - 1
            if base + self._rows > self._max_cache_len:
                # Unreachable when the up-front budget check ran, but the
                # kernel would corrupt memory rather than raise, so keep the
                # authoritative check on the hot path too. Host-side ints only,
                # no device sync, no graph impact.
                raise ValueError(
                    f"verify window writes slots {base}..{base + self._rows - 1} "
                    f"but max_cache_len is {self._max_cache_len}"
                )
            pad = len(draft) + 1
            # Row r lands at slot base+r (padded rows re-write the last real
            # slot's value at a later slot) and attends [0, base+r+1). Only
            # rows < pad are verified; padded logits are discarded.
            self._pos.fill_(base)
            self._valid_lens.copy_(self._lens_offsets)
            self._valid_lens.add_(base)
            self._window.copy_(self._window_cpu, non_blocking=True)
            if not self._graphs:
                self._capture()
            self._graphs[0].replay()
            steps += 1
            # Verify: row r predicts the token AFTER window[r]. The greedy
            # continuation of the confirmed prefix is computed left to right:
            # a draft d_r is accepted iff it equals argmax(row r-1).
            logits = self._logits_out  # [ROWS, vocab]
            targets_list = logits.argmax(-1).tolist()
            emitted: list[int] = []
            accepted = 0
            for r in range(1, pad):
                draft_tok = filled[r]
                target = targets_list[r - 1]
                if draft_tok == target:
                    accepted += 1
                    emitted.append(draft_tok)
                else:
                    # greedy correction token replaces the wrong draft
                    emitted.append(target)
                    break
            else:
                # every drafted position accepted: the row AFTER the last
                # accepted draft supplies one bonus fresh token (this is the
                # greedy continuation because all drafts matched).
                emitted.append(targets_list[pad - 1])
            total_accepted += accepted
            self._after_verify(base=base, accepted=accepted, rows=self._rows)

            if is_dspark:
                assert self._final_hidden is not None
                last_hidden = self._final_hidden[accepted : accepted + 1, :]
            elif pld_drafter is not None:
                pld_drafter.observe(emitted)

            for tok in emitted:
                tokens.append(tok)
                if tok in eos_token_ids:
                    hit_eos = True
                    break
            # roll position back to the last accepted slot: the next verify
            # window starts at the last confirmed token; slots beyond it are
            # overwritten by the next pass's rope/scatter, so no rollback of
            # the KV cache is needed (contiguous writes cover them).
        stats = {
            "verify_steps": steps,
            "drafted_tokens": total_draft,
            "accepted_draft_tokens": total_accepted,
            "mean_accepted_per_step": (total_accepted / steps if steps else 0.0),
        }
        return tokens, stats


__all__ = ["DSparkDrafter", "PromptLookupDrafter", "SpecDecodeSession"]
