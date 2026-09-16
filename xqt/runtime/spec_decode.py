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
    ) -> None:
        if not isinstance(session, CudaGraphDecodeSession):
            raise XQTBackendError("SpecDecodeSession wraps a CudaGraphDecodeSession")
        if not 1 <= draft_tokens <= MAX_DRAFT - 1:
            raise ValueError(f"draft_tokens must be in [1, {MAX_DRAFT - 1}]")
        self._session = session
        self._k = int(draft_tokens)
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
        self._attention_splits = 16
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
        self._pos = torch.zeros(1, dtype=torch.long, device=self._device)
        self._logits_out = None  # filled by _verify_body

    # ------------------------------------------------------------- drafting
    def _drafter_for(self, input_ids: list[int]) -> PromptLookupDrafter:
        return PromptLookupDrafter(input_ids, max_ngram=4)

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
        quantize = self._session._int8_activations
        rows = self._rows
        k_caches = self._session.k_cache
        v_caches = self._session.v_cache
        session = self._session
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            normed = session._norm(layer.input_layernorm, hidden, quantize)
            attention = layer.self_attn
            fused = session._fused_qkv[index]
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
        final = _rms_norm_rowwise(model.model.norm, hidden)
        head = self._session._lm_head_override
        if head is not None:
            logits = head(final)
        else:
            logits = model.lm_head(final)
        self._logits_out = logits

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
    ) -> tuple[list[int], dict[str, Any]]:
        """Greedy generation with exact speculative verification."""

        session = self._session
        prompt_tokens = int(input_ids.shape[1])
        first = session.prefill(input_ids)
        drafter = self._drafter_for(input_ids[0].tolist() + [first])
        tokens: list[int] = [first]
        hit_eos = first in eos_token_ids
        steps = 0
        total_draft = 0
        total_accepted = 0
        while len(tokens) < max_new_tokens and not hit_eos:
            k = min(self._k, max_new_tokens - len(tokens) - 1)
            draft = drafter.draft(k) if k > 0 else []
            # Padded positions repeat the last real token: they are written to
            # the cache by the fixed graph and must not contain garbage.
            filled = [tokens[-1]] + draft
            window = filled + [filled[-1]] * (self._rows - len(filled))
            total_draft += len(draft)
            base = prompt_tokens + len(tokens) - 1
            pad = len(draft) + 1
            # Row r lands at slot base+r (padded rows re-write the last real
            # slot's value at a later slot) and attends [0, base+r+1). Only
            # rows < pad are verified; padded logits are discarded.
            lens = [base + r + 1 for r in range(self._rows)]
            self._pos.fill_(base)
            self._valid_lens.copy_(
                torch.tensor(lens, dtype=torch.int32, device=self._device)
            )
            self._window.copy_(
                torch.tensor(window, dtype=torch.long, device=self._device).reshape(
                    self._rows, 1
                )
            )
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
                draft_tok = window[r]
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
            for tok in emitted:
                tokens.append(tok)
                if tok in eos_token_ids:
                    hit_eos = True
                    break
            # roll position back to the last accepted slot: the next verify
            # window starts at the last confirmed token; slots beyond it are
            # overwritten by the next pass's rope/scatter, so no rollback of
            # the KV cache is needed (contiguous writes cover them).
            drafter.observe(emitted)
        stats = {
            "verify_steps": steps,
            "drafted_tokens": total_draft,
            "accepted_draft_tokens": total_accepted,
            "mean_accepted_per_step": (total_accepted / steps if steps else 0.0),
        }
        return tokens, stats


def _rms_norm_rowwise(module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """RMSNorm over a [ROWS, hidden] tensor using the fused aten op."""

    weight = module.weight
    eps = float(module.variance_epsilon)
    return torch.nn.functional.rms_norm(hidden, (hidden.shape[-1],), weight, eps)


__all__ = ["PromptLookupDrafter", "SpecDecodeSession"]
