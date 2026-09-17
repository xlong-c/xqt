"""Speculative decoding driven by the official MiniCPM5-2B DSpark draft.

This wires :class:`xqt.model.dspark_draft.DSparkDraftModel` -- the released
``openbmb/MiniCPM5-2B-DSpark`` checkpoint -- into the existing CUDA-graph
verification path. Verification, acceptance and rollback are inherited from
:class:`SpecDecodeSession`; only proposal and cross-round draft state are new,
because re-implementing the accept rule would risk diverging from the gate that
proves the verified stream matches single-step greedy decoding.

What makes the official draft different from the training-free drafters: it does
not read one hidden vector, it *attends over the target's own representations*.
Each round the target model's outputs from layers ``[1, 10, 20, 30, 39]`` are
projected through the draft's ``fc`` and turned into K/V, so the draft's
attention sees the whole context the way the target does.

Per-round shape chain::

    target verify rows [rows, 5*2048]        (captured by the decode session)
      -> project_target_hidden -> ctx_kv     (draft KV cache, positions base+1..)
    block [anchor, mask x 6] at base..base+6
      -> 5 draft layers over (ctx KV ++ block) -> hidden [7, 2048]
      -> target lm_head -> base logits
      -> rank-256 Markov chain -> 7 draft tokens
      -> 8-row verify (inherited)

Cache validity invariant: before a round, the draft KV cache holds valid
entries for ``[0, base)`` where ``base`` is the position of ``tokens[-1]``; the
block itself supplies the K/V for ``base``. After the round, the accepted rows
are committed into ``base+1 .. base+accepted``, which restores the invariant for
the next round's ``base``.
"""

from __future__ import annotations

from typing import Any

import torch

from xqt.core.errors import XQTBackendError
from xqt.model.dspark_draft import DSparkDraftModel
from xqt.runtime.graph_decode import CudaGraphDecodeSession
from xqt.runtime.spec_decode import SpecDecodeSession


class DSparkSpecDecodeSession(SpecDecodeSession):
    """Greedy speculative decoding with the official DSpark draft."""

    def __init__(
        self,
        session: CudaGraphDecodeSession,
        drafter: DSparkDraftModel,
        *,
        draft_tokens: int | None = None,
    ) -> None:
        config = drafter.config
        k = int(config.block_size if draft_tokens is None else draft_tokens)
        if not 1 <= k <= config.block_size:
            raise XQTBackendError(
                f"draft_tokens must be in [1, {config.block_size}] for the "
                f"released DSpark draft; got {k}"
            )
        if session.captured_prefill is None or session.captured_rows is None:
            raise XQTBackendError(
                "DSparkSpecDecodeSession needs the target session to capture its "
                "layer outputs; build it with "
                f"capture_layer_ids={list(config.target_layer_ids)}"
            )
        super().__init__(session, draft_tokens=k, drafter=None)
        self._dspark = drafter
        self._block = k
        self._mask_token_id = int(config.mask_token_id)
        self._config = config
        features = len(config.target_layer_ids) * session.hidden_size
        if int(session.captured_prefill.shape[1]) != features:
            raise XQTBackendError(
                f"the target session captures {session.captured_prefill.shape[1]} "
                f"features per position but the draft expects {features} "
                f"(layers {list(config.target_layer_ids)})"
            )
        # The draft's own KV cache. Its K/V come from projected *target* hidden
        # states for context positions, and from the draft's own block for the
        # positions currently under speculation.
        shape = (
            1,
            config.num_key_value_heads,
            session.max_cache_len,
            config.head_dim,
        )
        self._draft_k = [
            torch.zeros(shape, dtype=torch.bfloat16, device=self._device)
            for _ in range(config.num_hidden_layers)
        ]
        self._draft_v = [
            torch.zeros(shape, dtype=torch.bfloat16, device=self._device)
            for _ in range(config.num_hidden_layers)
        ]
        self._ctx_valid = 0
        self._last_confidence: torch.Tensor | None = None
        self._primed = False

    # ------------------------------------------------------------- ctx build
    @torch.inference_mode()
    def _commit_context(self, source: torch.Tensor, low: int) -> None:
        """Project target hidden rows and write them at positions ``low..``.

        ``low`` is passed as a plain int rather than derived from a device
        tensor: reading it back would force a synchronization on every round,
        and the caller already knows it on the host.
        """

        count = int(source.shape[0])
        if count == 0:
            return
        positions = torch.arange(low, low + count, device=source.device)
        ctx_hidden = self._dspark.project_target_hidden(source)
        keys, values = self._dspark.ctx_kv(ctx_hidden, positions)
        for index in range(self._config.num_hidden_layers):
            self._draft_k[index][:, :, low : low + count] = keys[index].view(
                1, count, self._config.num_key_value_heads, self._config.head_dim
            ).transpose(1, 2)
            self._draft_v[index][:, :, low : low + count] = values[index].view(
                1, count, self._config.num_key_value_heads, self._config.head_dim
            ).transpose(1, 2)
        self._ctx_valid = max(self._ctx_valid, low + count)

    @torch.inference_mode()
    def _prime_context(self, prompt_tokens: int) -> None:
        """Build the draft KV cache for the whole prompt after prefill."""

        self._commit_context(self._session.captured_prefill[:prompt_tokens], 0)

    # ----------------------------------------------------------------- hooks
    @torch.inference_mode()
    def _propose_tokens(self, k: int, tokens: list[int]) -> list[int] | None:
        session = self._session
        if not self._primed:
            # ``super().generate`` runs prefill before the first proposal, so the
            # capture buffer is only valid now; priming here avoids paying for a
            # second prefill just to fill the draft context.
            self._prime_context(self._prompt_tokens)
            self._primed = True
        base = self._prompt_tokens + len(tokens) - 1
        anchor = tokens[-1]
        query = self._config.query_token_num

        block_ids = torch.full(
            (query,), int(self._mask_token_id), dtype=torch.long, device=self._device
        )
        block_ids[0] = anchor
        embeds = session.model.model.embed_tokens(block_ids)
        positions = torch.arange(base, base + query, device=self._device)

        head = session._lm_head_override
        if head is None:
            head = session.model.lm_head
        drafts, confidence = self._dspark.propose(
            embeds,
            positions,
            self._draft_k,
            self._draft_v,
            base,
            anchor,
            head,
        )
        self._last_confidence = confidence
        return drafts[:k]

    @torch.inference_mode()
    def _after_verify(self, *, base: int, accepted: int, rows: int) -> None:
        """Commit rows ``0 .. accepted`` of the verify window into the ctx cache.

        Row ``r`` holds the token at position ``base + r``, and its hidden state
        was produced with the correct prefix iff every draft before it survived.
        That is true for row ``0`` (the anchor, already confirmed) and for rows
        ``1 .. accepted`` (the accepted drafts), so those rows are exactly the
        ones whose target hidden states are trustworthy.

        Row ``accepted + 1`` and beyond are computed from *rejected* drafts and
        must be dropped. The next round's window starts at ``base + accepted + 1``
        and its block supplies that position, so the cache only ever has to cover
        ``[0, base + accepted + 1)`` -- which rows ``0 .. accepted`` complete.

        Note the anchor row is *not* optional: when ``accepted == 0`` it is the
        only valid row, and skipping it (as an ``accepted <= 0`` early return
        once did) leaves one position uncommitted per round, so the draft's
        context silently falls further and further behind the sequence.
        """

        self._commit_context(
            self._session.captured_rows[: accepted + 1], base
        )

    # ------------------------------------------------------------ public API
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
        on_cache_full: str = "raise",
    ) -> tuple[list[int], dict[str, Any]]:
        """Greedy generation with the official DSpark draft."""

        self._primed = False
        self._ctx_valid = 0
        self._last_confidence = None
        self._prompt_tokens = int(input_ids.shape[1])
        tokens, stats = super().generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            on_cache_full=on_cache_full,
        )
        stats["ctx_valid_positions"] = self._ctx_valid
        if self._last_confidence is not None:
            stats["last_draft_confidence"] = [
                float(value) for value in self._last_confidence.tolist()
            ]
        return tokens, stats
