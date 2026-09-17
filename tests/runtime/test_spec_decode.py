"""SpecDecodeSession correctness: exact greedy verification + draft oracle."""

from __future__ import annotations

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.runtime.graph_decode import CudaGraphDecodeSession
from xqt.runtime.spec_decode import (
    NullDrafter,
    PromptLookupDrafter,
    SpecDecodeSession,
)
from tests.runtime.test_graph_decode import _tiny_llama

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="spec decode requires CUDA"
)


def test_prompt_lookup_drafter_continues_ngram() -> None:
    prompt = [1, 2, 3, 4, 5, 1, 2, 3, 9, 1, 2]
    drafter = PromptLookupDrafter(prompt, max_ngram=4)
    # tail [.., 1, 2]: ngram (1,2) last valid occurrence starts at 5
    # (followed by 3); drafting continues 3, then (2,3)->8 gives 9, then
    # (3,9)->9 gives 1. The greedy prompt-lookup chain.
    assert drafter.draft(3) == [3, 9, 1]
    # after observing, the tail is [2,1,2,3]: n=3 ngram (1,2,3) occurs at 5
    # (latest valid occurrence, followed by 9), so drafting continues 9, 1
    drafter.observe([1, 2, 3])
    assert drafter.draft(2) == [9, 1]


def test_prompt_lookup_drafter_stops_without_match() -> None:
    drafter = PromptLookupDrafter([7, 7, 7], max_ngram=2)
    # [7,7] matches at 0 -> continuation 7; then tail [7,7] again -> 7...
    assert drafter.draft(3) == [7, 7, 7]
    drafter2 = PromptLookupDrafter([1, 9], max_ngram=4)
    # tail [1,9] has no continuation (n=2 needs a following token)
    assert drafter2.draft(2) == []


@requires_cuda
@pytest.mark.parametrize("draft_tokens", [1, 3])
def test_spec_decode_matches_greedy_reference(draft_tokens: int) -> None:
    model = _tiny_llama().cuda()
    torch.manual_seed(1)
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    with torch.inference_mode():
        reference = model.generate(
            input_ids=input_ids,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=0,
        ).tolist()[0][12:]

    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    spec = SpecDecodeSession(session, draft_tokens=draft_tokens)
    tokens, stats = spec.generate(input_ids, max_new_tokens=40, eos_token_ids=())
    # accepted output must be the exact greedy sequence
    assert tokens[: len(reference)] == reference
    assert stats["verify_steps"] > 0
    assert stats["accepted_draft_tokens"] >= 0


@requires_cuda
def test_spec_decode_hits_eos_inside_window() -> None:
    model = _tiny_llama().cuda()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    probe = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    probe.prefill(input_ids)
    eos = probe.decode_step()

    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    spec = SpecDecodeSession(session, draft_tokens=4)
    tokens, stats = spec.generate(input_ids, max_new_tokens=60, eos_token_ids=(eos,))
    assert tokens[0] == eos or eos in tokens
    assert tokens.count(eos) <= 1


@requires_cuda
def test_spec_decode_accepts_drafts_on_repetitive_prompt() -> None:
    """A repetitive prompt makes prompt-lookup draft non-empty; acceptance must
    still produce exactly the greedy sequence."""

    model = _tiny_llama().cuda()
    torch.manual_seed(4)
    pattern = torch.randint(0, 200, (1, 6), device="cuda")
    input_ids = pattern.repeat(1, 5)  # 30 tokens, heavy n-gram repetition
    with torch.inference_mode():
        reference = model.generate(
            input_ids=input_ids,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=0,
        ).tolist()[0][30:]

    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    spec = SpecDecodeSession(session, draft_tokens=5)
    tokens, stats = spec.generate(input_ids, max_new_tokens=24, eos_token_ids=())
    assert tokens[: len(reference)] == reference
    assert stats["drafted_tokens"] > 0
    assert stats["accepted_draft_tokens"] > 0


@requires_cuda
def test_spec_decode_partial_acceptance_does_not_leak_stale_slots() -> None:
    """A step that accepts fewer drafts than it wrote must not let the next
    window read the unaccepted slots' K/V (regression: padded rows used to
    write garbage, and stale draft slots were read after a partial accept)."""

    model = _tiny_llama().cuda()
    torch.manual_seed(11)
    input_ids = torch.randint(0, 200, (1, 10), device="cuda")
    with torch.inference_mode():
        reference = model.generate(
            input_ids=input_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=0,
        ).tolist()[0][10:]

    session = CudaGraphDecodeSession(model, max_cache_len=128, fuse_norms=False)
    # k=7 forces ROWS=8 with 7 real rows: plenty of room for stale slots.
    spec = SpecDecodeSession(session, draft_tokens=7)
    tokens, stats = spec.generate(input_ids, max_new_tokens=32, eos_token_ids=())

    assert tokens[: len(reference)] == reference
    # the padding rows are covered by the non-power-of-two draft budget
    assert stats["verify_steps"] > 0


@requires_cuda
def test_spec_decode_rejects_cache_overflow() -> None:
    """An over-long run must fail loudly: the rows kernel writes slot
    ``position + row`` with no mask, so the alternative is silent corruption."""

    model = _tiny_llama()
    session = CudaGraphDecodeSession(model, max_cache_len=64, fuse_norms=False)
    spec = SpecDecodeSession(session, draft_tokens=1)
    input_ids = torch.randint(0, 200, (1, 20), device="cuda")

    # rows = 2 for draft_tokens=1, so the budget is 64 - 20 - 2 = 42.
    with pytest.raises(ValueError, match="exceeds max_cache_len"):
        spec.generate(input_ids, max_new_tokens=100, eos_token_ids=())

    # the refusing path must not have touched the cache
    assert session.length == 0


@requires_cuda
def test_spec_decode_truncate_keeps_window_in_bounds() -> None:
    model = _tiny_llama()
    session = CudaGraphDecodeSession(model, max_cache_len=64, fuse_norms=False)
    spec = SpecDecodeSession(session, draft_tokens=1)
    input_ids = torch.randint(0, 200, (1, 20), device="cuda")

    tokens, _ = spec.generate(
        input_ids, max_new_tokens=100, eos_token_ids=(), on_cache_full="truncate"
    )
    assert 20 + len(tokens) + spec._rows <= 64


@requires_cuda
@pytest.mark.parametrize("kv_quant", ["int8", "int4"])
def test_spec_rejects_quantized_kv(kv_quant: str) -> None:
    """The rows attention kernel takes no k_scale/v_scale, so a quantized cache
    can only be rejected up front."""

    model = _tiny_llama()
    session = CudaGraphDecodeSession(
        model, max_cache_len=64, fuse_norms=False, kv_quant=kv_quant
    )
    with pytest.raises(XQTBackendError, match="kv_quant='none'"):
        SpecDecodeSession(session, draft_tokens=1)


@requires_cuda
def test_spec_honors_session_attention_splits() -> None:
    model = _tiny_llama()
    session = CudaGraphDecodeSession(
        model, max_cache_len=64, fuse_norms=False, attention_splits=8
    )
    spec = SpecDecodeSession(session, draft_tokens=1)
    assert spec._attention_splits == 8


@requires_cuda
def test_layer_capture_records_prefill_and_verify_rows() -> None:
    """The DSpark draft consumes five target-layer hidden states, so the
    runtime must record them without disturbing the verified numerics."""

    model = _tiny_llama()
    session = CudaGraphDecodeSession(
        model, max_cache_len=64, fuse_norms=False, capture_layer_ids=(0, 1)
    )
    assert session.captured_prefill is not None
    assert session.captured_rows is not None

    torch.manual_seed(5)
    input_ids = torch.randint(0, 200, (1, 12), device="cuda")
    reference = CudaGraphDecodeSession(
        model, max_cache_len=64, fuse_norms=False
    ).generate(input_ids, max_new_tokens=8).token_ids

    session.prefill(input_ids)
    assert session.captured_prefill.shape == (64, 512)
    assert session.captured_prefill[:12].abs().sum() > 0
    assert session.captured_prefill[12:].abs().sum() == 0
    # the two captured layers must not be copies of each other
    assert not torch.equal(
        session.captured_prefill[:12, :256], session.captured_prefill[:12, 256:]
    )

    spec = SpecDecodeSession(session, draft_tokens=1, drafter=NullDrafter())
    tokens, _ = spec.generate(input_ids, max_new_tokens=8, eos_token_ids=())
    assert session.captured_rows[:2].abs().sum() > 0
    # capture is a pure copy: the verified stream must be unchanged
    assert tokens == reference


@requires_cuda
def test_spec_rejects_unknown_cache_policy() -> None:
    model = _tiny_llama()
    session = CudaGraphDecodeSession(model, max_cache_len=64, fuse_norms=False)
    spec = SpecDecodeSession(session, draft_tokens=1)
    input_ids = torch.randint(0, 200, (1, 8), device="cuda")

    with pytest.raises(ValueError, match="on_cache_full must be"):
        spec.generate(
            input_ids, max_new_tokens=4, eos_token_ids=(), on_cache_full="grow"
        )


@requires_cuda
@pytest.mark.parametrize("int8_activations", [False, True])
def test_verify_numerics_match_single_step(int8_activations: bool) -> None:
    """The verify path must score the same model the single-step route
    deploys, so with no drafts it has to emit the identical greedy sequence.

    This is the gate for the exactness claim: ``_verify_body`` used to run a
    bf16-GEMV / dequantized-norm approximation of the W4A8 graph, which made
    the verifier a different model than the one being benchmarked."""

    model = _tiny_llama()
    torch.manual_seed(3)
    input_ids = torch.randint(0, 200, (1, 16), device="cuda")

    def build() -> CudaGraphDecodeSession:
        return CudaGraphDecodeSession(
            model,
            max_cache_len=128,
            fuse_norms=False,
            int8_activations=int8_activations,
        )

    reference = build().generate(input_ids, max_new_tokens=24).token_ids

    spec = SpecDecodeSession(build(), draft_tokens=1, drafter=NullDrafter())
    tokens, stats = spec.generate(input_ids, max_new_tokens=24, eos_token_ids=())

    assert stats["drafted_tokens"] == 0
    assert len(tokens) == len(reference)
    assert tokens == reference
