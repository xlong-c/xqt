"""SpecDecodeSession correctness: exact greedy verification + draft oracle."""

from __future__ import annotations

import pytest
import torch

from xqt.runtime.graph_decode import CudaGraphDecodeSession
from xqt.runtime.spec_decode import PromptLookupDrafter, SpecDecodeSession
from tests.xqt.runtime.test_graph_decode import _tiny_llama

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
