"""Tests for :class:`DSparkSpecDecodeSession`'s cross-round position accounting.

The draft's context cache is only useful if it covers a *contiguous* prefix of
the sequence at the start of every round. That property is pure integer
bookkeeping -- no GPU, no weights -- so it can be pinned down here rather than
discovered as a collapsed accept rate on a long run.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from xqt.runtime.dspark_session import DSparkSpecDecodeSession


def _stub_session(rows: int = 8) -> tuple[DSparkSpecDecodeSession, list[tuple[int, int]]]:
    """A session whose only real behaviour is recording commits."""

    session = DSparkSpecDecodeSession.__new__(DSparkSpecDecodeSession)
    committed: list[tuple[int, int]] = []

    def record(source: torch.Tensor, low: int) -> None:
        committed.append((int(low), int(source.shape[0])))

    session._commit_context = record  # type: ignore[method-assign]
    session._session = SimpleNamespace(captured_rows=torch.zeros(rows, 4))
    return session, committed


def _covered(committed: list[tuple[int, int]]) -> set[int]:
    return {
        position
        for low, count in committed
        for position in range(low, low + count)
    }


def test_after_verify_keeps_ctx_cache_contiguous() -> None:
    """Every position below the next round's ``base`` must be committed.

    ``base`` is the slot of the last confirmed token, and the next round's block
    only supplies that one slot, so the cache has to carry ``[0, base)``.
    """

    session, committed = _stub_session()
    base = 100
    committed.append((0, base))  # prefill prime covers [0, base)

    for accepted in (0, 3, 0, 7, 1, 0, 0, 6):
        session._after_verify(base=base, accepted=accepted, rows=8)
        base += accepted + 1  # the round emits the accepted drafts plus one more
        assert _covered(committed) == set(range(base)), (
            f"ctx cache has gaps after a round with accepted={accepted}; "
            f"missing={sorted(set(range(base)) - _covered(committed))}"
        )


def test_after_verify_commits_the_anchor_row_when_nothing_is_accepted() -> None:
    """``accepted == 0`` still advances ``base``, so it still has to commit."""

    session, committed = _stub_session()
    session._after_verify(base=42, accepted=0, rows=8)

    assert committed == [(42, 1)]


def test_after_verify_never_commits_rejected_rows() -> None:
    """Rows past the last accepted draft were computed from wrong tokens."""

    session, committed = _stub_session()
    session._after_verify(base=7, accepted=2, rows=8)

    # Rows 0..2 land at slots 7..9. Row 3 holds a rejected draft and its hidden
    # was computed from that wrong token, so slot 10 must stay untouched.
    assert committed == [(7, 3)]
    assert _covered(committed) == {7, 8, 9}


def test_after_verify_fits_the_capture_buffer() -> None:
    """A full-acceptance round reads rows ``0..k``, all of which must exist."""

    for rows in (2, 4, 8):
        session, committed = _stub_session(rows=rows)
        session._after_verify(base=0, accepted=rows - 1, rows=rows)
        assert committed == [(0, rows)]
