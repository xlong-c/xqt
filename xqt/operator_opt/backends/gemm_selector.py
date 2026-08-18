"""Registry-projected GEMM engine selection for operator optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from xqt.contracts import PrecisionPolicy
from xqt.contracts.engine_resolve import get_engine_registration
from xqt.gemm import GemmProblem, default_registry

GemmFusedOp = Literal["bias", "relu", "gelu", "silu", "activation_quant"]
GemmGoal = Literal["latency", "throughput", "accuracy"]
_KNOWN_GOALS = {"latency", "throughput", "accuracy"}


@dataclass(frozen=True)
class GemmShape:
    """Dense 2D GEMM shape.

    Field names (``m``/``n``/``k``) match the naming already used across
    ``gemm_precision.py`` and the TileLang/Triton kernel builders in this
    package. ``batch`` is reserved for future batched-GEMM candidates and is
    not consumed by any Phase 1 selection logic.
    """

    m: int
    n: int
    k: int
    batch: int = 1


@dataclass(frozen=True)
class GemmEngineCandidate:
    """One engine option considered by the selector for a given problem.

    ``maturity`` and ``dispatchable_by_gemm_with_precision`` are orthogonal
    on purpose: a candidate can be a mature, hardware-validated kernel that
    already exists elsewhere in XQT (``maturity="executable"``) while
    ``gemm_with_precision`` still cannot reach it
    (``dispatchable_by_gemm_with_precision=False``) because this dispatcher
    has not wired a call path to it yet. Conflating the two would either
    hide a real capability or falsely imply gemm_with_precision can execute
    it today.

    ``maturity`` reuses the four-level vocabulary already defined in
    ``xqt/operator_opt/capability.py``: ``executable``, ``reference_guarded``,
    ``metadata_only``, ``planned``.
    """

    engine: str
    kernel_name: str
    maturity: str
    dispatchable_by_gemm_with_precision: bool
    rationale: str
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "kernel_name": self.kernel_name,
            "maturity": self.maturity,
            "dispatchable_by_gemm_with_precision": self.dispatchable_by_gemm_with_precision,
            "rationale": self.rationale,
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class GemmEngineSelection:
    """Result of ``select_gemm_engine()``.

    ``selected_engine`` is guaranteed to be one gemm_with_precision can
    actually execute today. ``candidates`` is the full ranked list and may
    include non-dispatchable entries; callers that only need the dispatch
    decision (like ``gemm_precision._select_engine``) read
    ``selected_engine``, while reporting/benchmark code can inspect
    ``candidates`` for the complete picture.
    """

    selected_engine: str
    selected_kernel: str
    candidates: tuple[GemmEngineCandidate, ...]
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "selected_kernel": self.selected_kernel,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rationale": list(self.rationale),
        }


def _sm_from_device(device: torch.device) -> int | None:
    """Return ``major*10+minor`` SM version for a CUDA device, or ``None``.

    Deliberately not shared with ``advisor.py``'s ``_normalize_sm``/
    ``_sm_to_int`` or ``wrappers/_common.py``'s ``_resolved_target_arch``:
    ``advisor.py`` already imports
    ``xqt.operator_opt.backends.gemm_precision``, so having
    ``xqt/operator_opt/backends/gemm_selector.py`` import back into
    ``advisor.py`` would form a circular import. This is a recorded, known
    duplication (see the TODO document's "已知不做" section), not an
    oversight.
    """

    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def select_gemm_engine(
    *,
    precision: PrecisionPolicy,
    shape: GemmShape,
    device: torch.device,
    fused_ops: frozenset[str] = frozenset(),
    goal: str = "latency",
) -> GemmEngineSelection:
    """Rank registry entries for ``gemm_with_precision(engine="auto")``."""

    if goal not in _KNOWN_GOALS:
        raise ValueError(f"unsupported GEMM selection goal: {goal!r}. Known: {sorted(_KNOWN_GOALS)}")

    sm = _sm_from_device(device)
    problem = GemmProblem(
        m=shape.m,
        n=shape.n,
        k=shape.k,
        batch=shape.batch,
        device=str(device),
        sm=sm,
    )
    registrations = default_registry().matching_precision(
        mma=precision.mma,
        problem=problem,
        fused_ops=fused_ops,
        goal=goal,
    )
    candidates: list[GemmEngineCandidate] = []
    selected: str | None = None
    selected_kernel: str | None = None
    rationale: list[str] = []
    for entry in registrations:
        engine_registration = get_engine_registration(entry.backend)
        maturity = entry.maturity
        dispatchable = entry.dispatchable_by_precision
        if entry.backend == "ptx_sm89":
            required_sm = (
                engine_registration.min_capability
                if engine_registration is not None
                else 89
            )
            maturity = "executable" if sm == required_sm else "planned"
            dispatchable = False
        reason = entry.selection_reason.format(mma=precision.mma, goal=goal)
        caveats = (
            ()
            if entry.precision_caveats is None
            else entry.precision_caveats(problem, fused_ops)
        )
        candidate = GemmEngineCandidate(
            engine=entry.backend,
            kernel_name=entry.name,
            maturity=maturity,
            dispatchable_by_gemm_with_precision=dispatchable,
            rationale=reason,
            caveats=caveats,
        )
        candidates.append(candidate)
        if selected is None and dispatchable:
            selected = entry.backend
            selected_kernel = entry.name
            rationale.append(reason)
    if selected is None or selected_kernel is None:
        raise ValueError(
            f"no registered precision GEMM route for mma={precision.mma!r}, "
            f"device={device}"
        )
    return GemmEngineSelection(
        selected,
        selected_kernel,
        tuple(candidates),
        rationale,
    )


__all__ = [
    "GemmEngineCandidate",
    "GemmEngineSelection",
    "GemmFusedOp",
    "GemmGoal",
    "GemmShape",
    "select_gemm_engine",
]
