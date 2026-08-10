"""GEMM engine selector for XQT operator optimization.

Ranks GEMM engine candidates for ``gemm_with_precision(engine="auto")`` by
precision, shape, and device SM. Phase 1 (see
``docs/md/architecture/xqt-cuda-gemm-selector-todo.md``) keeps
``selected_engine`` behavior-identical to the legacy ``_select_engine`` in
``gemm_precision.py``; it only adds structured, honest candidate reporting
(including real engines gemm_with_precision cannot dispatch to yet, such as
``ptx_sm89``). Fused-op- and goal-aware differentiation land in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from xqt.contracts import PrecisionPolicy
from xqt.runtime.engine_resolve import get_engine_registration

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
    maturity: str
    dispatchable_by_gemm_with_precision: bool
    rationale: str
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
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
    candidates: tuple[GemmEngineCandidate, ...]
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
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


def _aligned_to_block(shape: GemmShape, block: int = 64) -> bool:
    return shape.m % block == 0 and shape.n % block == 0 and shape.k % block == 0


def _k_aligned_to_block(shape: GemmShape, block: int = 64) -> bool:
    return shape.k % block == 0


def select_gemm_engine(
    *,
    precision: PrecisionPolicy,
    shape: GemmShape,
    device: torch.device,
    fused_ops: frozenset[str] = frozenset(),
    goal: str = "latency",
) -> GemmEngineSelection:
    """Rank GEMM engine candidates for ``gemm_with_precision(engine="auto")``.

    Only ``triton``, ``tilelang``, and ``torch`` are ever returned as
    ``selected_engine``: those are the only engines
    ``gemm_precision.gemm_with_precision`` can actually execute today. Real,
    hardware-validated engines that exist elsewhere in XQT but are not wired
    into ``gemm_with_precision`` yet (for example ``ptx_sm89``, see
    ``xqt.runtime.modules.int8_mma_linear.Int8MmaLinear``) are still surfaced in
    ``candidates`` with ``dispatchable_by_gemm_with_precision=False`` so
    callers know they exist without this function claiming
    ``gemm_with_precision`` can reach them.

    ``fused_ops`` and ``goal`` are accepted and validated but do not change
    ``selected_engine`` in Phase 1 for any precision (see the TODO
    document); they are recorded in candidate rationale/caveats where
    relevant and are wired into real branching starting Phase 2.
    """

    if goal not in _KNOWN_GOALS:
        raise ValueError(f"unsupported GEMM selection goal: {goal!r}. Known: {sorted(_KNOWN_GOALS)}")

    mma = precision.mma

    if device.type != "cuda":
        if mma in {"fp4", "nvfp4"}:
            selected = "tilelang"
            reason = (
                f"{mma} has no CPU tensor-core path; the TileLang reference "
                "fallback is the only engine implemented off CUDA."
            )
            maturity = "reference_guarded"
        else:
            selected = "torch"
            reason = "non-CUDA device: torch fallback for all other precisions."
            maturity = "executable"
        candidate = GemmEngineCandidate(
            engine=selected,
            maturity=maturity,
            dispatchable_by_gemm_with_precision=True,
            rationale=reason,
        )
        return GemmEngineSelection(selected, (candidate,), [reason])

    # CUDA from here on.
    sm = _sm_from_device(device)
    candidates: list[GemmEngineCandidate] = []
    rationale: list[str] = []

    if mma in {"fp16", "bf16"}:
        selected = "triton"
        rationale.append(f"{mma} dense GEMM: triton has the broadest real kernel coverage today.")
        candidates.append(
            GemmEngineCandidate(
                engine="triton",
                maturity="executable",
                dispatchable_by_gemm_with_precision=True,
                rationale=rationale[-1],
            )
        )
        tilelang_caveats: tuple[str, ...] = ()
        if not _k_aligned_to_block(shape):
            tilelang_caveats = (
                "kernels/tilelang/gemm_builder.py requires k to be a multiple "
                "of block_k (default 64); this shape would raise with the "
                "default TileLang schedule.",
            )
        candidates.append(
            GemmEngineCandidate(
                engine="tilelang",
                maturity="executable",
                dispatchable_by_gemm_with_precision=True,
                rationale=f"real direct dense {mma} TileLang kernel exists as an alternative to triton.",
                caveats=tilelang_caveats,
            )
        )
    elif mma == "int8":
        prequantized = "activation_quant" not in fused_ops
        aligned = _aligned_to_block(shape)
        if prequantized and aligned:
            selected = "tilelang"
            rationale.append(
                "int8: pre-quantized torch.int8 a/b, shape aligned to 64 - "
                "TileLang true W8A8 kernel is the fastest real path."
            )
        else:
            selected = "triton"
            if not prequantized:
                rationale.append(
                    "int8: activation is not pre-quantized (activation_quant "
                    "is in fused_ops); triton weight-only (W8A16) reference is "
                    "the only available path for this contract."
                )
            else:
                rationale.append(
                    "int8: pre-quantized but shape is not aligned to block "
                    "size 64 - TileLang int8 kernel would raise; triton "
                    "reference fallback."
                )
        candidates.append(
            GemmEngineCandidate(
                engine="triton",
                maturity="reference_guarded",
                dispatchable_by_gemm_with_precision=True,
                rationale=(
                    "gemm_int8_triton falls back to gemm_int8_reference; the "
                    "activation stays in its original dtype (weight-only "
                    "int8, not true W8A8)."
                ),
            )
        )
        tilelang_int8_caveats: tuple[str, ...] = ()
        if not (prequantized and aligned):
            tilelang_int8_caveats = (
                "requires a.dtype == b.dtype == torch.int8 "
                "and m/n/k aligned to block size 64",
            )
        candidates.append(
            GemmEngineCandidate(
                engine="tilelang",
                maturity="executable",
                dispatchable_by_gemm_with_precision=True,
                rationale=(
                    "kernels/tilelang/int8_mma.py::int8_linear_tilelang is a "
                    "real true-W8A8 kernel; gemm_with_precision's _gemm_tilelang "
                    "int8 branch now dispatches to it (wired in Phase 2)."
                ),
                caveats=tilelang_int8_caveats,
            )
        )

        ptx_registration = get_engine_registration("ptx_sm89")
        ptx_min_capability = (
            ptx_registration.min_capability
            if ptx_registration is not None
            and ptx_registration.min_capability is not None
            else 89
        )
        candidates.append(
            GemmEngineCandidate(
                engine="ptx_sm89",
                maturity="executable" if sm == ptx_min_capability else "planned",
                dispatchable_by_gemm_with_precision=(
                    ptx_registration.dispatchable
                    if ptx_registration is not None
                    else False
                ),
                rationale=(
                    "hand-written Ada sm_89 PTX INT8 tensor-core kernel "
                    "(xqt/operator_opt/kernels/cute/int8mma_binding.py), "
                    "already used by "
                    "xqt.runtime.modules.int8_mma_linear.Int8MmaLinear with a real "
                    "m_rows>=192 auto heuristic and RTX 4070 Ti SUPER "
                    "measurements, but not reachable through "
                    "gemm_with_precision: it needs a prepacked-B caching "
                    "strategy this stateless function-call API does not "
                    "have yet."
                ),
                caveats=(
                    f"sm_{ptx_min_capability} only; batch=64 measured ~0.7x "
                    "vs tilelang (loses there), M>=256 single-layer can "
                    "exceed tilelang (~100 TOPS).",
                ),
            )
        )
    elif mma == "int4":
        selected = "triton"
        rationale.append("int4: gemm_int4_dequant_triton is the only implemented path.")
        candidates.append(
            GemmEngineCandidate(
                engine="triton",
                maturity="reference_guarded",
                dispatchable_by_gemm_with_precision=True,
                rationale="unpack + dequant reference fallback; no fused int4 tensor-core kernel yet.",
            )
        )
    elif mma in {"fp4", "nvfp4"}:
        selected = "tilelang"
        if goal == "accuracy":
            rationale.append(
                f"{mma}: goal=accuracy still selects tilelang packed dequant "
                "(only executable fused path); numeric gate should compare "
                "against a dense dequant+GEMM reference outside this selector."
            )
        elif goal == "throughput":
            rationale.append(
                f"{mma}: goal=throughput prefers tilelang fused dequant GEMM "
                f"for large K (shape k={shape.k}); single fused launch."
            )
        else:
            rationale.append(
                f"{mma}: goal=latency prefers tilelang fused dequant GEMM "
                f"(minimize host launches; shape m={shape.m})."
            )
        candidates.append(
            GemmEngineCandidate(
                engine="tilelang",
                maturity="executable",
                dispatchable_by_gemm_with_precision=True,
                rationale=(
                    f"kernels/tilelang/gemm_builder.py fused dequant epilogue "
                    f"for {mma}; goal={goal}."
                ),
            )
        )
        if goal == "accuracy":
            candidates.append(
                GemmEngineCandidate(
                    engine="torch",
                    maturity="executable",
                    dispatchable_by_gemm_with_precision=True,
                    rationale=(
                        f"{mma} accuracy goal: torch dense path is the portable "
                        "numeric reference for offline diff, not the production "
                        "fused kernel."
                    ),
                    caveats=(
                        "not selected as auto winner; use for correctness "
                        "baselines only.",
                    ),
                )
            )
        elif goal == "throughput" and shape.k >= 2048:
            candidates.append(
                GemmEngineCandidate(
                    engine="triton",
                    maturity="reference_guarded",
                    dispatchable_by_gemm_with_precision=True,
                    rationale=(
                        f"{mma} throughput goal: triton dequant reference listed "
                        "as guarded alternative when tilelang is unavailable."
                    ),
                    caveats=("reference_guarded; not the fused production path.",),
                )
            )
    elif mma in {"fp8", "mxfp8", "mxfp6", "mxfp4"}:
        selected = "triton"
        rationale.append(f"{mma}: no dedicated accelerated kernel yet; triton reference fallback.")
        candidates.append(
            GemmEngineCandidate(
                engine="triton",
                maturity="reference_guarded",
                dispatchable_by_gemm_with_precision=True,
                rationale="reference fallback; not yet a distinct accelerated path.",
            )
        )
    else:
        selected = "torch"
        rationale.append(f"unrecognized mma precision {mma!r}: torch safety fallback.")
        candidates.append(
            GemmEngineCandidate(
                engine="torch",
                maturity="executable",
                dispatchable_by_gemm_with_precision=True,
                rationale=rationale[-1],
            )
        )

    return GemmEngineSelection(selected, tuple(candidates), rationale)


__all__ = [
    "GemmEngineCandidate",
    "GemmEngineSelection",
    "GemmFusedOp",
    "GemmGoal",
    "GemmShape",
    "select_gemm_engine",
]
