"""Explicit grouped GEMM dispatch with a reported reference fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from xqt.core.errors import XQTBackendError

from .contracts import GroupedGemmProblem, PackedWeight, QuantSpec
from .reference import reference_packed_grouped_gemm
from .tuning_cache import GemmTuningLookup


_CANDIDATE_MATURITIES = frozenset({"executable", "metadata_only", "planned"})


@dataclass(frozen=True, slots=True)
class GroupedGemmCandidateOutput:
    """Normalized output returned by one executable grouped candidate."""

    output: torch.Tensor
    details: Mapping[str, Any] = field(default_factory=dict)
    native: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.output, torch.Tensor) or self.output.ndim != 2:
            raise TypeError("grouped candidate output must be a rank-2 tensor")
        if not isinstance(self.details, Mapping):
            raise TypeError("grouped candidate details must be a mapping")
        if not isinstance(self.native, bool):
            raise TypeError("grouped candidate native flag must be bool")


@dataclass(frozen=True, slots=True)
class GroupedGemmNativeCandidate:
    """One ordered native candidate for ``dispatch_grouped_gemm``."""

    name: str
    backend: str
    executor: Callable[[], GroupedGemmCandidateOutput]
    maturity: str = "executable"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("grouped candidate name cannot be empty")
        if not self.backend:
            raise ValueError("grouped candidate backend cannot be empty")
        if self.maturity not in _CANDIDATE_MATURITIES:
            raise ValueError(
                "grouped candidate maturity must be executable, metadata_only, "
                "or planned"
            )
        if not callable(self.executor):
            raise TypeError("grouped candidate executor must be callable")


@dataclass(frozen=True, slots=True)
class GroupedGemmDispatchReport:
    """Machine-readable grouped selection and fallback result."""

    requested_kernel: str | None
    selected_kernel: str
    backend: str
    maturity: str
    execution_mode: str
    native: bool
    group_count: int
    expert_rows: tuple[int, ...]
    m_offsets: tuple[int, ...]
    output_scatter: bool
    python_group_loop: bool
    fallback_reason: str | None
    fallback_chain: tuple[str, ...]
    native_details: Mapping[str, Any] | None = None
    tuning_cache_status: str = "not_applicable"
    tuning_cache_key: Mapping[str, Any] | None = None
    tuning_cache_key_id: str | None = None
    tuning_cache_reason: str | None = None
    tuning_source: str = "candidate_order"
    tuning_record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready grouped dispatch report."""

        return {
            "requested_kernel": self.requested_kernel,
            "selected_kernel": self.selected_kernel,
            "backend": self.backend,
            "maturity": self.maturity,
            "execution_mode": self.execution_mode,
            "native": self.native,
            "group_count": self.group_count,
            "expert_rows": list(self.expert_rows),
            "m_offsets": list(self.m_offsets),
            "output_scatter": self.output_scatter,
            "python_group_loop": self.python_group_loop,
            "fallback_reason": self.fallback_reason,
            "fallback_chain": list(self.fallback_chain),
            "native_details": (
                None
                if self.native_details is None
                else dict(self.native_details)
            ),
            "tuning_cache_status": self.tuning_cache_status,
            "tuning_cache_key": (
                None
                if self.tuning_cache_key is None
                else dict(self.tuning_cache_key)
            ),
            "tuning_cache_key_id": self.tuning_cache_key_id,
            "tuning_cache_reason": self.tuning_cache_reason,
            "tuning_source": self.tuning_source,
            "tuning_record_id": self.tuning_record_id,
        }


@dataclass(frozen=True, slots=True)
class GroupedGemmDispatchResult:
    """Packed grouped output plus the explicit selection report."""

    output: torch.Tensor
    report: GroupedGemmDispatchReport


def _offsets_for_problem(
    grouped_problem: GroupedGemmProblem,
) -> tuple[int, ...]:
    if grouped_problem.m_offsets is not None:
        return grouped_problem.m_offsets
    offsets = [0]
    for problem in grouped_problem.problems:
        offsets.append(offsets[-1] + problem.m)
    return tuple(offsets)


def _ordered_candidates(
    candidates: Sequence[GroupedGemmNativeCandidate],
    *,
    requested_kernel: str | None,
) -> tuple[GroupedGemmNativeCandidate, ...]:
    ordered = tuple(candidates)
    names = [candidate.name for candidate in ordered]
    if len(set(names)) != len(names):
        raise ValueError("grouped candidate names must be unique")
    if requested_kernel is None:
        return ordered
    requested = next(
        (candidate for candidate in ordered if candidate.name == requested_kernel),
        None,
    )
    if requested is None:
        raise ValueError(
            f"requested grouped kernel {requested_kernel!r} is not a candidate"
        )
    return (requested,) + tuple(
        candidate for candidate in ordered if candidate.name != requested_kernel
    )


def dispatch_grouped_gemm(
    activation: torch.Tensor,
    weights: Sequence[torch.Tensor | PackedWeight] | None,
    *,
    grouped_problem: GroupedGemmProblem,
    quant_specs: Sequence[QuantSpec] | QuantSpec,
    weight_scales: Sequence[torch.Tensor | None] | None = None,
    activation_scales: torch.Tensor
    | Sequence[torch.Tensor | None]
    | None = None,
    weight_zero_points: Sequence[torch.Tensor | None] | None = None,
    activation_zero_points: torch.Tensor
    | Sequence[torch.Tensor | None]
    | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
    candidates: Sequence[GroupedGemmNativeCandidate] = (),
    requested_kernel: str | None = None,
    allow_reference: bool = True,
    reference_executor: Callable[[], torch.Tensor] | None = None,
    tuning: GemmTuningLookup | None = None,
) -> GroupedGemmDispatchResult:
    """Try ordered native grouped candidates, then one explicit reference path."""

    if not isinstance(grouped_problem, GroupedGemmProblem):
        raise TypeError("dispatch_grouped_gemm requires GroupedGemmProblem")
    if tuning is not None and not isinstance(tuning, GemmTuningLookup):
        raise TypeError("dispatch_grouped_gemm tuning must be GemmTuningLookup")
    tuning_fields = (
        {
            "tuning_cache_status": "not_applicable",
            "tuning_cache_key": None,
            "tuning_cache_key_id": None,
            "tuning_cache_reason": None,
            "tuning_source": "candidate_order",
            "tuning_record_id": None,
        }
        if tuning is None
        else tuning.to_report_dict()
    )
    ordered = _ordered_candidates(
        candidates,
        requested_kernel=requested_kernel,
    )
    fallback_chain = tuple(candidate.name for candidate in ordered) + (
        "grouped_reference",
    )
    fallback_reasons: list[str] = []
    expected_shape = (grouped_problem.total_m, grouped_problem.n)
    for candidate in ordered:
        if candidate.maturity != "executable":
            fallback_reasons.append(
                f"candidate {candidate.name!r} is maturity={candidate.maturity}"
            )
            continue
        try:
            candidate_output = candidate.executor()
        except XQTBackendError as exc:
            fallback_reasons.append(
                f"candidate {candidate.name!r} unavailable: {exc}"
            )
            if not allow_reference:
                raise
            continue
        if not isinstance(candidate_output, GroupedGemmCandidateOutput):
            raise TypeError(
                f"grouped candidate {candidate.name!r} must return "
                "GroupedGemmCandidateOutput"
            )
        if tuple(candidate_output.output.shape) != expected_shape:
            raise RuntimeError(
                f"grouped candidate {candidate.name!r} returned shape "
                f"{tuple(candidate_output.output.shape)}, expected {expected_shape}"
            )
        return GroupedGemmDispatchResult(
            output=candidate_output.output,
            report=GroupedGemmDispatchReport(
                requested_kernel=requested_kernel,
                selected_kernel=candidate.name,
                backend=candidate.backend,
                maturity=candidate.maturity,
                execution_mode=(
                    "native_grouped"
                    if candidate_output.native
                    else "empty_grouped"
                ),
                native=candidate_output.native,
                group_count=grouped_problem.group_count,
                expert_rows=tuple(
                    problem.m for problem in grouped_problem.problems
                ),
                m_offsets=_offsets_for_problem(grouped_problem),
                output_scatter=grouped_problem.output_rows is not None,
                python_group_loop=False,
                fallback_reason=(
                    "; ".join(fallback_reasons)
                    if fallback_reasons
                    else None
                ),
                fallback_chain=fallback_chain,
                native_details=dict(candidate_output.details),
                **tuning_fields,
            ),
        )

    if not ordered and not fallback_reasons:
        fallback_reasons.append("no native grouped candidates were provided")
    if not allow_reference:
        reason = "; ".join(fallback_reasons)
        suffix = f" ({reason})" if reason else ""
        raise RuntimeError(
            "no executable grouped candidate and reference fallback is disabled"
            f"{suffix}"
        )
    if reference_executor is None:
        if weights is None:
            raise ValueError(
                "grouped reference fallback requires weights or "
                "reference_executor"
            )
        output = reference_packed_grouped_gemm(
            grouped_problem,
            activation,
            weights,
            quant_specs=quant_specs,
            weight_scales=weight_scales,
            activation_scales=activation_scales,
            weight_zero_points=weight_zero_points,
            activation_zero_points=activation_zero_points,
            bias=bias,
        )
    else:
        output = reference_executor()
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise TypeError(
                "grouped reference_executor must return a rank-2 tensor"
            )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            "grouped reference returned shape "
            f"{tuple(output.shape)}, expected {expected_shape}"
        )
    return GroupedGemmDispatchResult(
        output=output,
        report=GroupedGemmDispatchReport(
            requested_kernel=requested_kernel,
            selected_kernel="grouped_reference",
            backend="torch",
            maturity="reference_guarded",
            execution_mode="reference_grouped",
            native=False,
            group_count=grouped_problem.group_count,
            expert_rows=tuple(
                problem.m for problem in grouped_problem.problems
            ),
            m_offsets=_offsets_for_problem(grouped_problem),
            output_scatter=grouped_problem.output_rows is not None,
            python_group_loop=True,
            fallback_reason=(
                "; ".join(fallback_reasons) if fallback_reasons else None
            ),
            fallback_chain=fallback_chain,
            native_details=None,
            **tuning_fields,
        ),
    )


__all__ = [
    "GroupedGemmCandidateOutput",
    "GroupedGemmDispatchReport",
    "GroupedGemmDispatchResult",
    "GroupedGemmNativeCandidate",
    "dispatch_grouped_gemm",
]
