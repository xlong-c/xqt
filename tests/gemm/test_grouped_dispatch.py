from __future__ import annotations

import json

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import (
    GemmProblem,
    GroupedGemmCandidateOutput,
    GroupedGemmNativeCandidate,
    GroupedGemmProblem,
    QuantSpec,
    dispatch_grouped_gemm,
    reference_packed_grouped_gemm,
)


def _problem(
    rows: tuple[int, ...] = (0, 2, 1),
    *,
    n: int = 3,
    k: int = 4,
    scatter: bool = True,
) -> GroupedGemmProblem:
    offsets = [0]
    for row_count in rows:
        offsets.append(offsets[-1] + row_count)
    return GroupedGemmProblem(
        problems=tuple(
            GemmProblem(m=row_count, n=n, k=k)
            for row_count in rows
        ),
        m_offsets=tuple(offsets),
        output_rows=(
            tuple(reversed(range(offsets[-1]))) if scatter else None
        ),
    )


def _dense_inputs(
    problem: GroupedGemmProblem,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    activation = torch.arange(
        problem.total_m * problem.k,
        dtype=torch.float32,
    ).reshape(problem.total_m, problem.k)
    weights = tuple(
        torch.arange(
            problem.n * problem.k,
            dtype=torch.float32,
        ).reshape(problem.n, problem.k)
        + index
        for index in range(problem.group_count)
    )
    bias = tuple(
        torch.full((problem.n,), float(index), dtype=torch.float32)
        for index in range(problem.group_count)
    )
    return activation, weights, bias


def _dense_expected(
    problem: GroupedGemmProblem,
    activation: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    bias: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    assert problem.m_offsets is not None
    outputs = tuple(
        activation[
            problem.m_offsets[index] : problem.m_offsets[index + 1]
        ]
        @ weights[index].transpose(0, 1)
        + bias[index]
        for index in range(problem.group_count)
    )
    packed = torch.cat(outputs, dim=0)
    if problem.output_rows is None:
        return packed
    output = torch.empty_like(packed)
    output.index_copy_(
        0,
        torch.tensor(problem.output_rows, dtype=torch.int64),
        packed,
    )
    return output


def test_grouped_dispatch_uses_reported_reference_fallback() -> None:
    problem = _problem()
    activation, weights, bias = _dense_inputs(problem)
    quant = QuantSpec(
        weight_dtype="fp32",
        activation_dtype="fp32",
        output_dtype="fp32",
    )
    candidate = GroupedGemmNativeCandidate(
        name="planned_grouped",
        backend="custom_cuda",
        maturity="metadata_only",
        executor=lambda: GroupedGemmCandidateOutput(
            torch.empty(problem.total_m, problem.n)
        ),
    )

    result = dispatch_grouped_gemm(
        activation,
        weights,
        grouped_problem=problem,
        quant_specs=quant,
        bias=bias,
        candidates=(candidate,),
        requested_kernel="planned_grouped",
    )

    torch.testing.assert_close(
        result.output,
        _dense_expected(problem, activation, weights, bias),
    )
    assert result.report.selected_kernel == "grouped_reference"
    assert result.report.execution_mode == "reference_grouped"
    assert result.report.native is False
    assert result.report.python_group_loop is True
    assert result.report.output_scatter is True
    assert result.report.fallback_chain == (
        "planned_grouped",
        "grouped_reference",
    )
    assert "maturity=metadata_only" in (result.report.fallback_reason or "")
    assert json.loads(json.dumps(result.report.to_dict()))[
        "execution_mode"
    ] == "reference_grouped"


def test_grouped_dispatch_prefers_executable_native_candidate() -> None:
    problem = _problem(scatter=False)
    activation, _, _ = _dense_inputs(problem)
    expected = torch.full(
        (problem.total_m, problem.n),
        7.0,
        dtype=torch.float32,
    )
    candidate = GroupedGemmNativeCandidate(
        name="native_grouped",
        backend="custom_cuda",
        executor=lambda: GroupedGemmCandidateOutput(
            output=expected,
            details={"launch_count": 1},
        ),
    )

    def unexpected_reference() -> torch.Tensor:
        raise AssertionError("native dispatch must not materialize reference inputs")

    result = dispatch_grouped_gemm(
        activation,
        None,
        grouped_problem=problem,
        quant_specs=QuantSpec(
            weight_dtype="fp32",
            activation_dtype="fp32",
            output_dtype="fp32",
        ),
        candidates=(candidate,),
        reference_executor=unexpected_reference,
    )

    assert result.output is expected
    assert result.report.selected_kernel == "native_grouped"
    assert result.report.execution_mode == "native_grouped"
    assert result.report.native is True
    assert result.report.python_group_loop is False
    assert result.report.native_details == {"launch_count": 1}


def test_grouped_dispatch_reports_native_runtime_failure() -> None:
    problem = _problem(scatter=False)
    activation, weights, bias = _dense_inputs(problem)

    def unavailable() -> GroupedGemmCandidateOutput:
        raise XQTBackendError("artifact is not correctness-promoted")

    candidate = GroupedGemmNativeCandidate(
        name="sm89_grouped",
        backend="custom_cuda",
        executor=unavailable,
    )
    result = dispatch_grouped_gemm(
        activation,
        weights,
        grouped_problem=problem,
        quant_specs=QuantSpec(
            weight_dtype="fp32",
            activation_dtype="fp32",
            output_dtype="fp32",
        ),
        bias=bias,
        candidates=(candidate,),
    )

    assert result.report.selected_kernel == "grouped_reference"
    assert "artifact is not correctness-promoted" in (
        result.report.fallback_reason or ""
    )

    with pytest.raises(
        XQTBackendError,
        match="not correctness-promoted",
    ):
        dispatch_grouped_gemm(
            activation,
            weights,
            grouped_problem=problem,
            quant_specs=QuantSpec(
                weight_dtype="fp32",
                activation_dtype="fp32",
                output_dtype="fp32",
            ),
            bias=bias,
            candidates=(candidate,),
            allow_reference=False,
        )


def test_packed_grouped_reference_splits_per_token_scales() -> None:
    problem = _problem((2, 1), n=2, k=3, scatter=False)
    activation = torch.tensor(
        [[1, -2, 3], [4, 5, -6], [-7, 8, 9]],
        dtype=torch.int8,
    )
    weights = (
        torch.tensor([[1, 2, -1], [-2, 1, 3]], dtype=torch.int8),
        torch.tensor([[3, -1, 2], [1, 4, -2]], dtype=torch.int8),
    )
    weight_scales = (
        torch.tensor([0.25, 0.5]),
        torch.tensor([0.75, 0.125]),
    )
    activation_scales = torch.tensor([0.1, 0.2, 0.3])
    quant = QuantSpec(
        weight_dtype="int8",
        activation_dtype="int8",
        output_dtype="fp32",
        weight_granularity="per_channel",
        activation_granularity="per_token",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
    )

    actual = reference_packed_grouped_gemm(
        problem,
        activation,
        weights,
        quant_specs=quant,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
    )
    expected = torch.cat(
        (
            (activation[:2].float() * activation_scales[:2, None])
            @ (weights[0].float() * weight_scales[0][:, None]).transpose(0, 1),
            (activation[2:].float() * activation_scales[2:, None])
            @ (weights[1].float() * weight_scales[1][:, None]).transpose(0, 1),
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected)


def test_grouped_dispatch_without_candidate_requires_reference() -> None:
    problem = _problem(scatter=False)
    activation, weights, _ = _dense_inputs(problem)
    with pytest.raises(
        RuntimeError,
        match="reference fallback is disabled",
    ):
        dispatch_grouped_gemm(
            activation,
            weights,
            grouped_problem=problem,
            quant_specs=QuantSpec(
                weight_dtype="fp32",
                activation_dtype="fp32",
                output_dtype="fp32",
            ),
            allow_reference=False,
        )
