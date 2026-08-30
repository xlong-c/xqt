from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm import (
    GemmProblem,
    GroupedGemmProblem,
    PackedWeight,
    QuantSpec,
    build_packed_weight,
    build_sm89_grouped_w8a8_schedule,
    dispatch_sm89_grouped_w8a8,
    pack_sm89_grouped_w8a8_weights,
    query_sm89_grouped_w8a8_resources,
    sm89_grouped_w8a8_executor,
)


_ARTIFACT = Path.home() / ".cache/xqt/gemm/sm89/w8a8_grouped_sm89.so"


def _quant(*, per_token: bool) -> QuantSpec:
    return QuantSpec(
        weight_dtype="int8",
        activation_dtype="int8",
        output_dtype="fp16",
        weight_granularity="per_channel",
        activation_granularity="per_token" if per_token else "per_tensor",
        symmetric=True,
        weight_scale_source="weight_offline",
        activation_scale_source="activation_dynamic",
        storage_layout="xqt_int8_nk_v1",
        pack_version="xqt-int8-v1",
    )


def _problem(
    rows: tuple[int, ...],
    *,
    n: int,
    k: int,
    scatter: bool,
) -> GroupedGemmProblem:
    problems = tuple(
        GemmProblem(
            m=row_count,
            n=n,
            k=k,
            phase="decode" if row_count <= 8 else "prefill",
            sm=89,
            device="cuda:0",
        )
        for row_count in rows
    )
    offsets = [0]
    for row_count in rows:
        offsets.append(offsets[-1] + row_count)
    output_rows = tuple(reversed(range(offsets[-1]))) if scatter else None
    return GroupedGemmProblem(
        problems=problems,
        m_offsets=tuple(offsets),
        output_rows=output_rows,
    )


def _make_experts(
    *,
    expert_count: int,
    n: int,
    k: int,
    quant: QuantSpec,
    bias: bool,
    seed: int,
) -> tuple[tuple[PackedWeight, ...], tuple[torch.Tensor | None, ...]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    experts: list[PackedWeight] = []
    biases: list[torch.Tensor | None] = []
    for _ in range(expert_count):
        qweight = torch.randint(-127, 128, (n, k), generator=generator, dtype=torch.int8).cuda()
        scales = (
            torch.rand((n,), generator=generator, dtype=torch.float32) * 0.02 + 0.01
        ).cuda()
        experts.append(
            build_packed_weight(
                qweight,
                logical_shape=(n, k),
                spec=quant,
                scales=scales,
                padded_k=k,
                storage_layout="xqt_int8_nk_v1",
                pack_version="xqt-int8-v1",
            )
        )
        biases.append(torch.randn(n, device="cuda", dtype=torch.float32) if bias else None)
    return tuple(experts), tuple(biases)


def _reference(
    problem: GroupedGemmProblem,
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    experts: tuple[PackedWeight, ...],
    biases: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    offsets = problem.m_offsets
    assert offsets is not None
    packed = torch.empty(
        (problem.total_m, problem.n), device=activation.device, dtype=torch.float32
    )
    per_token = int(activation_scales.numel()) == problem.total_m
    for index, expert_problem in enumerate(problem.problems):
        if expert_problem.m == 0:
            continue
        expert = experts[index]
        qweight = expert.qweight
        weight_scales = expert.scales
        start, end = offsets[index], offsets[index + 1]
        scale_a = activation_scales[start:end, None] if per_token else activation_scales.reshape(1, 1)
        value = (
            activation[start:end].float() @ qweight.float().transpose(0, 1)
        ) * scale_a * weight_scales.reshape(1, -1)
        if biases[index] is not None:
            value = value + biases[index].reshape(1, -1)
        packed[start:end] = value
    if problem.output_rows is None:
        return packed
    output = torch.empty_like(packed)
    destination = torch.tensor(problem.output_rows, device=activation.device, dtype=torch.int64)
    output.index_copy_(0, destination, packed)
    return output


def test_grouped_w8a8_schedule_requires_cuda() -> None:
    problem = _problem((1,), n=8, k=32, scatter=False)
    with pytest.raises(ValueError, match="requires CUDA"):
        build_sm89_grouped_w8a8_schedule(problem, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("shape", [(8, 32), (17, 100), (32, 127)])
def test_grouped_w8a8_matches_integer_reference(
    per_token: bool,
    bias: bool,
    scatter: bool,
    shape: tuple[int, int],
) -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    if not _ARTIFACT.is_file():
        pytest.skip("grouped SM89 W8A8 artifact is not built")
    n, k = shape
    rows = (0, 1, 2, 8, 9)
    problem = _problem(rows, n=n, k=k, scatter=scatter)
    quant = _quant(per_token=per_token)
    experts, biases = _make_experts(
        expert_count=len(rows),
        n=n,
        k=k,
        quant=quant,
        bias=bias,
        seed=61000 + n + k,
    )
    packed = pack_sm89_grouped_w8a8_weights(
        experts,
        quant=quant,
        bias=biases,
    )
    schedule = build_sm89_grouped_w8a8_schedule(problem, device="cuda")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(62000 + n + k)
    activation = torch.randint(
        -127,
        128,
        (problem.total_m, k),
        generator=generator,
        dtype=torch.int8,
    ).cuda()
    scale_count = problem.total_m if per_token else 1
    activation_scales = (
        torch.rand((scale_count,), generator=generator, dtype=torch.float32) * 0.01 + 0.001
    ).cuda()
    expected = _reference(problem, activation, activation_scales, experts, biases)

    result = sm89_grouped_w8a8_executor(
        activation,
        activation_scales,
        packed,
        schedule,
        artifact=_ARTIFACT,
        allow_unverified_artifact=True,
    )

    torch.testing.assert_close(result.output.float(), expected, atol=0.0625, rtol=2e-3)
    assert result.report.launch_count == 1
    assert result.report.scheduler == "direct_task_grid"
    assert result.report.expert_rows == rows
    assert result.report.activation_scale_mode == ("per_token" if per_token else "per_tensor")
    assert result.report.scatter_launch_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("per_token", [False, True])
def test_grouped_w8a8_dispatch_falls_back_explicitly(
    tmp_path: Path,
    per_token: bool,
) -> None:
    rows = (0, 1, 2)
    n, k = 8, 32
    problem = _problem(rows, n=n, k=k, scatter=True)
    quant = _quant(per_token=per_token)
    experts, biases = _make_experts(
        expert_count=len(rows),
        n=n,
        k=k,
        quant=quant,
        bias=True,
        seed=62500,
    )
    packed = pack_sm89_grouped_w8a8_weights(
        experts,
        quant=quant,
        bias=biases,
    )
    schedule = build_sm89_grouped_w8a8_schedule(problem, device="cuda")
    activation = torch.randint(
        -127,
        128,
        (problem.total_m, k),
        dtype=torch.int8,
        device="cuda",
    )
    scale_count = problem.total_m if per_token else 1
    activation_scales = torch.rand(
        (scale_count,),
        dtype=torch.float32,
        device="cuda",
    ).mul_(0.01).add_(0.001)
    expected = _reference(
        problem,
        activation,
        activation_scales,
        experts,
        biases,
    )

    result = dispatch_sm89_grouped_w8a8(
        activation,
        activation_scales,
        packed,
        schedule,
        quant=quant,
        artifact=tmp_path / "missing-grouped-w8a8.so",
    )

    torch.testing.assert_close(
        result.output.float(),
        expected,
        atol=0.0625,
        rtol=2e-3,
    )
    assert result.report.selected_kernel == "grouped_reference"
    assert result.report.execution_mode == "reference_grouped"
    assert result.report.python_group_loop is True
    assert "not correctness-promoted" in (
        result.report.fallback_reason or ""
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w8a8_dispatch_selects_native_candidate() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    if not _ARTIFACT.is_file():
        pytest.skip("grouped SM89 W8A8 artifact is not built")
    rows = (1, 2)
    n, k = 8, 32
    problem = _problem(rows, n=n, k=k, scatter=True)
    quant = _quant(per_token=True)
    experts, biases = _make_experts(
        expert_count=len(rows),
        n=n,
        k=k,
        quant=quant,
        bias=True,
        seed=62600,
    )
    packed = pack_sm89_grouped_w8a8_weights(
        experts,
        quant=quant,
        bias=biases,
    )
    schedule = build_sm89_grouped_w8a8_schedule(problem, device="cuda")
    activation = torch.randint(
        -127,
        128,
        (problem.total_m, k),
        dtype=torch.int8,
        device="cuda",
    )
    activation_scales = torch.rand(
        (problem.total_m,),
        dtype=torch.float32,
        device="cuda",
    ).mul_(0.01).add_(0.001)
    expected = _reference(
        problem,
        activation,
        activation_scales,
        experts,
        biases,
    )

    result = dispatch_sm89_grouped_w8a8(
        activation,
        activation_scales,
        packed,
        schedule,
        quant=quant,
        artifact=_ARTIFACT,
        allow_unverified_artifact=True,
    )

    torch.testing.assert_close(
        result.output.float(),
        expected,
        atol=0.0625,
        rtol=2e-3,
    )
    assert result.report.selected_kernel == "sm89_w8a8_grouped_mma"
    assert result.report.execution_mode == "native_grouped"
    assert result.report.native is True
    assert result.report.python_group_loop is False
    assert result.report.native_details is not None
    assert result.report.native_details["launch_count"] == 1
    assert result.report.native_details["fallback_chain"][-1] == "grouped_reference"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w8a8_resource_query_reports_native_mma() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    if not _ARTIFACT.is_file():
        pytest.skip("grouped SM89 W8A8 artifact is not built")
    report = query_sm89_grouped_w8a8_resources(_ARTIFACT)
    assert report.block_threads == 128
    assert report.registers_per_thread > 0
    assert report.static_shared_bytes >= 16 * 64 + 8 * 32
    assert report.max_active_blocks_per_sm > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w8a8_requires_manifest_promotion(tmp_path: Path) -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    source_manifest = _ARTIFACT.with_suffix(_ARTIFACT.suffix + ".manifest.json")
    if not _ARTIFACT.is_file() or not source_manifest.is_file():
        pytest.skip("grouped SM89 W8A8 artifact and manifest are not built")
    temp_artifact = tmp_path / "grouped-w8a8-unverified.so"
    shutil.copy2(_ARTIFACT, temp_artifact)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["artifact"] = str(temp_artifact)
    payload["maturity"] = "metadata_only"
    payload["metadata"]["correctness_verified"] = False
    payload["metadata"].pop("correctness", None)
    temp_artifact.with_suffix(temp_artifact.suffix + ".manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    n, k = 8, 32
    quant = _quant(per_token=True)
    experts, biases = _make_experts(
        expert_count=1, n=n, k=k, quant=quant, bias=False, seed=63000
    )
    problem = _problem((1,), n=n, k=k, scatter=False)
    packed = pack_sm89_grouped_w8a8_weights(experts, quant=quant, bias=biases)
    schedule = build_sm89_grouped_w8a8_schedule(problem, device="cuda")
    activation = torch.ones((1, k), device="cuda", dtype=torch.int8)
    activation_scales = torch.ones((1,), device="cuda", dtype=torch.float32)

    with pytest.raises(XQTBackendError, match="not correctness-promoted"):
        sm89_grouped_w8a8_executor(
            activation,
            activation_scales,
            packed,
            schedule,
            artifact=temp_artifact,
        )
