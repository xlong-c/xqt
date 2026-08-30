from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    GroupedGemmProblem,
    PackedWeight,
    QuantSpec,
    build_packed_weight,
    build_sm89_grouped_w4a16_schedule,
    dispatch_sm89_grouped_w4a16,
    pack_int4_signed,
    pack_int4_unsigned,
    pack_sm89_grouped_w4a16_weights,
    pack_sm89_grouped_w4a16_weights_multi_stream,
    query_sm89_grouped_w4a16_persistent_resources,
    reference_w4a16_gemm,
    sm89_grouped_w4a16_executor,
    warmup_sm89_grouped_w4a16,
)


def _grouped_problem(
    rows: tuple[int, ...],
    *,
    n: int = 32,
    k: int = 1001,
    output_rows: tuple[int, ...] | None = None,
) -> GroupedGemmProblem:
    problems = tuple(
        GemmProblem(
            m=m,
            n=n,
            k=k,
            phase="decode" if m <= 8 else "prefill",
            sm=89,
            device="cuda:0",
        )
        for m in rows
    )
    offsets = [0]
    for m in rows:
        offsets.append(offsets[-1] + m)
    return GroupedGemmProblem(
        problems=problems,
        m_offsets=tuple(offsets),
        output_rows=output_rows,
    )


def test_grouped_problem_keeps_empty_experts_and_exact_offsets() -> None:
    problem = _grouped_problem((0, 1, 2, 8, 9))
    assert problem.total_m == 20
    assert problem.group_count == 5
    assert problem.m_offsets == (0, 0, 1, 3, 11, 20)


def test_grouped_problem_rejects_offsets_that_only_match_total() -> None:
    with pytest.raises(ValueError, match="cumulative expert M offsets"):
        GroupedGemmProblem.from_dict(
            {
                "problems": [
                    {
                        "m": 2,
                        "n": 32,
                        "k": 1001,
                        "phase": "decode",
                        "sm": 89,
                        "device": "cuda:0",
                    },
                    {
                        "m": 3,
                        "n": 32,
                        "k": 1001,
                        "phase": "decode",
                        "sm": 89,
                        "device": "cuda:0",
                    },
                ],
                "m_offsets": [0, 1, 5],
            }
        )


def test_grouped_problem_rejects_non_permutation_output_rows() -> None:
    with pytest.raises(ValueError, match="permutation"):
        _grouped_problem((1, 2), output_rows=(0, 0, 2))


def test_grouped_schedule_requires_cuda_device() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        build_sm89_grouped_w4a16_schedule(_grouped_problem((1,)), device="cpu")


def _quant(group_size: int, *, unsigned: bool) -> QuantSpec:
    return QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        symmetric=not unsigned,
        weight_zero_point=unsigned,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )


def _make_weight(
    *,
    n: int,
    k: int,
    group_size: int,
    unsigned: bool,
    seed: int,
) -> PackedWeight:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    padded_k = ((k + group_size - 1) // group_size) * group_size
    group_count = padded_k // group_size
    if unsigned:
        qweight = pack_int4_unsigned(
            torch.randint(0, 16, (n, padded_k), generator=generator, dtype=torch.uint8)
        )
        zero_points = torch.randint(
            0, 16, (n, group_count), generator=generator, dtype=torch.int16
        ).to(torch.float32)
    else:
        qweight = pack_int4_signed(
            torch.randint(-8, 8, (n, padded_k), generator=generator, dtype=torch.int8)
        )
        zero_points = None
    scales = torch.rand((n, group_count), generator=generator, dtype=torch.float32) * 0.2 + 0.02
    packed = build_packed_weight(
        qweight,
        logical_shape=(n, k),
        spec=_quant(group_size, unsigned=unsigned),
        scales=scales,
        zero_points=zero_points,
        padded_k=padded_k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    return replace(
        packed,
        qweight=packed.qweight.cuda(),
        scales=packed.scales.cuda(),
        zero_points=None if packed.zero_points is None else packed.zero_points.cuda(),
    )


def _grouped_reference(
    problem: GroupedGemmProblem,
    weights: tuple[PackedWeight, ...],
    activation: torch.Tensor,
    quant: QuantSpec,
    bias: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    offsets = problem.m_offsets
    assert offsets is not None
    for index, item in enumerate(problem.problems):
        if item.m == 0:
            continue
        spec = GemmSpec(
            problem=item,
            quant=quant,
            epilogue=EpilogueSpec(has_bias=True, output_dtype="fp16"),
        )
        outputs.append(
            reference_w4a16_gemm(
                activation[offsets[index] : offsets[index + 1]],
                weights[index],
                spec=spec,
                bias=bias[index],
            )
        )
    packed_output = torch.cat(outputs, dim=0)
    if problem.output_rows is None:
        return packed_output
    expected = torch.empty_like(packed_output)
    destination = torch.tensor(problem.output_rows, dtype=torch.int64, device=activation.device)
    expected.index_copy_(0, destination, packed_output)
    return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("scatter", [False, True])
def test_grouped_w4a16_matches_per_expert_reference(
    unsigned: bool,
    group_size: int,
    scatter: bool,
) -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_grouped_sm89.so"
    if not artifact.is_file():
        pytest.skip("grouped SM89 W4A16 artifact is not built")
    rows = (0, 1, 2, 8, 9)
    n, k = 32, 1001
    total_m = sum(rows)
    output_rows = tuple(reversed(range(total_m))) if scatter else None
    problem = _grouped_problem(rows, n=n, k=k, output_rows=output_rows)
    quant = _quant(group_size, unsigned=unsigned)
    experts = tuple(
        _make_weight(
            n=n,
            k=k,
            group_size=group_size,
            unsigned=unsigned,
            seed=51000 + index,
        )
        for index in range(len(rows))
    )
    bias = tuple(torch.randn(n, dtype=torch.float32, device="cuda") for _ in rows)
    packed = pack_sm89_grouped_w4a16_weights(experts, quant=quant, bias=bias)
    schedule = build_sm89_grouped_w4a16_schedule(problem, device="cuda")
    activation = torch.randn(total_m, k, dtype=torch.float16, device="cuda")
    expected = _grouped_reference(problem, experts, activation, quant, bias)

    for scheduler in ("direct", "bucketed", "persistent", "auto"):
        result = sm89_grouped_w4a16_executor(
            activation,
            packed,
            schedule,
            artifact=artifact,
            scheduler=scheduler,
            persistent_blocks_per_sm=2,
            allow_unverified_artifact=True,
        )
        torch.testing.assert_close(result.output.float(), expected.float(), atol=0.125, rtol=2e-2)
        assert result.report.expert_rows == rows
        assert result.report.output_scatter is scatter
        assert result.report.scatter_mode == (
            "in_kernel_permutation" if scatter else "identity"
        )
        assert result.report.scatter_launch_count == 0
        if scheduler == "direct":
            assert result.report.launch_count == 1
        elif scheduler == "bucketed":
            assert result.report.launch_count == 3
        elif scheduler == "persistent":
            assert result.report.scheduler == "persistent_grid_stride"
            assert result.report.launch_count == 1
            assert result.report.persistent_blocks_per_sm == 2
            assert result.report.persistent_grid_blocks == min(
                schedule.task_count * n,
                schedule.multiprocessor_count * 2,
            )
            assert result.report.fallback_chain[0].endswith(
                "persistent_grid_stride"
            )
        else:
            assert result.report.scheduler == "direct_task_grid"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w4a16_persistent_resource_query() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_grouped_sm89.so"
    if not artifact.is_file():
        pytest.skip("grouped SM89 W4A16 artifact is not built")

    report = query_sm89_grouped_w4a16_persistent_resources(
        artifact,
        blocks_per_sm=2,
    )

    assert report.requested_blocks_per_sm == 2
    assert report.resident_blocks_per_sm == min(2, report.max_active_blocks_per_sm)
    assert report.block_threads == 256
    assert report.registers_per_thread > 0
    assert report.static_shared_bytes >= 0
    assert report.max_active_blocks_per_sm > 0
    assert report.multiprocessor_count > 0
    assert report.capability == "sm_89"


@pytest.mark.parametrize("blocks_per_sm", [-1, 9])
def test_grouped_w4a16_persistent_resource_query_rejects_invalid_blocks(
    blocks_per_sm: int,
) -> None:
    with pytest.raises(ValueError, match="persistent_blocks_per_sm"):
        query_sm89_grouped_w4a16_persistent_resources(
            "unused.so",
            blocks_per_sm=blocks_per_sm,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w4a16_persistent_max_active_blocks() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_grouped_sm89.so"
    if not artifact.is_file():
        pytest.skip("grouped SM89 W4A16 artifact is not built")

    report = query_sm89_grouped_w4a16_persistent_resources(
        artifact,
        blocks_per_sm=0,
    )
    assert report.requested_blocks_per_sm == 0
    assert report.resident_blocks_per_sm == report.max_active_blocks_per_sm
    assert report.resident_blocks_per_sm > 0


def test_grouped_w4a16_persistent_resource_query_rejects_bool_blocks() -> None:
    with pytest.raises(TypeError, match="persistent_blocks_per_sm"):
        query_sm89_grouped_w4a16_persistent_resources(
            "unused.so",
            blocks_per_sm=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("unsigned", [False, True])
def test_grouped_w4a16_dispatch_falls_back_explicitly(
    tmp_path: Path,
    unsigned: bool,
) -> None:
    rows = (0, 1, 2)
    n, k = 8, 64
    problem = _grouped_problem(
        rows,
        n=n,
        k=k,
        output_rows=tuple(reversed(range(sum(rows)))),
    )
    quant = _quant(32, unsigned=unsigned)
    experts = tuple(
        _make_weight(
            n=n,
            k=k,
            group_size=32,
            unsigned=unsigned,
            seed=51500 + index,
        )
        for index in range(len(rows))
    )
    bias = tuple(
        torch.randn(n, dtype=torch.float32, device="cuda") for _ in rows
    )
    packed = pack_sm89_grouped_w4a16_weights(
        experts,
        quant=quant,
        bias=bias,
    )
    schedule = build_sm89_grouped_w4a16_schedule(problem, device="cuda")
    activation = torch.randn(
        problem.total_m,
        k,
        dtype=torch.float16,
        device="cuda",
    )
    expected = _grouped_reference(problem, experts, activation, quant, bias)

    result = dispatch_sm89_grouped_w4a16(
        activation,
        packed,
        schedule,
        quant=quant,
        artifact=tmp_path / "missing-grouped-w4a16.so",
    )

    torch.testing.assert_close(
        result.output.float(),
        expected.float(),
        atol=0.125,
        rtol=2e-2,
    )
    assert result.report.selected_kernel == "grouped_reference"
    assert result.report.execution_mode == "reference_grouped"
    assert result.report.python_group_loop is True
    assert "not correctness-promoted" in (
        result.report.fallback_reason or ""
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w4a16_dispatch_selects_native_candidate() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_grouped_sm89.so"
    if not artifact.is_file():
        pytest.skip("grouped SM89 W4A16 artifact is not built")
    rows = (1, 2)
    n, k = 8, 64
    problem = _grouped_problem(
        rows,
        n=n,
        k=k,
        output_rows=tuple(reversed(range(sum(rows)))),
    )
    quant = _quant(32, unsigned=False)
    experts = tuple(
        _make_weight(
            n=n,
            k=k,
            group_size=32,
            unsigned=False,
            seed=51600 + index,
        )
        for index in range(len(rows))
    )
    bias = tuple(
        torch.randn(n, dtype=torch.float32, device="cuda") for _ in rows
    )
    packed = pack_sm89_grouped_w4a16_weights(
        experts,
        quant=quant,
        bias=bias,
    )
    schedule = build_sm89_grouped_w4a16_schedule(problem, device="cuda")
    activation = torch.randn(
        problem.total_m,
        k,
        dtype=torch.float16,
        device="cuda",
    )
    expected = _grouped_reference(problem, experts, activation, quant, bias)

    result = dispatch_sm89_grouped_w4a16(
        activation,
        packed,
        schedule,
        quant=quant,
        artifact=artifact,
        scheduler="direct",
        allow_unverified_artifact=True,
    )

    torch.testing.assert_close(
        result.output.float(),
        expected.float(),
        atol=0.125,
        rtol=2e-2,
    )
    assert result.report.selected_kernel == "sm89_w4a16_grouped_decode"
    assert result.report.execution_mode == "native_grouped"
    assert result.report.native is True
    assert result.report.python_group_loop is False
    assert result.report.native_details is not None
    assert result.report.native_details["launch_count"] == 1
    assert result.report.native_details["fallback_chain"][-1] == "grouped_reference"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w4a16_requires_manifest_promotion(tmp_path: Path) -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_grouped_sm89.so"
    source_manifest = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if not artifact.is_file() or not source_manifest.is_file():
        pytest.skip("grouped SM89 W4A16 artifact and manifest are not built")
    temp_artifact = tmp_path / "grouped-unverified.so"
    shutil.copy2(artifact, temp_artifact)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["artifact"] = str(temp_artifact)
    payload["maturity"] = "metadata_only"
    payload["metadata"]["correctness_verified"] = False
    payload["metadata"].pop("correctness", None)
    temp_artifact.with_suffix(temp_artifact.suffix + ".manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    n, k = 16, 64
    quant = _quant(32, unsigned=False)
    expert = _make_weight(n=n, k=k, group_size=32, unsigned=False, seed=52000)
    problem = _grouped_problem((1,), n=n, k=k, output_rows=(0,))
    packed = pack_sm89_grouped_w4a16_weights(
        (expert,), quant=quant, bias=(torch.zeros(n, device="cuda"),)
    )
    schedule = build_sm89_grouped_w4a16_schedule(problem, device="cuda")
    activation = torch.randn(1, k, dtype=torch.float16, device="cuda")
    with pytest.raises(XQTBackendError, match="correctness-promoted"):
        sm89_grouped_w4a16_executor(
            activation,
            packed,
            schedule,
            artifact=temp_artifact,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("stream_count", [1, 3, 5])
def test_multi_stream_prepack_matches_canonical(stream_count: int) -> None:
    rows = (0, 1, 2, 8, 9)
    n, k = 32, 1001
    quant = _quant(128, unsigned=False)
    experts = tuple(
        _make_weight(n=n, k=k, group_size=128, unsigned=False, seed=53000 + index)
        for index in range(len(rows))
    )
    bias = tuple(torch.randn(n, dtype=torch.float32, device="cuda") for _ in rows)
    canonical = pack_sm89_grouped_w4a16_weights(experts, quant=quant, bias=bias)
    multi = pack_sm89_grouped_w4a16_weights_multi_stream(
        experts, quant=quant, bias=bias, stream_count=stream_count
    )

    assert multi.expert_count == canonical.expert_count
    assert torch.equal(multi.qweight, canonical.qweight)
    assert torch.equal(multi.scales, canonical.scales)
    assert torch.equal(multi.bias, canonical.bias)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_w4a16_warmup_reports_runs() -> None:
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_grouped_sm89.so"
    if not artifact.is_file():
        pytest.skip("grouped SM89 W4A16 artifact is not built")
    n, k = 16, 64
    quant = _quant(32, unsigned=False)
    expert = _make_weight(n=n, k=k, group_size=32, unsigned=False, seed=54000)
    problem = _grouped_problem((1,), n=n, k=k, output_rows=(0,))
    packed = pack_sm89_grouped_w4a16_weights(
        (expert,), quant=quant, bias=(torch.zeros(n, device="cuda"),)
    )
    schedule = build_sm89_grouped_w4a16_schedule(problem, device="cuda")
    activation = torch.randn(1, k, dtype=torch.float16, device="cuda")

    report = warmup_sm89_grouped_w4a16(
        activation,
        packed,
        schedule,
        quant=quant,
        artifact=artifact,
        runs=3,
        allow_unverified_artifact=True,
    )
    assert report["runs"] == 3
    assert report["expert_count"] == 1
