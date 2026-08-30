from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
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
    build_sm89_grouped_fp8_schedule,
    calibrate_fp8_scale,
    dispatch_sm89_grouped_fp8,
    pack_sm89_grouped_fp8_weights,
    quantize_fp8,
    query_sm89_grouped_fp8_resources,
    sm89_grouped_fp8_executor,
)


_ARTIFACT = Path.home() / ".cache/xqt/gemm/sm89/fp8_grouped_sm89.so"
_ROWS = (0, 1, 2, 8, 9)


@dataclass(frozen=True, slots=True)
class _Case:
    problem: GroupedGemmProblem
    quant: QuantSpec
    weights: tuple[PackedWeight, ...]
    bias: tuple[torch.Tensor | None, ...]
    activation: torch.Tensor
    activation_scales: torch.Tensor
    expected: torch.Tensor


def _quant(
    format_name: str,
    *,
    block_k: int | None,
) -> QuantSpec:
    granularity = "per_tensor" if block_k is None else "blockwise"
    return QuantSpec(
        weight_dtype=format_name,
        activation_dtype=format_name,
        output_dtype="fp16",
        weight_granularity=granularity,
        activation_granularity=granularity,
        group_size=block_k,
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )


def _problem(*, n: int, k: int, scatter: bool) -> GroupedGemmProblem:
    offsets = [0]
    for row_count in _ROWS:
        offsets.append(offsets[-1] + row_count)
    return GroupedGemmProblem(
        problems=tuple(
            GemmProblem(
                m=row_count,
                n=n,
                k=k,
                phase="decode" if row_count <= 8 else "prefill",
                sm=89,
                device="cuda:0",
            )
            for row_count in _ROWS
        ),
        m_offsets=tuple(offsets),
        output_rows=tuple(reversed(range(offsets[-1]))) if scatter else None,
    )


def _make_case(
    *,
    format_name: str,
    block_k: int | None,
    with_bias: bool,
    scatter: bool,
    seed: int,
    n: int = 17,
    k: int = 100,
) -> _Case:
    torch.manual_seed(seed)
    problem = _problem(n=n, k=k, scatter=scatter)
    quant = _quant(format_name, block_k=block_k)
    weights: list[PackedWeight] = []
    decoded_weights: list[torch.Tensor] = []
    biases: list[torch.Tensor | None] = []
    for _ in _ROWS:
        original = torch.randn((n, k), device="cuda", dtype=torch.float16)
        scale = calibrate_fp8_scale(
            original,
            format_name=format_name,
            granularity=quant.weight_granularity,
            role="weight",
            block_k=block_k,
        )
        encoded = quantize_fp8(
            original,
            format_name=format_name,
            granularity=quant.weight_granularity,
            role="weight",
            source="weight_offline",
            scale=scale,
            block_k=block_k,
        )
        weights.append(
            build_packed_weight(
                encoded.storage,
                logical_shape=(n, k),
                spec=quant,
                scales=scale,
                padded_k=k,
                storage_layout="xqt_fp8_rowmajor_v1",
                pack_version="xqt-fp8-v1",
            )
        )
        decoded_weights.append(encoded.dequantize())
        biases.append(
            torch.randn((n,), device="cuda", dtype=torch.float32)
            if with_bias
            else None
        )

    total_m = problem.total_m
    if block_k is None:
        encoded_parts: list[torch.Tensor] = []
        decoded_parts: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        for row_count in _ROWS:
            if row_count == 0:
                scales.append(torch.ones((), device="cuda", dtype=torch.float32))
                continue
            original = torch.randn(
                (row_count, k), device="cuda", dtype=torch.float16
            )
            scale = calibrate_fp8_scale(
                original,
                format_name=format_name,
                granularity="per_tensor",
                role="activation",
            )
            encoded = quantize_fp8(
                original,
                format_name=format_name,
                granularity="per_tensor",
                role="activation",
                source="activation_static",
                scale=scale,
            )
            encoded_parts.append(encoded.storage)
            decoded_parts.append(encoded.dequantize())
            scales.append(scale.reshape(()))
        activation = torch.cat(encoded_parts, dim=0)
        activation_scales = torch.stack(scales).contiguous()
        decoded_activation = torch.cat(decoded_parts, dim=0)
    else:
        original = torch.randn((total_m, k), device="cuda", dtype=torch.float16)
        activation_scales = calibrate_fp8_scale(
            original,
            format_name=format_name,
            granularity="blockwise",
            role="activation",
            block_k=block_k,
        )
        encoded = quantize_fp8(
            original,
            format_name=format_name,
            granularity="blockwise",
            role="activation",
            source="activation_static",
            scale=activation_scales,
            block_k=block_k,
        )
        activation = encoded.storage
        decoded_activation = encoded.dequantize()

    offsets = problem.m_offsets
    assert offsets is not None
    expected_packed = torch.empty(
        (total_m, n), device="cuda", dtype=torch.float32
    )
    for index, row_count in enumerate(_ROWS):
        if row_count == 0:
            continue
        start, end = offsets[index], offsets[index + 1]
        value = decoded_activation[start:end] @ decoded_weights[index].transpose(0, 1)
        if biases[index] is not None:
            value = value + biases[index].reshape(1, -1)
        expected_packed[start:end] = value
    if problem.output_rows is None:
        expected = expected_packed
    else:
        expected = torch.empty_like(expected_packed)
        destination = torch.tensor(
            problem.output_rows, device="cuda", dtype=torch.int64
        )
        expected.index_copy_(0, destination, expected_packed)
    return _Case(
        problem=problem,
        quant=quant,
        weights=tuple(weights),
        bias=tuple(biases),
        activation=activation,
        activation_scales=activation_scales,
        expected=expected,
    )


def test_grouped_fp8_schedule_requires_cuda() -> None:
    problem = _problem(n=8, k=32, scatter=False)
    with pytest.raises(ValueError, match="requires CUDA"):
        build_sm89_grouped_fp8_schedule(problem, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("block_k", [None, 32, 64, 128])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("scatter", [False, True])
def test_grouped_fp8_matches_dequantized_reference(
    format_name: str,
    block_k: int | None,
    with_bias: bool,
    scatter: bool,
) -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    if not _ARTIFACT.is_file():
        pytest.skip("grouped SM89 FP8 artifact is not built")
    case = _make_case(
        format_name=format_name,
        block_k=block_k,
        with_bias=with_bias,
        scatter=scatter,
        seed=81000 + _FORMAT_OFFSET[format_name] + (block_k or 0),
    )
    packed = pack_sm89_grouped_fp8_weights(
        case.weights, quant=case.quant, bias=case.bias
    )
    schedule = build_sm89_grouped_fp8_schedule(case.problem, device="cuda")

    result = sm89_grouped_fp8_executor(
        case.activation,
        case.activation_scales,
        packed,
        schedule,
        artifact=_ARTIFACT,
        allow_unverified_artifact=True,
    )

    torch.testing.assert_close(
        result.output.float(), case.expected, atol=0.25, rtol=3e-2
    )
    assert result.report.launch_count == 1
    assert result.report.expert_rows == _ROWS
    assert result.report.scale_mode == (
        "tensorwise" if block_k is None else "blockwise"
    )
    assert result.report.scatter_launch_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_k", [None, 32])
def test_grouped_fp8_dispatch_falls_back_explicitly(
    tmp_path: Path,
    block_k: int | None,
) -> None:
    case = _make_case(
        format_name="fp8_e4m3",
        block_k=block_k,
        with_bias=True,
        scatter=True,
        seed=81500,
        n=8,
        k=64,
    )
    packed = pack_sm89_grouped_fp8_weights(
        case.weights,
        quant=case.quant,
        bias=case.bias,
    )
    schedule = build_sm89_grouped_fp8_schedule(
        case.problem,
        device="cuda",
    )

    result = dispatch_sm89_grouped_fp8(
        case.activation,
        case.activation_scales,
        packed,
        schedule,
        quant=case.quant,
        artifact=tmp_path / "missing-grouped-fp8.so",
    )

    torch.testing.assert_close(
        result.output.float(),
        case.expected,
        atol=0.25,
        rtol=3e-2,
    )
    assert result.report.selected_kernel == "grouped_reference"
    assert result.report.execution_mode == "reference_grouped"
    assert result.report.python_group_loop is True
    assert "not correctness-promoted" in (
        result.report.fallback_reason or ""
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_fp8_dispatch_selects_native_candidate() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    if not _ARTIFACT.is_file():
        pytest.skip("grouped SM89 FP8 artifact is not built")
    case = _make_case(
        format_name="fp8_e4m3",
        block_k=None,
        with_bias=True,
        scatter=True,
        seed=81600,
        n=8,
        k=64,
    )
    packed = pack_sm89_grouped_fp8_weights(
        case.weights,
        quant=case.quant,
        bias=case.bias,
    )
    schedule = build_sm89_grouped_fp8_schedule(
        case.problem,
        device="cuda",
    )

    result = dispatch_sm89_grouped_fp8(
        case.activation,
        case.activation_scales,
        packed,
        schedule,
        quant=case.quant,
        artifact=_ARTIFACT,
        allow_unverified_artifact=True,
    )

    torch.testing.assert_close(
        result.output.float(),
        case.expected,
        atol=0.25,
        rtol=3e-2,
    )
    assert result.report.selected_kernel == "sm89_fp8_grouped_mma"
    assert result.report.execution_mode == "native_grouped"
    assert result.report.native is True
    assert result.report.python_group_loop is False
    assert result.report.native_details is not None
    assert result.report.native_details["launch_count"] == 1
    assert result.report.native_details["fallback_chain"][-1] == "grouped_reference"


_FORMAT_OFFSET = {"fp8_e4m3": 0, "fp8_e5m2": 1000}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_fp8_all_warp_candidates_match() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    if not _ARTIFACT.is_file():
        pytest.skip("grouped SM89 FP8 artifact is not built")
    case = _make_case(
        format_name="fp8_e4m3",
        block_k=64,
        with_bias=True,
        scatter=True,
        seed=83000,
    )
    packed = pack_sm89_grouped_fp8_weights(
        case.weights, quant=case.quant, bias=case.bias
    )
    schedule = build_sm89_grouped_fp8_schedule(case.problem, device="cuda")
    for warps in (1, 2, 4, 8):
        output = sm89_grouped_fp8_executor(
            case.activation,
            case.activation_scales,
            packed,
            schedule,
            artifact=_ARTIFACT,
            warps_per_block=warps,
            allow_unverified_artifact=True,
        ).output
        torch.testing.assert_close(
            output.float(), case.expected, atol=0.25, rtol=3e-2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_fp8_resource_query_reports_mma_variants() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    if not _ARTIFACT.is_file():
        pytest.skip("grouped SM89 FP8 artifact is not built")
    for format_name, block_k in (
        ("fp8_e4m3", None),
        ("fp8_e4m3", 64),
        ("fp8_e5m2", 128),
    ):
        report = query_sm89_grouped_fp8_resources(
            _ARTIFACT,
            format_name=format_name,
            block_k=block_k,
            warps_per_block=4,
        )
        assert report.block_threads == 128
        assert report.registers_per_thread > 0
        assert report.static_shared_bytes >= 16 * 64
        assert report.max_active_blocks_per_sm > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_fp8_requires_manifest_promotion(tmp_path: Path) -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"requires SM89, got sm_{major}{minor}")
    source_manifest = _ARTIFACT.with_suffix(_ARTIFACT.suffix + ".manifest.json")
    if not _ARTIFACT.is_file() or not source_manifest.is_file():
        pytest.skip("grouped SM89 FP8 artifact and manifest are not built")
    temp_artifact = tmp_path / "grouped-fp8-unverified.so"
    shutil.copy2(_ARTIFACT, temp_artifact)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["artifact"] = str(temp_artifact)
    payload["maturity"] = "metadata_only"
    payload["metadata"]["correctness_verified"] = False
    payload["metadata"].pop("correctness", None)
    temp_artifact.with_suffix(temp_artifact.suffix + ".manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    case = _make_case(
        format_name="fp8_e4m3",
        block_k=None,
        with_bias=False,
        scatter=False,
        seed=84000,
        n=8,
        k=32,
    )
    packed = pack_sm89_grouped_fp8_weights(
        case.weights, quant=case.quant, bias=case.bias
    )
    schedule = build_sm89_grouped_fp8_schedule(case.problem, device="cuda")
    with pytest.raises(XQTBackendError, match="not correctness-promoted"):
        sm89_grouped_fp8_executor(
            case.activation,
            case.activation_scales,
            packed,
            schedule,
            artifact=temp_artifact,
        )
