from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    QuantSpec,
    build_packed_weight,
    default_registry,
    dispatch_gemm,
    pack_int4_signed,
    pack_int4_unsigned,
    reference_w4a16_gemm,
    reference_gemm,
    query_sm89_w4a16_resources,
    sm89_w4a16_fused_executor,
)
from xqt.kernels.ops._impl.gemm_backends.sm89 import (
    install_sm89_w8a8_executor,
    prepack_sm89_int8_weight,
    sm89_artifact_available,
)
from xqt.kernels.ops._impl.gemm_backends.sm89.w4a16_fused_sm89 import (
    install_sm89_w4a16_fused_executor,
    select_fused_split_k,
    split_k_partition,
)
from xqt.kernels.ops._impl.gemm_backends.sm89.w4a16_sm89 import install_sm89_w4a16_dequant_executor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm89_w4a16_resource_query_reports_runtime_occupancy() -> None:
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_dequant_fallback_sm89.so"
    if not artifact.is_file():
        pytest.skip("SM89 W4A16 artifact is not built")
    reports = []
    for variant in ("m1_gemv", "small_m_2_8", "tile_m_8x16x32"):
        try:
            reports.append(query_sm89_w4a16_resources(artifact, variant=variant))
        except XQTBackendError as exc:
            pytest.skip(str(exc))
    assert [report.variant for report in reports] == [
        "m1_gemv",
        "small_m_2_8",
        "tile_m_8x16x32",
    ]
    assert all(report.registers_per_thread > 0 for report in reports)
    assert all(report.max_active_blocks_per_sm > 0 for report in reports)
    assert all(report.occupancy is not None and report.occupancy > 0.0 for report in reports)


def _static_spec() -> GemmSpec:
    return GemmSpec(
        problem=GemmProblem(m=4, n=8, k=16, sm=89, device="cuda:0"),
        quant=QuantSpec(
            weight_dtype="int8",
            activation_dtype="int8",
            output_dtype="fp32",
            weight_granularity="per_channel",
            activation_granularity="per_tensor",
            weight_scale_source="weight_offline",
            activation_scale_source="activation_static",
        ),
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )


def test_missing_sm89_artifact_does_not_promote_registry() -> None:
    registry = default_registry()
    assert install_sm89_w8a8_executor(registry, artifact="/tmp/xqt-missing-sm89.so") is False
    assert registry.get("sm89_int8_mma_cutlass").maturity == "metadata_only"
    assert sm89_artifact_available("/tmp/xqt-missing-sm89.so") is False


def test_missing_w4a16_fallback_does_not_promote_registry() -> None:
    registry = default_registry()
    assert install_sm89_w4a16_dequant_executor(
        registry, artifact="/tmp/xqt-missing-w4a16-sm89.so"
    ) is False
    assert registry.get("sm89_w4a16_cutlass").maturity == "metadata_only"
    assert registry.get("sm89_w4a16_dequant_fallback").maturity == "metadata_only"


def test_missing_fused_w4a16_does_not_promote_registry() -> None:
    registry = default_registry()
    assert install_sm89_w4a16_fused_executor(
        registry, artifact="/tmp/xqt-missing-fused-w4a16-sm89.so"
    ) is False
    assert registry.get("sm89_w4a16_cutlass_fused").maturity == "metadata_only"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_w4a16_cutlass_mma_matches_reference() -> None:
    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_cutlass_fused_sm89.so"
    if not artifact.is_file():
        pytest.skip("fused SM89 W4A16 artifact is not built")
    n, k, group_size = 32, 64, 32
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        symmetric=True,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = build_packed_weight(
        pack_int4_signed(torch.randint(-8, 8, (n, k), dtype=torch.int8)),
        logical_shape=(n, k),
        spec=quant,
        scales=torch.rand(n, k // group_size) * 0.2 + 0.02,
        padded_k=k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = replace(packed, qweight=packed.qweight.cuda(), scales=packed.scales.cuda())
    spec = GemmSpec(
        problem=GemmProblem(m=16, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    activation = torch.randn(16, k, dtype=torch.float16, device="cuda")
    actual = sm89_w4a16_fused_executor(activation, packed, spec=spec, artifact=artifact)
    expected = reference_w4a16_gemm(activation, packed, spec=spec)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_fused_w4a16_supports_odd_logical_k(
    unsigned: bool, group_size: int
) -> None:
    """The adapter pads only the execution K tile, never the logical result."""

    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_cutlass_fused_sm89.so"
    if not artifact.is_file():
        pytest.skip("fused SM89 W4A16 artifact is not built")
    torch.manual_seed(9100 + group_size + int(unsigned))
    n, k = 32, 1001
    padded_k = ((k + group_size - 1) // group_size) * group_size
    groups = padded_k // group_size
    if unsigned:
        qweight = pack_int4_unsigned(
            torch.randint(0, 16, (n, padded_k), dtype=torch.uint8)
        )
        zero_points = torch.randint(0, 16, (n, groups), dtype=torch.int16).float()
        symmetric = False
    else:
        qweight = pack_int4_signed(
            torch.randint(-8, 8, (n, padded_k), dtype=torch.int8)
        )
        zero_points = None
        symmetric = True
    scales = torch.rand(n, groups, dtype=torch.float32) * 0.2 + 0.02
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        symmetric=symmetric,
        weight_zero_point=zero_points is not None,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = build_packed_weight(
        qweight,
        logical_shape=(n, k),
        spec=quant,
        scales=scales,
        zero_points=zero_points,
        padded_k=padded_k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = replace(
        packed,
        qweight=packed.qweight.cuda(),
        scales=packed.scales.cuda(),
        zero_points=None if packed.zero_points is None else packed.zero_points.cuda(),
    )
    spec = GemmSpec(
        problem=GemmProblem(m=16, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    activation = torch.randn(16, k, dtype=torch.float16, device="cuda")
    actual = sm89_w4a16_fused_executor(
        activation, packed, spec=spec, artifact=artifact
    )
    expected = reference_w4a16_gemm(activation, packed, spec=spec)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_fused_w4a16_splitk_matches_reference_with_odd_logical_k(
    unsigned: bool, group_size: int
) -> None:
    """The split-K workspace/reduction path keeps group scale semantics."""

    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_cutlass_fused_sm89.so"
    if not artifact.is_file():
        pytest.skip("fused SM89 W4A16 artifact is not built")
    torch.manual_seed(9700 + group_size + int(unsigned))
    n, k = 32, 1001
    padded_k = ((k + group_size - 1) // group_size) * group_size
    groups = padded_k // group_size
    if unsigned:
        qweight = pack_int4_unsigned(
            torch.randint(0, 16, (n, padded_k), dtype=torch.uint8)
        )
        zero_points = torch.randint(0, 16, (n, groups), dtype=torch.int16).float()
        symmetric = False
    else:
        qweight = pack_int4_signed(
            torch.randint(-8, 8, (n, padded_k), dtype=torch.int8)
        )
        zero_points = None
        symmetric = True
    scales = torch.rand(n, groups, dtype=torch.float32) * 0.2 + 0.02
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        symmetric=symmetric,
        weight_zero_point=zero_points is not None,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = build_packed_weight(
        qweight,
        logical_shape=(n, k),
        spec=quant,
        scales=scales,
        zero_points=zero_points,
        padded_k=padded_k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = replace(
        packed,
        qweight=packed.qweight.cuda(),
        scales=packed.scales.cuda(),
        zero_points=None if packed.zero_points is None else packed.zero_points.cuda(),
    )
    spec = GemmSpec(
        problem=GemmProblem(m=16, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(has_bias=True, output_dtype="fp16"),
    )
    activation = torch.randn(16, k, dtype=torch.float16, device="cuda")
    bias = torch.randn(n, dtype=torch.float32, device="cuda")
    splitk = sm89_w4a16_fused_executor(
        activation, packed, spec=spec, bias=bias, artifact=artifact, split_k=4
    )
    full = sm89_w4a16_fused_executor(
        activation, packed, spec=spec, bias=bias, artifact=artifact
    )
    expected = reference_w4a16_gemm(activation, packed, spec=spec, bias=bias)
    torch.testing.assert_close(splitk, expected, atol=0.125, rtol=2e-2)
    torch.testing.assert_close(splitk, full, atol=0.125, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_fused_w4a16_decode_matches_reference_and_padded_mma(
    unsigned: bool, group_size: int, m: int
) -> None:
    """The M=1..8 fused decode kernel keeps the single nibble/scale semantics.

    Double check: against the dense reference and against the full-K MMA path
    with the activation zero-padded to 16 rows.
    """

    artifact = Path.home() / ".cache/xqt/gemm/sm89/w4a16_cutlass_fused_sm89.so"
    if not artifact.is_file():
        pytest.skip("fused SM89 W4A16 artifact is not built")
    torch.manual_seed(9900 + group_size + m + int(unsigned))
    n, k = 64, 1001
    padded_k = ((k + group_size - 1) // group_size) * group_size
    groups = padded_k // group_size
    if unsigned:
        qweight = pack_int4_unsigned(
            torch.randint(0, 16, (n, padded_k), dtype=torch.uint8)
        )
        zero_points = torch.randint(0, 16, (n, groups), dtype=torch.int16).float()
        symmetric = False
    else:
        qweight = pack_int4_signed(
            torch.randint(-8, 8, (n, padded_k), dtype=torch.int8)
        )
        zero_points = None
        symmetric = True
    scales = torch.rand(n, groups, dtype=torch.float32) * 0.2 + 0.02
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        symmetric=symmetric,
        weight_zero_point=zero_points is not None,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = build_packed_weight(
        qweight,
        logical_shape=(n, k),
        spec=quant,
        scales=scales,
        zero_points=zero_points,
        padded_k=padded_k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = replace(
        packed,
        qweight=packed.qweight.cuda(),
        scales=packed.scales.cuda(),
        zero_points=None if packed.zero_points is None else packed.zero_points.cuda(),
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(has_bias=True, output_dtype="fp16"),
    )
    activation = torch.randn(m, k, dtype=torch.float16, device="cuda")
    bias = torch.randn(n, dtype=torch.float32, device="cuda")
    actual = sm89_w4a16_fused_executor(
        activation, packed, spec=spec, bias=bias, artifact=artifact
    )
    expected = reference_w4a16_gemm(activation, packed, spec=spec, bias=bias)
    spec_m16 = replace(spec, problem=replace(spec.problem, m=16))
    activation_m16 = F.pad(activation, (0, 0, 0, 16 - m))
    full_k = sm89_w4a16_fused_executor(
        activation_m16, packed, spec=spec_m16, bias=bias, artifact=artifact
    )[:m]
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=2e-2)
    torch.testing.assert_close(actual, full_k, atol=0.125, rtol=2e-2)


def test_select_fused_split_k_heuristic_and_partition() -> None:
    assert select_fused_split_k(m=16, n=32, k=64, sm_count=80) == 1
    split = select_fused_split_k(m=32, n=1024, k=1024, sm_count=80)
    assert 2 <= split <= 16
    k_per_split, split_count = split_k_partition(1024, split)
    assert k_per_split % 16 == 0
    assert (split_count - 1) * k_per_split < 1024 <= split_count * k_per_split
    with pytest.raises(XQTBackendError, match="positive m/n/k/sm_count"):
        select_fused_split_k(m=0, n=8, k=64, sm_count=80)


@pytest.mark.parametrize(("m", "n"), [(9, 32), (16, 10)])
def test_fused_w4a16_rejects_non_aligned_m_or_n_before_loading_artifact(
    m: int, n: int
) -> None:
    """Non-aligned M/N are an explicit policy boundary, not silent padding."""

    k = 64
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=32,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = build_packed_weight(
        pack_int4_signed(torch.zeros(n, k, dtype=torch.int8)),
        logical_shape=(n, k),
        spec=quant,
        scales=torch.ones(n, k // 32),
        padded_k=k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    with pytest.raises(XQTBackendError, match="requires M in 1..8"):
        sm89_w4a16_fused_executor(
            torch.zeros(m, k, dtype=torch.float16),
            packed,
            spec=spec,
            artifact="/tmp/xqt-missing-fused-w4a16.so",
        )


def test_native_executor_error_is_reported_as_reference_fallback() -> None:
    registry = default_registry()
    entry = registry.get("sm89_int8_mma_cutlass")

    def unavailable(*args: object, **kwargs: object) -> torch.Tensor:
        raise XQTBackendError("test artifact unavailable")

    registry.replace(
        replace(entry, maturity="executable", executor=unavailable)
    )
    spec = _static_spec()
    result = dispatch_gemm(
        torch.randint(-4, 4, (4, 16), dtype=torch.int8),
        torch.randint(-4, 4, (8, 16), dtype=torch.int8),
        spec=spec,
        weight_scales=torch.ones(8),
        activation_scales=torch.tensor(0.02),
        registry=registry,
    )
    assert result.report.native is False
    assert result.report.selected_kernel == "quantized_dequant_reference"
    assert "test artifact unavailable" in (result.report.fallback_reason or "")


def test_sm89_prepacked_weight_keeps_reference_canonical_values() -> None:
    weight = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
    scales = torch.rand(8) + 0.1
    packed = prepack_sm89_int8_weight(weight, scales=scales)
    spec = _static_spec()
    activation = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
    actual = reference_gemm(
        activation,
        packed,
        spec=spec,
        weight_scales=scales,
        activation_scales=torch.tensor(0.02),
    )
    expected = reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=scales,
        activation_scales=torch.tensor(0.02),
    )
    torch.testing.assert_close(actual, expected)
