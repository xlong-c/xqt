from __future__ import annotations

from dataclasses import replace

import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    QuantSpec,
    build_packed_weight,
    default_registry,
    dispatch_gemm,
    pack_int4_signed,
    reference_w4a16_gemm,
    select_kernel,
)


def _sm89_w8a8_spec() -> GemmSpec:
    return GemmSpec(
        problem=GemmProblem(m=4, n=8, k=16, phase="decode", sm=89, device="cuda:0"),
        quant=QuantSpec(
            weight_dtype="int8",
            activation_dtype="int8",
            weight_granularity="per_channel",
            activation_granularity="per_token",
            weight_scale_source="weight_offline",
            activation_scale_source="activation_dynamic",
            output_dtype="fp32",
        ),
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )


def test_registry_exposes_metadata_only_sm89_candidate_and_reference() -> None:
    candidates = select_kernel(_sm89_w8a8_spec(), registry=default_registry())
    assert candidates[0].name == "sm89_int8_mma_cutlass"
    assert candidates[0].maturity == "metadata_only"
    assert any(candidate.name == "quantized_dequant_reference" for candidate in candidates)


def test_w4a16_registry_exposes_main_and_alternate_cutlass_ladder() -> None:
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
    spec = GemmSpec(
        problem=GemmProblem(m=9, n=257, k=1001, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    candidates = select_kernel(spec, registry=default_registry())
    names = [candidate.name for candidate in candidates]
    assert names[:4] == [
        "sm89_w4a16_cutlass_fused",
        "sm89_w4a16_cutlass",
        "sm89_w4a16_cutlass_alt_tile",
        "sm89_w4a16_dequant_fallback",
    ]
    assert candidates[0].maturity == "metadata_only"
    assert candidates[2].implementation == "cutlass_alternate_tile_pending"
    assert any(
        candidate.name == "sm89_w4a16_triton_dequant" and candidate.maturity == "planned"
        for candidate in candidates
    )


def test_dispatch_reports_fallback_instead_of_claiming_native() -> None:
    spec = _sm89_w8a8_spec()
    result = dispatch_gemm(
        torch.randn(4, 16),
        torch.randint(-8, 8, (8, 16), dtype=torch.int8),
        spec=spec,
        weight_scales=torch.ones(8),
        registry=default_registry(),
    )
    assert result.report.selected_kernel == "quantized_dequant_reference"
    assert result.report.native is False
    assert result.report.fallback_reason is not None
    assert result.output.shape == (4, 8)


def test_native_w4a16_report_exposes_shape_variant() -> None:
    n, k, group_size = 8, 32, 32
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = build_packed_weight(
        pack_int4_signed(torch.zeros(n, k, dtype=torch.int8)),
        scales=torch.ones(n, 1),
        spec=quant,
        logical_shape=(n, k),
        padded_k=k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    registry = default_registry()
    entry = registry.get("sm89_w4a16_dequant_fallback")

    def fake_executor(activation: torch.Tensor, weight: object, **kwargs: object) -> torch.Tensor:
        return reference_w4a16_gemm(activation, packed, spec=kwargs["spec"])

    registry.replace(replace(entry, maturity="executable", executor=fake_executor))
    for m, expected_variant in ((1, "m1_gemv"), (4, "small_m_2_8"), (16, "tile_m_8x16x32")):
        spec = GemmSpec(
            problem=GemmProblem(m=m, n=n, k=k, sm=89, device="cuda:0"),
            quant=quant,
            epilogue=EpilogueSpec(output_dtype="fp16"),
        )
        result = dispatch_gemm(torch.ones(m, k, dtype=torch.float16), packed, spec=spec, registry=registry)
        assert result.report.native is True
        assert result.report.shape_variant == expected_variant
        assert result.report.group_size == group_size
        assert result.report.fallback_chain[:4] == (
            "sm89_w4a16_cutlass_fused",
            "sm89_w4a16_cutlass",
            "sm89_w4a16_cutlass_alt_tile",
            "sm89_w4a16_dequant_fallback",
        )
        assert "sm89_w4a16_triton_dequant" in result.report.fallback_chain


def test_dispatch_tries_alternate_executable_before_reference() -> None:
    n, k, group_size = 8, 33, 32
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        weight_scale_source="weight_offline",
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    packed = build_packed_weight(
        pack_int4_signed(torch.zeros(n, 64, dtype=torch.int8)),
        scales=torch.ones(n, 2),
        spec=quant,
        logical_shape=(n, k),
        padded_k=64,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    registry = default_registry()
    alternate = registry.get("sm89_w4a16_cutlass_alt_tile")

    def fake_executor(activation: torch.Tensor, weight: object, **kwargs: object) -> torch.Tensor:
        assert weight is packed
        return reference_w4a16_gemm(activation, packed, spec=kwargs["spec"])

    registry.replace(replace(alternate, maturity="executable", executor=fake_executor))
    spec = GemmSpec(
        problem=GemmProblem(m=4, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    result = dispatch_gemm(torch.ones(4, k, dtype=torch.float16), packed, spec=spec, registry=registry)
    assert result.report.selected_kernel == "sm89_w4a16_cutlass_alt_tile"
    assert result.report.native is True
    assert "sm89_w4a16_cutlass" in result.report.fallback_chain
    assert "maturity=metadata_only" in (result.report.fallback_reason or "")


def test_dispatch_falls_back_when_fused_rejects_non_aligned_shape() -> None:
    n, k = 32, 64
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
        scales=torch.ones(n, 2),
        logical_shape=(n, k),
        spec=quant,
        padded_k=k,
        storage_layout="xqt_int4_nk_v1",
        pack_version="xqt-int4-v1",
    )
    registry = default_registry()
    fused = registry.get("sm89_w4a16_cutlass_fused")
    fallback = registry.get("sm89_w4a16_dequant_fallback")

    def fused_executor(*args: object, **kwargs: object) -> torch.Tensor:
        raise XQTBackendError("requires M%16=0 and N%8=0")

    def fallback_executor(
        activation: torch.Tensor, weight: object, **kwargs: object
    ) -> torch.Tensor:
        assert weight is packed
        return reference_w4a16_gemm(activation, packed, spec=kwargs["spec"])

    registry.replace(replace(fused, maturity="executable", executor=fused_executor))
    registry.replace(replace(fallback, maturity="executable", executor=fallback_executor))
    spec = GemmSpec(
        problem=GemmProblem(m=8, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    result = dispatch_gemm(
        torch.ones(8, k, dtype=torch.float16), packed, spec=spec, registry=registry
    )
    assert result.report.selected_kernel == "sm89_w4a16_dequant_fallback"
    assert result.report.native is True
    assert "requires M%16=0 and N%8=0" in (result.report.fallback_reason or "")


def test_w8a16_registry_exposes_main_cutlass_ladder_and_reference() -> None:
    quant = QuantSpec(
        weight_dtype="int8",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="per_channel",
        weight_scale_source="weight_offline",
        storage_layout="xqt_int8_nk_v1",
        pack_version="xqt-int8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=9, n=257, k=1001, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    candidates = select_kernel(spec, registry=default_registry())
    names = [candidate.name for candidate in candidates]
    assert names[:2] == [
        "sm89_w8a16_cutlass",
        "w8a16_packed_reference",
    ]
    assert candidates[0].maturity == "metadata_only"
    assert any(candidate.name == "quantized_dequant_reference" for candidate in candidates)
