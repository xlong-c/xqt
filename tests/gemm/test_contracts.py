from __future__ import annotations

import pytest

from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    GroupedGemmProblem,
    PackedWeightMetadata,
    QuantSpec,
    build_packed_weight,
    canonical_weight_view,
    cutlass_weight_view,
    pack_int4_signed,
    reference_weight_view,
)
import torch


def test_contract_round_trip_preserves_logical_and_quant_fields() -> None:
    problem = GemmProblem(
        m=7,
        n=19,
        k=33,
        phase="decode",
        device="cuda:0",
        sm=89,
        cuda_graph=True,
    )
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        compute_dtype="fp32",
        output_dtype="bf16",
        weight_granularity="groupwise",
        group_size=32,
        symmetric=False,
        weight_zero_point=True,
        weight_scale_source="weight_offline",
    )
    spec = GemmSpec(problem=problem, quant=quant, epilogue=EpilogueSpec(output_dtype="bf16"))
    assert GemmSpec.from_dict(spec.to_dict()) == spec
    assert quant.scale_mode == "w:groupwise/a:per_tensor"


def test_quant_spec_rejects_mixed_source_semantics() -> None:
    with pytest.raises(ValueError, match="weight_scale_source"):
        QuantSpec(weight_dtype="int4", activation_dtype="fp16")
    with pytest.raises(ValueError, match="activation_scale_source"):
        QuantSpec(
            weight_dtype="int8",
            activation_dtype="int8",
            weight_scale_source="weight_offline",
        )


def test_int8_mma_can_declare_int32_accumulator() -> None:
    quant = QuantSpec(
        weight_dtype="int8",
        activation_dtype="int8",
        accum_dtype="int32",
        output_dtype="fp16",
        weight_granularity="per_channel",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
    )
    assert quant.accum_dtype == "int32"


def test_grouped_problem_requires_common_n_k_and_valid_offsets() -> None:
    first = GemmProblem(m=2, n=8, k=16)
    second = GemmProblem(m=3, n=8, k=16)
    grouped = GroupedGemmProblem((first, second), m_offsets=(0, 2, 5))
    assert grouped.group_count == 2
    assert GroupedGemmProblem.from_dict(grouped.to_dict()) == grouped
    with pytest.raises(ValueError, match="common N and K"):
        GroupedGemmProblem((first, GemmProblem(m=3, n=7, k=16)))


def test_packed_metadata_reports_padding_ratio() -> None:
    metadata = PackedWeightMetadata(
        logical_shape=(8, 33),
        storage_layout="xqt_int4_nk_v1",
        pack_version="v1",
        weight_dtype="int4",
        padded_k=64,
        group_size=32,
        packed_bits=4,
        nibble_order="low_high",
    )
    assert metadata.padding_ratio == pytest.approx(31.0 / 33.0)
    assert PackedWeightMetadata.from_dict(metadata.to_dict()) == metadata


def test_build_packed_weight_rejects_wrong_group_scale_shape() -> None:
    spec = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        weight_granularity="groupwise",
        group_size=4,
        weight_scale_source="weight_offline",
    )
    packed = pack_int4_signed(torch.zeros((3, 8), dtype=torch.int8))
    with pytest.raises(ValueError, match="scales shape"):
        build_packed_weight(
            packed,
            logical_shape=(3, 8),
            spec=spec,
            scales=torch.ones(3, 1),
        )


def test_canonical_reference_and_cutlass_views_preserve_logical_values() -> None:
    weight = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    canonical = canonical_weight_view(weight)
    reference = reference_weight_view(weight)
    cutlass = cutlass_weight_view(weight)
    torch.testing.assert_close(reference, canonical)
    torch.testing.assert_close(cutlass, canonical.t())
