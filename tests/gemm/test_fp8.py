from __future__ import annotations

import pytest
import torch

from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    QuantSpec,
    build_packed_weight,
    default_registry,
    dispatch_gemm,
    select_kernel,
    calibrate_fp8_scale,
    decode_fp8_storage,
    dequantize_fp8,
    fp8_format_spec,
    quantize_fp8,
    reference_gemm,
)


@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
def test_fp8_format_round_trip_uses_one_byte_row_major_storage(format_name: str) -> None:
    values = torch.tensor([[-1.0, 0.0, 1.0, 3.5]], dtype=torch.float32)
    quantized = quantize_fp8(
        values,
        format_name=format_name,
        granularity="per_tensor",
        role="activation",
        source="activation_static",
        scale=torch.ones(1, 1),
    )
    assert quantized.storage.dtype == torch.uint8
    assert tuple(quantized.storage.shape) == (1, 4)
    assert quantized.storage.numel() == values.numel()
    assert quantized.as_float8().dtype == fp8_format_spec(format_name).torch_dtype
    torch.testing.assert_close(
        decode_fp8_storage(quantized.storage, format_name=format_name),
        dequantize_fp8(
            quantized.storage,
            format_name=format_name,
            scale=torch.ones(1, 1),
            granularity="per_tensor",
            role="activation",
        ),
    )


def test_fp8_e4m3_saturates_nonfinite_values_deterministically() -> None:
    values = torch.tensor([[-float("inf"), float("nan"), -500.0, 500.0, 0.0]])
    quantized = quantize_fp8(
        values,
        format_name="fp8_e4m3",
        granularity="per_tensor",
        role="activation",
        source="activation_static",
        scale=torch.ones(1, 1),
    )
    assert quantized.saturation_count == 3
    assert quantized.nan_count == 1
    assert quantized.inf_count == 1
    torch.testing.assert_close(
        quantized.dequantize(),
        torch.tensor([[-448.0, 0.0, -448.0, 448.0, 0.0]]),
    )


def test_fp8_dynamic_per_token_scale_is_canonical_and_reproducible() -> None:
    values = torch.tensor([[1.0, -2.0], [0.25, 0.5]])
    quantized = quantize_fp8(
        values,
        format_name="fp8_e5m2",
        granularity="per_token",
        role="activation",
        source="activation_dynamic",
    )
    expected_scale = values.abs().amax(dim=1, keepdim=True) / fp8_format_spec(
        "fp8_e5m2"
    ).max_finite
    assert tuple(quantized.scale.shape) == (2, 1)
    torch.testing.assert_close(quantized.scale, expected_scale)
    torch.testing.assert_close(
        quantized.dequantize(), values, atol=0.03, rtol=0.03
    )


def test_fp8_scale_shape_is_not_implicitly_broadcast() -> None:
    with pytest.raises(ValueError, match="per_channel scale"):
        quantize_fp8(
            torch.ones(3, 4),
            format_name="fp8_e4m3",
            granularity="per_channel",
            role="weight",
            source="weight_offline",
            scale=torch.ones(2, 1),
        )
    with pytest.raises(ValueError, match="per_tensor scale"):
        quantize_fp8(
            torch.ones(3, 4),
            format_name="fp8_e4m3",
            granularity="per_tensor",
            role="activation",
            source="activation_static",
            scale=torch.ones(3, 1),
        )


@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
def test_fp8_reference_gemm_matches_explicit_quantize_dequantize(
    format_name: str,
) -> None:
    torch.manual_seed(20260730)
    m, n, k = 3, 5, 7
    weight = torch.randn(n, k) * 0.5
    activation = torch.randn(m, k) * 0.5
    weight_scale = calibrate_fp8_scale(
        weight,
        format_name=format_name,
        granularity="per_channel",
        role="weight",
    )
    quantized_weight = quantize_fp8(
        weight,
        format_name=format_name,
        granularity="per_channel",
        role="weight",
        source="weight_offline",
        scale=weight_scale,
    )
    quant = QuantSpec(
        weight_dtype=format_name,
        activation_dtype=format_name,
        output_dtype="fp32",
        weight_granularity="per_channel",
        activation_granularity="per_token",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_dynamic",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    packed = build_packed_weight(
        quantized_weight.storage,
        logical_shape=(n, k),
        spec=quant,
        scales=quantized_weight.scale,
        padded_k=k,
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    actual = reference_gemm(activation, packed, spec=spec)
    expected_activation = quantize_fp8(
        activation,
        format_name=format_name,
        granularity="per_token",
        role="activation",
        source="activation_dynamic",
    ).dequantize()
    expected = expected_activation @ quantized_weight.dequantize().transpose(0, 1)
    torch.testing.assert_close(actual, expected)


def test_fp8_quant_spec_rejects_zero_points() -> None:
    with pytest.raises(ValueError, match="FP8 weights"):
        QuantSpec(
            weight_dtype="fp8_e4m3",
            activation_dtype="fp16",
            weight_zero_point=True,
            weight_scale_source="weight_offline",
        )
    with pytest.raises(ValueError, match="FP8 activations"):
        QuantSpec(
            weight_dtype="fp16",
            activation_dtype="fp8_e4m3",
            activation_zero_point=True,
            activation_scale_source="activation_dynamic",
        )


def test_fp8_static_per_token_activation_scale_is_explicit() -> None:
    values = torch.tensor([[1.0, -2.0], [4.0, -8.0]], dtype=torch.float32)
    scale = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    quantized = quantize_fp8(
        values,
        format_name="fp8_e4m3",
        granularity="per_token",
        role="activation",
        source="activation_static",
        scale=scale,
    )
    assert tuple(quantized.scale.shape) == (2, 1)
    torch.testing.assert_close(quantized.scale, scale)
    torch.testing.assert_close(quantized.dequantize(), values, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize(("m", "k"), [(1, 32), (32, 33), (256, 65)])
def test_fp8_reference_covers_aligned_and_non_aligned_k_shapes(
    format_name: str, m: int, k: int
) -> None:
    torch.manual_seed(20260731 + m + k)
    n = 17
    activation = torch.randn(m, k) * 0.25
    weight = torch.randn(n, k) * 0.25
    activation_scale = calibrate_fp8_scale(
        activation,
        format_name=format_name,
        granularity="per_tensor",
        role="activation",
    )
    weight_scale = calibrate_fp8_scale(
        weight,
        format_name=format_name,
        granularity="per_channel",
        role="weight",
    )
    encoded_activation = quantize_fp8(
        activation,
        format_name=format_name,
        granularity="per_tensor",
        role="activation",
        source="activation_static",
        scale=activation_scale,
    )
    encoded_weight = quantize_fp8(
        weight,
        format_name=format_name,
        granularity="per_channel",
        role="weight",
        source="weight_offline",
        scale=weight_scale,
    )
    quant = QuantSpec(
        weight_dtype=format_name,
        activation_dtype=format_name,
        output_dtype="fp32",
        weight_granularity="per_channel",
        activation_granularity="per_tensor",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    packed = build_packed_weight(
        encoded_weight.storage,
        logical_shape=(n, k),
        spec=quant,
        scales=encoded_weight.scale,
        padded_k=k,
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    actual = reference_gemm(
        encoded_activation.storage,
        packed,
        spec=spec,
        activation_scales=encoded_activation.scale,
    )
    expected = encoded_activation.dequantize() @ encoded_weight.dequantize().T
    torch.testing.assert_close(actual, expected)


def test_fp8_packed_weight_metadata_round_trip() -> None:
    weight = torch.randn(5, 7) * 0.25
    scale = calibrate_fp8_scale(
        weight,
        format_name="fp8_e4m3",
        granularity="per_channel",
        role="weight",
    )
    encoded = quantize_fp8(
        weight,
        format_name="fp8_e4m3",
        granularity="per_channel",
        role="weight",
        source="weight_offline",
        scale=scale,
    )
    quant = QuantSpec(
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        output_dtype="fp32",
        weight_granularity="per_channel",
        activation_granularity="per_tensor",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    packed = build_packed_weight(
        encoded.storage,
        logical_shape=(5, 7),
        spec=quant,
        scales=encoded.scale,
        padded_k=7,
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    metadata = packed.to_metadata_dict()
    assert metadata["storage_layout"] == "xqt_fp8_rowmajor_v1"
    assert metadata["pack_version"] == "xqt-fp8-v1"
    assert metadata["weight_dtype"] == "fp8_e4m3"
    assert metadata["packed_bits"] is None
    torch.testing.assert_close(
        reference_gemm(
            torch.ones(2, 7),
            packed,
            spec=GemmSpec(
                problem=GemmProblem(m=2, n=5, k=7),
                quant=quant,
                epilogue=EpilogueSpec(output_dtype="fp32"),
            ),
            activation_scales=torch.ones(1, 1),
        ),
        torch.ones(2, 7) @ encoded.dequantize().T,
    )


def test_fp8_registry_keeps_sm89_and_sm90_native_entries_separate() -> None:
    quant = QuantSpec(
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        output_dtype="fp32",
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    sm89_spec = GemmSpec(
        problem=GemmProblem(m=4, n=8, k=32, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    sm89_names = [candidate.name for candidate in select_kernel(sm89_spec)]
    assert "sm89_fp8_e4m3_cutlass" in sm89_names
    assert "sm90_fp8_e4m3_wgmma" not in sm89_names
    sm90_spec = GemmSpec(
        problem=GemmProblem(m=4, n=8, k=32, sm=90, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    sm90_names = [candidate.name for candidate in select_kernel(sm90_spec)]
    assert "sm90_fp8_e4m3_wgmma" in sm90_names
    assert "sm89_fp8_e4m3_cutlass" not in sm90_names


def test_fp8_metadata_only_candidate_falls_back_to_reference() -> None:
    activation = torch.randn(4, 32) * 0.1
    weight = torch.randn(8, 32) * 0.1
    activation_scale = calibrate_fp8_scale(
        activation,
        format_name="fp8_e4m3",
        granularity="per_tensor",
        role="activation",
    )
    weight_scale = calibrate_fp8_scale(
        weight,
        format_name="fp8_e4m3",
        granularity="per_tensor",
        role="weight",
    )
    encoded_activation = quantize_fp8(
        activation,
        format_name="fp8_e4m3",
        granularity="per_tensor",
        role="activation",
        source="activation_static",
        scale=activation_scale,
    )
    encoded_weight = quantize_fp8(
        weight,
        format_name="fp8_e4m3",
        granularity="per_tensor",
        role="weight",
        source="weight_offline",
        scale=weight_scale,
    )
    quant = QuantSpec(
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        output_dtype="fp32",
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=4, n=8, k=32, sm=89, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    result = dispatch_gemm(
        encoded_activation.storage,
        encoded_weight.storage,
        spec=spec,
        weight_scales=weight_scale,
        activation_scales=activation_scale,
        registry=default_registry(),
    )
    assert result.report.selected_kernel == "quantized_dequant_reference"
    assert result.report.native is False
    assert "maturity=metadata_only" in (result.report.fallback_reason or "")
    torch.testing.assert_close(
        result.output,
        reference_gemm(
            encoded_activation.storage,
            encoded_weight.storage,
            spec=spec,
            weight_scales=weight_scale,
            activation_scales=activation_scale,
        ),
    )


@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("block_k", [32, 64, 128])
def test_fp8_blockwise_scale_layout_covers_partial_trailing_block(
    format_name: str, block_k: int
) -> None:
    torch.manual_seed(20260803 + block_k)
    cols = block_k + 17
    values = torch.randn(3, cols) * 0.5
    scale = calibrate_fp8_scale(
        values,
        format_name=format_name,
        granularity="blockwise",
        role="activation",
        block_k=block_k,
    )
    assert tuple(scale.shape) == (3, 2)
    trailing = values[:, block_k:].abs().amax(dim=1, keepdim=True) / fp8_format_spec(
        format_name
    ).max_finite
    torch.testing.assert_close(scale[:, 1:2], trailing)
    quantized = quantize_fp8(
        values,
        format_name=format_name,
        granularity="blockwise",
        role="activation",
        source="activation_static",
        scale=scale,
        block_k=block_k,
    )
    assert quantized.block_k == block_k
    assert tuple(quantized.scale.shape) == (3, 2)
    assert tuple(quantized.storage.shape) == (3, cols)
    # E5M2 keeps only two mantissa bits, so its round-trip noise is wider.
    atol, rtol = (0.03, 0.06) if format_name == "fp8_e4m3" else (0.06, 0.15)
    torch.testing.assert_close(quantized.dequantize(), values, atol=atol, rtol=rtol)


@pytest.mark.parametrize("block_k", [32, 64, 128])
def test_fp8_blockwise_dynamic_scale_is_per_row_per_block(block_k: int) -> None:
    torch.manual_seed(20260804 + block_k)
    values = torch.randn(4, 2 * block_k) * 0.25
    quantized = quantize_fp8(
        values,
        format_name="fp8_e4m3",
        granularity="blockwise",
        role="activation",
        source="activation_dynamic",
        block_k=block_k,
    )
    expected = (
        values.view(4, 2, block_k).abs().amax(dim=2) / fp8_format_spec("fp8_e4m3").max_finite
    )
    assert tuple(quantized.scale.shape) == (4, 2)
    torch.testing.assert_close(quantized.scale, expected)


def test_fp8_blockwise_rejects_invalid_block_k_and_scale_shape() -> None:
    with pytest.raises(ValueError, match="block_k must be one of"):
        quantize_fp8(
            torch.ones(2, 64),
            format_name="fp8_e4m3",
            granularity="blockwise",
            role="activation",
            source="activation_dynamic",
            block_k=16,
        )
    with pytest.raises(ValueError, match="block_k is only valid for blockwise"):
        quantize_fp8(
            torch.ones(2, 64),
            format_name="fp8_e4m3",
            granularity="per_tensor",
            role="activation",
            source="activation_static",
            scale=torch.ones(1, 1),
            block_k=32,
        )
    with pytest.raises(ValueError, match="blockwise scale must be"):
        quantize_fp8(
            torch.ones(2, 64),
            format_name="fp8_e4m3",
            granularity="blockwise",
            role="weight",
            source="weight_offline",
            scale=torch.ones(2, 3),
            block_k=32,
        )
    with pytest.raises(ValueError, match="requires block_k"):
        quantize_fp8(
            torch.ones(2, 64),
            format_name="fp8_e4m3",
            granularity="blockwise",
            role="activation",
            source="activation_dynamic",
        )


@pytest.mark.parametrize("format_name", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("block_k", [32, 64, 128])
def test_fp8_blockwise_reference_gemm_matches_manual_dequantize(
    format_name: str, block_k: int
) -> None:
    torch.manual_seed(20260805 + block_k)
    m, n, k = 3, 5, 100
    activation = torch.randn(m, k) * 0.25
    weight = torch.randn(n, k) * 0.25
    activation_scale = calibrate_fp8_scale(
        activation,
        format_name=format_name,
        granularity="blockwise",
        role="activation",
        block_k=block_k,
    )
    weight_scale = calibrate_fp8_scale(
        weight,
        format_name=format_name,
        granularity="blockwise",
        role="weight",
        block_k=block_k,
    )
    encoded_activation = quantize_fp8(
        activation,
        format_name=format_name,
        granularity="blockwise",
        role="activation",
        source="activation_static",
        scale=activation_scale,
        block_k=block_k,
    )
    encoded_weight = quantize_fp8(
        weight,
        format_name=format_name,
        granularity="blockwise",
        role="weight",
        source="weight_offline",
        scale=weight_scale,
        block_k=block_k,
    )
    quant = QuantSpec(
        weight_dtype=format_name,
        activation_dtype=format_name,
        output_dtype="fp32",
        weight_granularity="blockwise",
        activation_granularity="blockwise",
        group_size=block_k,
        weight_scale_source="weight_offline",
        activation_scale_source="activation_static",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    packed = build_packed_weight(
        encoded_weight.storage,
        logical_shape=(n, k),
        spec=quant,
        scales=encoded_weight.scale,
        padded_k=k,
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-fp8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    actual = reference_gemm(
        encoded_activation.storage,
        packed,
        spec=spec,
        activation_scales=encoded_activation.scale,
    )
    expected = encoded_activation.dequantize() @ encoded_weight.dequantize().T
    torch.testing.assert_close(actual, expected)
