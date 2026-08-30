from __future__ import annotations

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    GroupedGemmProblem,
    QuantSpec,
    SVDGemmContract,
    Sm90Fp8WgmmaContract,
    W4A8Contract,
    build_w4a8_gemm_spec,
    calibrate_fp8_scale,
    default_registry,
    dequantize_weight_reference,
    dispatch_gemm,
    pack_fp4_weight,
    pack_w4a8_weight,
    quantize_fp4_reference,
    quantize_w4a8_activation,
    reference_grouped_w4a8_gemm,
    reference_svd_dual_path,
    reference_w4a8_gemm,
    sm90_fp8_wgmma_executor,
    Sparse2_4Contract,
    W3A16Contract,
    pack_sparse2_4_weight,
    pack_w3a16_weight,
    unpack_int3,
)

def test_sparse_2_4_contract_roundtrip_quant_dequant_and_dispatch() -> None:
    torch.manual_seed(20260812)
    contract = Sparse2_4Contract()
    m, n, k = 4, 8, 16
    spec = contract.quant_spec()
    assert spec.weight_dtype == "int8"
    assert spec.activation_dtype == "fp16"
    assert spec.weight_granularity == "groupwise"
    assert spec.group_size == 8
    assert spec.scale_mode == "w:groupwise/a:per_tensor"

    weight = torch.randn(n, k)
    mask = torch.zeros(n, k, dtype=torch.bool)
    mask[:, 0::4] = True
    mask[:, 1::4] = True
    packed = pack_sparse2_4_weight(weight, mask, contract=contract)
    activation = torch.randn(m, k)
    gemm_spec = GemmSpec(
        GemmProblem(m=m, n=n, k=k),
        contract.quant_spec(),
        EpilogueSpec(output_dtype="fp16"),
    )
    result = dispatch_gemm(activation, packed, spec=gemm_spec)
    decoded = dequantize_weight_reference(packed, spec=contract.quant_spec())
    expected = activation.to(torch.float32) @ decoded.transpose(0, 1)

    assert result.report.selected_kernel == "sparse2_4_reference"
    assert result.report.native is False
    torch.testing.assert_close(result.output, expected.to(torch.float16), atol=1e-4, rtol=1e-4)

    bad_mask = mask.clone()
    bad_mask[0, 2] = True
    with pytest.raises(ValueError, match="exactly two"):
        pack_sparse2_4_weight(weight, bad_mask, contract=contract)


@pytest.mark.parametrize("format_name", ("fp4", "mxfp4", "nvfp4"))
def test_fp4_value_scale_contract_roundtrip_and_dispatch(format_name: str) -> None:
    torch.manual_seed(20260811)
    values = torch.randn(5, 37)
    global_scale = torch.tensor(1.0) if format_name == "nvfp4" else None
    encoded = quantize_fp4_reference(
        values,
        format_name=format_name,
        global_scale=global_scale,
    )
    decoded = encoded.dequantize()

    assert encoded.storage.dtype == torch.uint8
    assert encoded.logical_shape == (5, 37)
    assert encoded.padded_k >= 37
    assert decoded.shape == values.shape
    assert torch.isfinite(decoded).all()

    packed = encoded.to_packed_weight()
    metadata = packed.to_metadata_dict()
    assert metadata["global_scale_present"] is (format_name == "nvfp4")
    assert metadata["global_scale_shape"] == ([] if format_name == "nvfp4" else None)
    assert (format_name == "nvfp4") == (packed.global_scale is not None)
    quant = QuantSpec(
        weight_dtype=format_name,
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=encoded.group_size,
        weight_scale_source="weight_offline",
        storage_layout="xqt_fp4_nk_v1",
    )
    spec = GemmSpec(
        GemmProblem(m=3, n=5, k=37),
        quant,
        EpilogueSpec(output_dtype="fp16"),
    )
    result = dispatch_gemm(torch.randn(3, 37), packed, spec=spec)

    assert result.report.selected_kernel == f"{format_name if format_name != 'fp4' else 'fp4_e2m1'}_reference"
    assert result.report.native is False
    assert result.output.shape == (3, 5)


def test_nvfp4_requires_explicit_global_scale() -> None:
    with pytest.raises(ValueError, match="global_scale"):
        quantize_fp4_reference(torch.randn(2, 16), format_name="nvfp4")

    values = quantize_fp4_reference(
        torch.randn(2, 16),
        format_name="nvfp4",
        global_scale=torch.tensor(1.0),
    )
    with pytest.raises(ValueError, match="global_scale"):
        dequantize_weight_reference(
            values.storage,
            spec=QuantSpec(
                weight_dtype="nvfp4",
                activation_dtype="fp16",
                weight_granularity="groupwise",
                group_size=16,
                weight_scale_source="weight_offline",
            ),
            scales=values.scale,
            logical_shape=values.logical_shape,
        )


def test_w4a8_int8_and_fp8_scale_combinations_are_explicit() -> None:
    torch.manual_seed(20260812)
    weight = torch.randn(7, 33)
    activation = torch.randn(4, 33)
    int8_contract = W4A8Contract()
    packed = pack_w4a8_weight(weight, contract=int8_contract)
    int8_spec = build_w4a8_gemm_spec(
        GemmProblem(m=4, n=7, k=33, sm=89, device="cpu"),
        int8_contract,
    )
    int8_result = dispatch_gemm(activation, packed, spec=int8_spec)

    assert int8_spec.quant.scale_mode == "w:groupwise/a:per_token"
    assert int8_result.report.selected_kernel == "w4a8_reference"
    assert int8_result.report.native is False

    shorthand_scales = torch.full((7,), 0.25)
    shorthand = pack_w4a8_weight(
        torch.randint(-8, 8, (7, 33), dtype=torch.int8),
        contract=int8_contract,
        scales=shorthand_scales,
    )
    assert shorthand.scales is not None
    assert tuple(shorthand.scales.shape) == (7, 2)

    odd_group_contract = W4A8Contract(weight_group_size=3)
    odd_group_weight = pack_w4a8_weight(torch.randn(7, 5), contract=odd_group_contract)
    odd_group_spec = build_w4a8_gemm_spec(
        GemmProblem(m=4, n=7, k=5),
        odd_group_contract,
    )
    odd_group_result = dispatch_gemm(torch.randn(4, 5), odd_group_weight, spec=odd_group_spec)
    assert odd_group_result.output.shape == (4, 7)

    fp8_contract = W4A8Contract(
        activation_dtype="fp8_e4m3",
        activation_granularity="per_token",
        activation_scale_source="activation_static",
    )
    fp8_scale = calibrate_fp8_scale(
        activation,
        format_name="fp8_e4m3",
        granularity="per_token",
        role="activation",
    )
    fp8_packed = pack_w4a8_weight(weight, contract=fp8_contract)
    fp8_spec = build_w4a8_gemm_spec(
        GemmProblem(m=4, n=7, k=33, sm=89, device="cpu"),
        fp8_contract,
    )
    fp8_result = reference_w4a8_gemm(
        activation,
        fp8_packed,
        contract=fp8_contract,
        spec=fp8_spec,
        activation_scales=fp8_scale,
    )

    assert fp8_spec.quant.scale_mode == "w:groupwise/a:per_token"
    assert fp8_result.shape == (4, 7)


def test_w4a8_grouped_moe_reference_handles_empty_expert_and_scatter() -> None:
    torch.manual_seed(20260813)
    contract = W4A8Contract()
    rows = (2, 0, 3)
    offsets = (0, 2, 2, 5)
    grouped = GroupedGemmProblem(
        tuple(GemmProblem(m=m, n=4, k=16) for m in rows),
        m_offsets=offsets,
        output_rows=(4, 2, 3, 1, 0),
    )
    activation = torch.randn(5, 16)
    encoded = quantize_w4a8_activation(activation, contract)
    weights = tuple(
        pack_w4a8_weight(torch.randn(4, 16), contract=contract) for _ in rows
    )

    result = reference_grouped_w4a8_gemm(
        grouped,
        encoded.values,
        weights,
        contract=contract,
        activation_scales=encoded.scales,
    )

    assert result.output.shape == (5, 4)
    assert result.report.empty_experts == (1,)
    assert result.report.output_scatter is True
    assert result.report.native is False


def test_svd_dual_path_reports_main_low_rank_and_outlier_branches() -> None:
    torch.manual_seed(20260814)
    contract = W4A8Contract()
    main_weight = pack_w4a8_weight(torch.randn(6, 16), contract=contract)
    main_spec = build_w4a8_gemm_spec(
        GemmProblem(m=3, n=6, k=16),
        contract,
    )
    dual = SVDGemmContract(
        main_spec=main_spec,
        rank=3,
        outlier_channels=(1, 7),
        fusion_boundary="fuse_up",
    )
    activation = torch.randn(3, 16)
    low_rank_down = torch.randn(3, 16)
    low_rank_up = torch.randn(6, 3)
    outlier_weight = torch.randn(6, 2)

    result = reference_svd_dual_path(
        activation,
        main_weight,
        contract=dual,
        low_rank_down=low_rank_down,
        low_rank_up=low_rank_up,
        outlier_weight=outlier_weight,
    )
    decoded_main = dequantize_weight_reference(
        main_weight,
        spec=main_spec.quant,
        scales=main_weight.scales,
    )
    encoded_activation = quantize_w4a8_activation(activation, contract)
    reference_main = encoded_activation.values.to(torch.float32) * encoded_activation.scales
    expected_main = (reference_main @ decoded_main.transpose(0, 1)).to(torch.float16).to(torch.float32)
    expected_low_rank = activation @ low_rank_down.transpose(0, 1) @ low_rank_up.transpose(0, 1)
    indices = torch.tensor((1, 7), dtype=torch.long)
    expected_outlier = activation.index_select(1, indices) @ (
        outlier_weight - decoded_main.index_select(1, indices)
    ).transpose(0, 1)

    torch.testing.assert_close(result.main_output, expected_main, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.low_rank_output, expected_low_rank, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.outlier_output, expected_outlier, atol=1e-5, rtol=1e-5)
    assert result.report.fusion_boundary == "fuse_up"
    assert result.report.outlier_channels == (1, 7)


def test_w3a16_contract_roundtrip_and_dispatch() -> None:
    torch.manual_seed(20260812)
    contract = W3A16Contract()
    m, n, k = 4, 8, 16
    spec = contract.quant_spec()
    assert spec.weight_dtype == "int3"
    assert spec.activation_dtype == "fp16"
    assert spec.weight_granularity == "groupwise"
    assert spec.group_size == 16
    assert spec.scale_mode == "w:groupwise/a:per_tensor"

    codes = torch.randint(-4, 4, (n, k), dtype=torch.int8)
    packed = pack_w3a16_weight(codes, contract=contract)
    assert packed.metadata.packed_bits == 3
    roundtrip = unpack_int3(packed.qweight, logical_k=packed.metadata.padded_k)
    assert torch.equal(roundtrip[:, :k], codes)

    activation = torch.randn(m, k)
    gemm_spec = GemmSpec(
        GemmProblem(m=m, n=n, k=k),
        contract.quant_spec(),
        EpilogueSpec(output_dtype="fp16"),
    )
    result = dispatch_gemm(activation, packed, spec=gemm_spec)
    decoded = dequantize_weight_reference(packed, spec=contract.quant_spec())
    expected = activation.to(torch.float32) @ decoded.transpose(0, 1)

    assert result.report.selected_kernel == "w3a16_reference"
    assert result.report.native is False
    torch.testing.assert_close(result.output, expected.to(torch.float16), atol=1e-4, rtol=1e-4)
