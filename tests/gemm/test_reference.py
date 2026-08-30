from __future__ import annotations

import torch
import pytest

from xqt.kernels.ops.gemm import (
    dense_gemm_reference,
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    GroupedGemmProblem,
    QuantSpec,
    pack_int4_signed,
    pack_int4_unsigned,
    reference_gemm,
    reference_grouped_gemm,
)


def test_dense_gemm_reference_matches_legacy_dense_signature() -> None:
    torch.manual_seed(0)
    a = torch.randn(3, 5)
    b = torch.randn(7, 5)
    bias = torch.randn(7)

    actual = dense_gemm_reference(a, b, bias, activation="gelu", transpose_b=True)
    expected = torch.nn.functional.gelu(a @ b.t() + bias)

    torch.testing.assert_close(actual, expected)


def test_dense_reference_matches_torch_with_epilogue() -> None:
    torch.manual_seed(0)
    activation = torch.randn(5, 7)
    weight = torch.randn(11, 7)
    bias = torch.randn(11)
    residual = torch.randn(5, 11)
    spec = GemmSpec(
        problem=GemmProblem(m=5, n=11, k=7),
        quant=QuantSpec(weight_dtype="fp32", activation_dtype="fp32", output_dtype="fp32"),
        epilogue=EpilogueSpec(
            activation="silu",
            has_bias=True,
            has_residual=True,
            output_dtype="fp32",
        ),
    )
    actual = reference_gemm(activation, weight, spec=spec, bias=bias, residual=residual)
    expected = torch.nn.functional.silu(activation @ weight.t() + bias + residual)
    torch.testing.assert_close(actual, expected)


def test_w4a16_reference_supports_signed_packed_and_group_scales() -> None:
    values = torch.tensor(
        [[-8, -4, 0, 4, 7], [7, 3, -1, -5, -8]], dtype=torch.int8
    )
    packed = pack_int4_signed(values)
    scales = torch.tensor([[0.5, 1.0, 2.0], [1.0, 0.5, 0.25]])
    activation = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp32",
        weight_granularity="groupwise",
        group_size=2,
        weight_scale_source="weight_offline",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=1, n=2, k=5),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    actual = reference_gemm(activation, packed, spec=spec, weight_scales=scales)
    decoded = values.to(torch.float32) * torch.tensor(
        [[0.5, 0.5, 1.0, 1.0, 2.0], [1.0, 1.0, 0.5, 0.5, 0.25]]
    )
    torch.testing.assert_close(actual, activation @ decoded.t())


def test_w4a16_reference_supports_unsigned_zero_point() -> None:
    codes = torch.tensor([[0, 4, 8, 15]], dtype=torch.int8)
    packed = pack_int4_unsigned(codes)
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp32",
        weight_granularity="groupwise",
        group_size=2,
        symmetric=False,
        weight_zero_point=True,
        weight_scale_source="weight_offline",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=1, n=1, k=4),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    actual = reference_gemm(
        torch.ones(1, 4),
        packed,
        spec=spec,
        weight_scales=torch.ones(1, 2),
        weight_zero_points=torch.tensor([[1.0, 7.0]]),
    )
    assert actual.item() == pytest.approx((0 - 1) + (4 - 1) + (8 - 7) + (15 - 7))


def test_w8a8_dynamic_per_token_reference_is_quantize_then_dequantize() -> None:
    activation = torch.tensor([[1.0, -2.0, 0.25, 3.0], [0.5, 0.0, -1.0, 2.0]])
    weight = torch.tensor([[2, -1, 3, 4], [-2, 4, 1, -3]], dtype=torch.int8)
    quant = QuantSpec(
        weight_dtype="int8",
        activation_dtype="int8",
        output_dtype="fp32",
        weight_granularity="per_channel",
        activation_granularity="per_token",
        weight_scale_source="weight_offline",
        activation_scale_source="activation_dynamic",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=2, n=2, k=4),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    actual = reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=torch.tensor([0.5, 0.25]),
    )
    activation_scale = activation.abs().amax(dim=1, keepdim=True) / 127.0
    quantized = torch.clamp(torch.round(activation / activation_scale), -128, 127)
    expected = (quantized * activation_scale) @ (weight.float() * torch.tensor([0.5, 0.25]).view(-1, 1)).t()
    torch.testing.assert_close(actual, expected)


def test_grouped_reference_keeps_one_output_per_group() -> None:
    problems = GroupedGemmProblem(
        (GemmProblem(m=2, n=3, k=4), GemmProblem(m=1, n=3, k=4)),
        m_offsets=(0, 2, 3),
    )
    quant = QuantSpec(weight_dtype="fp32", activation_dtype="fp32", output_dtype="fp32")
    activations = [torch.randn(2, 4), torch.randn(1, 4)]
    weights = [torch.randn(3, 4), torch.randn(3, 4)]
    outputs = reference_grouped_gemm(problems, activations, weights, quant_specs=quant)
    assert [tuple(output.shape) for output in outputs] == [(2, 3), (1, 3)]
