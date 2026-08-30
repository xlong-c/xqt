from __future__ import annotations

import pytest
import torch

from xqt.kernels.ops.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    QuantSpec,
    dequantize_weight_reference,
    dispatch_gemm,
    reference_gemm,
    repack_awq_int4,
    repack_gptq_int4,
)


def _pack_gptq_words(codes: torch.Tensor) -> torch.Tensor:
    n, k = (int(codes.shape[0]), int(codes.shape[1]))
    padded_k = ((k + 7) // 8) * 8
    values = torch.zeros((n, padded_k), dtype=torch.int64)
    values[:, :k] = codes.to(torch.int64)
    words = torch.zeros(((padded_k // 8), n), dtype=torch.int64)
    for lane in range(8):
        words |= values[:, lane::8].transpose(0, 1) << (lane * 4)
    return words.to(torch.int32)


def _pack_awq_words(codes: torch.Tensor) -> torch.Tensor:
    n, k = (int(codes.shape[0]), int(codes.shape[1]))
    if n % 8:
        raise ValueError("test AWQ fixture requires N divisible by 8")
    order = (0, 4, 1, 5, 2, 6, 3, 7)
    raw = torch.zeros((k, n // 8, 8), dtype=torch.int64)
    for word in range(n // 8):
        raw[:, word, list(order)] = codes[word * 8 : (word + 1) * 8, :].transpose(0, 1).to(torch.int64)
    packed = torch.zeros((k, n // 8), dtype=torch.int64)
    for lane in range(8):
        packed |= raw[..., lane] << (lane * 4)
    return packed.to(torch.int32)


def test_gptq_repack_decodes_signed_codes_and_group_padding() -> None:
    n, k, group_size = 8, 33, 32
    signed = torch.arange(n * k, dtype=torch.int16).reshape(n, k) % 16 - 8
    codes = (signed + 8).to(torch.uint8)
    scales = torch.tensor([[0.25, 0.5]]).expand(n, 2).contiguous()
    packed = repack_gptq_int4(
        _pack_gptq_words(codes),
        logical_shape=(n, k),
        scales=scales,
        group_size=group_size,
    )
    spec = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp32",
        weight_granularity="groupwise",
        group_size=group_size,
        weight_scale_source="weight_offline",
    )
    restored = dequantize_weight_reference(packed, spec=spec)
    expected = signed.to(torch.float32) * scales[:, :2].repeat_interleave(group_size, dim=1)[:, :k]
    torch.testing.assert_close(restored, expected)
    gemm_spec = GemmSpec(
        problem=GemmProblem(m=2, n=n, k=k),
        quant=spec,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    activation = torch.randn(2, k)
    torch.testing.assert_close(
        reference_gemm(activation, packed, spec=gemm_spec),
        activation @ expected.t(),
    )
    assert packed.metadata.padded_k == 64
    assert packed.metadata.packed_bits == 4
    assert packed.metadata.nibble_signed is True


def test_awq_repack_decodes_reverse_order_and_zero_points() -> None:
    n, k, group_size = 8, 33, 32
    codes = (torch.arange(n * k, dtype=torch.int16).reshape(n, k) % 16).to(torch.uint8)
    scales = torch.tensor([[0.5, 0.75]]).expand(n, 2).contiguous()
    zero_points = torch.tensor([[7.0, 8.0]]).expand(n, 2).contiguous()
    packed = repack_awq_int4(
        _pack_awq_words(codes),
        logical_shape=(n, k),
        scales=scales,
        zero_points=zero_points,
        group_size=group_size,
    )
    spec = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp32",
        weight_granularity="groupwise",
        group_size=group_size,
        symmetric=False,
        weight_zero_point=True,
        weight_scale_source="weight_offline",
    )
    restored = dequantize_weight_reference(packed, spec=spec)
    expected = (codes.to(torch.float32) - zero_points.repeat_interleave(group_size, dim=1)[:, :k]) * scales.repeat_interleave(group_size, dim=1)[:, :k]
    torch.testing.assert_close(restored, expected)
    gemm_spec = GemmSpec(
        problem=GemmProblem(m=2, n=n, k=k),
        quant=spec,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    activation = torch.randn(2, k)
    torch.testing.assert_close(
        reference_gemm(activation, packed, spec=gemm_spec),
        activation @ expected.t(),
    )
    assert packed.metadata.nibble_signed is False


def test_repack_rejects_non_sequential_g_idx() -> None:
    with pytest.raises(ValueError, match="non-sequential g_idx"):
        repack_gptq_int4(
            _pack_gptq_words(torch.zeros((8, 33), dtype=torch.uint8)),
            logical_shape=(8, 33),
            scales=torch.ones(8, 2),
            group_size=32,
            g_idx=torch.arange(33).flip(0),
        )


def test_gptq_signed_repack_rejects_zero_points() -> None:
    with pytest.raises(ValueError, match="does not support zero_points"):
        repack_gptq_int4(
            _pack_gptq_words(torch.zeros((8, 33), dtype=torch.uint8)),
            logical_shape=(8, 33),
            scales=torch.ones(8, 2),
            zero_points=torch.ones(8, 2),
            group_size=32,
        )


def test_dispatch_uses_packed_w4_reference_contract() -> None:
    n, k, group_size = 8, 33, 32
    signed = (torch.arange(n * k, dtype=torch.int16).reshape(n, k) % 16 - 8).to(torch.int8)
    packed = repack_gptq_int4(
        _pack_gptq_words((signed + 8).to(torch.uint8)),
        logical_shape=(n, k),
        scales=torch.ones(n, 2),
        group_size=group_size,
    )
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp32",
        weight_granularity="groupwise",
        group_size=group_size,
        weight_scale_source="weight_offline",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=3, n=n, k=k),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    activation = torch.randn(3, k, dtype=torch.float16)
    result = dispatch_gemm(activation, packed, spec=spec)
    assert result.report.selected_kernel == "w4a16_packed_reference"
    assert result.report.maturity == "reference_guarded"
    assert result.report.native is False
    expected = reference_gemm(activation, packed, spec=spec)
    torch.testing.assert_close(result.output, expected)


def test_dispatch_rejects_signedness_mismatch_before_reference() -> None:
    n, k = 8, 33
    packed = repack_gptq_int4(
        _pack_gptq_words(torch.zeros((n, k), dtype=torch.uint8)),
        logical_shape=(n, k),
        scales=torch.ones(n, 2),
        group_size=32,
    )
    quant = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp32",
        weight_granularity="groupwise",
        group_size=32,
        symmetric=False,
        weight_zero_point=True,
        weight_scale_source="weight_offline",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=1, n=n, k=k),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )
    with pytest.raises(ValueError, match="signedness mismatch"):
        dispatch_gemm(torch.ones(1, k), packed, spec=spec)
