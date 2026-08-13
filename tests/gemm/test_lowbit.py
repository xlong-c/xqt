from __future__ import annotations

import pytest
import torch

from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    Sparse2_4Contract,
    VectorCodebookContract,
    W2A16Contract,
    W3A16Contract,
    build_vector_codebook_weight,
    dequantize_weight_reference,
    dispatch_gemm,
    pack_int2_signed,
    pack_int3_signed,
    pack_sparse2_4_weight,
    pack_w2a16_weight,
    pack_w3a16_weight,
    quantize_vector_codebook_reference,
    unpack_int2,
    unpack_int3,
)


@pytest.mark.parametrize("k", (1, 4, 8, 13, 33))
def test_int2_pack_unpack_roundtrip(k: int) -> None:
    codes = torch.randint(-2, 2, (5, k), dtype=torch.int8)
    packed = pack_int2_signed(codes)
    decoded = unpack_int2(packed, logical_k=k)
    assert packed.dtype == torch.uint8
    assert torch.equal(decoded, codes)


@pytest.mark.parametrize("k", (1, 8, 13, 33, 61))
def test_int3_pack_unpack_roundtrip(k: int) -> None:
    codes = torch.randint(-4, 4, (5, k), dtype=torch.int8)
    packed = pack_int3_signed(codes)
    decoded = unpack_int3(packed, logical_k=k)
    assert packed.dtype == torch.uint8
    assert torch.equal(decoded, codes)


@pytest.mark.parametrize("contract", (W3A16Contract(), W2A16Contract()))
@pytest.mark.parametrize("k", (16, 33))
def test_lowbit_contract_roundtrip_and_dispatch(contract: object, k: int) -> None:
    torch.manual_seed(20260813)
    n = 6
    values = torch.randn(n, k)
    packer = pack_w3a16_weight if isinstance(contract, W3A16Contract) else pack_w2a16_weight
    packed = packer(values, contract=contract)
    spec = contract.quant_spec()
    expected_dtype = "int3" if isinstance(contract, W3A16Contract) else "int2"
    assert spec.weight_dtype == expected_dtype
    assert spec.weight_scale_source == "weight_offline"

    activation = torch.randn(4, k)
    gemm_spec = GemmSpec(
        GemmProblem(m=4, n=n, k=k),
        spec,
        EpilogueSpec(output_dtype="fp16"),
    )
    result = dispatch_gemm(activation, packed, spec=gemm_spec)
    decoded = dequantize_weight_reference(packed, spec=spec)
    expected = activation.to(torch.float32) @ decoded.transpose(0, 1)
    expected_name = f"{'w3a16' if isinstance(contract, W3A16Contract) else 'w2a16'}_reference"

    assert result.report.selected_kernel == expected_name
    assert result.report.native is False
    torch.testing.assert_close(result.output, expected.to(torch.float16), atol=1e-4, rtol=1e-4)


def test_lowbit_integer_codes_keep_unit_scales() -> None:
    contract = W3A16Contract()
    codes = torch.randint(-4, 4, (3, 16), dtype=torch.int8)
    packed = pack_w3a16_weight(codes, contract=contract)
    decoded = dequantize_weight_reference(packed, spec=contract.quant_spec())
    assert torch.equal(decoded.to(torch.int64), codes.to(torch.int64))


def test_lowbit_rejects_out_of_range_codes() -> None:
    with pytest.raises(ValueError, match="\\[-4, 3\\]"):
        pack_w3a16_weight(torch.full((3, 16), 4, dtype=torch.int8), contract=W3A16Contract())
    with pytest.raises(ValueError, match="\\[-2, 1\\]"):
        pack_w2a16_weight(torch.full((3, 16), 2, dtype=torch.int8), contract=W2A16Contract())


@pytest.mark.parametrize("weight_dtype", ("fp16", "bf16", "int8"))
def test_sparse2_4_dense_and_int8_dispatch(weight_dtype: str) -> None:
    torch.manual_seed(20260813)
    n, k = 6, 16
    activation = torch.randn(3, k)
    mask = torch.zeros(n, k, dtype=torch.bool)
    mask[:, 0::4] = True
    mask[:, 1::4] = True
    contract = Sparse2_4Contract(weight_dtype=weight_dtype)
    raw = torch.randn(n, k).to(torch.float16 if weight_dtype == "fp16" else torch.bfloat16)
    packed = pack_sparse2_4_weight(raw, mask, contract=contract)
    spec = contract.quant_spec()
    gemm_spec = GemmSpec(GemmProblem(m=3, n=n, k=k), spec, EpilogueSpec(output_dtype="fp16"))
    result = dispatch_gemm(activation, packed, spec=gemm_spec)
    decoded = dequantize_weight_reference(packed, spec=spec)
    expected = activation.to(torch.float32) @ decoded.transpose(0, 1)

    assert result.report.selected_kernel == "sparse2_4_reference"
    torch.testing.assert_close(result.output, expected.to(torch.float16), atol=1e-4, rtol=1e-4)


def test_sparse2_4_mask_pattern_gates() -> None:
    n, k = 4, 16
    mask = torch.zeros(n, k, dtype=torch.bool)
    mask[:, 0::4] = True
    mask[:, 1::4] = True
    mask[0, 2] = True
    with pytest.raises(ValueError, match="exactly two"):
        pack_sparse2_4_weight(torch.randn(n, k), mask, contract=Sparse2_4Contract())
    with pytest.raises(ValueError, match="divisible by four"):
        pack_sparse2_4_weight(
            torch.randn(n, 15),
            torch.zeros(n, 15, dtype=torch.bool),
            contract=Sparse2_4Contract(),
        )


def test_vector_codebook_roundtrip_and_dispatch() -> None:
    torch.manual_seed(20260813)
    contract = VectorCodebookContract(num_vectors=8, vector_size=4, vectors_per_group=2)
    n, k = 6, 16
    codebook = torch.randn(8, 4)
    indices = torch.randint(0, 8, (n, k // 4), dtype=torch.int16)
    scales = torch.full((n, k // contract.group_size), 0.5)
    packed = build_vector_codebook_weight(
        codebook, indices, scales, logical_shape=(n, k), contract=contract
    )
    spec = contract.quant_spec()
    gemm_spec = GemmSpec(GemmProblem(m=3, n=n, k=k), spec, EpilogueSpec(output_dtype="fp16"))
    activation = torch.randn(3, k)
    result = dispatch_gemm(activation, packed, spec=gemm_spec)
    decoded = dequantize_weight_reference(packed, spec=spec)
    expected = activation.to(torch.float32) @ decoded.transpose(0, 1)

    assert spec.weight_dtype == "codebook"
    assert result.report.selected_kernel == "vector_codebook_reference"
    torch.testing.assert_close(result.output, expected.to(torch.float16), atol=1e-4, rtol=1e-4)


def test_vector_codebook_quantize_reconstructs_through_dispatch() -> None:
    torch.manual_seed(20260813)
    contract = VectorCodebookContract(num_vectors=8, vector_size=4, vectors_per_group=2)
    values = torch.randn(6, 16)
    packed = quantize_vector_codebook_reference(values, contract=contract)
    spec = contract.quant_spec()
    activation = torch.randn(3, 16)
    gemm_spec = GemmSpec(GemmProblem(m=3, n=6, k=16), spec, EpilogueSpec(output_dtype="fp16"))
    result = dispatch_gemm(activation, packed, spec=gemm_spec)
    decoded = dequantize_weight_reference(packed, spec=spec)
    expected = activation.to(torch.float32) @ decoded.transpose(0, 1)

    assert result.report.selected_kernel == "vector_codebook_reference"
    torch.testing.assert_close(result.output, expected.to(torch.float16), atol=1e-4, rtol=1e-4)


def test_vector_codebook_rejects_out_of_range_indices() -> None:
    contract = VectorCodebookContract(num_vectors=4, vector_size=4, vectors_per_group=2)
    with pytest.raises(ValueError, match="indices out of range"):
        build_vector_codebook_weight(
            torch.randn(4, 4),
            torch.full((2, 4), 4, dtype=torch.int16),
            torch.ones((2, 2)),
            logical_shape=(2, 16),
            contract=contract,
        )
