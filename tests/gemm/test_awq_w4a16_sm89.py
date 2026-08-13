from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    PackedWeight,
    PackedWeightMetadata,
    QuantSpec,
    reference_w4a16_gemm,
)
from xqt.gemm.backends.awq_w4a16_decode_sm89 import (
    prepare_sm89_awq_w4a16_decode_parameters,
    prepack_sm89_awq_w4a16_decode,
    sm89_awq_w4a16_decode_executor,
)
from xqt.operator_opt.kernels.cuda.awq_w4a16_sm89 import (
    bind_awq_w4a16_decode,
    native_awq_w4a16_available,
    native_awq_w4a16_version,
)


def _weight(*, n: int = 64, k: int = 128) -> PackedWeight:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(37)
    codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.uint8)
    qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales = (
        torch.rand((n, k // 64), generator=generator, dtype=torch.float32) * 0.1
        + 0.01
    )
    zero_points = torch.randint(
        0,
        16,
        tuple(scales.shape),
        generator=generator,
        dtype=torch.int16,
    ).to(torch.float32)
    return PackedWeight(
        qweight=qweight,
        scales=scales,
        zero_points=zero_points,
        metadata=PackedWeightMetadata(
            logical_shape=(n, k),
            storage_layout="xqt_int4_nk_v1",
            pack_version="xqt-w4a16-awq-v1",
            weight_dtype="int4",
            padded_k=k,
            group_size=64,
            packed_bits=4,
            nibble_order="low_high",
            nibble_signed=False,
        ),
    )


def _to_cuda(weight: PackedWeight) -> PackedWeight:
    return replace(
        weight,
        qweight=weight.qweight.cuda(),
        scales=weight.scales.cuda(),
        zero_points=weight.zero_points.cuda(),
    )


def _spec(
    *,
    rows: int,
    n: int = 64,
    k: int = 128,
    dtype: torch.dtype = torch.float16,
    bias: bool = False,
) -> GemmSpec:
    dtype_name = "fp16" if dtype == torch.float16 else "bf16"
    return GemmSpec(
        problem=GemmProblem(
            m=rows,
            n=n,
            k=k,
            phase="decode",
            sm=89,
            device="cuda:0",
        ),
        quant=QuantSpec(
            weight_dtype="int4",
            activation_dtype=dtype_name,
            output_dtype=dtype_name,
            weight_granularity="groupwise",
            group_size=64,
            symmetric=False,
            weight_zero_point=True,
            weight_scale_source="weight_offline",
            storage_layout="xqt_int4_nk_v1",
            pack_version="xqt-w4a16-awq-v1",
        ),
        epilogue=EpilogueSpec(has_bias=bias, output_dtype=dtype_name),
    )


def test_prepack_rejects_signed_and_wrong_group_contracts() -> None:
    weight = _weight()
    signed = replace(
        weight,
        metadata=replace(weight.metadata, nibble_signed=True),
        zero_points=None,
    )
    with pytest.raises(XQTBackendError, match="asymmetric canonical INT4"):
        prepack_sm89_awq_w4a16_decode(signed)

    wrong_group = replace(
        weight,
        metadata=replace(weight.metadata, group_size=128),
    )
    with pytest.raises(XQTBackendError, match="group_size=64"):
        prepack_sm89_awq_w4a16_decode(wrong_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepack_requires_sm89_cuda() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    with pytest.raises(XQTBackendError, match="CUDA-resident"):
        prepack_sm89_awq_w4a16_decode(_weight())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_sm89_awq_decode_matches_reference(
    dtype: torch.dtype,
    rows: int,
) -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    canonical = _to_cuda(_weight())
    prepacked = prepack_sm89_awq_w4a16_decode(canonical)
    spec = _spec(rows=rows, dtype=dtype, bias=True)
    activation = torch.randn(rows, 128, device="cuda", dtype=dtype)
    bias = torch.randn(64, device="cuda", dtype=dtype)

    actual = sm89_awq_w4a16_decode_executor(
        activation,
        prepacked,
        spec=spec,
        bias=bias,
    )
    expected = reference_w4a16_gemm(
        activation,
        canonical,
        spec=spec,
        bias=bias,
    )

    tolerance = 0.04 if dtype == torch.float16 else 0.3
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bound_runner_matches_executor_and_reports_v2() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    canonical = _to_cuda(_weight())
    prepacked = prepack_sm89_awq_w4a16_decode(canonical)
    spec = _spec(rows=2, dtype=torch.float16, bias=True)
    prepared = prepare_sm89_awq_w4a16_decode_parameters(
        prepacked,
        spec=spec,
        dtype=torch.float16,
    )
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    bound = bind_awq_w4a16_decode(
        *prepared,
        rows=2,
        input_features=128,
        output_features=64,
        dtype=torch.float16,
        device=torch.device("cuda"),
        bias=bias,
    )
    activation = torch.randn(2, 128, device="cuda", dtype=torch.float16)
    expected = sm89_awq_w4a16_decode_executor(
        activation,
        prepacked,
        spec=spec,
        bias=bias,
        prepared_parameters=prepared,
    )

    assert native_awq_w4a16_available(build=True)
    assert native_awq_w4a16_version() == "xqt-awq-w4a16-sm89-v2"
    torch.testing.assert_close(bound(activation), expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bound_runner_rejects_static_shape_mismatch() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    qweight = torch.empty((16, 64), device="cuda", dtype=torch.int32)
    scales = torch.empty((2, 64), device="cuda", dtype=torch.float16)
    zeros = torch.empty_like(scales)
    with pytest.raises(XQTBackendError, match="qweight"):
        bind_awq_w4a16_decode(
            qweight[:, :-1],
            scales,
            zeros,
            rows=1,
            input_features=128,
            output_features=64,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
