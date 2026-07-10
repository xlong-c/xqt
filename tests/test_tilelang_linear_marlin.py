from __future__ import annotations

from collections.abc import Callable
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from xqt.operator_opt.backends.tilelang import get_tilelang_kernel_spec
from xqt.operator_opt.kernels.tilelang.linear_marlin import (
    linear_marlin_reference,
    linear_marlin_tilelang,
    quantize_int4_weight,
    quantize_int8_weight,
)
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang Marlin Linear CUDA tests",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang Marlin Linear tests",
)


def test_tilelang_registry_exposes_linear_marlin() -> None:
    spec = get_tilelang_kernel_spec("linear_marlin")

    assert spec.pattern == "linear_marlin"
    assert spec.metadata["supported_precisions"] == ["fp16", "bf16", "int8", "int4"]
    assert spec.metadata["unpack_stage"] == "tilelang_fused_weight_tile_decode"


def test_linear_marlin_reference_matches_dense_fp16() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.float16)
    weight = torch.randn(12, 16, dtype=torch.float16)
    bias = torch.randn(12, dtype=torch.float16)

    output = linear_marlin_reference(
        x,
        weight,
        bias=bias,
        precision="fp16",
        activation="relu",
    )
    reference = F.relu(F.linear(x, weight, bias))

    assert torch.allclose(output.float(), reference.float(), atol=0.0, rtol=0.0)


def test_linear_marlin_int8_quantized_reference_matches_manual_dequant() -> None:
    torch.manual_seed(1)
    x = torch.randn(8, 16, dtype=torch.float16)
    weight = torch.randn(12, 16, dtype=torch.float16)
    bias = torch.randn(12, dtype=torch.float16)
    qweight, scale = quantize_int8_weight(weight, group_size=8)

    output = linear_marlin_reference(
        x,
        qweight,
        scale,
        bias,
        precision="int8",
        group_size=8,
    )
    dequantized = (
        qweight.float().reshape(12, 2, 8) * scale.float()
    ).reshape(12, 16).to(dtype=x.dtype)
    reference = F.linear(x, dequantized, bias)

    assert qweight.dtype == torch.int8
    assert scale.shape == (12, 2, 1)
    assert torch.allclose(output.float(), reference.float(), atol=0.0, rtol=0.0)


def test_linear_marlin_int4_pack_reference_matches_manual_dequant_shape() -> None:
    torch.manual_seed(2)
    x = torch.randn(8, 16, dtype=torch.float16)
    weight = torch.randn(12, 16, dtype=torch.float16)
    qweight, scale = quantize_int4_weight(weight, group_size=8)

    output = linear_marlin_reference(
        x,
        qweight,
        scale,
        precision="int4",
        group_size=8,
    )

    assert qweight.dtype == torch.uint8
    assert qweight.shape == (12, 8)
    assert scale.shape == (12, 2, 1)
    assert output.shape == (8, 12)
    assert torch.isfinite(output).all()


@requires_cuda
@requires_tilelang
def test_linear_marlin_tilelang_fp16_matches_reference() -> None:
    torch.manual_seed(3)
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = linear_marlin_tilelang(
        x,
        weight,
        bias=bias,
        precision="fp16",
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = linear_marlin_reference(x, weight, bias=bias, precision="fp16")

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
@pytest.mark.parametrize(
    ("precision", "quantize"),
    [
        ("int8", quantize_int8_weight),
        ("int4", quantize_int4_weight),
    ],
)
def test_linear_marlin_tilelang_quantized_matches_reference(
    precision: str,
    quantize: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    torch.manual_seed(4)
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    qweight, scale = quantize(weight, group_size=64)

    output = linear_marlin_tilelang(
        x,
        qweight,
        scale,
        bias,
        precision=precision,
        group_size=64,
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = linear_marlin_reference(
        x,
        qweight,
        scale,
        bias,
        precision=precision,
        group_size=64,
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=2e-2, rtol=2e-2)
