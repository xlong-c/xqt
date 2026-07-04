from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.operator_opt.kernels.tilelang.conv import (
    conv2d_reference,
    conv2d_tilelang,
)
from xqt.operator_opt.kernels.tilelang.linear import (
    half_linear_reference,
    half_linear_tilelang,
)
from xqt.operator_opt.kernels.tilelang.norm import (
    layer_norm_reference,
    layer_norm_tilelang,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for TileLang half-op CUDA test",
)

requires_tilelang = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang package is required for TileLang half-op CUDA test",
)


@requires_cuda
@requires_tilelang
def test_tilelang_half_linear_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = half_linear_tilelang(
        x,
        weight,
        bias,
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = half_linear_reference(x, weight, bias)

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_half_conv_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 64, 8, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 64, 1, 1, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = conv2d_tilelang(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = conv2d_reference(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(0, 0),
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_half_conv_cuda_kernel_matches_reference_for_spatial_kernel() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 32, 8, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 32, 3, 3, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = conv2d_tilelang(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(1, 1),
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    )
    reference = conv2d_reference(
        x,
        weight,
        bias,
        stride=(1, 1),
        padding=(1, 1),
    )

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_tilelang
def test_tilelang_half_norm_cuda_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)

    output = layer_norm_tilelang(x, weight, bias, eps=1e-5)
    reference = layer_norm_reference(x, weight, bias, eps=1e-5)

    assert output.shape == reference.shape
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2)
