from __future__ import annotations

import importlib.util

import pytest
import torch

from xqt.operator_opt.kernels.fp4_quant_common import (
    dequantize_nvfp4_codes,
)
from xqt.operator_opt.kernels.tilelang.fp4_quant import (
    scaled_mxfp4_quant_reference,
    scaled_mxfp4_quant_tilelang,
    scaled_nvfp4_quant_reference,
    scaled_nvfp4_quant_tilelang,
)
from xqt.operator_opt.kernels.triton.fp4_quant import (
    scaled_mxfp4_quant_triton,
    scaled_nvfp4_quant_triton,
)
from xqt.operator_opt.kernels.tilelang._common import tilelang_runtime_usable


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for backend FP4 quantization tests",
)

requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton package is required for Triton FP4 quantization tests",
)

requires_tilelang = pytest.mark.skipif(
    not tilelang_runtime_usable(),
    reason="a runtime-compatible TileLang adapter is required for TileLang FP4 quantization tests",
)


def test_scaled_nvfp4_quant_reference_packs_expected_nibbles_cpu() -> None:
    inputs = torch.tensor(
        [
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        ],
        dtype=torch.float32,
    )
    packed, scale = scaled_nvfp4_quant_reference(inputs, torch.tensor(1.0))

    assert packed.shape == (1, 8)
    assert packed.dtype == torch.uint8
    assert packed[0].tolist() == [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
    assert scale.shape == (1, 1)
    assert torch.allclose(scale.to(torch.float32), torch.tensor([[1.0]], dtype=torch.float32))


def test_scaled_mxfp4_quant_reference_roundtrip_cpu() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(3, 20, dtype=torch.float32)
    packed, scale = scaled_mxfp4_quant_reference(inputs)

    assert packed.dtype == torch.uint8
    assert packed.shape == (3, 16)
    assert scale.shape == (3, 1)
    reconstructed = dequantize_nvfp4_codes(
        packed,
        scale,
        input_features=32,
        group_size=32,
    )[:, : inputs.shape[1]]
    error = (reconstructed - inputs).abs().mean().item()
    assert error < 0.75


@requires_cuda
@requires_triton
def test_scaled_nvfp4_quant_triton_matches_reference_cuda() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(8, 48, device="cuda", dtype=torch.float16)
    global_scale = torch.tensor(448.0, device="cuda", dtype=torch.float32)

    actual_packed, actual_scale = scaled_nvfp4_quant_triton(inputs, global_scale)
    expected_packed, expected_scale = scaled_nvfp4_quant_reference(inputs, global_scale)

    assert torch.equal(actual_packed.cpu(), expected_packed.cpu())
    assert torch.allclose(
        actual_scale.to(torch.float32).cpu(),
        expected_scale.to(torch.float32).cpu(),
        atol=1e-4,
        rtol=1e-4,
    )


@requires_cuda
@requires_triton
def test_scaled_mxfp4_quant_triton_matches_reference_cuda() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(8, 48, device="cuda", dtype=torch.float16)

    actual_packed, actual_scale = scaled_mxfp4_quant_triton(inputs)
    expected_packed, expected_scale = scaled_mxfp4_quant_reference(inputs)

    assert torch.equal(actual_packed.cpu(), expected_packed.cpu())
    assert torch.allclose(
        actual_scale.to(torch.float32).cpu(),
        expected_scale.to(torch.float32).cpu(),
        atol=1e-4,
        rtol=1e-4,
    )


@requires_cuda
@requires_tilelang
def test_scaled_nvfp4_quant_tilelang_matches_reference_cuda() -> None:
    torch.manual_seed(2)
    inputs = torch.randn(8, 48, device="cuda", dtype=torch.float16)
    global_scale = torch.tensor(448.0, device="cuda", dtype=torch.float32)

    actual_packed, actual_scale = scaled_nvfp4_quant_tilelang(
        inputs,
        global_scale,
        target_arch=None,
    )
    expected_packed, expected_scale = scaled_nvfp4_quant_reference(inputs, global_scale)

    assert torch.equal(actual_packed.cpu(), expected_packed.cpu())
    assert torch.allclose(
        actual_scale.to(torch.float32).cpu(),
        expected_scale.to(torch.float32).cpu(),
        atol=1e-4,
        rtol=1e-4,
    )


@requires_cuda
@requires_tilelang
def test_scaled_mxfp4_quant_tilelang_matches_reference_cuda() -> None:
    torch.manual_seed(3)
    inputs = torch.randn(8, 48, device="cuda", dtype=torch.float16)

    actual_packed, actual_scale = scaled_mxfp4_quant_tilelang(
        inputs,
        target_arch=None,
    )
    expected_packed, expected_scale = scaled_mxfp4_quant_reference(inputs)

    assert torch.equal(actual_packed.cpu(), expected_packed.cpu())
    assert torch.allclose(
        actual_scale.to(torch.float32).cpu(),
        expected_scale.to(torch.float32).cpu(),
        atol=1e-4,
        rtol=1e-4,
    )
