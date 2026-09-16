"""MLP-only W8A8 INT8 prefill tests: rowwise quantization and the GEMM view."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear
from xqt.kernels.ops._impl.triton.gemm import quantize_int8_rowwise_triton
from xqt.model.minicpm5 import _MiniCPM5W4A16HybridLinear

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="INT8 prefill requires CUDA"
)


def _hybrid_view(
    *, input_features: int = 128, output_features: int = 96
) -> _MiniCPM5W4A16HybridLinear:
    torch.manual_seed(17)
    linear = nn.Linear(input_features, output_features, bias=True).to(torch.bfloat16)
    storage = AWQGPTQWeightOnlyLinear.from_linear(
        linear, bits=4, group_size=64, method="rtn"
    )
    return _MiniCPM5W4A16HybridLinear(storage, decode=None).eval()


@requires_cuda
@pytest.mark.parametrize("rows", [4, 300])
def test_forward_prequant_matches_bf16_reference(rows: int) -> None:
    view = _hybrid_view().cuda()
    assert view.supports_prequant() is True

    inputs = torch.randn(rows, 128, dtype=torch.bfloat16, device="cuda") * 2.0
    quantized, activation_scale = quantize_int8_rowwise_triton(inputs)

    with torch.no_grad():
        actual = view.forward_prequant(quantized, activation_scale)
        weight = view.storage.dense_weight(dtype=torch.bfloat16, device="cuda")
        bias = view.storage.dense_bias(dtype=torch.bfloat16, device="cuda")
        reference = F.linear(inputs, weight, bias)

    assert actual.shape == (rows, view.output_features)
    assert actual.dtype == torch.bfloat16
    relative_error = float(
        (actual.float() - reference.float()).abs().mean()
        / reference.float().abs().mean()
    )
    assert relative_error < 0.05


@requires_cuda
def test_forward_prequant_is_lazy_and_cached() -> None:
    view = _hybrid_view().cuda()

    assert view._int8_weight is None
    assert view._int8_scale is None

    inputs = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    quantized, activation_scale = quantize_int8_rowwise_triton(inputs)
    with torch.no_grad():
        view.forward_prequant(quantized, activation_scale)

    assert view._int8_weight is not None
    assert view._int8_scale is not None
    # K-major [K, N]: stored transposed once so the GEMM never transposes per call.
    assert view._int8_weight.shape == (view.input_features, view.output_features)
    assert view._int8_weight.dtype == torch.int8
    assert view._int8_scale.shape == (view.output_features,)
    cached = view._int8_weight
    with torch.no_grad():
        view.forward_prequant(quantized, activation_scale)
    assert view._int8_weight is cached


@requires_cuda
def test_quantize_int8_rowwise_round_trip_within_one_step() -> None:
    torch.manual_seed(5)
    inputs = torch.randn(37, 256, dtype=torch.bfloat16, device="cuda") * 3.0

    quantized, scales = quantize_int8_rowwise_triton(inputs)
    reconstructed = quantized.float() * scales[:, None]

    assert quantized.shape == inputs.shape
    assert quantized.dtype == torch.int8
    assert scales.shape == (37,)
    assert scales.dtype == torch.float32
    # Round-to-nearest stays within half a quantization step per element, and
    # the per-row scale is at least that step.
    error = (reconstructed - inputs.float()).abs()
    assert torch.all(error <= scales[:, None] + 1e-6)
