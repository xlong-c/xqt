from __future__ import annotations

import pytest
import torch

from xqt.gemm import dequantize_int8_activation, quantize_int8_activation


def test_static_per_tensor_quantization_does_not_require_dynamic_scale() -> None:
    activation = torch.tensor([[0.0, 1.0, 2.0, 20.0]])
    result = quantize_int8_activation(
        activation,
        granularity="per_tensor",
        source="activation_static",
        scale=torch.tensor(0.1),
    )
    assert result.scales.shape == (1, 1)
    assert result.source == "activation_static"
    assert result.saturation_ratio > 0.0
    torch.testing.assert_close(
        dequantize_int8_activation(result), result.values.float() * 0.1
    )


def test_dynamic_per_token_quantization_records_row_scales() -> None:
    activation = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    result = quantize_int8_activation(
        activation,
        granularity="per_token",
        source="activation_dynamic",
    )
    assert tuple(result.scales.shape) == (2, 1)
    assert result.saturation_ratio == pytest.approx(0.0)
    restored = dequantize_int8_activation(result)
    torch.testing.assert_close(
        restored,
        activation,
        atol=float(activation.abs().max().item() / 127.0),
        rtol=0,
    )


def test_static_quantization_requires_positive_scale_artifact() -> None:
    with pytest.raises(ValueError, match="explicit scale artifact"):
        quantize_int8_activation(
            torch.ones(2, 3),
            granularity="per_token",
            source="activation_static",
        )
    with pytest.raises(ValueError, match="positive"):
        quantize_int8_activation(
            torch.ones(2, 3),
            granularity="per_tensor",
            source="activation_static",
            scale=torch.tensor(0.0),
        )
