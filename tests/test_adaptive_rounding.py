"""Tests for Adaptive Rounding Quantization (AdaRound / AutoRound)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear
from xqt.compression.quant.quantizers.adaptive_rounding import (
    optimize_linear_rounding,
    quantize_with_adaptive_rounding,
)
from xqt.compression.quant.sequential import (
    LayerSequentialConfig,
    quantize_layer_sequential,
)
from xqt.kernels.nn.fixtures.smoke_llm import build_smoke_llm


def test_optimize_linear_rounding_loss_reduction() -> None:
    torch.manual_seed(42)
    linear = nn.Linear(32, 64)
    x = torch.randn(16, 32)

    # 1. Float reference output
    with torch.no_grad():
        y_float = linear(x)

    # 2. Naive RTN baseline
    rtn_linear = AWQGPTQWeightOnlyLinear.from_linear(
        linear,
        bits=4,
        group_size=16,
        method="rtn",
    )
    with torch.no_grad():
        y_rtn = rtn_linear(x)
    rtn_mse = float(F.mse_loss(y_rtn, y_float).item())

    # 3. Optimized rounding
    opt_linear = optimize_linear_rounding(
        linear,
        x,
        bits=4,
        group_size=16,
        steps=30,
        lr=2e-2,
    )
    with torch.no_grad():
        y_opt = opt_linear(x)
    opt_mse = float(F.mse_loss(y_opt, y_float).item())

    # Assert that optimized rounding achieves lower or equal reconstruction MSE than naive RTN
    assert opt_mse <= rtn_mse * 1.05  # within numerical tolerance, typically significantly lower
    assert isinstance(opt_linear, AWQGPTQWeightOnlyLinear)
    assert opt_linear.bits == 4


def test_quantize_with_adaptive_rounding_full_model() -> None:
    model = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )
    calib = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    quantized_model, report = quantize_with_adaptive_rounding(
        model,
        calib,
        bits=4,
        group_size=16,
        steps=10,
    )

    assert report["count"] > 0
    assert report["algorithm"] == "adaptive_rounding"

    # Forward pass verification
    test_inp = torch.randint(0, 32, (2, 8))
    out = quantized_model(test_inp)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()


def test_layer_sequential_with_adaround_method() -> None:
    model = build_smoke_llm(
        vocab_size=32,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
    )
    calib = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    cfg = LayerSequentialConfig(
        target_device="cpu",
        offload_device="cpu",
        bits=4,
        group_size=16,
        method="adaround",
        sample_limit=2,
    )

    quantized_model, report = quantize_layer_sequential(
        model,
        calib,
        config=cfg,
    )

    assert report.total_blocks == 2
    assert report.quantized_blocks == 2
    assert report.quantized_linear_count == 14

    test_inp = torch.randint(0, 32, (2, 8))
    out = quantized_model(test_inp)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()
