"""Tests for XQT-004: Calibration state isolation and preservation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.compression.quant.calibration import (
    preserve_module_inference_state,
    run_calibration_batches,
)


class _MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout(0.5)
        self.conv = nn.Conv2d(4, 4, kernel_size=1)
        self.linear = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = self.dropout(out)
        out = self.conv(out)
        return self.linear(out.flatten(1))


def test_calibration_preserves_batchnorm_and_training_flags() -> None:
    """XQT-004: BatchNorm running stats and module training modes are restored."""
    model = _MixedModel()
    model.train()  # Explicitly in training mode

    # Set non-standard running mean & var
    with torch.no_grad():
        model.bn.running_mean.fill_(1.23)
        model.bn.running_var.fill_(4.56)

    initial_mean = model.bn.running_mean.clone()
    initial_var = model.bn.running_var.clone()

    data = [torch.randn(2, 4, 2, 2) for _ in range(5)]
    run_calibration_batches(model, data)

    # Invariants
    assert model.training is True
    assert model.bn.training is True
    assert model.dropout.training is True
    assert torch.equal(model.bn.running_mean, initial_mean)
    assert torch.equal(model.bn.running_var, initial_var)


def test_calibration_preserves_mixed_training_flags() -> None:
    """XQT-004: Models with mixed train/eval states retain exact per-module modes."""
    model = _MixedModel()
    model.train()
    model.conv.eval()  # Frozen/eval submodule

    assert model.training is True
    assert model.conv.training is False
    assert model.bn.training is True

    data = [torch.randn(2, 4, 2, 2) for _ in range(3)]
    run_calibration_batches(model, data)

    assert model.training is True
    assert model.conv.training is False
    assert model.bn.training is True


def test_calibration_exception_restores_state() -> None:
    """XQT-004: Forward exceptions during calibration safely restore flags and buffers."""
    model = _MixedModel()
    model.train()
    with torch.no_grad():
        model.bn.running_mean.fill_(0.5)

    initial_mean = model.bn.running_mean.clone()

    def failing_batches():
        yield torch.randn(2, 4, 2, 2)
        raise RuntimeError("calibration data error")

    with pytest.raises(RuntimeError, match="calibration data error"):
        run_calibration_batches(model, failing_batches())

    assert model.training is True
    assert torch.equal(model.bn.running_mean, initial_mean)


def test_calibration_rejects_empty_batches_and_tensors() -> None:
    """XQT-004: Empty batches or empty tensors raise ValueError."""
    model = _MixedModel().eval()

    # Empty iterable
    with pytest.raises(ValueError, match="at least one batch"):
        run_calibration_batches(model, [])

    # Batch containing empty tensor
    with pytest.raises(ValueError, match="empty tensor"):
        run_calibration_batches(model, [torch.empty(0, 4, 2, 2)])
