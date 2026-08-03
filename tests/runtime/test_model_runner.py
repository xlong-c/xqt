"""Y1: ModelRunner thin wrapper over quantized artifacts."""

from __future__ import annotations

import torch
from torch import nn

from xqt.quant.quantizers.int8_mma import quantize_with_int8_mma
from xqt.runtime.model_runner import ModelRunner


def test_model_runner_from_int8_quantized() -> None:
    class Shell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    result = quantize_with_int8_mma(
        Shell().eval(),
        policy={"include_module_types": ["Linear"]},
        engine="torch_int_mm",
        inplace=False,
    )
    runner = ModelRunner.from_quantized(result)
    x = torch.randn(3, 16)
    y = runner.forward(x)
    assert y.shape == (3, 8)
    assert torch.isfinite(y).all()
    report = runner.report()
    assert report.contract_ok is True
    assert report.storage_layout
    assert report.quantized_modules
    assert report.selected_kernel is not None or report.storage_layout


def test_model_runner_requires_contract() -> None:
    model = nn.Linear(4, 2)
    try:
        ModelRunner(model, metadata={}, require_contract=True)
        raised = False
    except ValueError:
        raised = True
    assert raised is True


def test_selection_fields_present_on_int8_report() -> None:
    class Shell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    result = quantize_with_int8_mma(
        Shell().eval(),
        policy={"include_module_types": ["Linear"]},
        engine="torch_int_mm",
        inplace=False,
    )
    assert result.quantized_modules
    assert "layout_kernel" in result.metadata
    layout = result.metadata["layout_kernel"]
    assert isinstance(layout, dict)
    assert layout.get("selected_kernel")
    assert result.resolve_runtime_quant_contract() is not None
