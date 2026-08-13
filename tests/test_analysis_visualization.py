from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.analysis import (
    plot_model_tensor_selections_bar3d,
    plot_tensor_bar3d,
    plot_tensor_bar3d_panels,
    prepare_tensor_plot_data,
    TensorSelection,
)


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_prepare_tensor_plot_data_selects_activation_matrix_and_preserves_peak() -> None:
    tensor = torch.zeros(2, 8, 10)
    tensor[1, 7, 9] = 42.0

    result = prepare_tensor_plot_data(
        tensor,
        matrix_index=1,
        max_bars=16,
        reduction="max",
    )

    assert result.source_shape == (8, 10)
    assert result.plotted_shape[0] * result.plotted_shape[1] <= 16
    assert result.reduced is True
    assert float(result.values.max()) == pytest.approx(42.0)


def test_prepare_tensor_plot_data_can_concatenate_leading_dimensions() -> None:
    tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)

    result = prepare_tensor_plot_data(
        tensor,
        matrix_index=None,
        max_bars=10_000,
    )

    assert result.source_shape == (24, 5)
    assert result.plotted_shape == (24, 5)
    assert result.reduced is False


def test_plot_tensor_bar3d_writes_png(tmp_path: Path) -> None:
    output = plot_tensor_bar3d(
        torch.arange(30, dtype=torch.float32).reshape(5, 6),
        tmp_path / "single" / "tensor.png",
        title="Tensor",
        max_bars=30,
    )

    assert output.is_file()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_tensor_bar3d_panels_writes_named_panels(tmp_path: Path) -> None:
    output = plot_tensor_bar3d_panels(
        {
            "Gate Projection": torch.rand(4, 5),
            "Matrix Multiplication": torch.rand(4, 5) * 8,
        },
        tmp_path / "panels.png",
        max_bars=20,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_model_tensor_selections_supports_activation_and_weight(
    tmp_path: Path,
) -> None:
    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 6)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.proj(inputs)

    output = plot_model_tensor_selections_bar3d(
        _TinyModel().eval(),
        torch.randn(2, 4),
        [
            TensorSelection("proj", "activation", label="Activation"),
            TensorSelection("proj", "weight", label="Weight"),
        ],
        tmp_path / "selected.png",
        max_bars=12,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
