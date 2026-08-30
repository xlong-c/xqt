from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.export import apply_pre_export_lowering
from xqt.compression.quant import FP4WeightOnlyLinear


class _FP4LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = FP4WeightOnlyLinear.from_linear(nn.Linear(8, 4), group_size=4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_fp4_weight_only_lowering_preserves_forward_without_mutating_source() -> None:
    torch.manual_seed(20260710)
    model = _FP4LinearModel().eval()
    inputs = torch.randn(3, 8)
    expected = model(inputs)

    result = apply_pre_export_lowering(
        model,
        {"enabled": True, "mode": "fp4_weight_only_to_dense_linear"},
    )

    assert result.applied is True
    assert result.mode == "fp4_weight_only_to_dense_linear"
    assert result.inplace is False
    assert isinstance(model.fc, FP4WeightOnlyLinear)
    assert isinstance(result.model.fc, nn.Linear)
    assert result.metadata["deployment_note"] == (
        "FP4 packed storage was materialized into dense dequantized weights for export; "
        "the resulting artifact is not a packed-FP4 runtime."
    )
    assert result.lowered_modules == [
        {
            "path": "fc",
            "source_module_type": "FP4WeightOnlyLinear",
            "target_module_type": "Linear",
            "source_weight_storage": "packed_fp4",
            "target_weight_storage": "dense_dequantized",
            "source_weight_dtype": "torch.uint8",
            "target_weight_dtype": "torch.float32",
        }
    ]
    torch.testing.assert_close(result.model(inputs), expected)


def test_pre_export_lowering_rejects_unknown_mode() -> None:
    with pytest.raises(XQTBackendError, match="Unsupported pre_export_lowering mode"):
        apply_pre_export_lowering(nn.Linear(2, 2), {"enabled": True, "mode": "unknown"})


def test_fp4_root_module_lowering_preserves_forward() -> None:
    torch.manual_seed(20260711)
    source = FP4WeightOnlyLinear.from_linear(nn.Linear(8, 4), group_size=4).eval()
    inputs = torch.randn(2, 8)

    result = apply_pre_export_lowering(
        source,
        {"enabled": True, "mode": "fp4_weight_only_to_dense_linear"},
    )

    assert isinstance(result.model, nn.Linear)
    assert result.lowered_modules[0]["path"] == ""
    torch.testing.assert_close(result.model(inputs), source(inputs))
