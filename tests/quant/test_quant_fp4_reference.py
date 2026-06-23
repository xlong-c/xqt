from __future__ import annotations

import torch

from xqt.core.config import load_xqt_config
from xqt.pipeline.runner import create_context
from xqt.quant import execute_quantization_plan
from xqt.quant.fp4_backend import ReferenceFP4Linear
from xqt.quant.plan import build_quantization_plan


class _TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.norm(hidden)
        return self.fc2(hidden)


def _base_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "quant_fp4_reference",
            "artifact_dir": "artifacts/xqt/tests/quant_fp4_reference",
        },
        "model": {
            "target": "torch.nn:Linear",
            "params": {"in_features": 8, "out_features": 8},
            "device": "cpu",
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "pytorch",
                "method": "awq",
                "strategy": "fp4_weight_only",
                "policy": {
                    "dtype": "fp4",
                    "scheme": "weight_only",
                    "include_module_types": ["Linear"],
                    "exclude_name_patterns": [],
                },
            }
        },
    }


def test_pytorch_fp4_weight_only_executes_reference_linear_rewrite() -> None:
    torch.manual_seed(0)
    config = load_xqt_config(_base_config())
    model = _TinyMLP().eval()
    sample = torch.randn(2, 8)
    baseline = model(sample)
    context = create_context(config, model=model)
    plan = build_quantization_plan(config.compression.quant)

    execution = execute_quantization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.backend == "pytorch"
    assert report.strategy == "fp4_weight_only"
    assert report.metadata["executed"] is True
    assert report.metadata["execution_state"] == "reference_fp4_weight_only"
    assert report.metadata["group_size"] == 128
    assert "fc1" in report.quantized_modules
    assert "fc2" in report.quantized_modules

    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, ReferenceFP4Linear)
    assert isinstance(quantized_model.fc2, ReferenceFP4Linear)
    assert quantized_model.fc1.group_size == 8
    assert quantized_model.fc1.weight_scale.shape == (8, 1, 1)

    quantized_output = quantized_model(sample)
    assert quantized_output.shape == baseline.shape
    max_diff = (baseline - quantized_output).abs().max().item()
    assert max_diff < 0.5


def test_pytorch_fp4_weight_only_respects_skip_quantize() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["skip_quantize"] = ["fc2"]
    config = load_xqt_config(config_dict)
    model = _TinyMLP().eval()
    context = create_context(config, model=model)
    plan = build_quantization_plan(config.compression.quant)

    execution = execute_quantization_plan(context, plan)

    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, ReferenceFP4Linear)
    assert isinstance(quantized_model.fc2, torch.nn.Linear)
    assert execution.reports[0].skipped_modules == ["fc2"]


def test_pytorch_fp4_weight_only_uses_group_size_policy() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["policy"]["group_size"] = 4
    config = load_xqt_config(config_dict)
    model = _TinyMLP().eval()
    context = create_context(config, model=model)
    plan = build_quantization_plan(config.compression.quant)

    execution = execute_quantization_plan(context, plan)

    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, ReferenceFP4Linear)
    assert quantized_model.fc1.group_size == 4
    assert quantized_model.fc1.weight_scale.shape == (8, 2, 1)
    assert execution.reports[0].metadata["group_size"] == 4
