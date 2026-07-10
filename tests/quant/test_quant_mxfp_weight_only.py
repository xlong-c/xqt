from __future__ import annotations

import copy
from typing import Any

import torch

from xqt.core.schema import QuantConfig
from xqt.core.types import XQTContext
from xqt.quant import execute_quantization_plan
from xqt.quant.plan import build_quantization_plan
from xqt.quant.quantizers.mxfp_weight_only import MXFPWeightOnlyLinear


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


def _runtime_context(
    quant_config: QuantConfig,
    *,
    model: Any = None,
    calibration_inputs: Any = None,
) -> XQTContext:
    return XQTContext(
        model=model,
        reference_model=copy.deepcopy(model) if model is not None else None,
        calibration_inputs=calibration_inputs,
        device="cpu",
        artifact_dir="artifacts/xqt/tests/quant_mxfp_weight_only",
        project_name="quant_mxfp_weight_only",
        quant_config=quant_config,
    )


def _quant_config(config_dict: dict[str, Any]) -> QuantConfig:
    return QuantConfig(**config_dict["compression"]["quant"])


def _base_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "quant_mxfp_weight_only",
            "artifact_dir": "artifacts/xqt/tests/quant_mxfp_weight_only",
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
                "strategy": "mxfp_weight_only",
                "policy": {
                    "dtype": "mxfp",
                    "scheme": "weight_only",
                    "precision": 8,
                    "block_size": 4,
                    "include_module_types": ["Linear"],
                    "exclude_name_patterns": [],
                },
            }
        },
    }


def test_pytorch_mxfp_weight_only_executes_weight_only_linear_rewrite() -> None:
    torch.manual_seed(0)
    quant_config = _quant_config(_base_config())
    model = _TinyMLP().eval()
    sample = torch.randn(2, 8)
    baseline = model(sample)
    context = _runtime_context(quant_config, model=model)
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.backend == "pytorch"
    assert report.strategy == "mxfp_weight_only"
    assert report.metadata["executed"] is True
    assert report.metadata["execution_state"] == "mxfp_weight_only"
    assert report.metadata["mx_precision"] == 8
    assert report.metadata["block_size"] == 4
    assert "fc1" in report.quantized_modules
    assert "fc2" in report.quantized_modules

    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, MXFPWeightOnlyLinear)
    assert isinstance(quantized_model.fc2, MXFPWeightOnlyLinear)
    assert quantized_model.fc1.mx_precision == 8
    assert quantized_model.fc1.block_size == 4
    assert quantized_model.fc1.weight_scale.shape == (8, 2)

    quantized_output = quantized_model(sample)
    assert quantized_output.shape == baseline.shape
    max_diff = (baseline - quantized_output).abs().max().item()
    assert max_diff < 0.2


def test_pytorch_mxfp_weight_only_supports_mxfp4_storage() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["policy"]["precision"] = 4
    quant_config = _quant_config(config_dict)
    model = _TinyMLP().eval()
    context = _runtime_context(quant_config, model=model)
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, MXFPWeightOnlyLinear)
    assert quantized_model.fc1.mx_precision == 4
    assert quantized_model.fc1.packed_weight.dtype == torch.uint8
    assert execution.reports[0].metadata["mx_precision"] == 4


def test_mxfp_weight_only_report_includes_calibration_summary_when_inputs_provided() -> None:
    quant_config = _quant_config(_base_config())
    model = _TinyMLP().eval()
    calibration_inputs = [torch.randn(2, 8), torch.randn(2, 8)]
    context = _runtime_context(
        quant_config,
        model=model,
        calibration_inputs=calibration_inputs,
    )
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    report = execution.reports[0]
    assert report.calibration_samples == 2
    assert report.calibration_summary is not None
    assert report.algorithm_executable is False
    assert report.method_semantics == "awq_gptq_label_only_groupwise_weight_only_storage_quantization"
    assert report.calibration_summary["sample_count"] == 2
    assert report.calibration_summary["batch_count"] == 2
