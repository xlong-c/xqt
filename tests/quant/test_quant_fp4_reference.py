from __future__ import annotations

import pytest
import torch

from xqt.core.config import load_xqt_config
import xqt.pipeline.passes as passes_module
from xqt.pipeline.passes import QuantPass
from xqt.pipeline.runner import create_context
from xqt.quant import execute_quantization_plan
from xqt.quant.fp4_backend import ReferenceFP4Linear
from xqt.quant.sensitivity import analyze_layer_sensitivity
from xqt.quant.plan import build_quantization_plan
from xqt.quant.torchao_backend import TorchAOQuantizationResult


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
    assert report.metadata["selection_policy"]["selectors"]["include_module_types"] == ["Linear"]
    assert (
        report.metadata["module_selection_reasons"]["quantized"]["fc1"]
        == "matched_selection_policy"
    )
    assert report.metadata["module_selection_reasons"]["fallback"] == {}
    assert report.calibration_summary is None
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
    assert (
        execution.reports[0].metadata["module_selection_reasons"]["skipped"]["fc2"]
        == "skip_quantize"
    )


def test_pytorch_fp4_weight_only_reports_high_precision_reason() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["keep_high_precision"] = ["fc2"]
    config = load_xqt_config(config_dict)
    model = _TinyMLP().eval()
    context = create_context(config, model=model)
    plan = build_quantization_plan(config.compression.quant)

    execution = execute_quantization_plan(context, plan)

    report = execution.reports[0]
    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, ReferenceFP4Linear)
    assert isinstance(quantized_model.fc2, torch.nn.Linear)
    assert report.high_precision_modules == ["fc2"]
    assert report.skipped_modules == ["fc2"]
    assert (
        report.metadata["module_selection_reasons"]["high_precision"]["fc2"]
        == "keep_high_precision"
    )
    assert (
        report.metadata["module_selection_reasons"]["skipped"]["fc2"]
        == "keep_high_precision"
    )


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


def test_quant_pass_records_layer_analysis_summary() -> None:
    config_dict = _base_config()
    config_dict["analysis"] = {
        "top_k": 4,
        "metrics": ["max_abs", "mean_abs", "cosine_similarity"],
    }
    config = load_xqt_config(config_dict)
    model = _TinyMLP().eval()
    context = create_context(
        config,
        model=model,
        example_inputs=torch.randn(2, 8),
    )

    output = QuantPass().run(context)

    layer_analysis = output.metrics["quant"]["layer_analysis"]
    assert layer_analysis["available"] is True
    assert layer_analysis["runtime"] == "quantized_pytorch"
    assert layer_analysis["layer_error_count"] >= 1
    assert layer_analysis["layer_sensitivity_count"] >= 1
    assert layer_analysis["layer_statistics_count"] >= 1
    assert layer_analysis["layer_errors"]
    assert layer_analysis["layer_sensitivity"]
    assert layer_analysis["layer_statistics"]
    assert any(row["variable"] == "output" for row in layer_analysis["layer_statistics"])
    assert any(row["variable"] == "weight" for row in layer_analysis["layer_statistics"])
    assert layer_analysis["avoid_list"]
    sensitivity = analyze_layer_sensitivity(
        context.reference_model,
        output.model,
        context.example_inputs,
        module_names=["fc1", "fc2"],
    )
    assert sensitivity
    assert {record.name for record in sensitivity} == {"fc1", "fc2"}


def test_quant_pass_degrades_gracefully_when_layer_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config()
    config_dict["analysis"] = {"top_k": 4}
    config = load_xqt_config(config_dict)
    model = _TinyMLP().eval()
    context = create_context(
        config,
        model=model,
        example_inputs=torch.randn(2, 8),
    )

    monkeypatch.setattr(
        passes_module,
        "build_layer_analysis_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    output = QuantPass().run(context)

    layer_analysis = output.metrics["quant"]["layer_analysis"]
    assert layer_analysis["available"] is False
    assert layer_analysis["reason"] == "analysis_failed"
    assert layer_analysis["error_type"] == "RuntimeError"
    assert output.metrics["quant"]["quantized_module_count"] >= 1


def test_reference_fp4_report_includes_calibration_summary_when_inputs_provided() -> None:
    config = load_xqt_config(_base_config())
    model = _TinyMLP().eval()
    calibration_inputs = [torch.randn(2, 8), torch.randn(2, 8)]
    context = create_context(config, model=model, calibration_inputs=calibration_inputs)
    plan = build_quantization_plan(config.compression.quant)

    execution = execute_quantization_plan(context, plan)

    report = execution.reports[0]
    assert report.calibration_samples == 2
    assert report.calibration_summary is not None
    assert report.calibration_summary["sample_count"] == 2
    assert report.calibration_summary["batch_count"] == 2
    assert report.calibration_summary["input_signature"]
    assert report.calibration_summary["calibrator_type"] == "XQTCalibrationInputSummary"
    assert report.calibration_summary["observer_type"] == "pytorch.calibration_inputs"


def test_torchao_report_includes_calibration_summary_when_inputs_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["backend"] = "torchao"
    config_dict["compression"]["quant"]["method"] = "dynamic_int8"
    config_dict["compression"]["quant"]["strategy"] = "dynamic_int8"
    config_dict["compression"]["quant"]["policy"] = {
        "dtype": "int8",
        "scheme": "dynamic",
        "include_module_types": ["Linear"],
        "exclude_name_patterns": [],
    }
    config = load_xqt_config(config_dict)
    model = _TinyMLP().eval()
    calibration_inputs = [torch.randn(2, 8), torch.randn(2, 8)]
    context = create_context(config, model=model, calibration_inputs=calibration_inputs)
    plan = build_quantization_plan(config.compression.quant)

    monkeypatch.setattr(
        passes_module,
        "build_layer_analysis_payload",
        lambda *args, **kwargs: {"runtime": "quantized_pytorch", "layer_errors": [], "layer_sensitivity": [], "avoid_list": []},
    )

    monkeypatch.setattr(
        "xqt.quant.executor.quantize_with_torchao",
        lambda model, **kwargs: TorchAOQuantizationResult(
            model=model,
            strategy="dynamic_int8",
            quantized_modules=["fc1", "fc2"],
            metadata={"policy": kwargs.get("policy", {})},
        ),
    )

    execution = execute_quantization_plan(context, plan)

    report = execution.reports[0]
    assert report.backend == "torchao"
    assert report.calibration_samples == 2
    assert report.calibration_summary is not None
    assert report.calibration_summary["sample_count"] == 2
    assert report.calibration_summary["observer_type"] == "torchao.calibration_inputs"
