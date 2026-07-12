from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from xqt.core.schema import AnalysisConfig, OutputDiffConfig, PruneConfig, QuantConfig
from xqt.core.types import XQTContext
import xqt.pipeline.passes as passes_module
from xqt.pipeline.passes import run_prune_stage, run_quant_stage
from xqt.quant import build_fake_qdq_surrogate, execute_quantization_plan
from xqt.quant.quantizers.fp4_weight_only import FP4WeightOnlyLinear
from xqt.quant.sensitivity import analyze_layer_sensitivity
from xqt.quant.plan import build_quantization_plan
from xqt.quant.backends.torchao import TorchAOQuantizationResult
from xqt.workflows.stage_specs import PruneStageSpec, QuantStageSpec


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
    example_inputs: Any = None,
    calibration_inputs: Any = None,
    analysis_config: AnalysisConfig | None = None,
    prune_config: PruneConfig | None = None,
) -> XQTContext:
    return XQTContext(
        model=model,
        reference_model=copy.deepcopy(model) if model is not None else None,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
        device="cpu",
        artifact_dir="artifacts/xqt/tests/quant_fp4_weight_only",
        project_name="quant_fp4_weight_only",
        task_type="classification",
        quant_config=quant_config,
        prune_config=prune_config or PruneConfig(),
        analysis_config=analysis_config or AnalysisConfig(),
        output_diff_config=OutputDiffConfig(),
    )


def _quant_config(config_dict: dict[str, Any]) -> QuantConfig:
    return QuantConfig(**config_dict["compression"]["quant"])


def _quant_stage_spec(quant_config: QuantConfig) -> QuantStageSpec:
    return QuantStageSpec(
        backend=quant_config.backend,
        method=quant_config.method,
        strategy=quant_config.strategy,
        policy=dict(quant_config.policy),
        keep_high_precision=list(quant_config.keep_high_precision),
        skip_quantize=list(quant_config.skip_quantize),
        force_quantize=list(quant_config.force_quantize),
        analysis_only_modules=list(quant_config.analysis_only_modules),
        component_policies=list(quant_config.component_policies),
    )


def _base_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "quant_fp4_weight_only",
            "artifact_dir": "artifacts/xqt/tests/quant_fp4_weight_only",
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


def test_pytorch_fp4_weight_only_executes_weight_only_linear_rewrite() -> None:
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
    assert report.strategy == "fp4_weight_only"
    assert report.metadata["executed"] is True
    assert report.metadata["execution_state"] == "fp4_weight_only"
    assert report.metadata["group_size"] == 128
    assert report.algorithm_executable is False
    assert report.method_semantics == "awq_label_only_groupwise_fp4_weight_only_storage_quantization"
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
    assert isinstance(quantized_model.fc1, FP4WeightOnlyLinear)
    assert isinstance(quantized_model.fc2, FP4WeightOnlyLinear)
    assert quantized_model.fc1.group_size == 8
    assert quantized_model.fc1.weight_scale.shape == (8, 1, 1)

    quantized_output = quantized_model(sample)
    assert quantized_output.shape == baseline.shape
    max_diff = (baseline - quantized_output).abs().max().item()
    assert max_diff < 0.5


def test_pytorch_fp4_weight_only_respects_skip_quantize() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["skip_quantize"] = ["fc2"]
    quant_config = _quant_config(config_dict)
    model = _TinyMLP().eval()
    context = _runtime_context(quant_config, model=model)
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, FP4WeightOnlyLinear)
    assert isinstance(quantized_model.fc2, torch.nn.Linear)
    assert execution.reports[0].skipped_modules == ["fc2"]
    assert (
        execution.reports[0].metadata["module_selection_reasons"]["skipped"]["fc2"]
        == "skip_quantize"
    )


def test_pytorch_fp4_weight_only_reports_high_precision_reason() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["keep_high_precision"] = ["fc2"]
    quant_config = _quant_config(config_dict)
    model = _TinyMLP().eval()
    context = _runtime_context(quant_config, model=model)
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    report = execution.reports[0]
    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, FP4WeightOnlyLinear)
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
    quant_config = _quant_config(config_dict)
    model = _TinyMLP().eval()
    context = _runtime_context(quant_config, model=model)
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    quantized_model = execution.model
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, FP4WeightOnlyLinear)
    assert quantized_model.fc1.group_size == 4
    assert quantized_model.fc1.weight_scale.shape == (8, 2, 1)
    assert execution.reports[0].metadata["group_size"] == 4


def test_run_quant_stage_records_layer_analysis_summary() -> None:
    config_dict = _base_config()
    config_dict["analysis"] = {
        "top_k": 4,
        "metrics": ["max_abs", "mean_abs", "cosine_similarity"],
    }
    quant_config = _quant_config(config_dict)
    model = _TinyMLP().eval()
    context = _runtime_context(
        quant_config,
        model=model,
        example_inputs=torch.randn(2, 8),
        analysis_config=AnalysisConfig(**config_dict["analysis"]),
    )

    output = run_quant_stage(context, _quant_stage_spec(quant_config))

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


def test_run_quant_stage_accepts_typed_stage_spec() -> None:
    quant_config = _quant_config(_base_config())
    model = _TinyMLP().eval()
    context = _runtime_context(
        quant_config,
        model=model,
        example_inputs=torch.randn(2, 8),
    )

    output = run_quant_stage(
        context,
        QuantStageSpec(
            backend="pytorch",
            method="awq",
            strategy="fp4_weight_only",
            policy={
                "dtype": "fp4",
                "scheme": "weight_only",
                "include_module_names": ["fc1", "fc2"],
                "group_size": 8,
            },
        ),
    )

    assert output is context
    assert output.metrics["quant"]["quantized_module_count"] >= 1
    assert isinstance(output.model, _TinyMLP)
    assert isinstance(output.model.fc1, FP4WeightOnlyLinear)
    assert isinstance(output.model.fc2, FP4WeightOnlyLinear)


def test_run_prune_stage_accepts_typed_stage_spec() -> None:
    quant_config = _quant_config(_base_config())
    model = _TinyMLP().eval()
    context = _runtime_context(
        quant_config,
        model=model,
        example_inputs=torch.randn(2, 8),
    )

    output = run_prune_stage(
        context,
        PruneStageSpec(
            method="global_l1_unstructured",
            target_sparsity=0.25,
        ),
    )

    assert output is context
    assert output.metrics["prune"]["method"] == "global_l1_unstructured"
    assert output.metrics["prune"]["target_sparsity"] == pytest.approx(0.25)
    assert output.metrics["prune"]["applied"] is True


def test_fake_qdq_surrogate_accepts_explicit_quant_stage_spec() -> None:
    quant_config = _quant_config(_base_config())
    model = _TinyMLP().eval()
    context = _runtime_context(
        quant_config,
        model=model,
        calibration_inputs=[torch.randn(2, 8)],
    )

    surrogate = build_fake_qdq_surrogate(
        context,
        QuantStageSpec(
            backend="onnxruntime_qdq",
            method="static_qdq_int8",
            strategy="static_qdq_int8",
            policy={
                "dtype": "int8",
                "scheme": "static",
                "include_module_names": ["fc1", "fc2"],
                "op_types_to_quantize": ["Gemm"],
            },
        ),
    )

    assert surrogate is not None
    assert isinstance(surrogate.model, _TinyMLP)
    assert set(surrogate.quantized_modules) >= {"fc1", "fc2"}
    assert surrogate.sample_count == 1


def test_fake_qdq_surrogate_prefers_context_runtime_quant_config() -> None:
    quant_config = _quant_config(_base_config())
    model = _TinyMLP().eval()
    context = _runtime_context(
        quant_config,
        model=model,
        calibration_inputs=[torch.randn(2, 8)],
    )
    context.quant_config = QuantConfig(
        enabled=True,
        backend="onnxruntime_qdq",
        method="static_qdq_int8",
        strategy="static_qdq_int8",
        policy={
            "dtype": "int8",
            "scheme": "static",
            "include_module_names": ["fc1", "fc2"],
            "op_types_to_quantize": ["Gemm"],
        },
    )

    surrogate = build_fake_qdq_surrogate(context)

    assert surrogate is not None
    assert isinstance(surrogate.model, _TinyMLP)
    assert set(surrogate.quantized_modules) >= {"fc1", "fc2"}
    assert surrogate.sample_count == 1


def test_run_quant_stage_degrades_gracefully_when_layer_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict = _base_config()
    config_dict["analysis"] = {"top_k": 4}
    quant_config = _quant_config(config_dict)
    model = _TinyMLP().eval()
    context = _runtime_context(
        quant_config,
        model=model,
        example_inputs=torch.randn(2, 8),
        analysis_config=AnalysisConfig(**config_dict["analysis"]),
    )

    monkeypatch.setattr(
        passes_module,
        "build_layer_analysis_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    output = run_quant_stage(context, _quant_stage_spec(quant_config))

    layer_analysis = output.metrics["quant"]["layer_analysis"]
    assert layer_analysis["available"] is False
    assert layer_analysis["reason"] == "analysis_failed"
    assert layer_analysis["error_type"] == "RuntimeError"
    assert output.metrics["quant"]["quantized_module_count"] >= 1


def test_fp4_weight_only_report_includes_calibration_summary_when_inputs_provided() -> None:
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
    assert report.algorithm_executable is True
    assert report.method_semantics == "awq_activation_aware_fp4_weight_only_quantization"
    assert report.metadata["calibration_algorithm"] == "activation_aware_scale_selection"
    assert report.metadata["calibrated_module_count"] == 2
    assert report.calibration_summary["sample_count"] == 2
    assert report.calibration_summary["batch_count"] == 2
    assert report.calibration_summary["input_signature"]
    assert report.calibration_summary["calibrator_type"] == "XQTCalibrationInputSummary"
    assert report.calibration_summary["observer_type"] == "pytorch.calibration_inputs"


def test_gptq_fp4_report_uses_hessian_calibration_when_inputs_provided() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["method"] = "gptq"
    quant_config = _quant_config(config_dict)
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
    assert report.algorithm_executable is True
    assert report.method_semantics == "gptq_hessian_aware_fp4_weight_only_quantization"
    assert report.metadata["calibration_algorithm"] == "hessian_diag_residual_compensation"
    assert report.metadata["calibrated_module_count"] == 2


def test_pytorch_awq_fp4_quant_method_executes_linear_rewrite() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["backend"] = "pytorch"
    quant_config = _quant_config(config_dict)
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
    quantized_model = execution.model
    assert report.backend == "pytorch"
    assert report.method == "awq"
    assert report.algorithm_executable is True
    assert isinstance(quantized_model, _TinyMLP)
    assert isinstance(quantized_model.fc1, FP4WeightOnlyLinear)
    # Operator bridge hook remains available for later engine=tilelang materialize.
    assert callable(getattr(quantized_model.fc1, "tilelang_packed_dequant_gemm_args"))


def test_quant_backend_tilelang_is_rejected() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["backend"] = "tilelang"
    quant_config = _quant_config(config_dict)
    with pytest.raises(ValueError, match="operator engine"):
        build_quantization_plan(quant_config)


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
    quant_config = _quant_config(config_dict)
    model = _TinyMLP().eval()
    calibration_inputs = [torch.randn(2, 8), torch.randn(2, 8)]
    context = _runtime_context(
        quant_config,
        model=model,
        calibration_inputs=calibration_inputs,
    )
    plan = build_quantization_plan(quant_config)

    monkeypatch.setattr(
        passes_module,
        "build_layer_analysis_payload",
        lambda *args, **kwargs: {"runtime": "quantized_pytorch", "layer_errors": [], "layer_sensitivity": [], "avoid_list": []},
    )

    monkeypatch.setattr(
        "xqt.quant.execution.executor.quantize_with_torchao",
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
    assert report.algorithm_executable is True
    assert report.method_semantics == "torchao_executable_quantization"
    assert report.metadata["algorithm_executable"] is True
    assert report.metadata["method_semantics"] == "torchao_executable_quantization"
    assert report.calibration_summary["sample_count"] == 2
    assert report.calibration_summary["observer_type"] == "torchao.calibration_inputs"
