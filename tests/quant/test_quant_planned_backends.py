from __future__ import annotations

import copy
from typing import Any

import torch

from xqt.core.schema import QuantConfig
from xqt.core.types import XQTContext
from xqt.quant.execution import execute_quantization_plan
from xqt.quant.plan import build_quantization_plan
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear


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
        artifact_dir="artifacts/xqt/tests/quant_planned_backends",
        project_name="quant_planned_backends",
        quant_config=quant_config,
    )


def _quant_config(config_dict: dict[str, Any]) -> QuantConfig:
    return QuantConfig(**config_dict["compression"]["quant"])


def _base_awq_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "quant_planned_backends",
            "artifact_dir": "artifacts/xqt/tests/quant_planned_backends",
        },
        "model": {
            "target": "torch.nn:Linear",
            "params": {"in_features": 4, "out_features": 4},
            "device": "cpu",
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "pytorch",
                "method": "awq",
                "strategy": "fp4_weight_only",
                "policy": {
                    "bits": 4,
                    "dtype": "fp4",
                    "group_size": 128,
                },
            }
        },
    }


def test_pytorch_awq_fp4_quantization_executes_storage_rewrite_without_calibration() -> None:
    quant_config = _quant_config(_base_awq_config())
    model = torch.nn.Linear(4, 4)
    context = _runtime_context(quant_config, model=model)
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.backend == "pytorch"
    assert report.method == "awq"
    assert report.strategy == "fp4_weight_only"
    assert report.algorithm_executable is False
    assert report.method_semantics == "awq_label_only_groupwise_fp4_weight_only_storage_quantization"
    assert report.metadata["execution_state"] == "fp4_weight_only"
    assert report.metadata["executed"] is True
    assert report.metadata["algorithm_executable"] is False
    assert report.metadata["method_semantics"] == "awq_label_only_groupwise_fp4_weight_only_storage_quantization"
    assert report.quantized_modules == [""]
    assert not report.artifacts
    assert not execution.artifacts


def test_pytorch_awq_int4_executes_algorithmic_calibration() -> None:
    config_dict = _base_awq_config()
    config_dict["compression"]["quant"]["backend"] = "pytorch"
    config_dict["compression"]["quant"]["strategy"] = "weight_only_int4"
    config_dict["compression"]["quant"]["policy"]["dtype"] = "int4"
    config_dict["compression"]["quant"]["policy"]["bits"] = 4
    quant_config = _quant_config(config_dict)
    model = torch.nn.Linear(4, 4)
    calibration_inputs = [torch.randn(2, 4), torch.randn(2, 4)]
    context = _runtime_context(
        quant_config,
        model=model,
        calibration_inputs=calibration_inputs,
    )
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.backend == "pytorch"
    assert report.method == "awq"
    assert report.strategy == "weight_only_int4"
    assert report.algorithm_executable is True
    assert report.method_semantics == "awq_activation_aware_weight_only_int4_quantization"
    assert report.metadata["calibration_algorithm"] == "activation_aware_scale_selection"
    assert report.metadata["execution_state"] == "weight_only_int4"
    assert report.metadata["bits"] == 4
    assert report.metadata["calibrated_module_count"] == 1
    assert report.quantized_modules == [""]
    assert isinstance(execution.model, AWQGPTQWeightOnlyLinear)
    assert execution.model.bits == 4
    assert execution.model.quantized_weight.dtype == torch.uint8


def test_pytorch_gptq_int8_executes_hessian_aware_calibration() -> None:
    config_dict = _base_awq_config()
    config_dict["compression"]["quant"]["method"] = "gptq"
    config_dict["compression"]["quant"]["strategy"] = "weight_only_int8"
    config_dict["compression"]["quant"]["policy"]["dtype"] = "int8"
    config_dict["compression"]["quant"]["policy"]["bits"] = 8
    quant_config = _quant_config(config_dict)
    model = torch.nn.Linear(4, 4)
    calibration_inputs = [torch.randn(2, 4), torch.randn(2, 4)]
    context = _runtime_context(
        quant_config,
        model=model,
        calibration_inputs=calibration_inputs,
    )
    plan = build_quantization_plan(quant_config)

    execution = execute_quantization_plan(context, plan)

    report = execution.reports[0]
    assert report.backend == "pytorch"
    assert report.method == "gptq"
    assert report.strategy == "weight_only_int8"
    assert report.algorithm_executable is True
    assert report.method_semantics == "gptq_hessian_aware_weight_only_int8_quantization"
    assert report.metadata["calibration_algorithm"] == "hessian_diag_residual_compensation"
    assert report.metadata["execution_state"] == "weight_only_int8"
    assert report.metadata["bits"] == 8
    assert report.quantized_modules == [""]
    assert isinstance(execution.model, AWQGPTQWeightOnlyLinear)
    assert execution.model.bits == 8
    assert execution.model.quantized_weight.dtype == torch.int8
