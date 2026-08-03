from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from xqt.core.schema import QuantConfig
from xqt.core.types import XQTContext
from xqt.quant.capability import describe_quant_backend_capability, list_quant_backend_capabilities
from xqt.quant.execution import execute_quantization_plan
from xqt.quant.plan import build_quantization_plan
from xqt.quant.quantizers.svd import (
    SVDQuantInt8MmaLinear,
    SVDQuantLinear,
    SVDQuantResult,
    quantize_with_svd,
)


def test_svdquant_is_not_a_quant_backend() -> None:
    caps = list_quant_backend_capabilities()
    assert "svdquant" not in caps
    with pytest.raises(ValueError, match="quant method"):
        describe_quant_backend_capability("svdquant", method="svd")


def test_pytorch_svd_method_capability() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="svd",
        strategy="w4a16_int4",
        compute="dequant_fp16",
        policy={"rank": 8, "quant_dtype": "int4"},
    )
    assert capability.backend == "pytorch"
    assert "svd" in capability.methods
    assert capability.maturity == "reference_guarded"
    assert capability.nature.value == "pseudo"


def test_pytorch_svd_int8_mma_capability_is_executable() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="svd",
        strategy="w4a16_fp4",
        compute="w8a8_int8_mma",
        policy={"rank": 8, "quant_dtype": "fp4", "residual_compute": "int8_mma"},
    )

    assert capability.maturity == "executable"
    assert capability.nature.value == "true"


def test_pytorch_svd_int4_int8_mma_capability_is_executable() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="svd",
        strategy="w4a16_int4",
        compute="w8a8_int8_mma",
        policy={"rank": 8, "quant_dtype": "int4", "residual_compute": "int8_mma"},
    )

    assert capability.maturity == "executable"
    assert capability.nature.value == "true"


def test_quantize_with_svd_reports_pytorch_backend_and_svd_method() -> None:
    model = nn.Sequential(nn.Linear(32, 32))
    result = quantize_with_svd(
        model,
        strategy="w4a16_int4",
        compute="dequant_fp16",
        rank=4,
        group_size=16,
        quant_dtype="int4",
        inplace=False,
    )
    assert isinstance(result, SVDQuantResult)
    assert result.backend == "pytorch"
    assert result.method == "svd"
    assert result.strategy == "w4a16_int4"
    assert result.compute == "dequant_fp16"
    assert result.quantized_modules == ["0"]
    assert isinstance(result.model[0], SVDQuantLinear)
    handoff = result.infer_handoff()
    assert "method" not in handoff
    assert handoff["model"] is result.model
    assert handoff["compute_config"] is not None
    assert handoff["compute_config"]["modules"][0]["compute_contract"] == "composite_add"
    assert handoff["compute_config"]["modules"][0]["combine"] == "add"
    branch_names = {
        b["name"] for b in handoff["compute_config"]["modules"][0]["branches"]
    }
    assert branch_names == {"low_rank", "quant_residual"}


def test_svd_fp4_int8_mma_keeps_packed_residual_and_declares_compute_contract() -> None:
    model = nn.Sequential(nn.Linear(32, 32)).eval()
    inputs = torch.randn(3, 32)
    result = quantize_with_svd(
        model,
        strategy="w4a16_fp4",
        compute="w8a8_int8_mma",
        rank=4,
        group_size=16,
        quant_dtype="fp4",
        residual_compute="int8_mma",
        engine="torch_int_mm",
        inplace=False,
    )

    assert result.strategy == "w4a16_fp4"
    assert result.compute == "w8a8_int8_mma"
    assert isinstance(result.model[0], SVDQuantInt8MmaLinear)
    assert result.model[0].quant_dtype == "fp4"
    output = result.model(inputs)
    assert output.shape == (3, 32)
    assert torch.isfinite(output).all()
    execution = result.model[0].execution_metadata()
    assert execution["residual_storage"] == "packed_signed_int4_group_scale"
    assert execution["residual_compute"] == "w8a8_int8_mma"
    assert execution["activation_dtype"] == "int8"
    assert execution["quant_dtype"] == "fp4"
    handoff = result.infer_handoff()
    assert handoff["compute_config"] is not None
    module_contract = handoff["compute_config"]["modules"][0]
    assert module_contract["compute_contract"] == "composite_add"
    assert module_contract["precision"] == "w8a8"
    assert module_contract["combine"] == "add"
    residual = next(
        b for b in module_contract["branches"] if b["name"] == "quant_residual"
    )
    assert residual["compute_contract"] == "w4_storage_int8_mma"


def test_svd_int4_int8_mma_keeps_packed_residual_and_declares_compute_contract() -> None:
    model = nn.Sequential(nn.Linear(32, 32)).eval()
    inputs = torch.randn(3, 32)
    result = quantize_with_svd(
        model,
        strategy="w4a16_int4",
        compute="w8a8_int8_mma",
        rank=4,
        group_size=16,
        quant_dtype="int4",
        residual_compute="int8_mma",
        engine="torch_int_mm",
        inplace=False,
    )

    assert result.strategy == "w4a16_int4"
    assert result.compute == "w8a8_int8_mma"
    assert result.metadata["quant_dtype"] == "int4"
    assert isinstance(result.model[0], SVDQuantInt8MmaLinear)
    assert result.model[0].quant_dtype == "int4"
    output = result.model(inputs)
    assert output.shape == (3, 32)
    assert torch.isfinite(output).all()
    execution = result.model[0].execution_metadata()
    assert execution["residual_storage"] == "packed_signed_int4_group_scale"
    assert execution["residual_compute"] == "w8a8_int8_mma"
    assert execution["quant_dtype"] == "int4"
    residual = next(
        b
        for b in result.infer_handoff()["compute_config"]["modules"][0]["branches"]
        if b["name"] == "quant_residual"
    )
    assert residual["storage"]["quant_dtype"] == "int4"
    assert residual["compute_contract"] == "w4_storage_int8_mma"


def test_svd_fp4_int8_mma_collects_static_scales_from_calibration_inputs() -> None:
    model = nn.Sequential(nn.Linear(32, 32)).eval()
    calibration_inputs = [torch.randn(2, 32), torch.randn(2, 32)]
    result = quantize_with_svd(
        model,
        strategy="w4a16_fp4",
        compute="w8a8_int8_mma",
        rank=4,
        group_size=16,
        quant_dtype="fp4",
        residual_compute="int8_mma",
        engine="torch_int_mm",
        activation_scale_mode="static",
        calibration_inputs=calibration_inputs,
        inplace=False,
    )

    module = result.model[0]
    assert isinstance(module, SVDQuantInt8MmaLinear)
    assert module.residual_int8.activation_scale_mode == "static"
    assert result.metadata["calibrated_static_scale_module_count"] == 1
    module(torch.randn(2, 32))
    assert module.execution_metadata()["activation_scale_mode"] == "static"


def test_execute_plan_pytorch_svd_strategy() -> None:
    quant_config = QuantConfig(
        enabled=True,
        backend="pytorch",
        method="svd",
        strategy="w4a16_int4",
        compute="dequant_fp16",
        policy={"rank": 4, "group_size": 16, "quant_dtype": "int4"},
    )
    model = nn.Sequential(nn.Linear(32, 32))
    context = XQTContext(
        model=model,
        device="cpu",
        artifact_dir="artifacts/xqt/tests/quant_svd",
        project_name="quant_svd",
        quant_config=quant_config,
    )
    plan = build_quantization_plan(quant_config)
    execution = execute_quantization_plan(context, plan)
    report = execution.reports[0]
    assert report.backend == "pytorch"
    assert report.method == "svd"
    assert report.strategy == "w4a16_int4"
    assert report.algorithm_executable is True
    assert isinstance(execution.model[0], SVDQuantLinear)


def test_execute_plan_svd_fp4_int8_mma_reports_true_quantization() -> None:
    quant_config = QuantConfig(
        enabled=True,
        backend="pytorch",
        method="svd",
        strategy="w4a16_fp4",
        compute="w8a8_int8_mma",
        policy={
            "rank": 4,
            "group_size": 16,
            "quant_dtype": "fp4",
            "residual_compute": "int8_mma",
            "engine": "torch_int_mm",
        },
    )
    context = XQTContext(
        model=nn.Sequential(nn.Linear(32, 32)),
        device="cpu",
        artifact_dir="artifacts/xqt/tests/quant_svd_int8_mma",
        project_name="quant_svd_int8_mma",
        quant_config=quant_config,
        calibration_inputs=[torch.randn(2, 32, dtype=torch.bfloat16)],
    )

    execution = execute_quantization_plan(context, build_quantization_plan(quant_config))

    report = execution.reports[0]
    assert report.strategy == "w4a16_fp4"
    assert report.nature.value == "true"
    assert report.metadata["execution_state"] == (
        "composite_add_materialized_int8_residual"
    )
    assert report.calibration_summary is not None
    assert report.calibration_summary["dtypes"]["input"] == ["bfloat16"]
    assert isinstance(execution.model[0], SVDQuantInt8MmaLinear)


def test_execute_plan_svd_int4_int8_mma_reports_true_quantization() -> None:
    quant_config = QuantConfig(
        enabled=True,
        backend="pytorch",
        method="svd",
        strategy="w4a16_int4",
        compute="w8a8_int8_mma",
        policy={
            "rank": 4,
            "group_size": 16,
            "quant_dtype": "int4",
            "residual_compute": "int8_mma",
            "engine": "torch_int_mm",
        },
    )
    context = XQTContext(
        model=nn.Sequential(nn.Linear(32, 32)),
        device="cpu",
        artifact_dir="artifacts/xqt/tests/quant_svd_int4_int8_mma",
        project_name="quant_svd_int4_int8_mma",
        quant_config=quant_config,
        calibration_inputs=[torch.randn(2, 32, dtype=torch.bfloat16)],
    )

    execution = execute_quantization_plan(context, build_quantization_plan(quant_config))

    report = execution.reports[0]
    assert report.strategy == "w4a16_int4"
    assert report.nature.value == "true"
    assert report.metadata["quant_dtype"] == "int4"
    assert report.metadata["execution_state"] == (
        "composite_add_materialized_int8_residual"
    )
    assert isinstance(execution.model[0], SVDQuantInt8MmaLinear)
    assert execution.model[0].quant_dtype == "int4"


def test_execute_plan_rejects_backend_svdquant() -> None:
    quant_config = QuantConfig(
        enabled=True,
        backend="svdquant",
        method="svd",
        strategy="w4a16_fp4",
        policy={"rank": 4},
    )
    with pytest.raises(ValueError, match="quant method"):
        build_quantization_plan(quant_config)


def test_svd_storage_shell_without_eager_materialize() -> None:
    model = nn.Sequential(nn.Linear(32, 32)).eval()
    result = quantize_with_svd(
        model,
        strategy="w4a16_fp4",
        compute="w8a8_int8_mma",
        rank=4,
        group_size=16,
        quant_dtype="fp4",
        residual_compute="int8_mma",
        engine="torch_int_mm",
        inplace=False,
    )
    # re-run with materialize_compute=False via policy
    result = quantize_with_svd(
        model,
        policy={
            "rank": 4,
            "group_size": 16,
            "quant_dtype": "fp4",
            "residual_compute": "int8_mma",
            "engine": "torch_int_mm",
            "materialize_compute": False,
        },
        strategy="w4a16_fp4",
        compute="w8a8_int8_mma",
        inplace=False,
    )
    assert isinstance(result.model[0], SVDQuantLinear)
    assert result.infer_handoff()["compute_config"]["modules"][0]["compute_contract"] == (
        "composite_add"
    )
    from xqt.runtime.composite_materialize import materialize_composite_compute

    materialized = materialize_composite_compute(
        result.model,
        result.compute_config,
        inplace=False,
    )
    assert isinstance(materialized[0], SVDQuantInt8MmaLinear)
