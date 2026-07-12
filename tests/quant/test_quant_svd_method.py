from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from xqt.core.schema import QuantConfig
from xqt.core.types import XQTContext
from xqt.quant.capability import describe_quant_backend_capability, list_quant_backend_capabilities
from xqt.quant.execution import execute_quantization_plan
from xqt.quant.plan import build_quantization_plan
from xqt.quant.quantizers.svd import SVDQuantLinear, SVDQuantResult, quantize_with_svd


def test_svdquant_is_not_a_quant_backend() -> None:
    caps = list_quant_backend_capabilities()
    assert "svdquant" not in caps
    with pytest.raises(ValueError, match="quant method"):
        describe_quant_backend_capability("svdquant", method="svd_fp4")


def test_pytorch_svd_method_capability() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="svd",
        strategy="svd_int4",
        policy={"rank": 8, "quant_dtype": "int4"},
    )
    assert capability.backend == "pytorch"
    assert "svd" in capability.methods
    assert "svd_fp4" in capability.methods
    assert capability.maturity == "reference_guarded"
    assert capability.nature.value == "pseudo"


def test_quantize_with_svd_reports_pytorch_backend_and_svd_method() -> None:
    model = nn.Sequential(nn.Linear(32, 32))
    result = quantize_with_svd(
        model,
        strategy="svd_int4",
        rank=4,
        group_size=16,
        quant_dtype="int4",
        inplace=False,
    )
    assert isinstance(result, SVDQuantResult)
    assert result.backend == "pytorch"
    assert result.method == "svd"
    assert result.strategy == "svd_int4"
    assert result.quantized_modules == ["0"]
    assert isinstance(result.model[0], SVDQuantLinear)
    handoff = result.infer_handoff()
    assert "method" not in handoff
    assert handoff["model"] is result.model


def test_execute_plan_pytorch_svd_strategy() -> None:
    quant_config = QuantConfig(
        enabled=True,
        backend="pytorch",
        method="svd",
        strategy="svd_int4",
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
    assert report.strategy == "svd_int4"
    assert report.algorithm_executable is True
    assert isinstance(execution.model[0], SVDQuantLinear)


def test_execute_plan_rejects_backend_svdquant() -> None:
    quant_config = QuantConfig(
        enabled=True,
        backend="svdquant",
        method="svd",
        strategy="svd_fp4",
        policy={"rank": 4},
    )
    with pytest.raises(ValueError, match="quant method"):
        build_quantization_plan(quant_config)
