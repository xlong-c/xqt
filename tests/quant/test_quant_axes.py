"""Tests for the DEBT-002 three-axis public API and DEBT-001 engine vocabulary."""

from __future__ import annotations

from typing import Literal, get_args

import pytest
import torch
from torch import nn

from xqt.conversion import EngineKind, convert
from xqt.core.errors import XQTBackendError
from xqt.core.schema import (
    CANONICAL_QUANT_COMPUTES,
    CANONICAL_QUANT_STRATEGIES,
    CONVERT_ENGINE_NAMES,
    OPERATOR_OPT_ENGINES,
)
from xqt.quant.axes import (
    quant_axis_report,
    quant_compute_specs,
    quant_method_specs,
    quant_storage_specs,
)
from xqt.quant.capability import describe_quant_backend_capability
from xqt.quant.strategy import (
    canonical_quant_strategies,
    resolve_scheme,
)
from xqt.contracts.engine_resolve import engine_registry_names, operator_engine_names
from xqt.contracts.quant_strategy import (
    QUANT_STRATEGY_DEFINITIONS,
    QuantizationNature,
    quantization_nature_for_compute,
    quantization_nature_for_strategy,
)


def test_quant_axis_report_covers_three_axes() -> None:
    report = quant_axis_report()

    assert report["axes"] == ["method", "storage", "compute"]
    assert report["strategy_is_scheme_alias"] is True

    methods = {item["name"] for item in report["method"]}
    assert {"none", "awq", "gptq", "svd", "convrot", "turboquant"} <= methods
    assert any(item["registered_route"] for item in report["method"])

    storage = {item["strategy"] for item in report["storage"]}
    assert storage == set(CANONICAL_QUANT_STRATEGIES)
    assert all(item["axis"] == "storage" for item in report["storage"])
    assert all(item["strategy_is_scheme_alias"] for item in report["storage"])

    compute = {item["name"] for item in report["compute"]}
    assert compute == set(CANONICAL_QUANT_COMPUTES)
    compute_by_name = {item["name"]: item for item in report["compute"]}
    assert compute_by_name["w8a8_int8_mma"]["nature"] == "true"
    assert compute_by_name["fp8_mma"]["nature"] == "true"
    assert compute_by_name["dequant_fp16"]["nature"] == "pseudo"
    assert compute_by_name["qdq_static"]["nature"] == "pseudo"


def test_strategy_vocabulary_is_single_sourced() -> None:
    assert set(CANONICAL_QUANT_STRATEGIES) == set(canonical_quant_strategies())
    assert len(CANONICAL_QUANT_STRATEGIES) == len(set(CANONICAL_QUANT_STRATEGIES))

    # Every canonical strategy resolves to an orthogonal QuantScheme.
    for strategy in CANONICAL_QUANT_STRATEGIES:
        scheme = resolve_scheme(strategy)
        assert scheme is not None
        assert scheme.weight_dtype in {
            "int4",
            "int8",
            "fp4",
            "nvfp4",
            "mxfp4",
            "mxfp8",
            "fp8_e4m3",
            "fp8_e5m2",
        }
        assert scheme == QUANT_STRATEGY_DEFINITIONS[strategy].scheme

    assert quantization_nature_for_strategy("w4a16_int4") is QuantizationNature.PSEUDO
    assert quantization_nature_for_strategy("w8a8_int8") is QuantizationNature.UNKNOWN
    assert quantization_nature_for_compute("w8a8_int8_mma") is QuantizationNature.TRUE


def test_backend_capability_is_split_into_three_axes() -> None:
    pytorch = describe_quant_backend_capability(
        "pytorch",
        method="awq",
        strategy="w4a16_int4",
        compute="dequant_fp16",
    )
    assert "awq" in pytorch.methods
    assert "gptq" in pytorch.methods
    assert "w4a16_int4" not in pytorch.methods
    assert pytorch.storage_strategies == CANONICAL_QUANT_STRATEGIES
    assert "w8a8_int8_mma" in pytorch.compute_contracts

    torchao = describe_quant_backend_capability("torchao", strategy="w8a8_int8")
    assert "w8a8_int8" in torchao.storage_strategies
    assert "awq" not in torchao.methods
    assert "dequant_fp16" in torchao.compute_contracts

    qdq = describe_quant_backend_capability(
        "onnxruntime_qdq",
        method="none",
        strategy="w8a8_int8",
        compute="qdq_static",
    )
    assert qdq.storage_strategies == ("w8a8_int8",)
    assert {"qdq_static", "qdq_dynamic"} <= set(qdq.compute_contracts)

    bitsandbytes = describe_quant_backend_capability("bitsandbytes")
    assert bitsandbytes.methods == ("none",)
    assert "w4a16_int4" in bitsandbytes.storage_strategies

    payload = pytorch.to_dict()
    assert "storage_strategies" in payload
    assert "compute_contracts" in payload
    capability = pytorch.to_optimization_capability().to_dict()
    assert capability["metadata"]["storage_strategies"] == list(
        pytorch.storage_strategies
    )
    assert "compute_contracts" in capability["metadata"]


def test_planned_online_method_preserves_axis_fields() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="fp8_per_tensor",
    )
    assert capability.status == "planned"
    assert capability.maturity == "planned"
    assert capability.storage_strategies == CANONICAL_QUANT_STRATEGIES
    assert capability.compute_contracts


def test_convert_engine_preference_uses_canonical_subset() -> None:
    assert set(get_args(EngineKind)) == set(CONVERT_ENGINE_NAMES)

    model = nn.Linear(4, 4)
    result = convert(model, engine="torch")
    assert isinstance(result, nn.Module)

    with pytest.raises(XQTBackendError, match="not a convert"):
        convert(nn.Linear(4, 4), engine="cutlass")


def test_engine_vocabulary_fact_sources_are_aligned() -> None:
    registry = set(engine_registry_names())
    config_engines = set(OPERATOR_OPT_ENGINES)

    assert {"tilelang", "triton", "torch", "cutile", "cutlass", "cute_dsl"} <= registry
    assert config_engines == set(operator_engine_names())
    assert config_engines <= registry
    assert set(CONVERT_ENGINE_NAMES) <= config_engines | {"torch"}
    assert "ptx_sm89" in registry
