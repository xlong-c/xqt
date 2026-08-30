"""Tests for the orthogonal QuantScheme value object (C1)."""

from __future__ import annotations

import pytest

from xqt.core.schema import CANONICAL_QUANT_STRATEGIES, QuantConfig
from xqt.compression.quant.plan import build_quantization_plan
from xqt.compression.quant.strategy import coerce_quant_scheme, resolve_scheme
from xqt.compression.quant.types import QuantScheme


def test_resolve_scheme_covers_every_canonical_strategy() -> None:
    for strategy in CANONICAL_QUANT_STRATEGIES:
        scheme = resolve_scheme(strategy)
        assert scheme is not None, strategy
        assert scheme.weight_dtype
        if scheme.weight_granularity in {"groupwise", "block"}:
            assert scheme.group_size is not None and scheme.group_size > 0
        if scheme.activation_dtype is None:
            assert scheme.activation_mode == "none"
        else:
            assert scheme.activation_mode in {"dynamic", "static"}


def test_resolve_scheme_defaults_match_quantizer_conventions() -> None:
    scheme = resolve_scheme("w4a16_int4")
    assert scheme == QuantScheme(
        weight_dtype="int4",
        weight_granularity="groupwise",
        group_size=128,
    )
    nvfp4 = resolve_scheme("w4a4_nvfp4")
    assert nvfp4 is not None
    assert (nvfp4.weight_granularity, nvfp4.group_size) == ("block", 16)
    assert nvfp4.activation_dtype == "nvfp4"
    assert nvfp4.activation_mode == "dynamic"
    mxfp8 = resolve_scheme("w8a16_mxfp8")
    assert mxfp8 is not None
    assert (mxfp8.weight_granularity, mxfp8.group_size) == ("block", 32)
    assert mxfp8.is_weight_only


def test_resolve_scheme_applies_policy_overrides() -> None:
    scheme = resolve_scheme("w4a16_int4", {"group_size": 64})
    assert scheme is not None
    assert scheme.group_size == 64
    static_int8 = resolve_scheme("w8a8_int8", {"activation": "static"})
    assert static_int8 is not None
    assert static_int8.activation_mode == "static"
    asymmetric = resolve_scheme("w4a16_int4", {"sym": False})
    assert asymmetric is not None
    assert asymmetric.sym is False
    # Block formats keep their format-inherent block size.
    nvfp4 = resolve_scheme("w4a4_nvfp4", {"group_size": 64})
    assert nvfp4 is not None
    assert nvfp4.group_size == 16


def test_resolve_scheme_alias_and_unknown() -> None:
    assert resolve_scheme(None) is None
    aliased = resolve_scheme("fp4_weight_only")
    assert aliased is not None
    assert aliased.weight_dtype == "fp4"
    with pytest.raises(ValueError, match="Cannot resolve a QuantScheme"):
        resolve_scheme("w4a4_int4_unknown")


def test_quant_scheme_validation() -> None:
    with pytest.raises(ValueError, match="group_size must be a positive int"):
        QuantScheme(weight_dtype="int4", weight_granularity="groupwise")
    with pytest.raises(ValueError, match="group_size must be None"):
        QuantScheme(
            weight_dtype="int8",
            weight_granularity="per_channel",
            group_size=128,
        )
    with pytest.raises(ValueError, match="weight-only"):
        QuantScheme(
            weight_dtype="int4",
            weight_granularity="groupwise",
            group_size=128,
            activation_mode="dynamic",
        )
    with pytest.raises(ValueError, match="activation_mode must be 'dynamic' or 'static'"):
        QuantScheme(
            weight_dtype="int8",
            weight_granularity="per_channel",
            activation_dtype="int8",
        )
    with pytest.raises(ValueError, match="weight_dtype"):
        QuantScheme(weight_dtype="fp16", weight_granularity="per_tensor")


def test_coerce_quant_scheme_inputs() -> None:
    assert coerce_quant_scheme(None, strategy=None) is None
    from_strategy = coerce_quant_scheme(None, strategy="w8a8_int8")
    assert from_strategy is not None and from_strategy.activation_dtype == "int8"
    from_name = coerce_quant_scheme("w4a16_fp4")
    assert from_name is not None and from_name.weight_dtype == "fp4"
    from_mapping = coerce_quant_scheme(
        {
            "weight_dtype": "int8",
            "weight_granularity": "per_channel",
            "activation_dtype": "int8",
            "activation_mode": "static",
        }
    )
    assert from_mapping is not None
    assert from_mapping.activation_mode == "static"
    existing = QuantScheme(weight_dtype="int8", weight_granularity="per_channel")
    assert coerce_quant_scheme(existing) is existing
    with pytest.raises(ValueError, match="unknown fields"):
        coerce_quant_scheme({"weight_dtype": "int8", "bogus": 1})
    with pytest.raises(TypeError, match="quant.scheme must be"):
        coerce_quant_scheme(42)


def _config(**overrides: object) -> QuantConfig:
    base = {
        "enabled": True,
        "backend": "pytorch",
        "method": "awq",
        "strategy": "w4a16_int4",
        "compute": "dequant_fp16",
    }
    base.update(overrides)
    return QuantConfig(**base)  # type: ignore[arg-type]


def test_plan_resolves_scheme_from_strategy_and_policy() -> None:
    quant_config = _config(policy={"group_size": 64})
    plan = build_quantization_plan(quant_config)
    component = plan.components[0]
    assert component.strategy == "w4a16_int4"
    assert component.scheme is not None
    assert component.scheme.group_size == 64
    assert component.to_dict()["scheme"]["weight_dtype"] == "int4"


def test_plan_scheme_string_doubles_as_strategy() -> None:
    quant_config = _config(strategy=None, scheme="w8a8_int8")
    plan = build_quantization_plan(quant_config)
    component = plan.components[0]
    assert component.strategy == "w8a8_int8"
    assert component.scheme is not None
    assert component.scheme.activation_dtype == "int8"


def test_plan_structured_scheme_wins_over_strategy() -> None:
    quant_config = _config(
        strategy="w4a16_int4",
        scheme={
            "weight_dtype": "int8",
            "weight_granularity": "per_channel",
            "activation_dtype": "int8",
            "activation_mode": "static",
        },
    )
    plan = build_quantization_plan(quant_config)
    component = plan.components[0]
    # Legacy strategy string stays for reporting, but the scheme is authoritative.
    assert component.strategy == "w4a16_int4"
    assert component.scheme is not None
    assert component.scheme.weight_dtype == "int8"
    assert component.scheme.activation_mode == "static"


def test_one_scheme_is_shared_across_methods() -> None:
    awq_plan = build_quantization_plan(_config(method="awq"))
    gptq_plan = build_quantization_plan(_config(method="gptq"))
    awq_scheme = awq_plan.components[0].scheme
    gptq_scheme = gptq_plan.components[0].scheme
    assert awq_scheme is not None and gptq_scheme is not None
    assert awq_scheme == gptq_scheme
