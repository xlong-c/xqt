import importlib

import pytest
from omegaconf import OmegaConf

from xqt.core.errors import XQTConfigError
from xqt.core.schema import QuantConfig
from xqt.quant.capability import describe_quant_backend_capability
from xqt.quant.plan import build_quantization_plan
from xqt.workflows import load_optimization_config


def _base_config() -> dict:
    return {
        "project": {
            "name": "quant_method_schema",
            "artifact_dir": "artifacts/xqt/tests/quant_method_schema",
        },
        "model": {
            "target": "xqt.quant.toy_models.build_hetero_quant_toy_model",
            "params": {
                "in_features": 4,
                "hidden_features": 4,
                "num_classes": 2,
            },
        },
        "compression": {
            "axes": ["precision"],
            "quant": {
                "enabled": True,
                "backend": "pytorch",
                "method": "awq",
                "strategy": "weight_only_int4",
                "policy": {
                    "bits": 4,
                    "group_size": 128,
                },
            },
        },
    }


def _quant_config(config: dict) -> QuantConfig:
    merged = OmegaConf.merge(
        OmegaConf.structured(QuantConfig),
        config["compression"]["quant"],
    )
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def _workflow_config(config: dict) -> dict:
    quant_params = dict(config["compression"]["quant"])
    quant_params.pop("enabled", None)
    return {
        "project": config["project"],
        "model": config["model"],
        "compression_axes": ["precision"],
        "stages": [
            {
                "name": "quant_model",
                "kind": "quant",
                "params": quant_params,
            }
        ],
    }


def test_quant_method_is_distinct_from_backend() -> None:
    quant_config = _quant_config(_base_config())

    plan = build_quantization_plan(quant_config)

    assert len(plan.components) == 1
    component = plan.components[0]
    assert component.backend == "pytorch"
    assert component.method == "awq"
    assert component.strategy == "weight_only_int4"
    assert "weight_only_int4" in plan.metadata["canonical_strategies"]


def test_legacy_strategy_alias_is_canonicalized_in_plan() -> None:
    config_dict = _base_config()
    config_dict["compression"]["quant"]["strategy"] = "int4_weight_only"
    quant_config = _quant_config(config_dict)

    plan = build_quantization_plan(quant_config)

    assert plan.components[0].strategy == "weight_only_int4"


def test_quant_config_requires_explicit_backend_when_enabled() -> None:
    config = _base_config()
    config["compression"]["quant"].pop("backend")

    with pytest.raises(XQTConfigError, match="quant.params.backend"):
        load_optimization_config(_workflow_config(config))


def test_quant_config_requires_explicit_method_strategy_or_policy() -> None:
    config = _base_config()
    quant = config["compression"]["quant"]
    quant["backend"] = "pytorch"
    quant.pop("method")
    quant.pop("strategy")
    quant.pop("policy")

    with pytest.raises(XQTConfigError, match="method, strategy, policy"):
        load_optimization_config(_workflow_config(config))


def test_component_policy_records_selection_policy_metadata() -> None:
    config = _base_config()
    config["compression"]["quant"]["component_policies"] = [
        {
            "name": "encoder",
            "target": "encoder",
            "backend": "pytorch",
            "method": "awq",
            "policy": {
                "include_module_types": ["Linear"],
                "include_name_patterns": ["layers\\.\\d+"],
                "exclude_module_names": ["lm_head"],
            },
            "keep_high_precision": ["norm"],
            "skip_quantize": ["router"],
            "force_quantize": ["fc1"],
        }
    ]

    quant_config = _quant_config(config)
    plan = build_quantization_plan(quant_config)
    selection_policy = plan.components[0].policy["selection_policy"]

    assert selection_policy["target_path"] == "encoder"
    assert selection_policy["selectors"]["include_module_types"] == ["Linear"]
    assert selection_policy["selectors"]["include_name_patterns"] == ["layers\\.\\d+"]
    assert selection_policy["selectors"]["exclude_module_names"] == ["lm_head"]
    assert selection_policy["keep_high_precision"] == ["norm"]
    assert selection_policy["skip_quantize"] == ["router"]
    assert selection_policy["force_quantize"] == ["fc1"]


def test_awq_is_not_accepted_as_quant_backend() -> None:
    config = _base_config()
    config["compression"]["quant"]["backend"] = "awq"
    config["compression"]["quant"].pop("method")

    with pytest.raises(XQTConfigError, match="Unsupported quantization backend"):
        load_optimization_config(_workflow_config(config))


def test_component_method_overrides_global_method() -> None:
    config = _base_config()
    config["compression"]["quant"]["method"] = "gptq"
    config["compression"]["quant"]["component_policies"] = [
        {
            "name": "encoder",
            "target": "encoder",
            "backend": "pytorch",
            "method": "awq",
        }
    ]

    quant_config = _quant_config(config)
    plan = build_quantization_plan(quant_config)

    assert len(plan.components) == 1
    assert plan.components[0].backend == "pytorch"
    assert plan.components[0].method == "awq"


def test_capability_rejects_method_backend_mismatch() -> None:
    with pytest.raises(ValueError, match="not supported by backend"):
        describe_quant_backend_capability("onnxruntime_qdq", method="awq")


def test_pytorch_awq_fp4_capability_is_quant_method_not_operator_engine() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="awq",
        strategy="fp4_weight_only",
        policy={"dtype": "fp4"},
    )

    assert capability.status == "available"
    assert capability.backend == "pytorch"
    assert "awq" in capability.methods
    assert "gptq" in capability.methods
    assert capability.requires_calibration is True
    assert capability.nature.value == "pseudo"
    assert capability.maturity == "executable"


def test_pytorch_gptq_int8_capability_is_executable() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="gptq",
        strategy="weight_only_int8",
        policy={"dtype": "int8", "bits": 8},
    )

    assert capability.status == "available"
    assert capability.maturity == "executable"


def test_tilelang_is_rejected_as_quant_backend() -> None:
    with pytest.raises(ValueError, match="operator engine"):
        describe_quant_backend_capability("tilelang", method="awq")


def test_awq_and_gptq_modules_are_explicit_reexports() -> None:
    awq_module = importlib.import_module("xqt.quant.quantizers.awq")
    gptq_module = importlib.import_module("xqt.quant.quantizers.gptq")

    assert awq_module.quantize_with_awq_weight_only.__module__ == (
        "xqt.quant.quantizers.awq_gptq_weight_only"
    )
    assert awq_module.quantize_with_awq_fp4.__module__ == "xqt.quant.quantizers.fp4_weight_only"
    assert gptq_module.quantize_with_gptq_weight_only.__module__ == (
        "xqt.quant.quantizers.awq_gptq_weight_only"
    )
    assert gptq_module.quantize_with_gptq_fp4.__module__ == "xqt.quant.quantizers.fp4_weight_only"


def test_removed_placeholder_quantizer_modules_are_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xqt.quant.quantizers.rtn")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xqt.quant.quantizers.smoothquant")


def test_svdquant_is_rejected_as_quant_backend() -> None:
    with pytest.raises(ValueError, match="quant method"):
        describe_quant_backend_capability("svdquant", method="svd_fp4")


def test_pytorch_hosts_svd_as_quant_method() -> None:
    capability = describe_quant_backend_capability(
        "pytorch",
        method="svd",
        strategy="svd_int4",
    )
    assert "svd" in capability.methods
    assert capability.backend == "pytorch"
