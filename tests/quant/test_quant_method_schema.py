import pytest

from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTConfigError
from xqt.quant.capability import describe_quant_backend_capability
from xqt.quant.plan import build_quantization_plan


def _base_config() -> dict:
    return {
        "config_version": 1,
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
                "strategy": "int4_weight_only",
                "calibration_split": "calibration",
                "policy": {
                    "bits": 4,
                    "group_size": 128,
                },
            },
        },
    }


def test_quant_method_is_distinct_from_backend() -> None:
    config = load_xqt_config(_base_config())

    plan = build_quantization_plan(config.compression.quant)

    assert len(plan.components) == 1
    component = plan.components[0]
    assert component.backend == "pytorch"
    assert component.method == "awq"
    assert component.strategy == "int4_weight_only"


def test_awq_is_not_accepted_as_quant_backend() -> None:
    config = _base_config()
    config["compression"]["quant"]["backend"] = "awq"
    config["compression"]["quant"].pop("method")

    with pytest.raises(XQTConfigError, match="compression.quant.backend"):
        load_xqt_config(config)


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

    loaded = load_xqt_config(config)
    plan = build_quantization_plan(loaded.compression.quant)

    assert len(plan.components) == 1
    assert plan.components[0].backend == "pytorch"
    assert plan.components[0].method == "awq"


def test_capability_rejects_method_backend_mismatch() -> None:
    with pytest.raises(ValueError, match="not supported by backend"):
        describe_quant_backend_capability("onnxruntime_qdq", method="awq")
