import pytest

from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTConfigError
from xqt.pipeline.preflight import preflight_xqt_config


def test_load_xqt_config_supports_operator_optimization_fields() -> None:
    config = load_xqt_config(
        {
            "operator_optimization": {
                "enabled": True,
                "default_backend": "torch_compile",
                "targets": [
                    {
                        "name": "encoder",
                        "target": "model.encoder",
                        "backend": "torch_compile",
                        "mode": "reduce-overhead",
                        "fullgraph": True,
                        "dynamic": False,
                        "options": {"backend": "inductor"},
                        "fallback": "eager",
                        "min_speedup": 1.1,
                        "validate": {"atol": 1e-4, "rtol": 1e-4},
                    },
                    {
                        "name": "attention_tilelang",
                        "target": "model.attn",
                        "backend": "tilelang",
                        "patterns": ["attention"],
                        "fallback": "eager",
                        "min_speedup": 1.2,
                        "tilelang": {
                            "target": "cuda",
                            "threads": 128,
                            "num_stages": 2,
                            "pass_configs": {"TL_ENABLE_FAST_MATH": True},
                        },
                    },
                ],
            }
        }
    )

    assert config.operator_optimization.enabled is True
    assert config.operator_optimization.default_backend == "torch_compile"
    assert len(config.operator_optimization.targets) == 2
    assert config.operator_optimization.targets[0].name == "encoder"
    assert config.operator_optimization.targets[1].tilelang.pass_configs == {
        "TL_ENABLE_FAST_MATH": True
    }


def test_load_xqt_config_allows_deployment_backend_without_module_target() -> None:
    config = load_xqt_config(
        {
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "postprocess",
                        "backend": "deployment_backend",
                        "options": {
                            "runtime": "onnxruntime",
                            "stage": "decode_nms",
                        },
                    },
                ],
            }
        }
    )

    target = config.operator_optimization.targets[0]
    assert target.name == "postprocess"
    assert target.target is None
    assert target.backend == "deployment_backend"
    assert target.options["stage"] == "decode_nms"


@pytest.mark.parametrize(
    ("raw_config", "message"),
    [
        (
            {
                "operator_optimization": {
                    "enabled": True,
                    "default_backend": "bad_backend",
                }
            },
            "operator_optimization.default_backend",
        ),
        (
            {
                "operator_optimization": {
                    "enabled": True,
                    "targets": [
                        {"name": "dup", "backend": "torch_compile", "target": "layer0"},
                        {"name": "dup", "backend": "torch_compile", "target": "layer1"},
                    ],
                }
            },
            "targets\\[\\*\\].name must be unique",
        ),
        (
            {
                "operator_optimization": {
                    "enabled": True,
                    "targets": [
                        {"name": "missing_target", "backend": "torch_compile"},
                    ],
                }
            },
            "operator_optimization.targets.0.target",
        ),
        (
            {
                "operator_optimization": {
                    "enabled": True,
                    "targets": [
                        {
                            "name": "bad_speedup",
                            "backend": "torch_compile",
                            "target": "layer0",
                            "min_speedup": 1.0,
                        },
                    ],
                }
            },
            "min_speedup",
        ),
        (
            {
                "operator_optimization": {
                    "enabled": True,
                    "targets": [
                        {
                            "name": "bad_tilelang_key",
                            "backend": "tilelang",
                            "target": "layer0",
                            "tilelang": {
                                "pass_configs": {
                                    "TL_ENABLE_FAST_MATH": True,
                                    "BAD_KEY": True,
                                }
                            },
                        },
                    ],
                }
            },
            "unknown keys",
        ),
        (
            {
                "operator_optimization": {
                    "enabled": True,
                    "targets": [
                        {
                            "name": "bad_mode",
                            "backend": "torch_compile",
                            "target": "layer0",
                            "mode": "very-fast",
                        },
                    ],
                }
            },
            "supported torch.compile mode",
        ),
    ],
)
def test_load_xqt_config_rejects_invalid_operator_optimization_config(
    raw_config: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(XQTConfigError, match=message):
        load_xqt_config(raw_config)


@pytest.mark.parametrize(
    ("recipe_path", "project_name", "backend", "target_name"),
    [
        (
            "xqt/recipes/operator_compile_cuda.yaml",
            "operator_compile_cuda",
            "torch_compile",
            "model",
        ),
        (
            "xqt/recipes/operator_vit_compile.yaml",
            "operator_vit_compile",
            "torch_compile",
            "encoder_compile",
        ),
        (
            "xqt/recipes/operator_llm_mlp_triton.yaml",
            "operator_llm_mlp_triton",
            "triton",
            "mlp_triton",
        ),
        (
            "xqt/recipes/operator_attention_tilelang.yaml",
            "operator_attention_tilelang",
            "tilelang",
            "attention_tilelang",
        ),
        (
            "xqt/recipes/operator_mlp_cutile.yaml",
            "operator_mlp_cutile",
            "cutile",
            "mlp_cutile",
        ),
        (
            "xqt/recipes/operator_mlp_cutlass.yaml",
            "operator_mlp_cutlass",
            "cutlass",
            "mlp_cutlass",
        ),
    ],
)
def test_operator_optimization_recipes_load_and_preflight(
    recipe_path: str,
    project_name: str,
    backend: str,
    target_name: str,
) -> None:
    config = load_xqt_config(recipe_path)

    assert config.project.name == project_name
    assert config.operator_optimization.enabled is True
    assert len(config.operator_optimization.targets) == 1
    target = config.operator_optimization.targets[0]
    assert target.name == target_name
    assert target.backend == backend
    assert target.min_speedup > 1.0

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["model.target"].passed is True
    assert checks["operator_optimization.targets"].passed is True
    assert checks["operator_optimization.targets.0.capability"].metadata["backend"] == backend
    if backend == "tilelang":
        assert checks["operator_optimization.targets.0.tilelang.config"].metadata["cache_dir"].startswith(
            "/tmp"
        )
    if backend == "cutile":
        assert checks["operator_optimization.targets.0.cutile.config"].metadata["cache_dir"].startswith(
            "/tmp"
        )
    if backend == "cutlass":
        assert checks["operator_optimization.targets.0.cutlass.config"].metadata["cache_dir"].startswith(
            "/tmp"
        )
