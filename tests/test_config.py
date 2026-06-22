import pytest

from xqt.core.config import load_xqt_config, xqt_config_to_dict
from xqt.core.errors import XQTConfigError
from xqt.core.schema import XQTConfig


def test_load_xqt_config_merges_mapping_and_overrides() -> None:
    config = load_xqt_config(
        {
            "project": {"name": "vit_quant"},
            "compression": {
                "axes": ["precision"],
                "quant": {
                    "enabled": True,
                    "backend": "torchao",
                    "policy": {"dtype": "fp8"},
                },
            },
        },
        overrides={"benchmark": {"warmup": 1, "iterations": 3}},
    )

    assert isinstance(config, XQTConfig)
    assert config.project.name == "vit_quant"
    assert config.compression.axes == ["precision"]
    assert config.compression.quant.enabled is True
    assert config.compression.quant.policy == {"dtype": "fp8"}
    assert config.benchmark.warmup == 1
    assert config.benchmark.iterations == 3
    assert config.analysis.enabled is False
    assert config.analysis.metrics == ["max_abs", "mean_abs", "cosine_similarity"]


def test_load_xqt_config_from_yaml_resolves_xdl_resolver(tmp_path) -> None:
    config_path = tmp_path / "xqt.yaml"
    config_path.write_text(
        """
project:
  name: resolver_case
  artifact_dir: ${xdl.join_path:artifacts,xqt,resolver_case}
compression:
  axes: [width, sparsity]
  prune:
    enabled: true
    target_sparsity: 0.5
benchmark:
  warmup: 0
  iterations: 1
""",
        encoding="utf-8",
    )

    config = load_xqt_config(config_path)

    assert config.project.artifact_dir == "artifacts/xqt/resolver_case"
    assert config.compression.prune.enabled is True
    assert config.compression.prune.target_sparsity == 0.5


def test_load_xqt_config_ignores_local_report_extension_block(tmp_path) -> None:
    config_path = tmp_path / "xqt_with_report.yaml"
    config_path.write_text(
        """
project:
  name: report_extension_case
report:
  enabled: true
  outputs:
    report: report.txt
compression:
  axes: [precision]
benchmark:
  warmup: 0
  iterations: 1
""",
        encoding="utf-8",
    )

    config = load_xqt_config(config_path)

    assert config.project.name == "report_extension_case"
    assert config.compression.axes == ["precision"]


def test_load_xqt_config_supports_structured_prune_fields() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "axes": ["width", "sparsity"],
                "prune": {
                    "enabled": True,
                    "method": "structured",
                    "granularity": "channel",
                    "scope": "global",
                    "target_sparsity": 0.25,
                    "importance": {"metric": "l1"},
                    "selection": {"min_keep": 1},
                    "rewrite": {"inplace": True},
                },
            }
        }
    )

    assert config.compression.prune.method == "structured"
    assert config.compression.prune.granularity == "channel"
    assert config.compression.prune.scope == "global"
    assert config.compression.prune.importance == {"metric": "l1"}
    assert config.compression.prune.selection == {"min_keep": 1}
    assert config.compression.prune.rewrite == {"inplace": True}


def test_load_xqt_config_supports_structured_prune_keep_indices_and_importance_type() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "axes": ["width", "depth"],
                "prune": {
                    "enabled": True,
                    "method": "structured",
                    "granularity": "block",
                    "importance": {"type": "l2"},
                    "selection": {"keep_indices": {"blocks": [0, 2]}},
                },
            }
        }
    )

    assert config.compression.prune.importance == {"type": "l2"}
    assert config.compression.prune.selection == {"keep_indices": {"blocks": [0, 2]}}


def test_load_xqt_config_supports_extended_structured_prune_granularities() -> None:
    for granularity in ("stage", "hidden_width", "embedding_width", "expert"):
        importance = {"metric": "usage"} if granularity == "expert" else {"metric": "l1"}
        config = load_xqt_config(
            {
                "compression": {
                    "axes": ["width", "depth"],
                    "prune": {
                        "enabled": True,
                        "method": "structured",
                        "granularity": granularity,
                        "importance": importance,
                        "selection": {"min_keep": 1},
                    },
                }
            }
        )

        assert config.compression.prune.granularity == granularity


def test_load_xqt_config_supports_extended_quant_fields() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "onnxruntime_qdq",
                    "strategy": "static_int8",
                    "calibration_split": "calibration_vision",
                    "validation_split": "validation_vision",
                    "keep_high_precision": ["head", "norm"],
                    "skip_quantize": ["head"],
                    "force_quantize": ["features.0"],
                    "analysis_only_modules": ["projector"],
                    "component_policies": [
                        {
                            "name": "vision_encoder",
                            "target": "encoder",
                            "backend": "torchao",
                            "strategy": "weight_only_int8",
                            "keep_high_precision": ["head"],
                            "skip_quantize": ["head"],
                            "force_quantize": ["blocks.0.attn.qkv"],
                            "policy": {"dtype": "int8"},
                        }
                    ],
                }
            }
        }
    )

    quant = config.compression.quant
    assert quant.strategy == "static_int8"
    assert quant.calibration_split == "calibration_vision"
    assert quant.validation_split == "validation_vision"
    assert quant.keep_high_precision == ["head", "norm"]
    assert quant.skip_quantize == ["head"]
    assert quant.force_quantize == ["features.0"]
    assert quant.analysis_only_modules == ["projector"]
    assert len(quant.component_policies) == 1
    assert quant.component_policies[0].name == "vision_encoder"
    assert quant.component_policies[0].backend == "torchao"
    assert quant.component_policies[0].target == "encoder"


def test_load_xqt_config_accepts_planned_quant_backends_as_interface_reservations() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "quant": {
                    "enabled": True,
                    "backend": "gptq",
                    "component_policies": [
                        {
                            "name": "decoder",
                            "backend": "awq",
                            "target": "model.decoder",
                        }
                    ],
                }
            }
        }
    )

    assert config.compression.quant.backend == "gptq"
    assert config.compression.quant.component_policies[0].backend == "awq"


@pytest.mark.parametrize(
    ("raw_config", "message"),
    [
        ({"config_version": 999}, "Unsupported config version"),
        ({"compression": {"axes": ["latency"]}}, "Unsupported compression axes"),
        ({"benchmark": {"warmup": -1}}, "benchmark.warmup"),
        ({"benchmark": {"iterations": 0}}, "benchmark.iterations"),
        ({"analysis": {"compare_to": "candidate"}}, "analysis.compare_to"),
        ({"analysis": {"top_k": 0}}, "analysis.top_k"),
        ({"analysis": {"metrics": []}}, "analysis.metrics"),
        (
            {"compression": {"quant": {"backend": "bad_backend"}}},
            "compression.quant.backend",
        ),
        (
            {"compression": {"quant": {"keep_high_precision": "head"}}},
            "compression.quant.keep_high_precision",
        ),
        (
            {"compression": {"quant": {"force_quantize": ["head"], "skip_quantize": ["head"]}}},
            "compression.quant.skip_quantize",
        ),
        (
            {
                "compression": {
                    "quant": {
                        "component_policies": [
                            {"name": "encoder"},
                            {"name": "encoder"},
                        ]
                    }
                }
            },
            "component_policies\\[\\*\\].name must be unique",
        ),
        (
            {
                "compression": {
                    "quant": {
                        "component_policies": [
                            {"name": "encoder", "backend": "bad_backend"},
                        ]
                    }
                }
            },
            "component_policies.0.backend",
        ),
        (
            {
                "compression": {
                    "quant": {
                        "component_policies": [
                            {
                                "name": "encoder",
                                "skip_quantize": ["head"],
                                "force_quantize": ["head"],
                            },
                        ]
                    }
                }
            },
            "component_policies.0.skip_quantize",
        ),
        (
            {"compression": {"prune": {"target_sparsity": 1.5}}},
            "target_sparsity",
        ),
        (
            {"compression": {"prune": {"enabled": True, "method": "structured"}}},
            "compression.prune.granularity",
        ),
        (
            {"compression": {"prune": {"granularity": "bad"}}},
            "compression.prune.granularity",
        ),
        (
            {"compression": {"prune": {"scope": "bad"}}},
            "compression.prune.scope",
        ),
        (
            {"compression": {"prune": {"importance": {"type": "bad"}}}},
            "compression.prune.importance.type",
        ),
        (
            {"compression": {"prune": {"selection": {"keep_indices": []}}}},
            "compression.prune.selection.keep_indices",
        ),
        (
            {"compression": {"prune": {"selection": {"keep_indices": {"blocks": []}}}}},
            "compression.prune.selection.keep_indices",
        ),
        (
            {"compression": {"prune": {"selection": {"keep_indices": {"blocks": ["0"]}}}}},
            "compression.prune.selection.keep_indices",
        ),
        (
            {"compression": {"diffusion_distill": {"teacher_steps": 0}}},
            "teacher_steps",
        ),
        (
            {"compression": {"diffusion_distill": {"student_steps": 0}}},
            "student_steps",
        ),
    ],
)
def test_load_xqt_config_rejects_invalid_values(
    raw_config: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(XQTConfigError, match=message):
        load_xqt_config(raw_config)


def test_load_xqt_config_validates_pre_export_fusion_under_export_and_quant() -> None:
    config = load_xqt_config(
        {
            "compression": {
                "quant": {
                    "policy": {
                        "pre_export_fusion": {
                            "enabled": True,
                            "mode": "fx",
                        }
                    }
                }
            },
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "params": {
                            "pre_export_fusion": {
                                "enabled": True,
                                "mode": "eager",
                                "modules_to_fuse": [["conv", "bn"]],
                            }
                        },
                    }
                ]
            },
        }
    )

    assert config.compression.quant.policy["pre_export_fusion"]["mode"] == "fx"
    assert config.export.targets[0].params["pre_export_fusion"]["mode"] == "eager"


@pytest.mark.parametrize(
    ("raw_config", "message"),
    [
        (
            {
                "compression": {
                    "quant": {
                        "policy": {
                            "pre_export_fusion": {"enabled": True, "mode": "bad"}
                        }
                    }
                }
            },
            "compression.quant.policy.pre_export_fusion.mode",
        ),
        (
            {
                "export": {
                    "targets": [
                        {
                            "format": "onnx",
                            "params": {
                                "pre_export_fusion": {
                                    "enabled": True,
                                    "mode": "eager",
                                }
                            },
                        }
                    ]
                }
            },
            "export.targets.0.params.pre_export_fusion.modules_to_fuse",
        ),
    ],
)
def test_load_xqt_config_rejects_invalid_pre_export_fusion(
    raw_config: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(XQTConfigError, match=message):
        load_xqt_config(raw_config)


def test_xqt_config_to_dict_returns_plain_mapping() -> None:
    config = load_xqt_config({"project": {"name": "plain_dict"}})

    data = xqt_config_to_dict(config)

    assert data["project"]["name"] == "plain_dict"
    assert data["compression"]["axes"] == []
    assert data["compression"]["quant"]["component_policies"] == []
