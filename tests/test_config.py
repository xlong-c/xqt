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
            {"compression": {"prune": {"target_sparsity": 1.5}}},
            "target_sparsity",
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
