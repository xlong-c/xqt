from pathlib import Path

from xqt.pipeline.preflight import PreflightReport, preflight_xqt_config


def test_preflight_report_serializes_and_aggregates() -> None:
    report = PreflightReport()
    report.add("ok", True, "fine")
    report.add("bad", False, "nope", reason="missing")

    assert report.passed is False
    data = report.to_dict()
    assert data["passed"] is False
    assert data["checks"][1]["metadata"]["reason"] == "missing"


def test_preflight_checks_targets_dependencies_and_export_commands(tmp_path) -> None:
    config = {
        "project": {"artifact_dir": str(tmp_path / "artifacts")},
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
        },
        "data": {
            "validation": {
                "target": "synthetic_classification",
                "root": str(tmp_path / "missing"),
            }
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "onnxruntime_qdq",
            }
        },
        "export": {
            "targets": [
                {
                    "format": "tensorrt",
                    "params": {"trtexec_path": "definitely_missing_trtexec"},
                },
                {
                    "format": "mnn",
                    "params": {"converter_path": "definitely_missing_MNNConvert"},
                },
            ]
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["model.target"].passed is True
    assert checks["data.validation.target"].passed is True
    assert checks["data.validation.root"].passed is False
    assert checks["dependency.onnxruntime"].passed is True
    assert checks["data.calibration"].passed is True
    assert checks["data.calibration"].level == "warning"
    assert checks["data.calibration"].metadata["fallback"] == "validation"
    assert checks["export.targets.0.tensorrt.trtexec"].passed is False
    assert checks["export.targets.1.mnn.MNNConvert"].passed is False


def test_preflight_accepts_config_path(tmp_path) -> None:
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
project:
  artifact_dir: artifacts
model:
  target: torch.nn.Linear
  params:
    in_features: 4
    out_features: 2
""",
        encoding="utf-8",
    )

    report = preflight_xqt_config(Path(config_path))

    assert report.passed is True


def test_preflight_marks_torchao_fp8_as_cuda_requirement(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)

    report = preflight_xqt_config("xqt/recipes/image_vit_torchao_fp8.yaml")
    checks = {check.name: check for check in report.checks}

    assert report.passed is False
    assert checks["model.device"].passed is False
    assert checks["dependency.torchao"].passed is True
    assert checks["hardware.cuda"].passed is False
    assert checks["compression.quant.capability"].metadata["requires_cuda"] is True
    assert checks["compression.quant.capability"].metadata["primary_module_types"] == ["Linear"]


def test_preflight_accepts_cuda_device_index(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)

    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
model:
  target: torch.nn.Linear
  params:
    in_features: 4
    out_features: 2
  device: cuda:0
data:
  validation:
    target: synthetic_classification
    sample_limit: 1
    batch_size: 1
compression:
  quant:
    enabled: true
    backend: torchao
    policy:
      strategy: fp8_dynamic
      include_module_types: [Linear]
""",
        encoding="utf-8",
    )

    report = preflight_xqt_config(config_path)
    checks = {check.name: check for check in report.checks}

    assert checks["model.device"].passed is True
    assert checks["hardware.cuda"].passed is True


def test_preflight_marks_configured_calibration_for_qdq(tmp_path) -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
        },
        "data": {
            "calibration": {
                "target": "synthetic_classification",
                "sample_limit": 1,
                "batch_size": 1,
            }
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "onnxruntime_qdq",
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["data.calibration"].passed is True
    assert checks["data.calibration"].level == "info"
    assert checks["data.calibration"].metadata["source"] == "calibration"
    assert checks["data.calibration"].metadata["calibration_split"] == "calibration"
    assert checks["compression.quant.capability"].metadata["requires_calibration"] is True
    assert checks["compression.quant.capability"].metadata["requires_exportable_graph"] is True


def test_preflight_treats_prompt_list_as_builtin_target() -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
        },
        "data": {
            "prompts": {
                "target": "prompt_list",
                "sample_limit": 1,
                "params": {"prompts": ["a castle"]},
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["data.prompts.target"].passed is True
    assert checks["data.prompts.target"].message == "built-in data target"


def test_preflight_treats_xdl_dataset_as_builtin_target() -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
        },
        "data": {
            "validation": {
                "target": "xdl_dataset",
                "batch_size": 1,
                "params": {
                    "dataset": {
                        "target": "registry:SyntheticClassificationDataset",
                        "params": {
                            "num_samples": 2,
                            "input_shape": [4],
                            "num_classes": 2,
                        },
                    }
                },
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["data.validation.target"].passed is True
    assert checks["data.validation.target"].message == "built-in data target"


def test_preflight_fails_without_calibration_or_validation_for_qdq() -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "onnxruntime_qdq",
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert report.passed is False
    assert checks["data.calibration"].passed is False
    assert checks["data.calibration"].level == "error"


def test_preflight_reports_component_level_quantization_checks() -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
        },
        "data": {
            "calibration": {
                "target": "synthetic_classification",
                "sample_limit": 1,
                "batch_size": 1,
            },
            "validation": {
                "target": "synthetic_classification",
                "sample_limit": 1,
                "batch_size": 1,
            },
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "torchao",
                "component_policies": [
                    {
                        "name": "vision_encoder",
                        "target": "encoder",
                        "backend": "onnxruntime_qdq",
                        "calibration_split": "calibration",
                        "validation_split": "validation",
                    },
                    {
                        "name": "decoder",
                        "target": "decoder",
                        "backend": "torchao",
                    },
                ],
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["compression.quant.component_policies"].passed is True
    assert checks["compression.quant.component_policies"].metadata["count"] == 2
    assert checks["compression.quant.component_policies.vision_encoder.target"].passed is True
    assert checks["compression.quant.component_policies.vision_encoder.backend"].metadata["backend"] == "onnxruntime_qdq"
    assert checks["compression.quant.component_policies.vision_encoder.capability"].metadata["component"] == "vision_encoder"
    assert checks["compression.quant.component_policies.vision_encoder.capability"].metadata["requires_exportable_graph"] is True
    assert checks["compression.quant.component_policies.vision_encoder.data_source"].passed is True
    assert checks["compression.quant.component_policies.vision_encoder.data_source"].metadata["component"] == "vision_encoder"
    assert checks["compression.quant.component_policies.decoder.backend"].metadata["backend"] == "torchao"
    assert checks["compression.quant.component_policies.decoder.capability"].metadata["runtime"] == "pytorch"
    assert checks["compression.quant.runtime_mix"].passed is True
    assert checks["compression.quant.runtime_mix"].level == "warning"
    assert checks["compression.quant.runtime_mix"].metadata["backends"] == [
        "onnxruntime_qdq",
        "torchao",
    ]


def test_preflight_reports_planned_quant_backend_as_warning() -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
        },
        "compression": {
            "quant": {
                "enabled": True,
                "backend": "gptq",
                "component_policies": [
                    {
                        "name": "decoder",
                        "backend": "awq",
                        "target": "decoder",
                    }
                ],
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["compression.quant.capability"].passed is False
    assert checks["compression.quant.capability"].level == "warning"
    assert checks["compression.quant.capability"].metadata["status"] == "planned"
    assert checks["compression.quant.component_policies.decoder.capability"].passed is False
    assert checks["compression.quant.component_policies.decoder.capability"].metadata["backend"] == "awq"
    assert checks["compression.quant.component_policies.decoder.capability"].metadata["status"] == "planned"


def test_preflight_checks_hf_text_data_dependencies(monkeypatch) -> None:
    def fake_package_available(package_name: str) -> bool:
        return package_name not in {"transformers", "datasets"}

    monkeypatch.setattr(
        "xqt.pipeline.preflight._package_available",
        fake_package_available,
    )
    config = {
        "model": {
            "params": {
                "teacher_name_or_path": "teacher",
                "dataset_name": "glue",
            }
        },
        "data": {
            "train": {
                "target": "hf_text_classification",
                "sample_limit": 1,
                "batch_size": 1,
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["dependency.transformers"].passed is False
    assert checks["dependency.datasets"].passed is False


def test_preflight_reports_nm_structured_backend_capability() -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
            "device": "cpu",
        },
        "compression": {
            "prune": {
                "enabled": True,
                "method": "nm_structured",
                "selection": {"pattern": [2, 4]},
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["compression.prune.nm_backend"].passed is False
    assert checks["compression.prune.nm_backend"].level == "warning"
    assert checks["compression.prune.nm_backend"].metadata["pattern"] == [2, 4]
    assert checks["compression.prune.nm_backend"].metadata["pattern_present"] is True
    assert checks["compression.prune.nm_backend"].metadata["speedup_verified"] is False


def test_preflight_reports_block_sparse_backend_capability() -> None:
    config = {
        "model": {
            "target": "torch.nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
            "device": "cpu",
        },
        "compression": {
            "prune": {
                "enabled": True,
                "method": "block_sparse",
                "selection": {"block_shape": [2, 2]},
            }
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert checks["compression.prune.block_sparse_backend"].passed is False
    assert checks["compression.prune.block_sparse_backend"].level == "warning"
    assert checks["compression.prune.block_sparse_backend"].metadata["block_shape"] == [2, 2]
    assert checks["compression.prune.block_sparse_backend"].metadata["pattern_present"] is True
    assert checks["compression.prune.block_sparse_backend"].metadata["speedup_verified"] is False
