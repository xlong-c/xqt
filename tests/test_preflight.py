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
