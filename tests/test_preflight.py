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
    assert checks["dependency.torchao"].passed is True
    assert checks["hardware.cuda"].passed is False
