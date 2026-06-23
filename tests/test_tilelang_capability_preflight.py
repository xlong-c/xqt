from __future__ import annotations

from unittest.mock import patch

from xqt.operator_opt.capability import describe_operator_backend_capability
from xqt.pipeline.preflight import preflight_xqt_config


def _tilelang_operator_config() -> dict:
    return {
        "config_version": 1,
        "project": {
            "name": "tilelang_capability_preflight",
            "artifact_dir": "artifacts/xqt/tests/tilelang_capability_preflight",
        },
        "model": {
            "target": "xqt.operator_opt.toy_models.build_toy_attention_classifier",
            "params": {
                "hidden_dim": 16,
                "num_heads": 4,
                "num_classes": 4,
            },
            "device": "cpu",
        },
        "operator_optimization": {
            "enabled": True,
            "default_backend": "tilelang",
            "targets": [
                {
                    "name": "model",
                    "backend": "tilelang",
                    "patterns": ["attention"],
                }
            ],
        },
    }


def test_tilelang_capability_reports_reference_fallback_when_package_missing() -> None:
    with patch("xqt.operator_opt.capability._package_available", return_value=False):
        capability = describe_operator_backend_capability("tilelang")

    assert capability.status == "available"
    assert capability.available is True
    assert any("reference fallback" in note for note in capability.notes)
    assert any("attention pattern" in limitation for limitation in capability.limitations)


def test_preflight_warns_but_does_not_fail_when_tilelang_package_missing() -> None:
    with patch("xqt.pipeline.preflight._package_available", return_value=False):
        report = preflight_xqt_config(_tilelang_operator_config())

    checks = {check.name: check for check in report.checks}
    capability_check = checks["operator_optimization.targets.0.capability"]
    runtime_check = checks["operator_optimization.targets.0.tilelang.runtime"]

    assert capability_check.passed is True
    assert runtime_check.passed is True
    assert runtime_check.level == "warning"
    assert "reference fallback" in runtime_check.message
