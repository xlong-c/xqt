from __future__ import annotations

from pathlib import Path

import pytest

from xqt.export.tensorrt import (
    _load_tensorrt_plugin_libraries,
    build_trtexec_command,
)
from xqt.pipeline.preflight import preflight_xqt_config


def test_build_trtexec_command_includes_plugin_libraries() -> None:
    command = build_trtexec_command(
        "model.onnx",
        "model.engine",
        plugin_libraries=["plugins/libcustom_a.so", "plugins/libcustom_b.so"],
    )

    assert "--dynamicPlugins=plugins/libcustom_a.so" in command
    assert "--dynamicPlugins=plugins/libcustom_b.so" in command
    assert "--setPluginsToSerialize=plugins/libcustom_a.so" in command
    assert "--setPluginsToSerialize=plugins/libcustom_b.so" in command


def test_load_tensorrt_plugin_libraries_requires_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing_plugin.so"

    with pytest.raises(Exception, match="plugin library not found"):
        _load_tensorrt_plugin_libraries([missing])


def test_preflight_reports_tensorrt_plugin_library_presence(tmp_path: Path) -> None:
    plugin_path = tmp_path / "libcustom_plugin.so"
    plugin_path.write_bytes(b"")
    config = {
        "config_version": 1,
        "project": {
            "name": "trt_plugin_preflight",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "model": {
            "target": "torch.nn:Identity",
        },
        "export": {
            "targets": [
                {
                    "format": "tensorrt",
                    "output_path": str(tmp_path / "model.engine"),
                    "params": {
                        "dry_run": True,
                        "plugin_libraries": [str(plugin_path)],
                    },
                }
            ]
        },
    }

    report = preflight_xqt_config(config)
    checks = {check.name: check for check in report.checks}

    assert f"export.targets.0.tensorrt.plugin_libraries.0" in checks
    assert checks["export.targets.0.tensorrt.plugin_libraries.0"].passed is True
