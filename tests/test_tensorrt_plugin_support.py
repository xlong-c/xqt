from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from xqt.export import (
    TensorRTPluginValidationResult,
    build_trtexec_command,
    validate_tensorrt_plugin_libraries,
)
from xqt.pipeline.preflight import preflight_optimization_config


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


def test_validate_tensorrt_plugin_libraries_reports_missing_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing_plugin.so"

    result = validate_tensorrt_plugin_libraries([missing])

    assert isinstance(result, TensorRTPluginValidationResult)
    assert result.status == "missing"
    assert result.passed is False
    assert result.plugin_libraries[0].exists is False
    assert "plugin library not found" in str(result.plugin_libraries[0].error)


def _build_dummy_shared_library(tmp_path: Path) -> Path:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is required to build a dummy shared library")
    source = tmp_path / "dummy_plugin.c"
    library = tmp_path / "libxqt_dummy_plugin.so"
    source.write_text(
        "int xqt_dummy_plugin_symbol(void) { return 7; }\n",
        encoding="utf-8",
    )
    subprocess.run(
        [gcc, "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


def test_validate_tensorrt_plugin_libraries_loads_real_shared_object(
    tmp_path: Path,
) -> None:
    plugin_path = _build_dummy_shared_library(tmp_path)

    result = validate_tensorrt_plugin_libraries(
        [plugin_path],
        validate_loadability=True,
    )

    assert result.status == "ok"
    assert result.passed is True
    assert result.loaded_plugin_libraries == [str(plugin_path)]
    assert result.plugin_libraries[0].loadable is True


def test_validate_tensorrt_plugin_libraries_can_only_check_presence(
    tmp_path: Path,
) -> None:
    plugin_path = tmp_path / "libcustom_plugin.so"
    plugin_path.write_bytes(b"")

    result = validate_tensorrt_plugin_libraries([plugin_path])

    assert result.status == "present"
    assert result.passed is True
    assert result.plugin_libraries[0].exists is True
    assert result.plugin_libraries[0].loadable is None


def test_preflight_reports_tensorrt_plugin_library_presence(tmp_path: Path) -> None:
    plugin_path = tmp_path / "libcustom_plugin.so"
    plugin_path.write_bytes(b"")
    config = {
        "project": {
            "name": "trt_plugin_preflight",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "model": {
            "target": "torch.nn:Identity",
        },
        "stages": [
            {
                "name": "export_tensorrt",
                "kind": "export",
                "params": {
                    "targets": [
                        {
                            "format": "tensorrt",
                            "output_path": str(tmp_path / "model.engine"),
                            "tensorrt": {
                                "dry_run": True,
                                "plugin_libraries": [str(plugin_path)],
                            },
                        }
                    ]
                },
            }
        ],
    }

    report = preflight_optimization_config(config)
    checks = {check.name: check for check in report.checks}

    check_name = "stages.0.export_tensorrt.targets.0.tensorrt.plugin_libraries.0"
    assert check_name in checks
    assert checks[check_name].passed is True


def test_preflight_can_validate_tensorrt_plugin_library_loadability(
    tmp_path: Path,
) -> None:
    plugin_path = _build_dummy_shared_library(tmp_path)
    config = {
        "project": {
            "name": "trt_plugin_preflight_loadable",
            "artifact_dir": str(tmp_path / "artifacts"),
        },
        "model": {
            "target": "torch.nn:Identity",
        },
        "stages": [
            {
                "name": "export_tensorrt",
                "kind": "export",
                "params": {
                    "targets": [
                        {
                            "format": "tensorrt",
                            "output_path": str(tmp_path / "model.engine"),
                            "tensorrt": {
                                "dry_run": True,
                                "plugin_libraries": [str(plugin_path)],
                                "validate_plugin_libraries_loadable": True,
                            },
                        }
                    ]
                },
            }
        ],
    }

    report = preflight_optimization_config(config)
    checks = {check.name: check for check in report.checks}
    loadable = checks[
        "stages.0.export_tensorrt.targets.0.tensorrt.plugin_libraries.0.loadable"
    ]

    assert loadable.passed is True
    assert loadable.metadata["loaded_plugin_libraries"] == [str(plugin_path)]
