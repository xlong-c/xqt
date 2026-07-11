"""Preflight checks for export stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xqt.export.tensorrt import validate_tensorrt_plugin_libraries

from ._base import (
    PreflightReport,
    _check_dependency,
    _check_optional_dependency,
    _check_optional_executable,
)


def _check_export_targets(
    report: PreflightReport,
    targets: list[Any],
    *,
    prefix: str = "export.targets",
) -> None:
    for index, target in enumerate(targets):
        target_format = getattr(target, "format", None)
        target_params = dict(getattr(target, "params", {}) or {})
        target_prefix = f"{prefix}.{index}.{target_format}"
        if target_format in {"torch_export", "torchscript"}:
            report.add(target_prefix, True, "built-in PyTorch export target")
        elif target_format == "onnx":
            _check_dependency(report, "onnx")
            onnx_config = getattr(target, "onnx", None)
            if onnx_config is None:
                report.add(
                    f"{target_prefix}.onnx",
                    False,
                    "ONNX export target requires typed onnx configuration",
                    level="error",
                )
                continue
            if onnx_config.runtime_diff:
                _check_dependency(report, "onnxruntime")
            optimization = onnx_config.optimization
            if optimization.enabled:
                backend = optimization.backend
                if backend == "onnxruntime":
                    _check_dependency(report, "onnxruntime")
                else:
                    report.add(
                        f"{target_prefix}.onnx_optimization",
                        False,
                        f"unsupported ONNX optimization backend: {backend}",
                        level="error",
                    )
        elif target_format == "tensorrt":
            tensorrt = target.tensorrt
            _check_optional_executable(
                report,
                tensorrt.trtexec_path,
                f"{target_prefix}.trtexec",
                dry_run=tensorrt.dry_run,
            )
            for plugin_index, plugin_path in enumerate(tensorrt.plugin_libraries):
                path = Path(plugin_path)
                plugin_validation = validate_tensorrt_plugin_libraries(
                    [path],
                    validate_loadability=tensorrt.validate_plugin_libraries_loadable,
                )
                plugin_check = plugin_validation.plugin_libraries[0]
                report.add(
                    f"{target_prefix}.plugin_libraries.{plugin_index}",
                    plugin_check.exists,
                    "TensorRT plugin library found"
                    if plugin_check.exists
                    else "TensorRT plugin library missing",
                    level="info" if plugin_check.exists else "warning",
                    path=plugin_check.path,
                    validation=plugin_check.to_dict(),
                )
                if tensorrt.validate_plugin_libraries_loadable:
                    if plugin_check.loadable is True:
                        report.add(
                            f"{target_prefix}.plugin_libraries.{plugin_index}.loadable",
                            True,
                            "TensorRT plugin library loaded with ctypes RTLD_GLOBAL",
                            level="info",
                            path=plugin_check.path,
                            loaded_plugin_libraries=plugin_check.loaded_plugin_libraries,
                            validation=plugin_check.to_dict(),
                        )
                    else:
                        report.add(
                            f"{target_prefix}.plugin_libraries.{plugin_index}.loadable",
                            False,
                            f"TensorRT plugin library failed to load: {plugin_check.error}",
                            level="error",
                            path=plugin_check.path,
                            validation=plugin_check.to_dict(),
                        )
        elif target_format == "openvino":
            openvino = target.openvino
            _check_optional_dependency(
                report,
                "openvino",
                "dependency.openvino",
                dry_run=openvino.dry_run,
            )
        elif target_format == "executorch":
            _check_optional_dependency(
                report,
                "executorch",
                "dependency.executorch",
                dry_run=target.executorch.dry_run,
            )
        elif target_format == "ncnn":
            ncnn = target.ncnn
            if ncnn.converter == "pnnx":
                _check_optional_executable(
                    report,
                    ncnn.pnnx_path,
                    f"{target_prefix}.pnnx",
                    dry_run=ncnn.dry_run,
                )
            else:
                _check_optional_executable(
                    report,
                    ncnn.onnx2ncnn_path,
                    f"{target_prefix}.onnx2ncnn",
                    dry_run=ncnn.dry_run,
                )
        elif target_format == "mnn":
            mnn = target.mnn
            _check_optional_executable(
                report,
                mnn.converter_path,
                f"{target_prefix}.MNNConvert",
                dry_run=mnn.dry_run,
            )
        else:
            report.add(target_prefix, False, "unsupported export target")
