"""Preflight checks for deploy runtime handles."""

from __future__ import annotations

from xqt.export.tensorrt import validate_tensorrt_plugin_libraries
from xqt.workflows.stage_specs import DeployRuntimeHandleSpec

from ._base import PreflightReport, _check_dependency


def _check_deploy_runtime_handle(
    report: PreflightReport,
    handle: DeployRuntimeHandleSpec | None,
    *,
    prefix: str,
    targets: list[object],
) -> None:
    if handle is None or not handle.materialize:
        return
    runtime = handle.runtime
    supported = runtime in {"onnxruntime", "tensorrt"}
    report.add(
        f"{prefix}.runtime_handle.runtime",
        supported,
        "runtime handle is configured"
        if supported
        else "unsupported materialized runtime handle",
        level="info" if supported else "error",
        runtime=runtime,
        handle_kind=handle.handle_kind,
    )
    if runtime == "onnxruntime":
        onnx_targets = [
            target for target in targets if getattr(target, "format", None) == "onnx"
        ]
        report.add(
            f"{prefix}.runtime_handle.targets",
            len(onnx_targets) == 1,
            "runtime handle has exactly one ONNX target"
            if len(onnx_targets) == 1
            else "runtime handle requires exactly one ONNX target",
            level="info" if len(onnx_targets) == 1 else "error",
            target_count=len(onnx_targets),
        )
        _check_dependency(report, "onnxruntime")
        return
    if runtime != "tensorrt":
        return
    tensorrt_targets = [
        target for target in targets if getattr(target, "format", None) == "tensorrt"
    ]
    report.add(
        f"{prefix}.runtime_handle.targets",
        len(tensorrt_targets) == 1,
        "runtime handle has exactly one TensorRT target"
        if len(tensorrt_targets) == 1
        else "runtime handle requires exactly one TensorRT target",
        level="info" if len(tensorrt_targets) == 1 else "error",
        target_count=len(tensorrt_targets),
    )
    _check_dependency(report, "tensorrt")
    for index, plugin_path in enumerate(handle.tensorrt.plugin_libraries):
        validation = validate_tensorrt_plugin_libraries(
            [plugin_path],
            validate_loadability=True,
        )
        check = validation.plugin_libraries[0]
        report.add(
            f"{prefix}.runtime_handle.tensorrt.plugin_libraries.{index}",
            check.loadable is True,
            "TensorRT runtime plugin library loaded"
            if check.loadable is True
            else f"TensorRT runtime plugin library failed to load: {check.error}",
            level="info" if check.loadable is True else "error",
            path=check.path,
            validation=check.to_dict(),
        )
