"""Internal helpers for session-side export and deploy target requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


_TYPED_TARGET_CONFIG_NAMES = (
    "onnx",
    "openvino",
    "tensorrt",
    "torch_export",
    "torchscript",
    "executorch",
    "ncnn",
    "mnn",
    "qnn",
)


def _copy_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if mapping is None:
        return None
    return dict(mapping)


def build_session_export_targets(
    operation: str,
    *,
    format: str | None,
    output_path: str | Path | None,
    targets: list[Mapping[str, Any]] | None,
    target_params: Mapping[str, Any] | None,
    opset: int | None,
    inference: Mapping[str, Any] | None = None,
    onnx: Mapping[str, Any] | None = None,
    openvino: Mapping[str, Any] | None = None,
    tensorrt: Mapping[str, Any] | None = None,
    torch_export: Mapping[str, Any] | None = None,
    torchscript: Mapping[str, Any] | None = None,
    executorch: Mapping[str, Any] | None = None,
    ncnn: Mapping[str, Any] | None = None,
    mnn: Mapping[str, Any] | None = None,
    qnn: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize one session export/deploy request into workflow targets."""

    typed_target_configs = {
        "onnx": _copy_mapping(onnx),
        "openvino": _copy_mapping(openvino),
        "tensorrt": _copy_mapping(tensorrt),
        "torch_export": _copy_mapping(torch_export),
        "torchscript": _copy_mapping(torchscript),
        "executorch": _copy_mapping(executorch),
        "ncnn": _copy_mapping(ncnn),
        "mnn": _copy_mapping(mnn),
        "qnn": _copy_mapping(qnn),
    }
    selected_typed_configs = [
        name
        for name in _TYPED_TARGET_CONFIG_NAMES
        if typed_target_configs[name] is not None
    ]
    if targets is not None and (selected_typed_configs or inference is not None):
        raise ValueError(
            f"{operation} accepts typed target config and inference only with "
            "format + output_path"
        )
    if len(selected_typed_configs) > 1:
        raise ValueError(f"{operation} accepts only one typed target config")
    if targets is not None:
        if not targets:
            raise ValueError(f"{operation} requires at least one target")
        return [dict(target) for target in targets]
    if format is None or output_path is None:
        raise ValueError(f"{operation} requires targets or format + output_path")

    target: dict[str, Any] = {
        "format": format,
        "output_path": str(output_path),
    }
    if opset is not None:
        target["opset"] = opset
    if inference is not None:
        target["inference"] = dict(inference)
    if target_params is not None:
        target["params"] = dict(target_params)
    for name in _TYPED_TARGET_CONFIG_NAMES:
        typed_config = typed_target_configs[name]
        if typed_config is not None:
            target[name] = typed_config
    return [target]


__all__ = ["build_session_export_targets"]
