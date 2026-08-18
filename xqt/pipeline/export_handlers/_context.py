"""Shared helpers used by export handlers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf
import torch
from torch import nn

from xqt.core.schema import ExportTargetConfig, OutputDiffConfig
from xqt.core.types import XQTContext
from xqt.contracts.input_utils import default_input_names, first_tensor_output


def call_model(model: nn.Module, inputs: Any) -> Any:
    """Call a module with mapping, tuple, or positional inputs."""

    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def resolve_export_model(context: XQTContext) -> tuple[nn.Module, dict[str, object]]:
    """Choose an export-friendly model when runtime optimization wrapped the current module."""

    model = context.require_model()
    operator_metrics = context.metrics.get("operator_optimization")
    if not isinstance(operator_metrics, dict):
        return model, {"guarded": False}

    targets = operator_metrics.get("targets")
    if not isinstance(targets, list) or not any(
        isinstance(item, Mapping) and bool(item.get("applied")) for item in targets
    ):
        return model, {"guarded": False}

    if hasattr(model, "_orig_mod") and isinstance(
        getattr(model, "_orig_mod"), nn.Module
    ):
        return getattr(model, "_orig_mod"), {
            "guarded": True,
            "reason": "compiled_runtime_unwrapped",
        }
    if isinstance(context.reference_model, nn.Module):
        return context.reference_model, {
            "guarded": True,
            "reason": "reference_model_fallback",
        }
    return model, {"guarded": False}


def update_structured_prune_export_status(
    context: XQTContext,
    exported: list[dict[str, object]],
) -> None:
    """Record whether structured pruning survived export checks."""

    prune_metrics = context.metrics.get("prune")
    if (
        not isinstance(prune_metrics, dict)
        or prune_metrics.get("method") != "structured"
    ):
        return
    checked_values = [
        bool(item.get("checked"))
        for item in exported
        if isinstance(item.get("checked"), bool)
    ]
    prune_metrics["export_status"] = {
        "attempted": bool(exported),
        "passed": all(checked_values)
        if checked_values
        else (True if exported else None),
        "artifact_count": len(exported),
        "formats": [
            str(item.get("format"))
            for item in exported
            if item.get("format") is not None
        ],
        "artifacts": [dict(item) for item in exported],
    }


def _target_summary(
    target: ExportTargetConfig,
    artifact: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format": target.format,
        "output_path": target.output_path,
        "precision": target.precision,
        "opset": target.opset,
        "dynamic_shapes": dict(target.dynamic_shapes),
        "profiles": dict(target.profiles),
        "onnx": asdict(target.onnx),
        "openvino": asdict(target.openvino),
        "tensorrt": asdict(target.tensorrt),
        "torch_export": asdict(target.torch_export),
        "torchscript": asdict(target.torchscript),
        "executorch": asdict(target.executorch),
        "ncnn": asdict(target.ncnn),
        "mnn": asdict(target.mnn),
        "params": dict(target.params),
        **dict(artifact),
    }


def _output_diff_runtime_config(
    spec: OutputDiffConfig | None,
) -> OutputDiffConfig | None:
    if spec is None:
        return None
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(OutputDiffConfig),
            OmegaConf.create(asdict(spec) if is_dataclass(spec) else dict(spec)),
        )
        return cast(OutputDiffConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ValueError(f"failed to load output diff config: {exc}") from exc
