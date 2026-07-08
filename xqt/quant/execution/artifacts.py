"""Artifact naming helpers for quantization execution."""

from __future__ import annotations

from xqt.quant.types import QuantizationComponentPlan


def component_source_name(component: QuantizationComponentPlan) -> str:
    """Return the default ONNX source artifact name for a component."""

    if component.name == "model":
        return "quant_source.onnx"
    return f"{component.name}_source.onnx"


def component_output_name(component: QuantizationComponentPlan) -> str:
    """Return the default QDQ output artifact name for a component."""

    if component.name == "model":
        return "model_qdq.onnx"
    return f"{component.name}_qdq.onnx"


def artifact_key(prefix: str, component_name: str) -> str:
    """Return the artifact dictionary key for a component."""

    if component_name == "model":
        return prefix
    return f"{prefix}_{component_name}"


__all__ = [
    "artifact_key",
    "component_output_name",
    "component_source_name",
]
