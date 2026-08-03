"""TensorRT engine inspection and summarization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from xqt.core.errors import XQTBackendError

from .trt_plugins import _import_tensorrt, _load_tensorrt_plugin_libraries
from .trt_types import TensorRTEngineInspectorSummary


def _normalize_trt_layer_names(raw_layers: Any) -> list[str]:
    if not isinstance(raw_layers, list):
        return []
    names: list[str] = []
    for item in raw_layers:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Mapping):
            name = item.get("Name") or item.get("LayerName") or item.get("name")
            if name is not None:
                names.append(str(name))
    return names


def summarize_tensorrt_engine_inspector(raw: Mapping[str, Any]) -> TensorRTEngineInspectorSummary:
    """Summarize TensorRT engine inspector JSON into quant/fusion-oriented signals."""

    layers = _normalize_trt_layer_names(raw.get("Layers"))
    quantize_layers = [name for name in layers if "QuantizeLinear" in name]
    dequantize_layers = [name for name in layers if "DequantizeLinear" in name]
    fusion_signatures = [name for name in layers if " + " in name]
    quantized_conv_layers = [
        name
        for name in layers
        if "quantized" in name.lower() and "conv" in name.lower()
    ]
    return TensorRTEngineInspectorSummary(
        profiling_verbosity=(
            str(raw.get("ProfilingVerbosity"))
            if raw.get("ProfilingVerbosity") is not None
            else None
        ),
        layer_count=len(layers),
        layers=layers,
        io_tensors=[
            dict(item) for item in raw.get("I/O Tensors", []) if isinstance(item, Mapping)
        ],
        engine_metadata=(
            dict(raw.get("Engine Metadata", {}))
            if isinstance(raw.get("Engine Metadata"), Mapping)
            else {}
        ),
        quantize_layer_count=len(quantize_layers),
        dequantize_layer_count=len(dequantize_layers),
        quantized_conv_count=len(quantized_conv_layers),
        fused_layer_count=len(fusion_signatures),
        has_quantization_layers=bool(quantize_layers or dequantize_layers),
        has_fused_quantized_conv=bool(quantized_conv_layers),
        fusion_signatures=fusion_signatures,
    )


def inspect_tensorrt_engine(
    engine_path: str | Path,
    *,
    profiling_verbosity: Optional[str] = None,
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTEngineInspectorSummary:
    """Inspect a TensorRT engine and summarize layer-level quantization/fusion signals."""

    engine = Path(engine_path)
    if not engine.is_file():
        raise XQTBackendError(f"TensorRT engine file not found: {engine}")

    loaded_plugins = _load_tensorrt_plugin_libraries(plugin_libraries)
    trt = _import_tensorrt()
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    deserialized = runtime.deserialize_cuda_engine(engine.read_bytes())
    if deserialized is None:
        raise XQTBackendError(f"Failed to deserialize TensorRT engine: {engine}")
    inspector = deserialized.create_engine_inspector()
    if inspector is None:
        raise XQTBackendError("Failed to create TensorRT engine inspector")

    info_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        raw = json.loads(info_json)
    except json.JSONDecodeError as exc:
        raise XQTBackendError("Failed to parse TensorRT engine inspector JSON") from exc
    if profiling_verbosity is not None:
        raw["ProfilingVerbosity"] = profiling_verbosity
    if loaded_plugins:
        raw["PluginLibraries"] = loaded_plugins
    return summarize_tensorrt_engine_inspector(raw)
