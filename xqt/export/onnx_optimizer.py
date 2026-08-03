"""ONNX graph optimization helpers for exported artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError

from .onnx_exporter import validate_onnx


_ORT_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "disabled": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


@dataclass(frozen=True)
class ONNXNativeQDQOptimizationResult:
    """Metadata for XQT-native safe QDQ graph cleanup."""

    removed_nodes: int
    removed_quantize_nodes: int
    removed_dequantize_nodes: int
    rewired_edges: int
    remaining_nodes: int

    def to_dict(self) -> dict[str, int]:
        """Return a manifest-safe mapping."""

        return {
            "removed_nodes": self.removed_nodes,
            "removed_quantize_nodes": self.removed_quantize_nodes,
            "removed_dequantize_nodes": self.removed_dequantize_nodes,
            "rewired_edges": self.rewired_edges,
            "remaining_nodes": self.remaining_nodes,
        }


@dataclass(frozen=True)
class ONNXOptimizationResult:
    """Metadata for an optimized ONNX artifact."""

    path: Path
    source_path: Path
    backend: str
    level: str
    checksum: str
    checked: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-safe mapping."""

        return {
            "path": str(self.path),
            "source_path": str(self.source_path),
            "backend": self.backend,
            "level": self.level,
            "checksum": self.checksum,
            "checked": self.checked,
            **dict(self.metadata),
        }


def _default_optimized_path(source: Path, suffix: str) -> Path:
    return source.with_name(f"{source.stem}{suffix}{source.suffix}")


def _node_input_counts(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                counts[name] = counts.get(name, 0) + 1
    for output in model.graph.output:
        counts[output.name] = counts.get(output.name, 0) + 1
    return counts


def _build_producer_map(model: Any) -> dict[str, Any]:
    producers: dict[str, Any] = {}
    for node in model.graph.node:
        for output in node.output:
            if output:
                producers[output] = node
    return producers


def _same_qdq_params(quantize_node: Any, dequantize_node: Any) -> bool:
    if len(quantize_node.input) < 3 or len(dequantize_node.input) < 3:
        return False
    return (
        quantize_node.input[1] == dequantize_node.input[1]
        and quantize_node.input[2] == dequantize_node.input[2]
    )


def _replace_node_input(model: Any, old_name: str, new_name: str) -> int:
    rewired = 0
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name
                rewired += 1
    for output in model.graph.output:
        if output.name == old_name:
            output.name = new_name
            rewired += 1
    return rewired


def _remove_nodes(model: Any, nodes_to_remove: set[int]) -> None:
    kept = [
        node
        for index, node in enumerate(model.graph.node)
        if index not in nodes_to_remove
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def optimize_qdq_native(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    *,
    validate: bool = True,
) -> ONNXNativeQDQOptimizationResult:
    """Remove only provably redundant QDQ round trips from an ONNX graph.

    The pass is intentionally conservative. It rewrites
    QuantizeLinear(DequantizeLinear(q, s, zp), s, zp) back to q only when the
    dequantized tensor has a single consumer and the scale/zero-point inputs
    match. It deliberately does not remove DequantizeLinear(QuantizeLinear(x)),
    because that fake-quant boundary changes floating-point values.
    """

    source = Path(onnx_path)
    if not source.is_file():
        raise XQTBackendError(f"ONNX file not found: {source}")
    output = Path(output_path) if output_path is not None else source

    try:
        import onnx
    except ImportError as exc:
        raise XQTBackendError("onnx is required for native QDQ optimization") from exc

    model = onnx.load(str(source))
    producers = _build_producer_map(model)
    input_counts = _node_input_counts(model)
    nodes_to_remove: set[int] = set()
    rewired_edges = 0
    removed_quantize_nodes = 0
    removed_dequantize_nodes = 0

    for index, node in enumerate(model.graph.node):
        if node.op_type != "QuantizeLinear" or len(node.input) < 1:
            continue
        dequantized_input = node.input[0]
        producer = producers.get(dequantized_input)
        if producer is None or producer.op_type != "DequantizeLinear":
            continue
        if not _same_qdq_params(producer, node):
            continue
        if input_counts.get(dequantized_input, 0) != 1:
            continue
        producer_index = next(
            candidate_index
            for candidate_index, candidate in enumerate(model.graph.node)
            if candidate is producer
        )
        if producer_index in nodes_to_remove:
            continue
        if len(node.output) != 1 or len(producer.input) < 1:
            continue
        rewired_edges += _replace_node_input(model, node.output[0], producer.input[0])
        nodes_to_remove.add(index)
        nodes_to_remove.add(producer_index)
        removed_quantize_nodes += 1
        removed_dequantize_nodes += 1

    _remove_nodes(model, nodes_to_remove)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    checked = validate_onnx(output) if validate else True
    if not checked:
        raise XQTBackendError(f"native QDQ optimized ONNX validation failed: {output}")

    return ONNXNativeQDQOptimizationResult(
        removed_nodes=len(nodes_to_remove),
        removed_quantize_nodes=removed_quantize_nodes,
        removed_dequantize_nodes=removed_dequantize_nodes,
        rewired_edges=rewired_edges,
        remaining_nodes=len(model.graph.node),
    )


def _optimize_with_onnxruntime(
    source: Path,
    output: Path,
    *,
    level: str,
    providers: list[str],
) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise XQTBackendError(
            "onnxruntime is required for ONNX graph optimization"
        ) from exc

    normalized_level = level.lower()
    if normalized_level not in _ORT_LEVELS:
        valid = ", ".join(sorted(_ORT_LEVELS))
        raise ValueError(f"unsupported onnxruntime optimization level: {level}; valid: {valid}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel,
        _ORT_LEVELS[normalized_level],
    )
    session_options.optimized_model_filepath = str(output)
    ort.InferenceSession(str(source), session_options, providers=providers)


def optimize_onnx(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    *,
    backend: str = "onnxruntime",
    level: str = "extended",
    output_suffix: str = ".optimized",
    validate: bool = True,
    providers: list[str] | None = None,
    native_qdq: bool | Mapping[str, Any] = True,
    metadata: Mapping[str, Any] | None = None,
) -> ONNXOptimizationResult:
    """Optimize an ONNX artifact without changing the owning model object."""

    source = Path(onnx_path)
    if not source.is_file():
        raise XQTBackendError(f"ONNX file not found: {source}")

    output = Path(output_path) if output_path is not None else _default_optimized_path(
        source,
        output_suffix,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    normalized_backend = backend.lower()
    if normalized_backend == "onnxruntime":
        _optimize_with_onnxruntime(
            source,
            output,
            level=level,
            providers=list(providers or ["CPUExecutionProvider"]),
        )
    else:
        raise ValueError(f"unsupported ONNX optimization backend: {backend}")

    native_qdq_metadata: dict[str, Any] | None = None
    if native_qdq:
        native_qdq_config = native_qdq if isinstance(native_qdq, Mapping) else {}
        native_qdq_result = optimize_qdq_native(
            output,
            output,
            validate=bool(native_qdq_config.get("validate", validate)),
        )
        native_qdq_metadata = native_qdq_result.to_dict()

    checked = validate_onnx(output) if validate else False
    return ONNXOptimizationResult(
        path=output,
        source_path=source,
        backend=normalized_backend,
        level=level.lower(),
        checksum=file_sha256(output),
        checked=checked,
        metadata={
            "output_suffix": output_suffix,
            "providers": list(providers or ["CPUExecutionProvider"]),
            "native_qdq_optimization": native_qdq_metadata,
            **dict(metadata or {}),
        },
    )


__all__ = [
    "ONNXNativeQDQOptimizationResult",
    "ONNXOptimizationResult",
    "optimize_qdq_native",
    "optimize_onnx",
]
