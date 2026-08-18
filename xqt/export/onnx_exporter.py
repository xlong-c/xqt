"""ONNX export and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError
from xqt.analysis.compare import TensorDiff, compare_tensors
from xqt.contracts.input_utils import (
    build_onnx_feed,
    default_input_names,
    split_example_input,
)
from xqt.export.fusion import apply_pre_export_fusion
from xqt.export.lowering import apply_pre_export_lowering
from xqt.export.base import ExportResultBase


_STANDARD_ONNX_DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml"})


@dataclass
class ONNXExportResult(ExportResultBase):
    """ONNX export metadata."""

    path: Path
    opset: Optional[int]
    checksum: str
    checked: bool
    output_diff: Optional[TensorDiff] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (self.path,)


def _legacy_dynamic_axes(
    dynamic_shapes: Mapping[str, Any] | None,
) -> dict[str, dict[int, str]] | None:
    """Translate the shared dynamic-shape mapping for the legacy ONNX exporter."""

    if not dynamic_shapes:
        return None
    dynamic_axes: dict[str, dict[int, str]] = {}
    for input_name, dimensions in dynamic_shapes.items():
        if not isinstance(dimensions, Mapping):
            raise XQTBackendError(
                "legacy ONNX dynamic_shapes entries must map axis indices to names"
            )
        axes: dict[int, str] = {}
        for axis, name in dimensions.items():
            try:
                normalized_axis = int(axis)
            except (TypeError, ValueError) as exc:
                raise XQTBackendError(
                    f"legacy ONNX dynamic shape axis for {input_name!r} must be an integer"
                ) from exc
            if normalized_axis < 0:
                raise XQTBackendError(
                    f"legacy ONNX dynamic shape axis for {input_name!r} must be non-negative"
                )
            axes[normalized_axis] = str(name)
        if not axes:
            raise XQTBackendError(
                f"legacy ONNX dynamic shape entry for {input_name!r} cannot be empty"
            )
        dynamic_axes[str(input_name)] = axes
    return dynamic_axes


def _dimension_value(dimension: Any) -> int | str | None:
    if getattr(dimension, "dim_param", ""):
        return str(dimension.dim_param)
    if callable(getattr(dimension, "HasField", None)) and dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    return None


def _value_info_signature(value_infos: Any) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for value in value_infos:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            shape: list[int | str | None] = []
        else:
            shape = [_dimension_value(dim) for dim in tensor_type.shape.dim]
        signatures.append({"name": value.name, "shape": shape})
    return signatures


def _dynamic_axes_from_signature(
    signatures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    for value in signatures:
        name = str(value.get("name", ""))
        shape = value.get("shape")
        if not isinstance(shape, list):
            continue
        for axis, dimension in enumerate(shape):
            if isinstance(dimension, str) and dimension:
                axes.append({"value_name": name, "axis": axis, "symbol": dimension})
    return axes


def onnx_graph_diagnostics_report(
    path: str | Path,
    *,
    supported_domains: Sequence[str] | None = None,
    supported_op_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Inspect an ONNX graph for op domains and materialized dynamic axes."""

    try:
        import onnx
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "onnx_not_installed",
            "unsupported_ops": [],
            "dynamic_axes": [],
        }

    model = onnx.load(str(path))
    domains = set(supported_domains or _STANDARD_ONNX_DOMAINS)
    op_types = set(supported_op_types or [])
    domain_counts: dict[str, int] = {}
    op_type_counts: dict[str, int] = {}
    unsupported_ops: list[dict[str, Any]] = []
    for node in model.graph.node:
        domain = str(node.domain or "")
        op_type = str(node.op_type or "")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        op_type_counts[op_type] = op_type_counts.get(op_type, 0) + 1
        reasons: list[str] = []
        if domain not in domains:
            reasons.append("unsupported_domain")
        if op_types and op_type not in op_types:
            reasons.append("unsupported_op_type")
        if not op_type:
            reasons.append("missing_op_type")
        if reasons:
            unsupported_ops.append(
                {
                    "name": str(node.name or ""),
                    "op_type": op_type,
                    "domain": domain,
                    "reasons": reasons,
                }
            )

    input_signature = _value_info_signature(model.graph.input)
    output_signature = _value_info_signature(model.graph.output)
    dynamic_axes = _dynamic_axes_from_signature(input_signature + output_signature)
    return {
        "status": "available",
        "op_count": len(model.graph.node),
        "domain_counts": domain_counts,
        "op_type_counts": op_type_counts,
        "unsupported_op_count": len(unsupported_ops),
        "unsupported_ops": unsupported_ops,
        "dynamic_axes": dynamic_axes,
        "dynamic_axis_count": len(dynamic_axes),
        "input_signature": input_signature,
        "output_signature": output_signature,
        "producer_name": model.producer_name,
        "ir_version": int(model.ir_version),
    }


def export_onnx(
    model: nn.Module,
    example_input: Any,
    output_path: str | Path,
    *,
    opset: Optional[int] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Mapping[str, Any]] = None,
    dynamo: bool = True,
    validate: bool = True,
    pre_export_fusion: Optional[Mapping[str, Any]] = None,
    pre_export_lowering: Optional[Mapping[str, Any]] = None,
) -> ONNXExportResult:
    """Export a PyTorch module to ONNX using the modern dynamo exporter by default."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example_spec = split_example_input(example_input)
    resolved_input_names = list(input_names or default_input_names(example_input))
    fusion_result = apply_pre_export_fusion(model, pre_export_fusion)
    lowering_result = apply_pre_export_lowering(
        fusion_result.model,
        pre_export_lowering,
    )
    export_model = lowering_result.model
    export_model.eval()

    kwargs: dict[str, Any] = {
        "dynamo": dynamo,
        "input_names": resolved_input_names,
        "output_names": list(output_names or ["output"]),
    }
    if opset is not None:
        kwargs["opset_version"] = opset
    if dynamic_shapes:
        if dynamo:
            kwargs["dynamic_shapes"] = dict(dynamic_shapes)
        else:
            kwargs["dynamic_axes"] = _legacy_dynamic_axes(dynamic_shapes)
    if example_spec.kwargs:
        kwargs["kwargs"] = dict(example_spec.kwargs)

    try:
        torch.onnx.export(export_model, example_spec.args, str(output), **kwargs)
    except TypeError:
        fallback_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"dynamo", "dynamic_shapes"}
        }
        torch.onnx.export(export_model, example_spec.args, str(output), **fallback_kwargs)

    checked = validate_onnx(output) if validate else False
    graph_diagnostics = onnx_graph_diagnostics_report(output)
    return ONNXExportResult(
        path=output,
        opset=opset,
        checksum=file_sha256(output),
        checked=checked,
        metadata={
            "dynamo": dynamo,
            "dynamic_shapes": dict(dynamic_shapes or {}),
            "input_names": resolved_input_names,
            "output_names": list(output_names or ["output"]),
            "onnx_graph_diagnostics": graph_diagnostics,
            "pre_export_fusion": dict(fusion_result.metadata),
            "pre_export_lowering": dict(lowering_result.metadata),
        },
    )


def validate_onnx(path: str | Path) -> bool:
    """Run ONNX checker on an exported model."""

    try:
        import onnx
    except ImportError as exc:
        raise XQTBackendError("onnx is required to validate ONNX exports") from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return True


def convert_onnx_to_fp16(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    keep_io_types: bool = False,
    validate: bool = True,
) -> ONNXExportResult:
    """Convert an ONNX model to FP16 using onnxconverter-common."""

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise XQTBackendError(
            "onnx and onnxconverter-common are required for ONNX FP16 conversion"
        ) from exc

    source = Path(onnx_path)
    if not source.is_file():
        raise XQTBackendError(f"ONNX file not found: {source}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
    )
    onnx.save(fp16_model, str(output))
    checked = validate_onnx(output) if validate else False
    return ONNXExportResult(
        path=output,
        opset=None,
        checksum=file_sha256(output),
        checked=checked,
        metadata={
            "precision": "fp16",
            "source_path": str(source),
            "keep_io_types": keep_io_types,
            "converter": "onnxconverter_common.float16",
        },
    )


def compare_onnxruntime_outputs(
    onnx_path: str | Path,
    reference_output: torch.Tensor,
    example_input: Any,
    *,
    input_name: str = "input",
    input_names: Optional[Sequence[str]] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TensorDiff:
    """Run ONNX Runtime and compare its first output with a PyTorch tensor."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise XQTBackendError("onnxruntime is required for ONNX Runtime diff") from exc

    resolved_input_names = list(input_names or [input_name])
    feeds = build_onnx_feed(example_input, input_names=resolved_input_names)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, feeds)[0]
    candidate = torch.from_numpy(np.asarray(ort_output))
    return compare_tensors(reference_output, candidate, atol=atol, rtol=rtol)


__all__ = [
    "ONNXExportResult",
    "compare_onnxruntime_outputs",
    "convert_onnx_to_fp16",
    "export_onnx",
    "onnx_graph_diagnostics_report",
    "validate_onnx",
]
