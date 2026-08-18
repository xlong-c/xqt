"""ONNX Runtime static QDQ quantization adapter."""

from __future__ import annotations

from itertools import chain
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTBackendError
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.types import XQTContext
from xqt.contracts.input_utils import build_onnx_feed, default_input_names
from ..calibration.summary import build_calibration_summary
from ..capability import _resolve_nature
from ..execution.artifacts import (
    artifact_key,
    component_output_name,
    component_source_name,
)
from ..execution.component import module_structure_name, prefix_module_names, resolve_component_model
from ..execution.selection import selection_policy_metadata
from ..types import QuantizationComponentPlan, QuantizationReport


@dataclass
class ONNXQDQQuantizationResult:
    """ONNX QDQ quantization metadata."""

    path: Path
    source_path: Path
    checksum: str
    calibration_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_numpy_inputs(
    batch: Any,
    input_names: Sequence[str],
) -> dict[str, np.ndarray]:
    return build_onnx_feed(
        extract_model_inputs(batch, expected_input_count=len(input_names)),
        input_names=input_names,
    )


class IterableCalibrationDataReader:
    """Small ONNX Runtime CalibrationDataReader for PyTorch iterables."""

    def __init__(
        self,
        dataloader: Iterable[Any],
        *,
        input_names: Sequence[str] = ("input",),
        sample_limit: Optional[int] = None,
    ) -> None:
        self.input_names = tuple(input_names)
        self.sample_limit = sample_limit
        self._records = [
            _as_numpy_inputs(batch, self.input_names)
            for index, batch in enumerate(dataloader)
            if sample_limit is None or index < sample_limit
        ]
        self._index = 0

    @property
    def samples(self) -> int:
        """Return the number of calibration batches captured."""

        return len(self._records)

    @property
    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of captured calibration inputs."""

        return build_calibration_summary(
            self._records,
            input_names=self.input_names,
            sample_limit=self.sample_limit,
            calibrator_type=type(self).__name__,
            observer_type="onnxruntime.quantization.CalibrationDataReader",
        )

    def get_next(self) -> Optional[dict[str, np.ndarray]]:
        """Return the next calibration sample for ONNX Runtime."""

        if self._index >= len(self._records):
            return None
        record = self._records[self._index]
        self._index += 1
        return record

    def rewind(self) -> None:
        """Reset iteration for tests or repeated calibration."""

        self._index = 0


def quantize_onnx_qdq_static(
    onnx_path: str | Path,
    output_path: str | Path,
    calibration_data: Iterable[Any],
    *,
    input_names: Sequence[str] = ("input",),
    sample_limit: Optional[int] = None,
    activation_type: str = "QUInt8",
    weight_type: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
    op_types_to_quantize: Optional[Sequence[str]] = None,
    extra_options: Optional[Mapping[str, Any]] = None,
) -> ONNXQDQQuantizationResult:
    """Run ONNX Runtime static QDQ quantization."""

    source = Path(onnx_path)
    if not source.is_file():
        raise XQTBackendError(f"ONNX file not found: {source}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise XQTBackendError(
            "onnxruntime.quantization is required for ONNX QDQ quantization"
        ) from exc

    quant_types = {
        "QInt8": QuantType.QInt8,
        "QUInt8": QuantType.QUInt8,
    }
    if activation_type not in quant_types:
        raise ValueError("activation_type must be QInt8 or QUInt8")
    if weight_type not in quant_types:
        raise ValueError("weight_type must be QInt8 or QUInt8")

    reader = IterableCalibrationDataReader(
        calibration_data,
        input_names=input_names,
        sample_limit=sample_limit,
    )
    if reader.samples == 0:
        raise ValueError("calibration_data must yield at least one batch")

    quantize_static(
        str(source),
        str(output),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=quant_types[activation_type],
        weight_type=quant_types[weight_type],
        per_channel=per_channel,
        reduce_range=reduce_range,
        op_types_to_quantize=list(op_types_to_quantize or []),
        extra_options=dict(extra_options or {}),
    )
    return ONNXQDQQuantizationResult(
        path=output,
        source_path=source,
        checksum=file_sha256(output),
        calibration_samples=reader.samples,
        metadata={
            "input_names": list(input_names),
            "activation_type": activation_type,
            "weight_type": weight_type,
            "per_channel": per_channel,
            "reduce_range": reduce_range,
            "op_types_to_quantize": list(op_types_to_quantize or []),
            "calibration_summary": reader.summary,
        },
    )


def resolve_calibration_inputs(
    context: XQTContext,
    component: QuantizationComponentPlan,
) -> Iterable[Any]:
    """Return required calibration inputs or raise a component-scoped error."""

    calibration_inputs = context.calibration_inputs
    if calibration_inputs is not None:
        return calibration_inputs
    raise ValueError(
        "calibration_inputs are required for component "
        f"'{component.name}' backend '{component.backend}'"
    )


def onnx_qdq_graph_summary(path: str | Path) -> dict[str, Any]:
    """Summarize QDQ graph node types when onnx is available."""

    try:
        import onnx
    except ImportError:
        return {}
    try:
        model = onnx.load(str(path))
    except Exception:
        return {}
    op_type_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1
    return {
        "node_count": len(model.graph.node),
        "op_type_counts": op_type_counts,
        "qdq_node_count": op_type_counts.get("QuantizeLinear", 0)
        + op_type_counts.get("DequantizeLinear", 0),
        "quantize_linear_count": op_type_counts.get("QuantizeLinear", 0),
        "dequantize_linear_count": op_type_counts.get("DequantizeLinear", 0),
    }


def infer_qdq_quantized_op_types(qdq_graph: dict[str, Any]) -> list[str]:
    """Infer quantized op types from a QDQ graph summary."""

    op_type_counts = qdq_graph.get("op_type_counts", {})
    if not isinstance(op_type_counts, dict):
        return []
    wrapper_op_types = {"QuantizeLinear", "DequantizeLinear", "Constant"}
    return sorted(
        str(op_type)
        for op_type in op_type_counts
        if str(op_type) not in wrapper_op_types
    )


def assess_qdq_runtime_diff(
    onnx_path: str | Path,
    *,
    run_diff: bool = False,
    example_input: Any | None = None,
    reference_model: nn.Module | None = None,
    input_names: Sequence[str] | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Probe ORT loadability and optional numeric diff for a QDQ ONNX artifact.

    Always returns a structured dict. Missing onnxruntime is recorded as
    ``runtime_diff_skip_reason``, never as a silent success.
    """

    path = Path(onnx_path)
    payload: dict[str, Any] = {
        "onnx_path": str(path),
        "onnxruntime_available": False,
        "session_created": False,
        "diff_ran": False,
        "runtime_diff_skip_reason": None,
        "diff": None,
    }
    if not path.is_file():
        payload["runtime_diff_skip_reason"] = f"onnx_file_missing:{path}"
        return payload
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        payload["runtime_diff_skip_reason"] = (
            "onnxruntime_not_installed: cannot create InferenceSession or run diff"
        )
        return payload

    payload["onnxruntime_available"] = True
    try:
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        payload["session_created"] = True
        payload["providers"] = list(session.get_providers())
    except Exception as exc:  # noqa: BLE001 - surface load failure honestly
        payload["runtime_diff_skip_reason"] = (
            f"onnxruntime_session_failed:{type(exc).__name__}:{exc}"
        )
        return payload

    if not run_diff:
        payload["runtime_diff_skip_reason"] = (
            "runtime_diff_not_requested: set policy.runtime_diff=true to compare"
        )
        return payload
    if example_input is None or reference_model is None:
        payload["runtime_diff_skip_reason"] = (
            "runtime_diff_requested_but_missing_example_input_or_reference_model"
        )
        return payload

    try:
        from xqt.export.onnx_exporter import compare_onnxruntime_outputs

        names = list(input_names or ["input"])
        model_inputs = extract_model_inputs(
            example_input,
            expected_input_count=infer_model_input_count(reference_model),
        )
        with torch.no_grad():
            if isinstance(model_inputs, (tuple, list)):
                ref = reference_model(*model_inputs)
            else:
                ref = reference_model(model_inputs)
        if isinstance(ref, (tuple, list)):
            ref_tensor = ref[0]
        else:
            ref_tensor = ref
        if not isinstance(ref_tensor, torch.Tensor):
            payload["runtime_diff_skip_reason"] = (
                f"reference_output_not_tensor:{type(ref_tensor).__name__}"
            )
            return payload
        diff = compare_onnxruntime_outputs(
            path,
            ref_tensor,
            example_input,
            input_names=names,
            atol=atol,
            rtol=rtol,
        )
        payload["diff_ran"] = True
        payload["diff"] = (
            diff.to_dict() if hasattr(diff, "to_dict") else dict(diff)
        )
        payload["runtime_diff_skip_reason"] = None
    except Exception as exc:  # noqa: BLE001
        payload["runtime_diff_skip_reason"] = (
            f"runtime_diff_failed:{type(exc).__name__}:{exc}"
        )
    return payload


def execute_onnx_qdq_component(
    context: XQTContext,
    root_model: nn.Module | None,
    component: QuantizationComponentPlan,
    *,
    export_onnx_fn: Any,
    quantize_onnx_qdq_static_fn: Any = quantize_onnx_qdq_static,
    graph_summary_fn: Any = onnx_qdq_graph_summary,
) -> tuple[nn.Module | None, QuantizationReport, dict[str, Any]]:
    """Execute ONNX Runtime static QDQ quantization for a component."""

    calibration_inputs = resolve_calibration_inputs(context, component)
    policy = dict(component.policy)
    calibration_iterator = iter(calibration_inputs)
    batch = next(calibration_iterator)
    calibration_data = chain([batch], calibration_iterator)
    artifact_dir = Path(context.artifact_dir)
    onnx_key = artifact_key("last_onnx", component.name)
    default_last_onnx = context.artifacts.get(onnx_key)
    if component.name == "model" and default_last_onnx is None:
        default_last_onnx = context.artifacts.get("last_onnx")
    onnx_path = policy.get("onnx_path") or default_last_onnx
    export_metadata: dict[str, Any] = {}
    input_names = list(policy.get("input_names") or [])
    input_structure = "external_onnx"
    if onnx_path is None:
        component_model = resolve_component_model(root_model, component.target_path)
        example_input = extract_model_inputs(
            batch,
            expected_input_count=infer_model_input_count(component_model),
        )
        if not input_names:
            input_names = default_input_names(example_input)
        input_structure = module_structure_name(example_input)
        onnx_path = artifact_dir / str(policy.get("source_name", component_source_name(component)))
        export_result = export_onnx_fn(
            component_model,
            example_input,
            onnx_path,
            opset=policy.get("opset"),
            input_names=input_names,
            output_names=policy.get("output_names"),
            dynamo=bool(policy.get("dynamo", True)),
            validate=bool(policy.get("validate", True)),
            pre_export_fusion=policy.get("pre_export_fusion"),
        )
        export_metadata = dict(export_result.metadata)
    elif not input_names:
        raise ValueError(
            "onnxruntime_qdq with external onnx_path requires policy.input_names"
        )
    output_path = policy.get("output_path")
    if output_path is None:
        output_path = str(artifact_dir / component_output_name(component))
    result = quantize_onnx_qdq_static_fn(
        onnx_path,
        output_path,
        calibration_data,
        input_names=input_names,
        sample_limit=policy.get("sample_limit"),
        activation_type=str(policy.get("activation_type", "QUInt8")),
        weight_type=str(policy.get("weight_type", "QInt8")),
        per_channel=bool(policy.get("per_channel", False)),
        reduce_range=bool(policy.get("reduce_range", False)),
        op_types_to_quantize=policy.get("op_types_to_quantize"),
        extra_options=policy.get("extra_options"),
    )
    metadata = dict(result.metadata)
    qdq_graph = graph_summary_fn(result.path)
    if qdq_graph:
        metadata["qdq_graph"] = qdq_graph
        requested_op_types = [
            str(item) for item in policy.get("op_types_to_quantize") or []
        ]
        op_type_counts = qdq_graph.get("op_type_counts", {})
        if requested_op_types:
            metadata["quantized_op_types"] = [
                op_type for op_type in requested_op_types if op_type in op_type_counts
            ]
        else:
            metadata["quantized_op_types"] = infer_qdq_quantized_op_types(qdq_graph)
        metadata["qdq_node_count"] = qdq_graph["qdq_node_count"]
    metadata.update(
        {
            "component_name": component.name,
            "calibration_source": "context.calibration_inputs",
            "input_structure": input_structure,
            "policy": policy,
            "selection_policy": selection_policy_metadata(component),
        }
    )
    if export_metadata.get("pre_export_fusion") is not None:
        metadata["pre_export_fusion"] = dict(export_metadata["pre_export_fusion"])
    runtime_diff_meta = assess_qdq_runtime_diff(
        result.path,
        run_diff=bool(policy.get("runtime_diff", False)),
        example_input=batch if root_model is not None else None,
        reference_model=root_model if bool(policy.get("runtime_diff", False)) else None,
        input_names=input_names,
    )
    metadata["runtime_diff"] = runtime_diff_meta
    nature = _resolve_nature(component.strategy, component.policy)
    method_semantics = "onnxruntime_static_qdq_graph_quantization"
    report = QuantizationReport(
        component_name=component.name,
        backend="onnxruntime_qdq",
        runtime="onnxruntime",
        method=component.method,
        strategy=component.strategy,
        target_path=component.target_path,
        quantized_modules=[
            f"onnx::{op_type}"
            for op_type in metadata.get("quantized_op_types", [])
        ],
        skipped_modules=prefix_module_names(component.skip_quantize, component.target_path),
        high_precision_modules=prefix_module_names(
            component.keep_high_precision,
            component.target_path,
        ),
        artifacts={"onnx": str(result.path)},
        calibration_samples=result.calibration_samples,
        calibration_summary=metadata.get("calibration_summary"),
        nature=nature,
        algorithm_executable=True,
        method_semantics=method_semantics,
        metadata={
            **metadata,
            "quantization_nature_scope": "qdq_graph_contract_not_runtime_observation",
            "runtime_precision_note": (
                "Q and DQ nodes encode the graph-level W/A contract. The ONNX Runtime "
                "execution provider decides whether it fuses or lowers them to a native "
                "integer kernel; this report does not claim that result."
            ),
            "path": str(result.path),
            "checksum": result.checksum,
            "analysis_only": component.analysis_only,
            "algorithm_executable": True,
            "method_semantics": method_semantics,
            "selection_policy": selection_policy_metadata(component),
        },
    )
    artifact_updates = {
        artifact_key("quant_onnx", component.name): result.path,
        artifact_key("last_onnx", component.name): result.path,
    }
    if component.name == "model":
        artifact_updates["quant_onnx"] = result.path
        artifact_updates["last_onnx"] = result.path
    return root_model, report, artifact_updates


__all__ = [
    "IterableCalibrationDataReader",
    "ONNXQDQQuantizationResult",
    "assess_qdq_runtime_diff",
    "execute_onnx_qdq_component",
    "infer_qdq_quantized_op_types",
    "onnx_qdq_graph_summary",
    "quantize_onnx_qdq_static",
    "resolve_calibration_inputs",
]
