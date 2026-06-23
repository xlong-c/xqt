"""Detection postprocess, evaluation, and decoded diff helpers for XQT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch

from xdl.metric import DetectionMeanAveragePrecision
from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.core.schema import DetectionMetricConfig, DetectionPostprocessConfig
from xqt.data.input_utils import split_batch
from xqt.export import create_tensorrt_runtime_session, execute_tensorrt_engine, execute_tensorrt_session
from xqt.export.input_utils import build_onnx_feed, default_input_names, first_tensor_output
from xqt.eval.compare import TensorDiff, compare_tensors
from xqt.integrations.detection import DetectionPrediction, decode_detection_output


@dataclass
class DetectionEvaluationReport:
    """Aggregated detection metrics."""

    samples: int
    metrics: dict[str, float]
    predictions: list[DetectionPrediction]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "metrics": dict(self.metrics),
            "prediction_count": len(self.predictions),
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class DetectionRuntimeEvaluationReport:
    """Aggregated detection metrics for a runtime artifact."""

    runtime: str
    samples: int
    metrics: dict[str, float]
    predictions: list[DetectionPrediction]
    raw_output_diff: Optional[TensorDiff] = None
    decoded_diff: Optional[DecodedDetectionDiff] = None
    latency: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "samples": self.samples,
            "metrics": dict(self.metrics),
            "prediction_count": len(self.predictions),
            "raw_output_diff": (
                self.raw_output_diff.to_dict() if self.raw_output_diff is not None else None
            ),
            "decoded_diff": (
                self.decoded_diff.to_dict() if self.decoded_diff is not None else None
            ),
            "latency": dict(self.latency or {}),
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class DecodedDetectionDiff:
    """Comparison between two decoded detection result sets."""

    box_mae: float
    score_mae: float
    label_match_rate: float
    prediction_count_reference: int
    prediction_count_candidate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_mae": self.box_mae,
            "score_mae": self.score_mae,
            "label_match_rate": self.label_match_rate,
            "prediction_count_reference": self.prediction_count_reference,
            "prediction_count_candidate": self.prediction_count_candidate,
        }


def _move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    return data


def _call_model(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        if "image" in inputs and len(inputs) == 1:
            return model(inputs["image"])
        if "image" in inputs and "images" not in inputs:
            filtered = {
                key: value
                for key, value in inputs.items()
                if key not in {"image_path", "class_names"}
            }
            if set(filtered.keys()) == {"image"}:
                return model(filtered["image"])
            return model(**filtered)
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _sanitize_runtime_inputs(inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        filtered = {
            key: value
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor) and key not in {"image_path", "class_names"}
        }
        if set(filtered.keys()) == {"image"}:
            return filtered["image"]
        return filtered
    return inputs


def _onnx_input_type_map(session: Any) -> dict[str, str]:
    return {str(item.name): str(item.type) for item in session.get_inputs()}


def _cast_onnx_feed(feed: Mapping[str, Any], input_types: Mapping[str, str]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise XQTBackendError("numpy is required for ONNX Runtime detection feeds") from exc

    dtype_map = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    casted: dict[str, Any] = {}
    for name, value in feed.items():
        target_dtype = dtype_map.get(str(input_types.get(name, "")))
        if target_dtype is None:
            casted[name] = value
            continue
        array = np.asarray(value)
        casted[name] = array.astype(target_dtype, copy=False)
    return casted


def _to_tensor_list(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim <= 1:
            return [value]
        return [item for item in value]
    if isinstance(value, list):
        return [item if isinstance(item, torch.Tensor) else torch.tensor(item) for item in value]
    raise TypeError(f"expected tensor or list of tensors, got {type(value).__name__}")


def _per_image_input_sizes(inputs: Any, batch_size: int) -> list[tuple[int, int]] | None:
    tensor: torch.Tensor | None = None
    if isinstance(inputs, torch.Tensor):
        tensor = inputs
    elif isinstance(inputs, Mapping):
        for key in ("pixel_values", "images", "image", "input"):
            candidate = inputs.get(key)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                break
    if tensor is None or tensor.ndim < 4:
        return None
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    return [(height, width) for _ in range(batch_size)]


def _normalize_detection_targets(targets: Any) -> list[dict[str, torch.Tensor]]:
    if not isinstance(targets, Mapping):
        raise TypeError("detection targets must be a mapping with boxes and labels")
    boxes_list = _to_tensor_list(targets["boxes"])
    labels_list = _to_tensor_list(targets["labels"])
    orig_sizes = None
    if "orig_size" in targets:
        orig_sizes = _to_tensor_list(targets["orig_size"])
    normalized: list[dict[str, torch.Tensor]] = []
    for index, boxes in enumerate(boxes_list):
        item: dict[str, torch.Tensor] = {
            "boxes": boxes.float(),
            "labels": labels_list[index].long(),
        }
        if orig_sizes is not None:
            item["orig_size"] = orig_sizes[index].long()
        normalized.append(item)
    return normalized


def _detection_loader_metadata(dataloader: Any) -> dict[str, Any]:
    metadata = getattr(dataloader, "xqt_detection_metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    return {str(key): value for key, value in metadata.items()}


def _image_paths_from_batch(batch: Any) -> list[str]:
    if not isinstance(batch, Mapping):
        return []
    value = batch.get("image_path")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _batch_detection_metadata(
    batch: Any,
    inputs: Any,
    targets: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "batch_size": int(len(targets)),
        "input_image_sizes": (
            [list(item) for item in _per_image_input_sizes(inputs, len(targets))] or None
        ),
        "orig_sizes": [
            [int(dim) for dim in target["orig_size"].tolist()]
            for target in targets
            if "orig_size" in target
        ],
    }
    image_paths = _image_paths_from_batch(batch)
    if image_paths:
        metadata["image_paths"] = image_paths
    return metadata


def evaluate_detection_predictions(
    predictions: Sequence[DetectionPrediction],
    targets: Sequence[Mapping[str, torch.Tensor]],
    metric_config: DetectionMetricConfig,
) -> dict[str, float]:
    """Evaluate decoded detections using COCO-style AP over configured IoU thresholds."""

    metric = DetectionMeanAveragePrecision(
        iou_thresholds=metric_config.iou_thresholds,
        box_format="xyxy",
    )
    return metric(predictions, targets)


def evaluate_detection_model(
    model: torch.nn.Module,
    dataloader: Iterable[Any],
    *,
    postprocess: DetectionPostprocessConfig,
    metric_config: DetectionMetricConfig,
    device: str | torch.device = "cpu",
    max_batches: Optional[int] = None,
) -> DetectionEvaluationReport:
    """Run detection evaluation on a dataloader."""

    torch_device = torch.device(device)
    model.to(torch_device)
    was_training = model.training
    model.eval()

    decoded_predictions: list[DetectionPrediction] = []
    decoded_targets: list[dict[str, torch.Tensor]] = []
    batch_metadata: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            split = split_batch(batch)
            if split.targets is None:
                raise ValueError("detection evaluation requires labeled targets")
            inputs = _move_to_device(split.inputs, torch_device)
            normalized_targets = _normalize_detection_targets(split.targets)
            outputs = _call_model(model, inputs)
            predictions = decode_detection_output(
                outputs,
                postprocess,
                image_sizes=_per_image_input_sizes(inputs, len(normalized_targets)),
            )
            decoded_predictions.extend(
                DetectionPrediction(
                    boxes=item.boxes.detach().cpu(),
                    scores=item.scores.detach().cpu(),
                    labels=item.labels.detach().cpu(),
                )
                for item in predictions
            )
            decoded_targets.extend(normalized_targets)
            batch_metadata.append(
                _batch_detection_metadata(batch, inputs, normalized_targets)
            )

    if was_training:
        model.train()
    metrics = evaluate_detection_predictions(
        decoded_predictions,
        decoded_targets,
        metric_config,
    )
    return DetectionEvaluationReport(
        samples=len(decoded_targets),
        metrics=metrics,
        predictions=decoded_predictions,
        metadata={
            "dataset": _detection_loader_metadata(dataloader),
            "batches": batch_metadata,
        },
    )


def evaluate_detection_runtime_model(
    model: torch.nn.Module,
    dataloader: Iterable[Any],
    *,
    postprocess: DetectionPostprocessConfig,
    metric_config: DetectionMetricConfig,
    reference_model: Optional[torch.nn.Module] = None,
    runtime: str = "pytorch",
    device: str | torch.device = "cpu",
    max_batches: Optional[int] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    benchmark_warmup: int = 0,
    benchmark_iterations: int = 1,
    benchmark_sync_cuda: bool = True,
) -> DetectionRuntimeEvaluationReport:
    """Run detection evaluation on a PyTorch runtime and optional diff checks."""

    torch_device = torch.device(device)
    model.to(torch_device)
    was_training = model.training
    model.eval()

    reference_was_training: Optional[bool] = None
    if reference_model is not None:
        reference_model.to(torch_device)
        reference_was_training = reference_model.training
        reference_model.eval()

    decoded_predictions: list[DetectionPrediction] = []
    decoded_targets: list[dict[str, torch.Tensor]] = []
    reference_predictions: list[DetectionPrediction] = []
    raw_reference_tensors: list[torch.Tensor] = []
    raw_candidate_tensors: list[torch.Tensor] = []
    benchmark_inputs: Any = None
    batch_metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            split = split_batch(batch)
            if split.targets is None:
                raise ValueError("detection runtime evaluation requires labeled targets")
            inputs = _move_to_device(split.inputs, torch_device)
            normalized_targets = _normalize_detection_targets(split.targets)
            benchmark_inputs = inputs if benchmark_inputs is None else benchmark_inputs
            outputs = _call_model(model, inputs)
            raw_candidate_tensors.append(first_tensor_output(outputs).detach().cpu())
            predictions = decode_detection_output(
                outputs,
                postprocess,
                image_sizes=_per_image_input_sizes(inputs, len(normalized_targets)),
            )
            decoded_predictions.extend(
                DetectionPrediction(
                    boxes=item.boxes.detach().cpu(),
                    scores=item.scores.detach().cpu(),
                    labels=item.labels.detach().cpu(),
                )
                for item in predictions
            )
            decoded_targets.extend(normalized_targets)
            batch_metadata.append(
                _batch_detection_metadata(batch, inputs, normalized_targets)
            )

            if reference_model is not None:
                reference_output = _call_model(reference_model, inputs)
                raw_reference_tensors.append(
                    first_tensor_output(reference_output).detach().cpu()
                )
                reference_predictions.extend(
                    DetectionPrediction(
                        boxes=item.boxes.detach().cpu(),
                        scores=item.scores.detach().cpu(),
                        labels=item.labels.detach().cpu(),
                    )
                    for item in decode_detection_output(
                        reference_output,
                        postprocess,
                        image_sizes=_per_image_input_sizes(inputs, len(normalized_targets)),
                    )
                )

    if was_training:
        model.train()
    if reference_model is not None and reference_was_training:
        reference_model.train()

    metrics = evaluate_detection_predictions(
        decoded_predictions,
        decoded_targets,
        metric_config,
    )
    raw_output_diff = None
    decoded_diff = None
    if raw_reference_tensors and raw_candidate_tensors:
        raw_output_diff = compare_tensors(
            torch.cat([tensor.reshape(-1) for tensor in raw_reference_tensors]),
            torch.cat([tensor.reshape(-1) for tensor in raw_candidate_tensors]),
            atol=atol,
            rtol=rtol,
        )
    if reference_predictions:
        decoded_diff = compare_decoded_detections(reference_predictions, decoded_predictions)

    latency = None
    if benchmark_inputs is not None:
        latency = benchmark_callable(
            lambda: _call_model(model, benchmark_inputs),
            warmup=benchmark_warmup,
            iterations=benchmark_iterations,
            sync_cuda=benchmark_sync_cuda,
            device=device,
        ).to_dict()

    return DetectionRuntimeEvaluationReport(
        runtime=runtime,
        samples=len(decoded_targets),
        metrics=metrics,
        predictions=decoded_predictions,
        raw_output_diff=raw_output_diff,
        decoded_diff=decoded_diff,
        latency=latency,
        metadata={
            "dataset": _detection_loader_metadata(dataloader),
            "batches": batch_metadata,
        },
    )


def compare_decoded_detections(
    reference: Sequence[DetectionPrediction],
    candidate: Sequence[DetectionPrediction],
) -> DecodedDetectionDiff:
    """Compare two decoded detection result sets."""

    image_count = min(len(reference), len(candidate))
    if image_count == 0:
        return DecodedDetectionDiff(
            box_mae=0.0,
            score_mae=0.0,
            label_match_rate=1.0,
            prediction_count_reference=sum(item.boxes.shape[0] for item in reference),
            prediction_count_candidate=sum(item.boxes.shape[0] for item in candidate),
        )
    box_diffs: list[torch.Tensor] = []
    score_diffs: list[torch.Tensor] = []
    label_matches: list[torch.Tensor] = []
    for ref_item, cand_item in zip(reference[:image_count], candidate[:image_count], strict=False):
        compare_count = min(ref_item.boxes.shape[0], cand_item.boxes.shape[0])
        if compare_count == 0:
            continue
        box_diffs.append(
            (ref_item.boxes[:compare_count] - cand_item.boxes[:compare_count]).abs().reshape(-1)
        )
        score_diffs.append(
            (ref_item.scores[:compare_count] - cand_item.scores[:compare_count]).abs().reshape(-1)
        )
        label_matches.append(
            (ref_item.labels[:compare_count] == cand_item.labels[:compare_count]).float()
        )
    return DecodedDetectionDiff(
        box_mae=float(torch.cat(box_diffs).mean().item()) if box_diffs else 0.0,
        score_mae=float(torch.cat(score_diffs).mean().item()) if score_diffs else 0.0,
        label_match_rate=float(torch.cat(label_matches).mean().item()) if label_matches else 1.0,
        prediction_count_reference=sum(item.boxes.shape[0] for item in reference),
        prediction_count_candidate=sum(item.boxes.shape[0] for item in candidate),
    )


def evaluate_onnx_detection_model(
    onnx_path: str,
    dataloader: Iterable[Any],
    *,
    postprocess: DetectionPostprocessConfig,
    metric_config: DetectionMetricConfig,
    input_names: Optional[Sequence[str]] = None,
    reference_model: Optional[torch.nn.Module] = None,
    device: str | torch.device = "cpu",
    max_batches: Optional[int] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    benchmark_warmup: int = 0,
    benchmark_iterations: int = 1,
    benchmark_sync_cuda: bool = True,
) -> DetectionRuntimeEvaluationReport:
    """Run ONNX Runtime detection evaluation and optional diff checks."""

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise XQTBackendError("onnxruntime is required for detection runtime evaluation") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_types = _onnx_input_type_map(session)
    torch_device = torch.device(device)
    resolved_input_names = list(input_names or [])
    decoded_predictions: list[DetectionPrediction] = []
    decoded_targets: list[dict[str, torch.Tensor]] = []
    reference_predictions: list[DetectionPrediction] = []
    raw_reference_tensors: list[torch.Tensor] = []
    raw_candidate_tensors: list[torch.Tensor] = []
    benchmark_inputs: Any = None
    batch_metadata: list[dict[str, Any]] = []

    if reference_model is not None:
        reference_model.to(torch_device)
        reference_model.eval()

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        split = split_batch(batch)
        if split.targets is None:
            raise ValueError("detection runtime evaluation requires labeled targets")
        inputs = _sanitize_runtime_inputs(split.inputs)
        normalized_targets = _normalize_detection_targets(split.targets)
        normalized_names = (
            resolved_input_names or default_input_names(inputs)
        )
        benchmark_inputs = inputs if benchmark_inputs is None else benchmark_inputs
        feed = _cast_onnx_feed(
            build_onnx_feed(inputs, input_names=normalized_names),
            input_types,
        )
        ort_outputs = session.run(
            None,
            feed,
        )
        ort_tensor = torch.from_numpy(np.asarray(ort_outputs[0]))
        raw_candidate_tensors.append(ort_tensor.detach().cpu())
        image_sizes = _per_image_input_sizes(inputs, len(normalized_targets))
        predictions = decode_detection_output(
            {"logits": torch.from_numpy(np.asarray(ort_outputs[0])), "pred_boxes": torch.from_numpy(np.asarray(ort_outputs[1]))}
            if len(ort_outputs) >= 2 and np.asarray(ort_outputs[0]).ndim == 3 and np.asarray(ort_outputs[1]).ndim == 3
            else ort_tensor,
            postprocess,
            image_sizes=image_sizes,
        )
        decoded_predictions.extend(
            DetectionPrediction(
                boxes=item.boxes.detach().cpu(),
                scores=item.scores.detach().cpu(),
                labels=item.labels.detach().cpu(),
            )
            for item in predictions
        )
        decoded_targets.extend(normalized_targets)
        batch_metadata.append(
            _batch_detection_metadata(batch, inputs, normalized_targets)
        )

        if reference_model is not None:
            moved_inputs = _move_to_device(inputs, torch_device)
            with torch.no_grad():
                reference_output = _call_model(reference_model, moved_inputs)
            raw_reference_tensors.append(first_tensor_output(reference_output).detach().cpu())
            reference_predictions.extend(
                DetectionPrediction(
                    boxes=item.boxes.detach().cpu(),
                    scores=item.scores.detach().cpu(),
                    labels=item.labels.detach().cpu(),
                )
                for item in decode_detection_output(
                    reference_output,
                    postprocess,
                    image_sizes=image_sizes,
                )
            )

    metrics = evaluate_detection_predictions(
        decoded_predictions,
        decoded_targets,
        metric_config,
    )
    raw_output_diff = None
    decoded_diff = None
    if raw_reference_tensors and raw_candidate_tensors:
        raw_output_diff = compare_tensors(
            torch.cat([tensor.reshape(-1) for tensor in raw_reference_tensors]),
            torch.cat([tensor.reshape(-1) for tensor in raw_candidate_tensors]),
            atol=atol,
            rtol=rtol,
        )
    if reference_predictions:
        decoded_diff = compare_decoded_detections(reference_predictions, decoded_predictions)

    latency = None
    if benchmark_inputs is not None:
        latency = benchmark_callable(
            lambda: session.run(
                None,
                _cast_onnx_feed(
                    build_onnx_feed(
                        benchmark_inputs,
                        input_names=resolved_input_names
                        or default_input_names(benchmark_inputs),
                    ),
                    input_types,
                ),
            ),
            warmup=benchmark_warmup,
            iterations=benchmark_iterations,
            sync_cuda=benchmark_sync_cuda,
            device="cpu",
        ).to_dict()

    return DetectionRuntimeEvaluationReport(
        runtime="onnxruntime",
        samples=len(decoded_targets),
        metrics=metrics,
        predictions=decoded_predictions,
        raw_output_diff=raw_output_diff,
        decoded_diff=decoded_diff,
        latency=latency,
        metadata={
            "dataset": _detection_loader_metadata(dataloader),
            "batches": batch_metadata,
        },
    )


def evaluate_tensorrt_detection_model(
    engine_path: str,
    dataloader: Iterable[Any],
    *,
    postprocess: DetectionPostprocessConfig,
    metric_config: DetectionMetricConfig,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    reference_model: Optional[torch.nn.Module] = None,
    device: str | torch.device = "cuda:0",
    max_batches: Optional[int] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    benchmark_warmup: int = 0,
    benchmark_iterations: int = 1,
    benchmark_sync_cuda: bool = True,
) -> DetectionRuntimeEvaluationReport:
    """Run TensorRT detection evaluation and optional diff checks."""

    torch_device = torch.device(device)
    resolved_input_names = list(input_names or [])
    resolved_output_names = [str(name) for name in output_names] if output_names else None
    decoded_predictions: list[DetectionPrediction] = []
    decoded_targets: list[dict[str, torch.Tensor]] = []
    reference_predictions: list[DetectionPrediction] = []
    raw_reference_tensors: list[torch.Tensor] = []
    raw_candidate_tensors: list[torch.Tensor] = []
    benchmark_inputs: dict[str, torch.Tensor] | None = None
    batch_metadata: list[dict[str, Any]] = []

    if reference_model is not None:
        reference_model.to(torch_device)
        reference_model.eval()

    trt_session = create_tensorrt_runtime_session(engine_path, device=str(device))

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        split = split_batch(batch)
        if split.targets is None:
            raise ValueError("detection runtime evaluation requires labeled targets")
        inputs = _sanitize_runtime_inputs(split.inputs)
        normalized_targets = _normalize_detection_targets(split.targets)
        normalized_names = resolved_input_names or default_input_names(inputs)
        tensor_inputs = {}
        if isinstance(inputs, Mapping):
            for name in normalized_names:
                value = inputs.get(name)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"TensorRT detection runtime input '{name}' must be a tensor")
                tensor_inputs[str(name)] = value
        elif isinstance(inputs, torch.Tensor):
            if len(normalized_names) != 1:
                raise TypeError("TensorRT tensor input requires exactly one input name")
            tensor_inputs[str(normalized_names[0])] = inputs
        else:
            raise TypeError("TensorRT detection runtime expects tensor or mapping inputs")
        benchmark_inputs = tensor_inputs if benchmark_inputs is None else benchmark_inputs

        execution = execute_tensorrt_session(trt_session, inputs=tensor_inputs)
        output_items = list(execution.output_tensors.items())
        if resolved_output_names is not None:
            output_items = [
                (name, execution.output_tensors[name])
                for name in resolved_output_names
                if name in execution.output_tensors
            ]
        if not output_items:
            raise XQTBackendError("TensorRT detection runtime produced no outputs")

        candidate_output: Any
        if len(output_items) >= 2:
            candidate_output = {
                "logits": output_items[0][1],
                "pred_boxes": output_items[1][1],
            }
            raw_candidate_tensors.append(output_items[0][1].detach().cpu())
        else:
            candidate_output = output_items[0][1]
            raw_candidate_tensors.append(output_items[0][1].detach().cpu())
        image_sizes = _per_image_input_sizes(inputs, len(normalized_targets))
        predictions = decode_detection_output(
            candidate_output,
            postprocess,
            image_sizes=image_sizes,
        )
        decoded_predictions.extend(
            DetectionPrediction(
                boxes=item.boxes.detach().cpu(),
                scores=item.scores.detach().cpu(),
                labels=item.labels.detach().cpu(),
            )
            for item in predictions
        )
        decoded_targets.extend(normalized_targets)
        batch_metadata.append(
            _batch_detection_metadata(batch, inputs, normalized_targets)
        )

        if reference_model is not None:
            moved_inputs = _move_to_device(inputs, torch_device)
            with torch.no_grad():
                reference_output = _call_model(reference_model, moved_inputs)
            raw_reference_tensors.append(first_tensor_output(reference_output).detach().cpu())
            reference_predictions.extend(
                DetectionPrediction(
                    boxes=item.boxes.detach().cpu(),
                    scores=item.scores.detach().cpu(),
                    labels=item.labels.detach().cpu(),
                )
                for item in decode_detection_output(
                    reference_output,
                    postprocess,
                    image_sizes=image_sizes,
                )
            )

    metrics = evaluate_detection_predictions(
        decoded_predictions,
        decoded_targets,
        metric_config,
    )
    raw_output_diff = None
    decoded_diff = None
    if raw_reference_tensors and raw_candidate_tensors:
        raw_output_diff = compare_tensors(
            torch.cat([tensor.reshape(-1) for tensor in raw_reference_tensors]),
            torch.cat([tensor.reshape(-1) for tensor in raw_candidate_tensors]),
            atol=atol,
            rtol=rtol,
        )
    if reference_predictions:
        decoded_diff = compare_decoded_detections(reference_predictions, decoded_predictions)

    latency = None
    if benchmark_inputs is not None:
        latency = benchmark_callable(
            lambda: execute_tensorrt_session(
                trt_session,
                inputs=benchmark_inputs,
            ).output_tensors,
            warmup=benchmark_warmup,
            iterations=benchmark_iterations,
            sync_cuda=benchmark_sync_cuda,
            device=str(device),
        ).to_dict()

    return DetectionRuntimeEvaluationReport(
        runtime="tensorrt",
        samples=len(decoded_targets),
        metrics=metrics,
        predictions=decoded_predictions,
        raw_output_diff=raw_output_diff,
        decoded_diff=decoded_diff,
        latency=latency,
        metadata={
            "dataset": _detection_loader_metadata(dataloader),
            "batches": batch_metadata,
        },
    )


__all__ = [
    "DecodedDetectionDiff",
    "DetectionEvaluationReport",
    "DetectionRuntimeEvaluationReport",
    "DetectionPrediction",
    "compare_decoded_detections",
    "decode_detection_output",
    "evaluate_detection_model",
    "evaluate_detection_runtime_model",
    "evaluate_onnx_detection_model",
    "evaluate_tensorrt_detection_model",
    "evaluate_detection_predictions",
]
