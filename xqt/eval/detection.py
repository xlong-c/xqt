"""Detection postprocess, evaluation, and decoded diff helpers for XQT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torchvision.ops import nms

from xdl.metric.metrics import _aligned_box_iou, _boxes_to_xyxy
from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.core.schema import DetectionMetricConfig, DetectionPostprocessConfig
from xqt.data.input_utils import split_batch
from xqt.export.input_utils import build_onnx_feed, default_input_names, first_tensor_output
from xqt.eval.compare import TensorDiff, compare_tensors


@dataclass
class DetectionPrediction:
    """One image worth of decoded detections."""

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxes": self.boxes.detach().cpu().tolist(),
            "scores": self.scores.detach().cpu().tolist(),
            "labels": self.labels.detach().cpu().tolist(),
        }


@dataclass
class DetectionEvaluationReport:
    """Aggregated detection metrics."""

    samples: int
    metrics: dict[str, float]
    predictions: list[DetectionPrediction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "metrics": dict(self.metrics),
            "prediction_count": len(self.predictions),
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


def _score_activation(logits: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "identity":
        return logits
    if activation == "sigmoid":
        return logits.sigmoid()
    if activation == "softmax":
        return logits.softmax(dim=-1)
    raise ValueError(f"Unsupported score activation: {activation}")


def _apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    config: DetectionPostprocessConfig,
) -> DetectionPrediction:
    if boxes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=boxes.device)
        return DetectionPrediction(
            boxes=boxes.reshape(0, 4),
            scores=empty,
            labels=torch.empty((0,), dtype=torch.long, device=boxes.device),
        )
    if config.class_agnostic_nms:
        keep = nms(boxes, scores, config.iou_threshold)
    else:
        keep_indices: list[torch.Tensor] = []
        for label in labels.unique(sorted=True):
            class_mask = labels == label
            class_indices = torch.nonzero(class_mask, as_tuple=False).reshape(-1)
            selected = nms(boxes[class_mask], scores[class_mask], config.iou_threshold)
            keep_indices.append(class_indices[selected])
        keep = torch.cat(keep_indices, dim=0) if keep_indices else torch.empty((0,), dtype=torch.long)
        keep = keep[scores[keep].argsort(descending=True)]
    keep = keep[: config.max_detections]
    return DetectionPrediction(
        boxes=boxes[keep],
        scores=scores[keep],
        labels=labels[keep],
    )


def decode_detection_output(
    output: Any,
    config: DetectionPostprocessConfig,
) -> list[DetectionPrediction]:
    """Decode model outputs into per-image detections."""

    if isinstance(output, Mapping):
        if {"boxes", "scores", "labels"} <= set(output.keys()):
            return [
                DetectionPrediction(
                    boxes=output["boxes"].float(),
                    scores=output["scores"].float(),
                    labels=output["labels"].long(),
                )
            ]
        if "logits" in output:
            output = output["logits"]

    if isinstance(output, (tuple, list)):
        if len(output) == 1:
            output = output[0]
        elif output and isinstance(output[0], torch.Tensor):
            output = output[0]

    if not isinstance(output, torch.Tensor):
        raise TypeError("detection output must resolve to a tensor or mapping")
    if output.ndim == 2 and output.shape[-1] == 6:
        output = output.unsqueeze(0)
    if output.ndim != 3:
        raise TypeError("detection output tensor must have shape [B, N, 6] or [B, C, N]")

    predictions: list[DetectionPrediction] = []
    if config.format == "end2end" or (config.format == "auto" and output.shape[-1] == 6):
        for image_output in output:
            boxes = image_output[:, :4].float()
            scores = image_output[:, 4].float()
            labels = image_output[:, 5].long()
            keep = scores >= config.score_threshold
            predictions.append(
                _apply_nms(boxes[keep], scores[keep], labels[keep], config)
            )
        return predictions

    if config.format not in {"auto", "yolo_raw"}:
        raise ValueError(f"Unsupported detection postprocess format: {config.format}")

    if output.shape[1] < 5:
        raise ValueError("yolo_raw output must have at least 5 channels")

    batch_predictions = output.permute(0, 2, 1).contiguous()
    for image_output in batch_predictions:
        boxes = _boxes_to_xyxy(image_output[:, :4].float(), config.box_format)
        class_logits = image_output[:, 4:].float()
        if config.has_objectness:
            objectness = class_logits[:, :1].sigmoid()
            class_scores = _score_activation(class_logits[:, 1:], config.score_activation)
            scores_per_class = objectness * class_scores
        else:
            scores_per_class = _score_activation(class_logits, config.score_activation)
        scores, labels = scores_per_class.max(dim=-1)
        keep = scores >= config.score_threshold
        predictions.append(
            _apply_nms(boxes[keep], scores[keep], labels[keep], config)
        )
    return predictions


def _build_tp_flags(
    prediction: DetectionPrediction,
    target: Mapping[str, torch.Tensor],
    iou_threshold: float,
) -> tuple[torch.Tensor, int]:
    if prediction.boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool), int(target["boxes"].shape[0])
    target_boxes = target["boxes"].float()
    target_labels = target["labels"].long()
    matched_targets = torch.zeros((target_boxes.shape[0],), dtype=torch.bool)
    tp = torch.zeros((prediction.boxes.shape[0],), dtype=torch.bool)
    order = prediction.scores.argsort(descending=True)
    sorted_boxes = prediction.boxes[order]
    sorted_labels = prediction.labels[order]
    for pred_position, (box, label) in enumerate(zip(sorted_boxes, sorted_labels, strict=False)):
        label_mask = target_labels == label
        if not bool(label_mask.any()):
            continue
        candidate_indices = torch.nonzero(label_mask, as_tuple=False).reshape(-1)
        ious = _aligned_box_iou(
            box.unsqueeze(0).expand(candidate_indices.shape[0], -1),
            target_boxes[candidate_indices],
            "xyxy",
            1e-7,
        )
        best_iou, best_index = ious.max(dim=0)
        candidate_index = int(candidate_indices[int(best_index.item())].item())
        if float(best_iou.item()) >= iou_threshold and not bool(matched_targets[candidate_index]):
            tp[pred_position] = True
            matched_targets[candidate_index] = True
    reordered = torch.zeros_like(tp)
    reordered[order] = tp
    return reordered, int(target_boxes.shape[0])


def _average_precision(
    predictions: Sequence[DetectionPrediction],
    targets: Sequence[Mapping[str, torch.Tensor]],
    iou_threshold: float,
) -> float:
    all_scores: list[torch.Tensor] = []
    all_tp: list[torch.Tensor] = []
    total_gt = 0
    for prediction, target in zip(predictions, targets, strict=False):
        tp_flags, gt_count = _build_tp_flags(prediction, target, iou_threshold)
        all_scores.append(prediction.scores.detach().cpu())
        all_tp.append(tp_flags.detach().cpu())
        total_gt += gt_count
    if total_gt == 0:
        return 0.0
    if not all_scores:
        return 0.0
    scores = torch.cat(all_scores) if all_scores else torch.empty((0,))
    tp = torch.cat(all_tp) if all_tp else torch.empty((0,), dtype=torch.bool)
    if scores.numel() == 0:
        return 0.0
    order = scores.argsort(descending=True)
    tp_sorted = tp[order].float()
    fp_sorted = 1.0 - tp_sorted
    tp_cum = tp_sorted.cumsum(dim=0)
    fp_cum = fp_sorted.cumsum(dim=0)
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)
    precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    for index in range(precision.numel() - 1, 0, -1):
        precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
    delta = recall[1:] - recall[:-1]
    return float((delta * precision[1:]).sum().item())


def evaluate_detection_predictions(
    predictions: Sequence[DetectionPrediction],
    targets: Sequence[Mapping[str, torch.Tensor]],
    metric_config: DetectionMetricConfig,
) -> dict[str, float]:
    """Evaluate decoded detections using COCO-style AP over configured IoU thresholds."""

    aps = [
        _average_precision(predictions, targets, iou_threshold=float(threshold))
        for threshold in metric_config.iou_thresholds
    ]
    metric_map = {
        "map50_95": float(sum(aps) / len(aps)) if aps else 0.0,
        "map50": 0.0,
        "map75": 0.0,
    }
    for threshold, ap in zip(metric_config.iou_thresholds, aps, strict=False):
        if abs(float(threshold) - 0.5) < 1e-6:
            metric_map["map50"] = float(ap)
        if abs(float(threshold) - 0.75) < 1e-6:
            metric_map["map75"] = float(ap)
    return metric_map


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
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            split = split_batch(batch)
            if split.targets is None:
                raise ValueError("detection evaluation requires labeled targets")
            inputs = _move_to_device(split.inputs, torch_device)
            outputs = _call_model(model, inputs)
            predictions = decode_detection_output(outputs, postprocess)
            decoded_predictions.extend(
                DetectionPrediction(
                    boxes=item.boxes.detach().cpu(),
                    scores=item.scores.detach().cpu(),
                    labels=item.labels.detach().cpu(),
                )
                for item in predictions
            )
            decoded_targets.extend(_normalize_detection_targets(split.targets))

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

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            split = split_batch(batch)
            if split.targets is None:
                raise ValueError("detection runtime evaluation requires labeled targets")
            inputs = _move_to_device(split.inputs, torch_device)
            benchmark_inputs = inputs if benchmark_inputs is None else benchmark_inputs
            outputs = _call_model(model, inputs)
            raw_candidate_tensors.append(first_tensor_output(outputs).detach().cpu())
            predictions = decode_detection_output(outputs, postprocess)
            decoded_predictions.extend(
                DetectionPrediction(
                    boxes=item.boxes.detach().cpu(),
                    scores=item.scores.detach().cpu(),
                    labels=item.labels.detach().cpu(),
                )
                for item in predictions
            )
            decoded_targets.extend(_normalize_detection_targets(split.targets))

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
                    for item in decode_detection_output(reference_output, postprocess)
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
        predictions = decode_detection_output(ort_tensor, postprocess)
        decoded_predictions.extend(
            DetectionPrediction(
                boxes=item.boxes.detach().cpu(),
                scores=item.scores.detach().cpu(),
                labels=item.labels.detach().cpu(),
            )
            for item in predictions
        )
        decoded_targets.extend(_normalize_detection_targets(split.targets))

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
                for item in decode_detection_output(reference_output, postprocess)
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
    "evaluate_detection_predictions",
]
