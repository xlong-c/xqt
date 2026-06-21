"""Evaluation helpers for XQT."""

from .accuracy import EvaluationReport, evaluate_pytorch_model, topk_accuracy
from .compare import TensorDiff, TensorSummary, compare_tensors, summarize_tensor
from .detection import (
    DecodedDetectionDiff,
    DetectionEvaluationReport,
    DetectionRuntimeEvaluationReport,
    DetectionPrediction,
    compare_decoded_detections,
    decode_detection_output,
    evaluate_detection_model,
    evaluate_detection_runtime_model,
    evaluate_onnx_detection_model,
    evaluate_detection_predictions,
)
from .report import (
    build_pareto_points,
    flatten_metrics,
    records_to_dataframe,
    records_to_rows,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)

__all__ = [
    "DecodedDetectionDiff",
    "DetectionEvaluationReport",
    "DetectionRuntimeEvaluationReport",
    "DetectionPrediction",
    "EvaluationReport",
    "TensorDiff",
    "TensorSummary",
    "build_pareto_points",
    "compare_decoded_detections",
    "compare_tensors",
    "decode_detection_output",
    "evaluate_detection_model",
    "evaluate_detection_runtime_model",
    "evaluate_onnx_detection_model",
    "evaluate_detection_predictions",
    "evaluate_pytorch_model",
    "flatten_metrics",
    "records_to_dataframe",
    "records_to_rows",
    "summarize_tensor",
    "topk_accuracy",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
]
