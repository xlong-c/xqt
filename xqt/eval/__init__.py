"""Evaluation helpers for XQT."""

from .accuracy import EvaluationReport, evaluate_pytorch_model, topk_accuracy
from .compare import TensorDiff, compare_tensors
from .report import (
    flatten_metrics,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)

__all__ = [
    "EvaluationReport",
    "TensorDiff",
    "compare_tensors",
    "evaluate_pytorch_model",
    "flatten_metrics",
    "topk_accuracy",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
]
