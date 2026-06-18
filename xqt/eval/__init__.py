"""Evaluation helpers for XQT."""

from .accuracy import EvaluationReport, evaluate_pytorch_model, topk_accuracy
from .compare import TensorDiff, TensorSummary, compare_tensors, summarize_tensor
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
    "EvaluationReport",
    "TensorDiff",
    "TensorSummary",
    "build_pareto_points",
    "compare_tensors",
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
