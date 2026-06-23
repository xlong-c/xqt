"""Evaluation helpers for XQT."""

from .compare import TensorDiff, TensorSummary, compare_tensors, summarize_tensor
from .detection import (
    DecodedDetectionDiff,
    compare_decoded_detections,
)
from .layer_analysis import (
    build_avoid_list,
    build_layer_analysis_events,
    build_layer_analysis_payload,
    collect_top_layer_errors,
    layer_error_rows,
    layer_statistics_rows,
    layer_sensitivity_rows,
    top_layer_error_for_scenario,
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
    "TensorDiff",
    "TensorSummary",
    "build_pareto_points",
    "compare_decoded_detections",
    "compare_tensors",
    "flatten_metrics",
    "build_avoid_list",
    "build_layer_analysis_events",
    "build_layer_analysis_payload",
    "collect_top_layer_errors",
    "layer_error_rows",
    "layer_statistics_rows",
    "layer_sensitivity_rows",
    "records_to_dataframe",
    "records_to_rows",
    "summarize_tensor",
    "top_layer_error_for_scenario",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
]
