"""Evaluation helpers for XQT."""

from .compare import TensorDiff, TensorSummary, compare_tensors, summarize_tensor
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
from .svd_analysis import (
    SVDDecomposition,
    SVDQuantAnalysis,
    compute_residual_weight,
    decompose_weight_svd,
)
from .visualization import (
    TensorSelection,
    TensorPlotData,
    plot_model_tensor_selections_bar3d,
    plot_tensor_bar3d,
    plot_tensor_bar3d_panels,
    prepare_tensor_plot_data,
)

__all__ = [
    "DecodedDetectionDiff",
    "SVDDecomposition",
    "SVDQuantAnalysis",
    "TensorDiff",
    "TensorSelection",
    "TensorPlotData",
    "TensorSummary",
    "build_pareto_points",
    "compare_decoded_detections",
    "compare_tensors",
    "compute_residual_weight",
    "decompose_weight_svd",
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
    "plot_model_tensor_selections_bar3d",
    "plot_tensor_bar3d",
    "plot_tensor_bar3d_panels",
    "prepare_tensor_plot_data",
    "summarize_tensor",
    "top_layer_error_for_scenario",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
]

# detection diff is xqt-local (xqt.analysis.detection.DecodedDetectionDiff), no xdl dependency.
_LAZY = {"DecodedDetectionDiff", "compare_decoded_detections"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import detection

        return getattr(detection, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
