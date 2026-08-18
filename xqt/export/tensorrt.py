"""TensorRT build adapters (facade)."""

from .trt_build import build_tensorrt_engine, build_trtexec_command
from .trt_diagnostics import (
    evaluate_tensorrt_performance_thresholds,
    inspect_tensorrt_engine,
    parse_trtexec_performance,
    summarize_tensorrt_engine_inspector,
    validate_tensorrt_plugin_libraries,
)
from .trt_runtime import (
    benchmark_tensorrt_engine,
    create_tensorrt_runtime_session,
    execute_tensorrt_engine,
    execute_tensorrt_session,
)
from .trt_types import (
    TensorRTBuildResult,
    TensorRTEngineInspectorSummary,
    TensorRTPerformanceCheck,
    TensorRTPerformanceMetrics,
    TensorRTPerformanceThresholdReport,
    TensorRTPluginLibraryCheck,
    TensorRTPluginValidationResult,
    TensorRTRuntimeBenchmarkResult,
    TensorRTRuntimeExecutionResult,
    TensorRTRuntimeSession,
)

__all__ = [
    "TensorRTBuildResult",
    "TensorRTEngineInspectorSummary",
    "TensorRTPluginLibraryCheck",
    "TensorRTPluginValidationResult",
    "TensorRTRuntimeBenchmarkResult",
    "TensorRTRuntimeExecutionResult",
    "TensorRTRuntimeSession",
    "TensorRTPerformanceCheck",
    "TensorRTPerformanceMetrics",
    "TensorRTPerformanceThresholdReport",
    "benchmark_tensorrt_engine",
    "build_tensorrt_engine",
    "build_trtexec_command",
    "create_tensorrt_runtime_session",
    "execute_tensorrt_engine",
    "execute_tensorrt_session",
    "evaluate_tensorrt_performance_thresholds",
    "inspect_tensorrt_engine",
    "parse_trtexec_performance",
    "summarize_tensorrt_engine_inspector",
    "validate_tensorrt_plugin_libraries",
]
