"""Quantization helpers for XQT."""

from .calibration import (
    ActivationDriftRecord,
    ActivationStatistic,
    analyze_activation_drift,
    calibrate_activation_statistics,
)
from .capability import (
    QuantBackendCapability,
    describe_quant_backend_capability,
    list_quant_backend_capabilities,
)
from .onnx_qdq import (
    IterableCalibrationDataReader,
    ONNXQDQQuantizationResult,
    quantize_onnx_qdq_static,
)
from .plan import build_quantization_plan
from .policy import (
    QuantizationCandidate,
    QuantizationPolicy,
    list_quantizable_modules,
    should_quantize_module,
)
from .sensitivity import (
    LayerAnalysisRecord,
    analyze_layer_errors,
    LayerSensitivityRecord,
    analyze_layer_sensitivity,
    recommend_high_precision_modules,
    suggest_high_precision_modules,
)
from .torchao_backend import TorchAOQuantizationResult, quantize_with_torchao
from .types import (
    QuantizationComponentPlan,
    QuantizationExecutionPlan,
    QuantizationExecutionResult,
    QuantizationReport,
)
from .executor import execute_quantization_plan, summarize_quantization_reports

__all__ = [
    "ActivationDriftRecord",
    "ActivationStatistic",
    "IterableCalibrationDataReader",
    "LayerAnalysisRecord",
    "LayerSensitivityRecord",
    "ONNXQDQQuantizationResult",
    "QuantBackendCapability",
    "QuantizationCandidate",
    "QuantizationComponentPlan",
    "QuantizationExecutionPlan",
    "QuantizationExecutionResult",
    "QuantizationPolicy",
    "QuantizationReport",
    "TorchAOQuantizationResult",
    "analyze_activation_drift",
    "analyze_layer_errors",
    "analyze_layer_sensitivity",
    "build_quantization_plan",
    "calibrate_activation_statistics",
    "describe_quant_backend_capability",
    "execute_quantization_plan",
    "list_quant_backend_capabilities",
    "list_quantizable_modules",
    "quantize_onnx_qdq_static",
    "quantize_with_torchao",
    "recommend_high_precision_modules",
    "should_quantize_module",
    "summarize_quantization_reports",
    "suggest_high_precision_modules",
]
