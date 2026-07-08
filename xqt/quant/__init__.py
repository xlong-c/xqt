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
from .backends.onnx_qdq import (
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
from .backends.torchao import TorchAOQuantizationResult, quantize_with_torchao
from .types import (
    QuantizationComponentPlan,
    QuantizationExecutionPlan,
    QuantizationExecutionResult,
    QuantizationNature,
    QuantizationReport,
)
from .execution import execute_quantization_plan, summarize_quantization_reports
from .quantizers import Quantizer, QuantizerOptions, QuantizerResult
from .quantizers.fake_qdq import FakeQDQSurrogateResult, build_fake_qdq_surrogate
from .quantizers.reference_fp4 import (
    FP4QuantizationResult,
    ReferenceFP4Linear,
    quantize_with_reference_fp4,
)
from .bridges.nvfp4 import (
    NVFP4LinearBridge,
    NVFP4TensorLayout,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    expand_group_scale,
    infer_nvfp4_tensor_layout,
    unpack_nvfp4e2m1,
)
from .quantizers.svd import (
    LowRankBranch,
    SVDQuantLinear,
    SVDQuantResult,
    quantize_with_svd,
)

__all__ = [
    "ActivationDriftRecord",
    "ActivationStatistic",
    "FakeQDQSurrogateResult",
    "FP4QuantizationResult",
    "IterableCalibrationDataReader",
    "LayerAnalysisRecord",
    "LayerSensitivityRecord",
    "LowRankBranch",
    "NVFP4LinearBridge",
    "NVFP4TensorLayout",
    "ONNXQDQQuantizationResult",
    "Quantizer",
    "QuantizerOptions",
    "QuantizerResult",
    "QuantBackendCapability",
    "QuantizationCandidate",
    "QuantizationComponentPlan",
    "QuantizationExecutionPlan",
    "QuantizationExecutionResult",
    "QuantizationNature",
    "QuantizationPolicy",
    "QuantizationReport",
    "ReferenceFP4Linear",
    "SVDQuantLinear",
    "SVDQuantResult",
    "TorchAOQuantizationResult",
    "analyze_activation_drift",
    "analyze_layer_errors",
    "analyze_layer_sensitivity",
    "build_quantization_plan",
    "build_fake_qdq_surrogate",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "calibrate_activation_statistics",
    "describe_quant_backend_capability",
    "execute_quantization_plan",
    "expand_group_scale",
    "infer_nvfp4_tensor_layout",
    "list_quant_backend_capabilities",
    "list_quantizable_modules",
    "quantize_onnx_qdq_static",
    "quantize_with_reference_fp4",
    "quantize_with_svd",
    "quantize_with_torchao",
    "recommend_high_precision_modules",
    "should_quantize_module",
    "summarize_quantization_reports",
    "suggest_high_precision_modules",
    "unpack_nvfp4e2m1",
]
