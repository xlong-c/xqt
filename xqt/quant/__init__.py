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
    CompositeQuantBranchArtifact,
    CompositeQuantizationArtifact,
    QuantizationComponentPlan,
    QuantizationExecutionPlan,
    QuantizationExecutionResult,
    QuantizationNature,
    QuantizationReport,
    build_composite_quantization_artifact,
)
from .execution import execute_quantization_plan, summarize_quantization_reports
from .quantizers import Quantizer, QuantizerOptions, QuantizerResult
from .quantizers.fake_qdq import FakeQDQSurrogateResult, build_fake_qdq_surrogate
from .quantizers.fp4_weight_only import (
    FP4QuantizationResult,
    FP4WeightOnlyLinear,
    quantize_with_awq_fp4,
    quantize_with_fp4_weight_only,
    quantize_with_gptq_fp4,
)
from .quantizers.awq_gptq_weight_only import (
    AWQGPTQWeightOnlyLinear,
    AWQGPTQWeightOnlyQuantizationResult,
    quantize_with_awq_weight_only,
    quantize_with_gptq_weight_only,
)
from .quantizers.int8_mma import (
    Int8MmaLinear,
    Int8MmaQuantizationResult,
    quantize_with_int8_mma,
)
from .quantizers.w4_storage_int8_mma import (
    W4StorageInt8MmaLinear,
    W4StorageInt8MmaQuantizationResult,
    quantize_with_w4_storage_int8_mma,
)
from .quantizers.convrot_4bit import (
    ConvRot4BitQuantizationResult,
    ConvRotMixedPrecisionLinear,
    build_regular_hadamard_matrix,
    quantize_with_convrot_4bit,
)
from .quantizers.mxfp_weight_only import (
    MXFPQuantizationResult,
    MXFPWeightOnlyLinear,
    quantize_with_mxfp_weight_only,
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
    "AWQGPTQWeightOnlyLinear",
    "AWQGPTQWeightOnlyQuantizationResult",
    "FakeQDQSurrogateResult",
    "FP4QuantizationResult",
    "IterableCalibrationDataReader",
    "ConvRot4BitQuantizationResult",
    "ConvRotMixedPrecisionLinear",
    "CompositeQuantBranchArtifact",
    "CompositeQuantizationArtifact",
    "Int8MmaLinear",
    "Int8MmaQuantizationResult",
    "LayerAnalysisRecord",
    "LayerSensitivityRecord",
    "LowRankBranch",
    "MXFPQuantizationResult",
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
    "FP4WeightOnlyLinear",
    "MXFPWeightOnlyLinear",
    "SVDQuantLinear",
    "SVDQuantResult",
    "TorchAOQuantizationResult",
    "W4StorageInt8MmaLinear",
    "W4StorageInt8MmaQuantizationResult",
    "analyze_activation_drift",
    "analyze_layer_errors",
    "analyze_layer_sensitivity",
    "build_quantization_plan",
    "build_composite_quantization_artifact",
    "build_regular_hadamard_matrix",
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
    "quantize_with_awq_fp4",
    "quantize_with_awq_weight_only",
    "quantize_with_convrot_4bit",
    "quantize_with_int8_mma",
    "quantize_onnx_qdq_static",
    "quantize_with_mxfp_weight_only",
    "quantize_with_fp4_weight_only",
    "quantize_with_gptq_fp4",
    "quantize_with_gptq_weight_only",
    "quantize_with_svd",
    "quantize_with_torchao",
    "quantize_with_w4_storage_int8_mma",
    "recommend_high_precision_modules",
    "should_quantize_module",
    "summarize_quantization_reports",
    "suggest_high_precision_modules",
    "unpack_nvfp4e2m1",
]
