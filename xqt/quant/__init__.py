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
from .quantizers.convrot_int8 import (
    ConvRotInt8Linear,
    ConvRotInt8QuantizationResult,
    quantize_with_convrot_int8,
)
from .comfy_quant import (
    DEFAULT_CONVROT_GROUP_SIZE,
    STOCK_INT8_FORMAT,
    build_int8_tensorwise_marker,
    decode_comfy_quant_marker,
    encode_comfy_quant_marker,
    encode_int8_tensorwise_marker,
    marker_convrot_groupsize,
    marker_reports_convrot,
    normalize_int8_tensorwise_marker,
)
from .quantizers.mxfp_weight_only import (
    MXFPQuantizationResult,
    MXFPWeightOnlyLinear,
    quantize_with_mxfp_weight_only,
)
from .quantizers.fp4_dynamic import (
    FP4DynamicLinear,
    FP4DynamicQuantizationResult,
    quantize_with_dynamic_fp4,
    quantize_with_mxfp4_dynamic,
    quantize_with_nvfp4_dynamic,
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
    SVDQuantInt8MmaLinear,
    SVDQuantLinear,
    SVDQuantResult,
    quantize_with_svd,
)
from .quantizers.turboquant import (
    TurboQuantCodec,
    TurboQuantEncoding,
    TurboQuantQuantizationResult,
    TurboQuantWeightOnlyLinear,
    execute_turboquant_component,
    quantize_with_turboquant,
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
    "ConvRotInt8Linear",
    "ConvRotInt8QuantizationResult",
    "ConvRotMixedPrecisionLinear",
    "DEFAULT_CONVROT_GROUP_SIZE",
    "STOCK_INT8_FORMAT",
    "TurboQuantCodec",
    "TurboQuantEncoding",
    "TurboQuantQuantizationResult",
    "TurboQuantWeightOnlyLinear",
    "execute_turboquant_component",
    "quantize_with_turboquant",
    "CompositeQuantBranchArtifact",
    "CompositeQuantizationArtifact",
    "FP4DynamicLinear",
    "FP4DynamicQuantizationResult",
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
    "SVDQuantInt8MmaLinear",
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
    "build_int8_tensorwise_marker",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "calibrate_activation_statistics",
    "decode_comfy_quant_marker",
    "describe_quant_backend_capability",
    "encode_comfy_quant_marker",
    "encode_int8_tensorwise_marker",
    "execute_quantization_plan",
    "expand_group_scale",
    "infer_nvfp4_tensor_layout",
    "list_quant_backend_capabilities",
    "list_quantizable_modules",
    "marker_convrot_groupsize",
    "marker_reports_convrot",
    "normalize_int8_tensorwise_marker",
    "quantize_with_awq_fp4",
    "quantize_with_awq_weight_only",
    "quantize_with_convrot_4bit",
    "quantize_with_convrot_int8",
    "quantize_with_dynamic_fp4",
    "quantize_with_int8_mma",
    "quantize_with_mxfp4_dynamic",
    "quantize_onnx_qdq_static",
    "quantize_with_mxfp_weight_only",
    "quantize_with_nvfp4_dynamic",
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
