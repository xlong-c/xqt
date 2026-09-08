"""Quantization helpers for XQT."""

from .calibration import (
    ActivationDriftRecord,
    ActivationScaleArtifact,
    ActivationStatistic,
    analyze_activation_drift,
    calibrate_activation_scales,
    calibrate_activation_statistics,
)
from xqt.contracts.external import (
    ExternalQuantInfo,
    iter_config_filenames,
    list_supported_external_formats,
    normalize_external_format,
    override_external_format,
    probe_external_quant_config,
    resolve_external_quantization,
)
from .transforms import (
    GraphQuantTransform,
    OrthogonalRotationTransform,
    RewriteTransactionReport,
    RotationAbsorbTransform,
    SpeculativeRewriteConfig,
    TransformPlan,
    TransformReport,
    apply_graph_transforms,
    build_random_orthogonal_matrix,
    speculative_graph_rewrite,
)
from .capability import (
    QuantBackendCapability,
    describe_quant_backend_capability,
    list_quant_backend_capabilities,
    supported_quant_backends,
)
from .axes import (
    quant_axis_report,
    quant_compute_specs,
    quant_method_specs,
    quant_storage_specs,
)
from .backends.onnx_qdq import (
    IterableCalibrationDataReader,
    ONNXQDQQuantizationResult,
    quantize_onnx_qdq_static,
)
from .plan import build_quantization_plan
from .policy import (
    DEFAULT_MOE_EXPERT_NAME_PATTERNS,
    DEFAULT_MOE_ROUTER_NAME_PATTERNS,
    QuantizationCandidate,
    QuantizationPolicy,
    classify_moe_module,
    is_moe_expert_module,
    is_moe_router_module,
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
    QuantScheme,
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
from .quantizers.adaptive_rounding import (
    optimize_linear_rounding,
    quantize_with_adaptive_rounding,
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
    ConvRotNormInt8Linear,
    ConvRotInt8QuantizationResult,
    quantize_with_convrot_int8,
)
from .sequential import (
    LayerSequentialConfig,
    LayerSequentialReport,
    SequentialBlockSpec,
    SequentialPartition,
    discover_sequential_partition,
    quantize_layer_sequential,
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
from xqt.kernels.ops.quantization.nvfp4 import (
    expand_group_scale,
    unpack_nvfp4e2m1,
)
_NVFP4_WRAPPERS_EXPORTS = {
    "NVFP4LinearBridge",
    "NVFP4TensorLayout",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "infer_nvfp4_tensor_layout",
}


def __getattr__(name: str):
    if name in _NVFP4_WRAPPERS_EXPORTS:
        import importlib

        mod = importlib.import_module("xqt.kernels.wrappers.nvfp4")
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from .quantizers.svd import (
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
from .quantizers.kv_scale import (
    KvScaleArtifact,
    calibrate_kv_scales,
    evaluate_kv_scale_cosine,
    execute_kv_scale_component,
    kv_scales_to_compute_metadata,
)
from .quantizers.moe_weight_only import (
    MoEExpertQuantizationResult,
    execute_moe_weight_only_component,
    list_moe_module_roles,
    quantize_moe_experts_weight_only,
)
from .quality import (
    ModelCompressionQualityReport,
    evaluate_model_compression_quality,
)

__all__ = [
    "ActivationDriftRecord",
    "ActivationScaleArtifact",
    "ActivationStatistic",
    "quant_axis_report",
    "quant_compute_specs",
    "quant_method_specs",
    "quant_storage_specs",
    "ExternalQuantInfo",
    "GraphQuantTransform",
    "RotationAbsorbTransform",
    "TransformPlan",
    "TransformReport",
    "AWQGPTQWeightOnlyLinear",
    "AWQGPTQWeightOnlyQuantizationResult",
    "FakeQDQSurrogateResult",
    "FP4QuantizationResult",
    "IterableCalibrationDataReader",
    "ConvRot4BitQuantizationResult",
    "ConvRotInt8Linear",
    "ConvRotNormInt8Linear",
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
    "KvScaleArtifact",
    "LayerAnalysisRecord",
    "LayerSensitivityRecord",
    "MoEExpertQuantizationResult",
    "MXFPQuantizationResult",
    "DEFAULT_MOE_EXPERT_NAME_PATTERNS",
    "DEFAULT_MOE_ROUTER_NAME_PATTERNS",
    "NVFP4LinearBridge",
    "NVFP4TensorLayout",
    "ONNXQDQQuantizationResult",
    "Quantizer",
    "QuantizerOptions",
    "QuantizerResult",
    "QuantBackendCapability",
    "QuantScheme",
    "QuantizationCandidate",
    "QuantizationComponentPlan",
    "QuantizationExecutionPlan",
    "QuantizationExecutionResult",
    "QuantizationNature",
    "QuantizationPolicy",
    "QuantizationReport",
    "FP4WeightOnlyLinear",
    "MXFPWeightOnlyLinear",
    "SVDQuantResult",
    "TorchAOQuantizationResult",
    "W4StorageInt8MmaLinear",
    "W4StorageInt8MmaQuantizationResult",
    "analyze_activation_drift",
    "analyze_layer_errors",
    "analyze_layer_sensitivity",
    "apply_graph_transforms",
    "build_quantization_plan",
    "build_composite_quantization_artifact",
    "build_random_orthogonal_matrix",
    "build_regular_hadamard_matrix",
    "build_fake_qdq_surrogate",
    "build_int8_tensorwise_marker",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "calibrate_activation_scales",
    "calibrate_activation_statistics",
    "calibrate_kv_scales",
    "classify_moe_module",
    "iter_config_filenames",
    "list_supported_external_formats",
    "normalize_external_format",
    "override_external_format",
    "OrthogonalRotationTransform",
    "RewriteTransactionReport",
    "SpeculativeRewriteConfig",
    "speculative_graph_rewrite",
    "probe_external_quant_config",
    "resolve_external_quantization",
    "decode_comfy_quant_marker",
    "describe_quant_backend_capability",
    "discover_sequential_partition",
    "encode_comfy_quant_marker",
    "encode_int8_tensorwise_marker",
    "evaluate_kv_scale_cosine",
    "execute_kv_scale_component",
    "execute_moe_weight_only_component",
    "execute_quantization_plan",
    "expand_group_scale",
    "infer_nvfp4_tensor_layout",
    "is_moe_expert_module",
    "is_moe_router_module",
    "kv_scales_to_compute_metadata",
    "LayerSequentialConfig",
    "LayerSequentialReport",
    "list_moe_module_roles",
    "list_quant_backend_capabilities",
    "list_quantizable_modules",
    "marker_convrot_groupsize",
    "marker_reports_convrot",
    "normalize_int8_tensorwise_marker",
    "optimize_linear_rounding",
    "quantize_layer_sequential",
    "quantize_moe_experts_weight_only",
    "quantize_with_adaptive_rounding",
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
    "ModelCompressionQualityReport",
    "evaluate_model_compression_quality",
]
