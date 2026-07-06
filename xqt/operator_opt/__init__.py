"""Operator optimization helpers for XQT."""

from .advisor import (
    PrecisionRecommendation,
    ProfilingPlan,
    build_profiling_plan,
    recommend_precision_strategy,
)
from .capability import (
    OperatorOptimizationEngineCapability,
    describe_operator_engine_capability,
    list_operator_engine_capabilities,
)
from .compile_backend import compile_with_torch
from .cuda_extension import (
    CUSTOM_CUDA_BUILD_ENV,
    CUSTOM_CUDA_EXTENSION_NAME,
    CUSTOM_CUDA_OPS,
    CustomCudaExtensionCapability,
    describe_custom_cuda_extension_capability,
    fused_bias_gelu_custom_cuda,
    run_custom_cuda_opcheck,
)
from .backends.cutile import (
    CuTileCompileSettings,
    CuTileKernelSpec,
    build_cutile_artifact_metadata,
    cutile_available,
    get_cutile_kernel_spec,
    list_cutile_kernel_specs,
    run_cutile_kernel,
)
from .backends.cutlass import (
    CutlassCompileSettings,
    CutlassKernelSpec,
    build_cutlass_artifact_metadata,
    get_cutlass_kernel_spec,
    list_cutlass_kernel_specs,
    run_cutlass_kernel,
)
from .backends.cute_dsl import (
    CuteDSLCompileSettings,
    CuteDSLKernelSpec,
    build_cute_dsl_artifact_metadata,
    get_cute_dsl_kernel_spec,
    list_cute_dsl_kernel_specs,
    run_cute_dsl_kernel,
)
from .executor import (
    build_operator_optimization_plan,
    execute_operator_optimization_plan,
    materialize_operator_candidate_model,
    materialize_operator_candidate_models,
    summarize_operator_optimization_reports,
)
from .patterns import (
    OperatorPatternCandidate,
    scan_export_candidates,
    scan_fx_candidates,
    summarize_candidate_report,
)
from .backends.tilelang import (
    TileLangCompileSettings,
    TileLangKernelSpec,
    build_tilelang_artifact_metadata,
    get_tilelang_kernel_spec,
    list_tilelang_kernel_specs,
    run_tilelang_kernel,
    tilelang_validation_thresholds,
)
from .backends.tilelang_validation import (
    TileLangFP4ValidationResult,
    validate_tilelang_packed_fp4_fused_gemm,
)
from .backends.triton import (
    TritonKernelSpec,
    get_triton_kernel_spec,
    list_triton_kernel_specs,
    run_triton_kernel,
)
from .types import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationExecutionResult,
    OperatorOptimizationReport,
    OperatorOptimizationTargetPlan,
)

__all__ = [
    "OperatorOptimizationEngineCapability",
    "OperatorOptimizationExecutionPlan",
    "OperatorOptimizationExecutionResult",
    "OperatorPatternCandidate",
    "OperatorOptimizationReport",
    "OperatorOptimizationTargetPlan",
    "PrecisionRecommendation",
    "ProfilingPlan",
    "build_operator_optimization_plan",
    "build_profiling_plan",
    "build_cutile_artifact_metadata",
    "cutile_available",
    "build_cutlass_artifact_metadata",
    "build_cute_dsl_artifact_metadata",
    "build_tilelang_artifact_metadata",
    "compile_with_torch",
    "CUSTOM_CUDA_BUILD_ENV",
    "CUSTOM_CUDA_EXTENSION_NAME",
    "CUSTOM_CUDA_OPS",
    "CustomCudaExtensionCapability",
    "CuTileCompileSettings",
    "CuTileKernelSpec",
    "CutlassCompileSettings",
    "CutlassKernelSpec",
    "CuteDSLCompileSettings",
    "CuteDSLKernelSpec",
    "describe_custom_cuda_extension_capability",
    "describe_operator_engine_capability",
    "execute_operator_optimization_plan",
    "get_cutile_kernel_spec",
    "get_cutlass_kernel_spec",
    "get_cute_dsl_kernel_spec",
    "fused_bias_gelu_custom_cuda",
    "get_tilelang_kernel_spec",
    "get_triton_kernel_spec",
    "list_cutile_kernel_specs",
    "list_cutlass_kernel_specs",
    "list_cute_dsl_kernel_specs",
    "list_operator_engine_capabilities",
    "list_tilelang_kernel_specs",
    "list_triton_kernel_specs",
    "materialize_operator_candidate_model",
    "materialize_operator_candidate_models",
    "recommend_precision_strategy",
    "run_tilelang_kernel",
    "run_triton_kernel",
    "run_cutile_kernel",
    "run_cutlass_kernel",
    "run_cute_dsl_kernel",
    "run_custom_cuda_opcheck",
    "scan_export_candidates",
    "scan_fx_candidates",
    "summarize_operator_optimization_reports",
    "summarize_candidate_report",
    "tilelang_validation_thresholds",
    "TileLangCompileSettings",
    "TileLangFP4ValidationResult",
    "TileLangKernelSpec",
    "TritonKernelSpec",
    "validate_tilelang_packed_fp4_fused_gemm",
]
