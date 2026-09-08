"""Operator wrappers, materialize and benchmark helpers for XQT kernels."""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # cutile
    "CuTileCompileSettings": ("xqt.kernels.ops._impl.engines.cutile", "CuTileCompileSettings"),
    "CuTileKernelSpec": ("xqt.kernels.ops._impl.engines.cutile", "CuTileKernelSpec"),
    "build_cutile_artifact_metadata": ("xqt.kernels.ops._impl.engines.cutile", "build_cutile_artifact_metadata"),
    "cutile_available": ("xqt.kernels.ops._impl.engines.cutile", "cutile_available"),
    "get_cutile_kernel_spec": ("xqt.kernels.ops._impl.engines.cutile", "get_cutile_kernel_spec"),
    "list_cutile_kernel_specs": ("xqt.kernels.ops._impl.engines.cutile", "list_cutile_kernel_specs"),
    "run_cutile_kernel": ("xqt.kernels.ops._impl.engines.cutile", "run_cutile_kernel"),
    # cutlass
    "CutlassCompileSettings": ("xqt.kernels.ops._impl.engines.cutlass", "CutlassCompileSettings"),
    "CutlassKernelSpec": ("xqt.kernels.ops._impl.engines.cutlass", "CutlassKernelSpec"),
    "build_cutlass_artifact_metadata": ("xqt.kernels.ops._impl.engines.cutlass", "build_cutlass_artifact_metadata"),
    "get_cutlass_kernel_spec": ("xqt.kernels.ops._impl.engines.cutlass", "get_cutlass_kernel_spec"),
    "list_cutlass_kernel_specs": ("xqt.kernels.ops._impl.engines.cutlass", "list_cutlass_kernel_specs"),
    "run_cutlass_kernel": ("xqt.kernels.ops._impl.engines.cutlass", "run_cutlass_kernel"),
    # cute_dsl
    "CuteDSLCompileSettings": ("xqt.kernels.ops._impl.engines.cute_dsl", "CuteDSLCompileSettings"),
    "CuteDSLKernelSpec": ("xqt.kernels.ops._impl.engines.cute_dsl", "CuteDSLKernelSpec"),
    "build_cute_dsl_artifact_metadata": ("xqt.kernels.ops._impl.engines.cute_dsl", "build_cute_dsl_artifact_metadata"),
    "get_cute_dsl_kernel_spec": ("xqt.kernels.ops._impl.engines.cute_dsl", "get_cute_dsl_kernel_spec"),
    "list_cute_dsl_kernel_specs": ("xqt.kernels.ops._impl.engines.cute_dsl", "list_cute_dsl_kernel_specs"),
    "run_cute_dsl_kernel": ("xqt.kernels.ops._impl.engines.cute_dsl", "run_cute_dsl_kernel"),
    # tilelang
    "TileLangCompileSettings": ("xqt.kernels.ops._impl.engines.tilelang", "TileLangCompileSettings"),
    "TileLangKernelSpec": ("xqt.kernels.ops._impl.engines.tilelang", "TileLangKernelSpec"),
    "build_tilelang_artifact_metadata": ("xqt.kernels.ops._impl.engines.tilelang", "build_tilelang_artifact_metadata"),
    "get_tilelang_kernel_spec": ("xqt.kernels.ops._impl.engines.tilelang", "get_tilelang_kernel_spec"),
    "list_tilelang_kernel_specs": ("xqt.kernels.ops._impl.engines.tilelang", "list_tilelang_kernel_specs"),
    "run_tilelang_kernel": ("xqt.kernels.ops._impl.engines.tilelang", "run_tilelang_kernel"),
    "tilelang_validation_thresholds": ("xqt.kernels.ops._impl.engines.tilelang", "tilelang_validation_thresholds"),
    "TileLangFP4ValidationResult": ("xqt.kernels.ops._impl.tilelang.validation", "TileLangFP4ValidationResult"),
    "validate_tilelang_packed_fp4_fused_gemm": ("xqt.kernels.ops._impl.tilelang.validation", "validate_tilelang_packed_fp4_fused_gemm"),
    # triton
    "TritonKernelSpec": ("xqt.kernels.ops._impl.engines.triton", "TritonKernelSpec"),
    "get_triton_kernel_spec": ("xqt.kernels.ops._impl.engines.triton", "get_triton_kernel_spec"),
    "list_triton_kernel_specs": ("xqt.kernels.ops._impl.engines.triton", "list_triton_kernel_specs"),
    "run_triton_kernel": ("xqt.kernels.ops._impl.engines.triton", "run_triton_kernel"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .advisor import (
    PrecisionRecommendation,
    ProfilingPlan,
    build_profiling_plan,
    recommend_precision_strategy,
)
from .block_kernels import (
    available_block_kernel_builders,
    register_block_kernel_builder,
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
from .execute import execute_operator_optimization_plan
from .nvfp4 import (
    NVFP4LinearBridge,
    NVFP4TensorLayout,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    infer_nvfp4_tensor_layout,
)
from .materialize import (
    materialize_module,
    materialize_operator_candidate_model,
    materialize_operator_candidate_models,
)
from .patterns import (
    OperatorPatternCandidate,
    scan_candidate_report,
    scan_export_candidates,
    scan_fx_candidates,
    scan_operator_candidate_reports,
    summarize_candidate_report,
)
from .plan import build_operator_optimization_plan
from .reporting import summarize_operator_optimization_reports
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
    "available_block_kernel_builders",
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
    "materialize_module",
    "NVFP4LinearBridge",
    "NVFP4TensorLayout",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "infer_nvfp4_tensor_layout",
    "recommend_precision_strategy",
    "register_block_kernel_builder",
    "run_tilelang_kernel",
    "run_triton_kernel",
    "run_cutile_kernel",
    "run_cutlass_kernel",
    "run_cute_dsl_kernel",
    "run_custom_cuda_opcheck",
    "scan_export_candidates",
    "scan_fx_candidates",
    "scan_candidate_report",
    "scan_operator_candidate_reports",
    "summarize_operator_optimization_reports",
    "summarize_candidate_report",
    "tilelang_validation_thresholds",
    "TileLangCompileSettings",
    "TileLangFP4ValidationResult",
    "TileLangKernelSpec",
    "TritonKernelSpec",
    "validate_tilelang_packed_fp4_fused_gemm",
]
