"""Structured config schema for XQT recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from xdl.metric.detection_utils import DetectionPostprocessConfig

COMPRESSION_AXES = ("width", "depth", "precision", "sparsity", "steps", "low_rank")
TASK_TYPES = ("classification", "detection")
PRUNE_GRANULARITIES = (
    "channel",
    "filter",
    "mlp_neuron",
    "head",
    "block",
    "stage",
    "hidden_width",
    "embedding_width",
    "expert",
    "nm",
    "block_sparse",
)
PRUNE_SCOPES = ("global", "per_layer", "per_stage", "custom")
OPERATOR_OPT_ENGINES = (
    "torch_compile",
    "deployment_engine",
    "triton",
    "tilelang",
    "cutile",
    "cutlass",
    "cute_dsl",
    "custom_cuda",
)
TILELANG_PASS_CONFIG_KEYS = (
    "TL_ENABLE_FAST_MATH",
    "TL_DISABLE_WARP_SPECIALIZED",
    "TL_DISABLE_TMA_LOWER",
)
CUTILE_PASS_CONFIG_KEYS = (
    "CUTILE_ENABLE_FAST_MATH",
    "CUTILE_ENABLE_PERSISTENT_CACHE",
)
CUTLASS_PASS_CONFIG_KEYS = (
    "CUTLASS_ENABLE_FAST_MATH",
    "CUTLASS_ENABLE_EPILOGUE_FUSION",
)
CUTE_DSL_PASS_CONFIG_KEYS = (
    "CUTE_DSL_ENABLE_FAST_MATH",
    "CUTE_DSL_ENABLE_EPILOGUE_FUSION",
    "CUTE_DSL_ENABLE_PERSISTENT_CACHE",
)
CANONICAL_QUANT_STRATEGIES = (
    "w4a16_int4",
    "w8a16_int8",
    "w4a16_fp4",
    "w4a16_nvfp4",
    "w4a16_mxfp4",
    "w8a16_mxfp8",
    "w8a16_fp8_e4m3",
    "w8a16_fp8_e5m2",
    "w8a8_int8",
    "w8a8_fp8_e4m3",
    "w8a8_fp8_e5m2",
    "w4a4_int4",
    "w4a4_fp4",
    "w4a4_nvfp4",
    "w4a4_mxfp4",
)
OPTIONAL_QUANT_STRATEGIES: tuple[str, ...] = ()
SUPPORTED_QUANT_STRATEGIES = CANONICAL_QUANT_STRATEGIES + OPTIONAL_QUANT_STRATEGIES

CANONICAL_QUANT_COMPUTES = (
    "dequant_fp16",
    "w8a8_int8_mma",
    "fp8_mma",
    "qdq_static",
    "qdq_dynamic",
    "dequant_gemm",
)
SUPPORTED_QUANT_COMPUTES = CANONICAL_QUANT_COMPUTES

CANONICAL_QUANT_METHODS = (
    "none",
    "awq",
    "gptq",
    "svd",
    "convrot",
    "turboquant",
)
SUPPORTED_QUANT_METHODS = CANONICAL_QUANT_METHODS

_QUANT_STRATEGY_ALIASES = {
    "fp4_weight_only": "w4a16_fp4",
    "weight_only_int4": "w4a16_int4",
    "convrot_w4a4": "w4a4_int4",
    "convrot_w8a8": "w8a8_int8",
}


def normalize_quant_strategy(
    strategy: Any | None,
    policy: Mapping[str, Any] | None = None,
) -> str | None:
    if strategy is None:
        return None
    text = str(strategy).strip().lower()
    if not text:
        return None
    if text == "mxfp_weight_only":
        precision = int((policy or {}).get("precision", 4))
        return "w8a16_mxfp8" if precision == 8 else "w4a16_mxfp4"
    return _QUANT_STRATEGY_ALIASES.get(text, text)


def normalize_quant_compute(
    compute: Any | None,
    policy: Mapping[str, Any] | None = None,
) -> str | None:
    del policy
    if compute is None:
        return None
    text = str(compute).strip()
    if not text:
        return None
    return text


def normalize_quant_method(method: Any | None) -> str | None:
    if method is None:
        return None
    text = str(method).strip().lower()
    if not text:
        return None
    return text


def is_supported_quant_strategy(strategy: Any | None) -> bool:
    normalized = normalize_quant_strategy(strategy)
    return normalized in SUPPORTED_QUANT_STRATEGIES


def is_supported_quant_compute(compute: Any | None) -> bool:
    normalized = normalize_quant_compute(compute)
    return normalized is None or normalized in SUPPORTED_QUANT_COMPUTES


def is_supported_quant_method(method: Any | None) -> bool:
    normalized = normalize_quant_method(method)
    return normalized is None or normalized in SUPPORTED_QUANT_METHODS


def require_supported_quant_strategy(
    strategy: Any | None,
    *,
    location: str,
    policy: Mapping[str, Any] | None = None,
) -> str:
    normalized = normalize_quant_strategy(strategy, policy)
    if normalized is None:
        allowed = ", ".join(CANONICAL_QUANT_STRATEGIES)
        raise ValueError(f"{location} must specify one of: {allowed}")
    if normalized not in SUPPORTED_QUANT_STRATEGIES:
        allowed = ", ".join(SUPPORTED_QUANT_STRATEGIES)
        raise ValueError(f"{location} must be one of: {allowed}")
    return normalized


def require_supported_quant_compute(
    compute: Any | None,
    *,
    location: str,
    allow_none: bool = True,
) -> str | None:
    normalized = normalize_quant_compute(compute)
    if normalized is None:
        if allow_none:
            return None
        allowed = ", ".join(CANONICAL_QUANT_COMPUTES)
        raise ValueError(f"{location} must specify one of: {allowed}")
    if normalized not in SUPPORTED_QUANT_COMPUTES:
        allowed = ", ".join(SUPPORTED_QUANT_COMPUTES)
        raise ValueError(f"{location} must be one of: {allowed}")
    return normalized


def require_supported_quant_method(
    method: Any | None,
    *,
    location: str,
    allow_none: bool = True,
) -> str | None:
    normalized = normalize_quant_method(method)
    if normalized is None:
        if allow_none:
            return None
        allowed = ", ".join(CANONICAL_QUANT_METHODS)
        raise ValueError(f"{location} must specify one of: {allowed}")
    if normalized not in SUPPORTED_QUANT_METHODS:
        allowed = ", ".join(SUPPORTED_QUANT_METHODS)
        raise ValueError(f"{location} must be one of: {allowed}")
    return normalized


@dataclass
class ComponentConfig:
    """Generic target + params config."""

    target: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig(ComponentConfig):
    """Model construction and checkpoint settings."""

    checkpoint: Optional[str] = None
    dtype: str = "float32"
    device: str = "cpu"


@dataclass
class TaskConfig:
    """Task metadata and task-specific runtime settings."""

    type: str = "classification"
    class_names: List[str] = field(default_factory=list)
    detection_postprocess: DetectionPostprocessConfig = field(
        default_factory=DetectionPostprocessConfig
    )
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantConfig:
    """Quantization pass settings."""

    enabled: bool = False
    backend: str = "torchao"
    method: Optional[str] = None
    strategy: Optional[str] = None
    compute: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    composite_gemm: Any | None = None
    keep_high_precision: List[str] = field(default_factory=list)
    skip_quantize: List[str] = field(default_factory=list)
    force_quantize: List[str] = field(default_factory=list)
    analysis_only_modules: List[str] = field(default_factory=list)
    component_policies: List["QuantComponentPolicyConfig"] = field(default_factory=list)

    def __post_init__(self) -> None:
        from xqt.contracts.module import coerce_composite_precision_gemm_spec

        self.composite_gemm = coerce_composite_precision_gemm_spec(self.composite_gemm)


@dataclass
class QuantComponentPolicyConfig:
    """Component-level quantization policy override."""

    name: str = ""
    target: Optional[str] = None
    enabled: bool = True
    backend: Optional[str] = None
    method: Optional[str] = None
    strategy: Optional[str] = None
    compute: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    composite_gemm: Any | None = None
    keep_high_precision: List[str] = field(default_factory=list)
    skip_quantize: List[str] = field(default_factory=list)
    force_quantize: List[str] = field(default_factory=list)
    analysis_only: bool = False

    def __post_init__(self) -> None:
        from xqt.contracts.module import coerce_composite_precision_gemm_spec

        self.composite_gemm = coerce_composite_precision_gemm_spec(self.composite_gemm)


@dataclass
class PruneConfig:
    """Pruning pass settings."""

    enabled: bool = False
    method: str = "global_l1_unstructured"
    granularity: Optional[str] = None
    scope: str = "global"
    target_sparsity: float = 0.0
    schedule: str = "one_shot"
    importance: Dict[str, Any] = field(default_factory=dict)
    selection: Dict[str, Any] = field(default_factory=dict)
    rewrite: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorOptimizationValidationConfig:
    """Numeric validation thresholds for one operator optimization target."""

    atol: float = 1e-5
    rtol: float = 1e-5


@dataclass
class TileLangKernelConfig:
    """Structured TileLang compile settings."""

    target: str = "cuda"
    target_arch: Optional[str] = None
    threads: int = 128
    num_stages: int = 2
    cache_dir: Optional[str] = None
    pass_configs: Dict[str, Any] = field(default_factory=dict)
    linear_runtime: str = "auto"
    linear_fastpath: str = "auto"
    attention_fastpath: str = "auto"
    conv_fastpath: str = "auto"
    norm_fastpath: str = "auto"


@dataclass
class CuTileKernelConfig:
    """Structured CuTile compile settings."""

    target: str = "cuda"
    target_arch: Optional[str] = None
    threads: int = 128
    cache_dir: Optional[str] = None
    pass_configs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CutlassKernelConfig:
    """Structured CUTLASS Python compile settings."""

    target_arch: Optional[str] = None
    cache_dir: Optional[str] = None
    tile_shape: List[int] = field(default_factory=lambda: [128, 128, 64])
    cluster_shape: Optional[List[int]] = None
    pass_configs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CuteDSLKernelConfig:
    """Structured CuTe DSL compile settings."""

    target_arch: Optional[str] = None
    cache_dir: Optional[str] = None
    tile_shape: List[int] = field(default_factory=lambda: [128, 128, 64])
    cluster_shape: Optional[List[int]] = None
    pass_configs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorOptimizationTargetConfig:
    """One runtime candidate replacement and block-level benchmark target."""

    name: str = ""
    target: Optional[str] = None
    candidate_kind: str = "single_kernel"
    benchmark_target: Optional[str] = None
    block_kernel: Optional[str] = None
    block_kernel_engine: Optional[str] = None
    engine: Optional[str] = None
    mode: Optional[str] = None
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    options: Dict[str, Any] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    fallback: str = "eager"
    fallback_policy: str = "prefer_fallback"
    min_speedup: float = 1.01
    validate: OperatorOptimizationValidationConfig = field(
        default_factory=OperatorOptimizationValidationConfig
    )
    tilelang: TileLangKernelConfig = field(default_factory=TileLangKernelConfig)
    cutile: CuTileKernelConfig = field(default_factory=CuTileKernelConfig)
    cutlass: CutlassKernelConfig = field(default_factory=CutlassKernelConfig)
    cute_dsl: CuteDSLKernelConfig = field(default_factory=CuteDSLKernelConfig)


@dataclass
class OperatorOptimizationConfig:
    """PyTorch runtime operator optimization settings."""

    enabled: bool = False
    stage: str = "after_compression"
    default_engine: str = "torch_compile"
    targets: List[OperatorOptimizationTargetConfig] = field(default_factory=list)


@dataclass
class PreExportFusionConfig:
    """PyTorch module fusion settings applied on an ONNX export copy."""

    enabled: bool = False
    mode: str = "eager"
    inplace: bool = False
    modules_to_fuse: List[List[str]] = field(default_factory=list)


@dataclass
class PreExportLoweringConfig:
    """Explicit deployment lowering settings applied on an ONNX export copy."""

    enabled: bool = False
    mode: str = "fp4_weight_only_to_dense_linear"
    inplace: bool = False


@dataclass
class ONNXOptimizationConfig:
    """Typed ONNX graph optimization settings for one ONNX export target."""

    enabled: bool = False
    backend: str = "onnxruntime"
    level: str = "extended"
    output_path: Optional[str] = None
    output_suffix: str = ".optimized"
    validate: bool = True
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    native_qdq: bool = True


@dataclass
class ONNXExportConfig:
    """Typed ONNX-specific settings for one export target."""

    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    dynamo: bool = True
    validate: bool = True
    runtime_diff: bool = True
    pre_export_fusion: PreExportFusionConfig = field(
        default_factory=PreExportFusionConfig
    )
    pre_export_lowering: PreExportLoweringConfig = field(
        default_factory=PreExportLoweringConfig
    )
    optimization: ONNXOptimizationConfig = field(default_factory=ONNXOptimizationConfig)


@dataclass
class OpenVINOExportConfig:
    """Typed OpenVINO-specific settings for one export target."""

    onnx_path: Optional[str] = None
    input_shape: Optional[List[int]] = None
    dry_run: bool = False
    runtime_diff: bool = True
    device: str = "CPU"


@dataclass
class TensorRTRuntimeBenchmarkConfig:
    """Optional runtime benchmark settings for one materialized TensorRT engine."""

    enabled: bool = False
    input_shapes: Dict[str, List[int]] = field(default_factory=dict)
    warmup: int = 10
    iterations: int = 50
    device: str = "cuda:0"
    fill_random: bool = True


@dataclass
class TensorRTExportConfig:
    """Typed TensorRT engine-build settings for one export target."""

    onnx_path: Optional[str] = None
    backend: str = "trtexec"
    trtexec_path: str = "trtexec"
    extra_args: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    dry_run: bool = False
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    workspace_mib: int = 4096
    builder_optimization_level: Optional[int] = None
    timing_cache_path: Optional[str] = None
    log_level: Optional[str] = None
    plugin_libraries: List[str] = field(default_factory=list)
    serialize_plugin_libraries: bool = True
    validate_plugin_libraries_loadable: bool = False
    runtime_benchmark: TensorRTRuntimeBenchmarkConfig = field(
        default_factory=TensorRTRuntimeBenchmarkConfig
    )


@dataclass
class TorchExportConfig:
    """Typed torch.export-specific settings for one export target."""

    strict: bool = False
    validate: bool = True
    runtime_diff: bool = True


@dataclass
class TorchScriptExportConfig:
    """Typed TorchScript-specific settings for one export target."""

    method: str = "trace"
    check_trace: bool = True
    runtime_diff: bool = True


@dataclass
class ExecuTorchExportConfig:
    """Typed ExecuTorch-specific settings for one export target."""

    dry_run: bool = False


@dataclass
class NCNNExportConfig:
    """Typed ncnn converter settings for one export target."""

    source_path: Optional[str] = None
    converter: str = "onnx2ncnn"
    onnx2ncnn_path: str = "onnx2ncnn"
    pnnx_path: str = "pnnx"
    bin_path: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    dry_run: bool = False


@dataclass
class MNNExportConfig:
    """Typed MNNConvert settings for one export target."""

    source_path: Optional[str] = None
    converter_path: str = "MNNConvert"
    framework: str = "ONNX"
    extra_args: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    dry_run: bool = False


@dataclass
class ExportTargetConfig:
    """Single export target settings."""

    format: str = "onnx"
    output_path: Optional[str] = None
    opset: Optional[int] = None
    precision: Optional[str] = None
    dynamic_shapes: Dict[str, Any] = field(default_factory=dict)
    profiles: Dict[str, Any] = field(default_factory=dict)
    onnx: ONNXExportConfig = field(default_factory=ONNXExportConfig)
    openvino: OpenVINOExportConfig = field(default_factory=OpenVINOExportConfig)
    tensorrt: TensorRTExportConfig = field(default_factory=TensorRTExportConfig)
    torch_export: TorchExportConfig = field(default_factory=TorchExportConfig)
    torchscript: TorchScriptExportConfig = field(
        default_factory=TorchScriptExportConfig
    )
    executorch: ExecuTorchExportConfig = field(default_factory=ExecuTorchExportConfig)
    ncnn: NCNNExportConfig = field(default_factory=NCNNExportConfig)
    mnn: MNNExportConfig = field(default_factory=MNNExportConfig)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputDiffConfig:
    """Output comparison thresholds."""

    atol: float = 1e-5
    rtol: float = 1e-5


@dataclass
class BenchmarkConfig:
    """Latency and memory benchmark settings."""

    warmup: int = 10
    iterations: int = 50
    percentiles: List[int] = field(default_factory=lambda: [50, 90, 99])
    sync_cuda: bool = True
    measure_memory: bool = True


@dataclass
class AnalysisExportConfig:
    """Artifact export settings for analysis reports."""

    json: bool = True
    csv: bool = True
    markdown: bool = True


@dataclass
class AnalysisStructuredConfig:
    """Structured diff output switches for analysis reports."""

    per_channel: bool = False
    per_token: bool = False


@dataclass
class AnalysisRecommendationConfig:
    """Recommendation toggles for analysis pass outputs."""

    mixed_precision: bool = True
    prune_candidates: bool = True


@dataclass
class AnalysisConfig:
    """Analysis and recommendation pass settings."""

    enabled: bool = False
    compare_to: str = "baseline"
    module_names: Optional[List[str]] = None
    top_k: Optional[int] = None
    metrics: List[str] = field(
        default_factory=lambda: ["max_abs", "mean_abs", "cosine_similarity"]
    )
    structured: AnalysisStructuredConfig = field(
        default_factory=AnalysisStructuredConfig
    )
    recommendations: AnalysisRecommendationConfig = field(
        default_factory=AnalysisRecommendationConfig
    )
    include_weight_diff: bool = True
    include_statistics: bool = False
    sample_budget: Optional[int] = None
    sample_seed: int = 0
    histogram_bins: int = 32
    export: AnalysisExportConfig = field(default_factory=AnalysisExportConfig)


__all__ = [
    "COMPRESSION_AXES",
    "CUTILE_PASS_CONFIG_KEYS",
    "CUTLASS_PASS_CONFIG_KEYS",
    "CUTE_DSL_PASS_CONFIG_KEYS",
    "OPERATOR_OPT_ENGINES",
    "PRUNE_GRANULARITIES",
    "PRUNE_SCOPES",
    "TASK_TYPES",
    "TILELANG_PASS_CONFIG_KEYS",
    "AnalysisConfig",
    "AnalysisExportConfig",
    "AnalysisRecommendationConfig",
    "AnalysisStructuredConfig",
    "BenchmarkConfig",
    "ComponentConfig",
    "CuTileKernelConfig",
    "CutlassKernelConfig",
    "CuteDSLKernelConfig",
    "DetectionPostprocessConfig",
    "ExecuTorchExportConfig",
    "ExportTargetConfig",
    "MNNExportConfig",
    "ModelConfig",
    "NCNNExportConfig",
    "ONNXExportConfig",
    "ONNXOptimizationConfig",
    "OpenVINOExportConfig",
    "OperatorOptimizationConfig",
    "OperatorOptimizationTargetConfig",
    "OperatorOptimizationValidationConfig",
    "OutputDiffConfig",
    "PreExportFusionConfig",
    "PreExportLoweringConfig",
    "PruneConfig",
    "QuantComponentPolicyConfig",
    "QuantConfig",
    "TaskConfig",
    "TensorRTExportConfig",
    "TensorRTRuntimeBenchmarkConfig",
    "TileLangKernelConfig",
    "TorchExportConfig",
    "TorchScriptExportConfig",
]
