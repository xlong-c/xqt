"""Structured config schema for XQT recipes."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from xdl.metric.detection_utils import DetectionPostprocessConfig

XQT_CONFIG_VERSION = 1

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
    "dynamic_int8",
    "weight_only_int8",
    "weight_only_int4",
    "static_qdq_int8",
    "fp8_dynamic",
    "fp4_weight_only",
    "mxfp_weight_only",
    "svd_fp4",
    "svd_int4",
)
OPTIONAL_QUANT_STRATEGIES = (
    "fp8_weight_only",
)
SUPPORTED_QUANT_STRATEGIES = CANONICAL_QUANT_STRATEGIES + OPTIONAL_QUANT_STRATEGIES
QUANT_STRATEGY_ALIASES = {
    "int8_dynamic_activation_int8_weight": "dynamic_int8",
    "int8_dynamic": "dynamic_int8",
    "int8": "dynamic_int8",
    "int8_weight_only": "weight_only_int8",
    "weight_only_int8": "weight_only_int8",
    "int4_weight_only": "weight_only_int4",
    "weight_only_int4": "weight_only_int4",
    "int4": "weight_only_int4",
    "static_int8": "static_qdq_int8",
    "qdq_int8": "static_qdq_int8",
    "static_qdq_int8": "static_qdq_int8",
    "float8_dynamic_activation_float8_weight": "fp8_dynamic",
    "float8_dynamic": "fp8_dynamic",
    "fp8": "fp8_dynamic",
    "fp4": "fp4_weight_only",
    "weight_only_fp4": "fp4_weight_only",
    "mxfp": "mxfp_weight_only",
    "weight_only_mxfp": "mxfp_weight_only",
    "mxfp_weight_only": "mxfp_weight_only",
    "svdquant_fp4": "svd_fp4",
    "svdquant_int4": "svd_int4",
    "svd_fp4": "svd_fp4",
    "svd_int4": "svd_int4",
    "svdquant": "svd_fp4",
}


def normalize_quant_strategy(
    strategy: Any | None,
    policy: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the canonical quantization strategy inferred from strategy or policy."""

    policy = policy or {}
    raw = strategy
    if raw is None:
        raw = policy.get("strategy")
    if raw is None:
        dtype = str(policy.get("dtype") or "").lower()
        scheme = str(policy.get("scheme") or "").lower()
        if dtype == "fp4" and scheme in {"svd", "svdquant"}:
            raw = "svd_fp4"
        elif dtype == "int4" and scheme in {"svd", "svdquant"}:
            raw = "svd_int4"
        elif dtype == "fp4" and scheme in {"", "weight_only", "weight-only"}:
            raw = "fp4_weight_only"
        elif dtype.startswith("mxfp") and scheme in {"", "weight_only", "weight-only"}:
            raw = "mxfp_weight_only"
        elif dtype == "int4" and scheme in {"", "weight_only", "weight-only"}:
            raw = "weight_only_int4"
        elif dtype == "int8" and scheme in {"weight_only", "weight-only"}:
            raw = "weight_only_int8"
        elif dtype == "int8" and scheme in {"", "dynamic"}:
            raw = "dynamic_int8"
        elif dtype in {"fp8", "float8"} and scheme in {"", "dynamic"}:
            raw = "fp8_dynamic"
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return QUANT_STRATEGY_ALIASES.get(text, text)


def is_supported_quant_strategy(strategy: Any | None) -> bool:
    """Return whether a strategy name is part of the supported XQT vocabulary."""

    normalized = normalize_quant_strategy(strategy)
    return normalized in SUPPORTED_QUANT_STRATEGIES


def require_supported_quant_strategy(
    strategy: Any | None,
    *,
    location: str,
    policy: Mapping[str, Any] | None = None,
) -> str:
    """Normalize a strategy or raise a clear configuration error."""

    normalized = normalize_quant_strategy(strategy, policy)
    if normalized is None:
        allowed = ", ".join(CANONICAL_QUANT_STRATEGIES)
        raise ValueError(f"{location} must specify one of: {allowed}")
    if normalized not in SUPPORTED_QUANT_STRATEGIES:
        allowed = ", ".join(SUPPORTED_QUANT_STRATEGIES)
        raise ValueError(f"{location} must be one of: {allowed}")
    return normalized


@dataclass
class ProjectConfig:
    """Project-level artifact settings."""

    name: str = "xqt_experiment"
    artifact_dir: str = "artifacts/xqt/default"


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
    policy: Dict[str, Any] = field(default_factory=dict)
    keep_high_precision: List[str] = field(default_factory=list)
    skip_quantize: List[str] = field(default_factory=list)
    force_quantize: List[str] = field(default_factory=list)
    analysis_only_modules: List[str] = field(default_factory=list)
    component_policies: List["QuantComponentPolicyConfig"] = field(default_factory=list)


@dataclass
class QuantComponentPolicyConfig:
    """Component-level quantization policy override."""

    name: str = ""
    target: Optional[str] = None
    enabled: bool = True
    backend: Optional[str] = None
    method: Optional[str] = None
    strategy: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    keep_high_precision: List[str] = field(default_factory=list)
    skip_quantize: List[str] = field(default_factory=list)
    force_quantize: List[str] = field(default_factory=list)
    analysis_only: bool = False


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
    """One operator optimization target entry."""

    name: str = ""
    target: Optional[str] = None
    engine: Optional[str] = None
    mode: Optional[str] = None
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    options: Dict[str, Any] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    fallback: str = "eager"
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
class CompressionConfig:
    """Enabled compression passes and compression axes."""

    axes: List[str] = field(default_factory=list)
    quant: QuantConfig = field(default_factory=QuantConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)


@dataclass
class ExportTargetConfig:
    """Single export target settings."""

    format: str = "onnx"
    output_path: Optional[str] = None
    opset: Optional[int] = None
    precision: Optional[str] = None
    dynamic_shapes: Dict[str, Any] = field(default_factory=dict)
    profiles: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """Export target collection."""

    targets: List[ExportTargetConfig] = field(default_factory=list)


@dataclass
class OutputDiffConfig:
    """Output comparison thresholds."""

    atol: float = 1e-5
    rtol: float = 1e-5


@dataclass
class ValidationConfig:
    """Model output numeric validation thresholds."""

    output_diff: OutputDiffConfig = field(default_factory=OutputDiffConfig)


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
    structured: AnalysisStructuredConfig = field(default_factory=AnalysisStructuredConfig)
    recommendations: AnalysisRecommendationConfig = field(
        default_factory=AnalysisRecommendationConfig
    )
    include_weight_diff: bool = True
    include_statistics: bool = False
    sample_budget: Optional[int] = None
    sample_seed: int = 0
    histogram_bins: int = 32
    export: AnalysisExportConfig = field(default_factory=AnalysisExportConfig)


@dataclass
class XQTConfig:
    """XQT recipe schema v1."""

    config_version: int = XQT_CONFIG_VERSION
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    operator_optimization: OperatorOptimizationConfig = field(
        default_factory=OperatorOptimizationConfig
    )
    export: ExportConfig = field(default_factory=ExportConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


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
    "XQT_CONFIG_VERSION",
    "AnalysisConfig",
    "AnalysisExportConfig",
    "AnalysisRecommendationConfig",
    "AnalysisStructuredConfig",
    "BenchmarkConfig",
    "ComponentConfig",
    "CompressionConfig",
    "CuTileKernelConfig",
    "CutlassKernelConfig",
    "CuteDSLKernelConfig",
    "DetectionPostprocessConfig",
    "ExportConfig",
    "ExportTargetConfig",
    "ModelConfig",
    "OperatorOptimizationConfig",
    "OperatorOptimizationTargetConfig",
    "OperatorOptimizationValidationConfig",
    "OutputDiffConfig",
    "ProjectConfig",
    "PruneConfig",
    "QuantComponentPolicyConfig",
    "QuantConfig",
    "TaskConfig",
    "TileLangKernelConfig",
    "ValidationConfig",
    "XQTConfig",
]
