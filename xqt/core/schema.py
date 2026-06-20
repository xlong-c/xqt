"""Structured config schema for XQT recipes."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

XQT_CONFIG_VERSION = 1

COMPRESSION_AXES = ("width", "depth", "precision", "sparsity", "steps")
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
OPERATOR_OPT_BACKENDS = (
    "torch_compile",
    "deployment_backend",
    "triton",
    "tilelang",
    "cutile",
    "cutlass",
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
class DataSplitConfig(ComponentConfig):
    """Dataset or sample split settings."""

    root: Optional[str] = None
    sample_limit: Optional[int] = None
    batch_size: int = 1


@dataclass
class DataConfig:
    """Data settings used by calibration, training, validation, and prompts."""

    calibration: Optional[DataSplitConfig] = None
    train: Optional[DataSplitConfig] = None
    validation: Optional[DataSplitConfig] = None
    prompts: Optional[DataSplitConfig] = None


@dataclass
class QuantConfig:
    """Quantization pass settings."""

    enabled: bool = False
    backend: str = "torchao"
    strategy: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    calibration_split: Optional[str] = None
    validation_split: Optional[str] = None
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
    strategy: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    calibration_split: Optional[str] = None
    validation_split: Optional[str] = None
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
class DistillConfig:
    """General distillation pass settings."""

    enabled: bool = False
    teacher: Optional[ModelConfig] = None
    temperature: float = 2.0
    alpha: float = 0.5
    cache_teacher: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionDistillConfig:
    """Diffusion few-step distillation settings."""

    enabled: bool = False
    teacher_steps: int = 20
    student_steps: int = 4
    scheduler: Optional[str] = None
    prediction_type: str = "epsilon"
    guidance_scale: float = 1.0
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
class OperatorOptimizationTargetConfig:
    """One operator optimization target entry."""

    name: str = ""
    target: Optional[str] = None
    backend: Optional[str] = None
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


@dataclass
class OperatorOptimizationConfig:
    """PyTorch runtime operator optimization settings."""

    enabled: bool = False
    stage: str = "after_compression"
    default_backend: str = "torch_compile"
    targets: List[OperatorOptimizationTargetConfig] = field(default_factory=list)


@dataclass
class CompressionConfig:
    """Enabled compression passes and compression axes."""

    axes: List[str] = field(default_factory=list)
    quant: QuantConfig = field(default_factory=QuantConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    diffusion_distill: DiffusionDistillConfig = field(
        default_factory=DiffusionDistillConfig
    )


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
class MetricThresholdConfig:
    """Task metric threshold."""

    name: str = ""
    max_drop: Optional[float] = None


@dataclass
class ValidationConfig:
    """Validation thresholds."""

    output_diff: OutputDiffConfig = field(default_factory=OutputDiffConfig)
    metric: MetricThresholdConfig = field(default_factory=MetricThresholdConfig)


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
    export: AnalysisExportConfig = field(default_factory=AnalysisExportConfig)


@dataclass
class XQTConfig:
    """XQT recipe schema v1."""

    config_version: int = XQT_CONFIG_VERSION
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
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
    "OPERATOR_OPT_BACKENDS",
    "PRUNE_GRANULARITIES",
    "PRUNE_SCOPES",
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
    "DataConfig",
    "DataSplitConfig",
    "DiffusionDistillConfig",
    "DistillConfig",
    "ExportConfig",
    "ExportTargetConfig",
    "MetricThresholdConfig",
    "ModelConfig",
    "OperatorOptimizationConfig",
    "OperatorOptimizationTargetConfig",
    "OperatorOptimizationValidationConfig",
    "OutputDiffConfig",
    "ProjectConfig",
    "PruneConfig",
    "QuantComponentPolicyConfig",
    "QuantConfig",
    "TileLangKernelConfig",
    "ValidationConfig",
    "XQTConfig",
]
