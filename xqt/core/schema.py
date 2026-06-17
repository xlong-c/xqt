"""Structured config schema for XQT recipes."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

XQT_CONFIG_VERSION = 1

COMPRESSION_AXES = ("width", "depth", "precision", "sparsity", "steps")


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
    policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruneConfig:
    """Pruning pass settings."""

    enabled: bool = False
    method: str = "global_l1_unstructured"
    target_sparsity: float = 0.0
    schedule: str = "one_shot"
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
class XQTConfig:
    """XQT recipe schema v1."""

    config_version: int = XQT_CONFIG_VERSION
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


__all__ = [
    "COMPRESSION_AXES",
    "XQT_CONFIG_VERSION",
    "BenchmarkConfig",
    "ComponentConfig",
    "CompressionConfig",
    "DataConfig",
    "DataSplitConfig",
    "DiffusionDistillConfig",
    "DistillConfig",
    "ExportConfig",
    "ExportTargetConfig",
    "MetricThresholdConfig",
    "ModelConfig",
    "OutputDiffConfig",
    "ProjectConfig",
    "PruneConfig",
    "QuantConfig",
    "ValidationConfig",
    "XQTConfig",
]
