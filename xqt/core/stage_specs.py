"""Typed parameter schemas for XQT optimization stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypeAlias, TypeVar

from omegaconf import MISSING, OmegaConf

from xqt.contracts.module import CompositePrecisionGemmSpec

from .errors import XQTConfigError
from .schema import (
    AnalysisExportConfig,
    AnalysisRecommendationConfig,
    AnalysisStructuredConfig,
    ExportTargetConfig,
    OperatorOptimizationTargetConfig,
    OutputDiffConfig,
    QuantComponentPolicyConfig,
)


@dataclass
class QuantStageSpec:
    """Parameters for one quantization stage."""

    backend: str = MISSING
    method: Optional[str] = None
    strategy: Optional[str] = None
    compute: Optional[str] = None
    scheme: Any | None = None
    policy: Dict[str, Any] = field(default_factory=dict)
    composite_gemm: Any | None = None
    keep_high_precision: List[str] = field(default_factory=list)
    skip_quantize: List[str] = field(default_factory=list)
    force_quantize: List[str] = field(default_factory=list)
    analysis_only_modules: List[str] = field(default_factory=list)
    component_policies: List[QuantComponentPolicyConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.composite_gemm = _composite_gemm_config(self.composite_gemm)


@dataclass
class PruneStageSpec:
    """Parameters for one pruning stage."""

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
class ExportStageSpec:
    """Parameters for one export or deploy stage."""

    targets: List[ExportTargetConfig] = field(default_factory=list)
    validate: Optional[OutputDiffConfig] = None


@dataclass
class AnalyzeStageSpec:
    """Parameters for one analysis stage."""

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


@dataclass
class BenchmarkStageSpec:
    """Parameters for one benchmark stage override."""

    warmup: Optional[int] = None
    iterations: Optional[int] = None
    percentiles: Optional[List[int]] = None
    sync_cuda: Optional[bool] = None
    measure_memory: Optional[bool] = None


@dataclass
class OperatorStageSpec:
    """Parameters for one operator optimization stage."""

    default_engine: str = "torch_compile"
    targets: List[OperatorOptimizationTargetConfig] = field(default_factory=list)
    benchmark: Optional[BenchmarkStageSpec] = None


@dataclass
class ONNXRuntimeHandleConfig:
    """Typed runtime settings for an ONNX Runtime deploy handle."""

    providers: List[str] = field(default_factory=list)


@dataclass
class TensorRTRuntimeHandleConfig:
    """Typed runtime settings for a TensorRT deploy handle."""

    device: Optional[str] = None
    plugin_libraries: List[str] = field(default_factory=list)


@dataclass
class DeployRuntimeHandleSpec:
    """Optional runtime-handle request attached to a deploy stage."""

    runtime: Optional[str] = None
    handle_kind: str = "inference_session"
    materialize: bool = False
    onnxruntime: ONNXRuntimeHandleConfig = field(
        default_factory=ONNXRuntimeHandleConfig
    )
    tensorrt: TensorRTRuntimeHandleConfig = field(
        default_factory=TensorRTRuntimeHandleConfig
    )


@dataclass
class DeployStageSpec:
    """Parameters for one deploy stage."""

    targets: List[ExportTargetConfig] = field(default_factory=list)
    validate: Optional[OutputDiffConfig] = None
    runtime_handle: Optional[DeployRuntimeHandleSpec] = None


StageSpec: TypeAlias = (
    QuantStageSpec
    | PruneStageSpec
    | OperatorStageSpec
    | ExportStageSpec
    | DeployStageSpec
    | AnalyzeStageSpec
    | BenchmarkStageSpec
)

ConfigT = TypeVar("ConfigT")


def stage_spec_to_params(spec: StageSpec, *, drop_none: bool = True) -> dict[str, Any]:
    """Convert a typed stage spec into the runtime mapping used by passes."""

    if is_dataclass(spec):
        params = asdict(spec)
    elif isinstance(spec, Mapping):
        params = dict(spec)
    else:
        raise XQTConfigError(f"unsupported stage spec type: {type(spec).__name__}")
    if drop_none:
        return {key: value for key, value in params.items() if value is not None}
    return params


def stage_spec_to_config(
    spec: StageSpec,
    config_type: type[ConfigT],
    *,
    base: ConfigT | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> ConfigT:
    """Derive one runtime config from a typed stage spec without field lists."""

    params = stage_spec_to_params(spec)
    for field_name in exclude:
        params.pop(field_name, None)
    nodes: list[Any] = [OmegaConf.structured(config_type)]
    if base is not None:
        nodes.append(asdict(base) if is_dataclass(base) else dict(base))
    nodes.append(OmegaConf.create(params))
    if overrides:
        nodes.append(OmegaConf.create(dict(overrides)))
    try:
        merged = OmegaConf.merge(*nodes)
        return OmegaConf.to_object(merged)
    except Exception as exc:
        raise XQTConfigError(
            f"failed to derive {config_type.__name__} from "
            f"{type(spec).__name__}: {exc}"
        ) from exc


def _composite_gemm_config(value: Any) -> CompositePrecisionGemmSpec | None:
    if value is None or isinstance(value, CompositePrecisionGemmSpec):
        return value
    try:
        return CompositePrecisionGemmSpec.from_mapping(dict(value))
    except Exception as exc:
        raise XQTConfigError(f"failed to load composite_gemm spec: {exc}") from exc


__all__ = [
    "AnalyzeStageSpec",
    "BenchmarkStageSpec",
    "DeployRuntimeHandleSpec",
    "DeployStageSpec",
    "ExportStageSpec",
    "ONNXRuntimeHandleConfig",
    "OperatorStageSpec",
    "PruneStageSpec",
    "QuantStageSpec",
    "StageSpec",
    "TensorRTRuntimeHandleConfig",
    "stage_spec_to_config",
    "stage_spec_to_params",
]
