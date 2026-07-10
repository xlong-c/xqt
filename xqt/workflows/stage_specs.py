"""Typed stage parameter specs for XQT optimization workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, TypeAlias, cast

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from xqt.core.errors import XQTConfigError
from xqt.core.schema import (
    AnalysisExportConfig,
    AnalysisRecommendationConfig,
    AnalysisStructuredConfig,
    ExportTargetConfig,
    OperatorOptimizationTargetConfig,
    OutputDiffConfig,
    PruneConfig,
    QuantComponentPolicyConfig,
    QuantConfig,
)


@dataclass
class QuantStageSpec:
    """Parameters for one quantization stage."""

    backend: str = MISSING
    method: Optional[str] = None
    strategy: Optional[str] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    keep_high_precision: List[str] = field(default_factory=list)
    skip_quantize: List[str] = field(default_factory=list)
    force_quantize: List[str] = field(default_factory=list)
    analysis_only_modules: List[str] = field(default_factory=list)
    component_policies: List[QuantComponentPolicyConfig] = field(default_factory=list)


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
class DeployRuntimeHandleSpec:
    """Optional runtime-handle request attached to a deploy stage."""

    runtime: Optional[str] = None
    handle_kind: str = "inference_session"
    materialize: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


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

_STAGE_SPEC_TYPES = {
    "quant": QuantStageSpec,
    "prune": PruneStageSpec,
    "operator": OperatorStageSpec,
    "export": ExportStageSpec,
    "deploy": DeployStageSpec,
    "analyze": AnalyzeStageSpec,
    "benchmark": BenchmarkStageSpec,
}


def build_stage_spec(kind: str, params: Mapping[str, Any] | None = None) -> StageSpec:
    """Build the typed spec for one workflow stage kind."""

    spec_type = _STAGE_SPEC_TYPES.get(kind)
    if spec_type is None:
        allowed = ", ".join(sorted(_STAGE_SPEC_TYPES))
        raise XQTConfigError(f"unsupported stage kind {kind}. Allowed: {allowed}")
    if kind == "quant" and (not params or not params.get("backend")):
        raise XQTConfigError("quant.params.backend is required")
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(spec_type),
            OmegaConf.create(dict(params or {})),
        )
        spec = cast(StageSpec, OmegaConf.to_object(merged))
    except OmegaConfBaseException as exc:
        raise XQTConfigError(f"failed to load {kind} stage params: {exc}") from exc
    _validate_stage_spec(kind, spec)
    return spec


def ensure_stage_spec(stage: Any, *, rebuild: bool = False) -> StageSpec:
    """Return ``stage.spec``, building it from ``stage.params`` when needed."""

    spec = getattr(stage, "spec", None)
    if spec is None or rebuild:
        spec = build_stage_spec(str(stage.kind), stage.params)
        setattr(stage, "spec", spec)
    return cast(StageSpec, spec)


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


def stage_params(stage: Any, *, drop_none: bool = True) -> dict[str, Any]:
    """Return normalized stage parameters from the typed spec."""

    return stage_spec_to_params(ensure_stage_spec(stage), drop_none=drop_none)


def _validate_stage_spec(kind: str, spec: StageSpec) -> None:
    if isinstance(spec, QuantStageSpec):
        _validate_quant_stage_spec(spec)
    elif isinstance(spec, PruneStageSpec):
        _validate_prune_stage_spec(spec)
    elif isinstance(spec, OperatorStageSpec):
        _validate_operator_stage_spec(spec)
    elif isinstance(spec, ExportStageSpec):
        _validate_export_stage_spec(spec, location=f"{kind}.params.validate")
    elif isinstance(spec, DeployStageSpec):
        _validate_deploy_stage_spec(spec, location=f"{kind}.params")
    elif isinstance(spec, AnalyzeStageSpec):
        _validate_analyze_stage_spec(spec)
    elif isinstance(spec, BenchmarkStageSpec):
        _validate_benchmark_stage_spec(spec, location=f"{kind}.params")


def _validate_quant_stage_spec(spec: QuantStageSpec) -> None:
    from xqt.quant.capability import describe_quant_backend_capability

    spec.component_policies = [
        _quant_component_policy_config(component)
        for component in spec.component_policies
    ]
    params = stage_spec_to_params(spec)
    quant = QuantConfig(enabled=True, **params)
    quant.component_policies = [
        _quant_component_policy_config(component)
        for component in quant.component_policies
    ]
    if not quant.backend:
        raise XQTConfigError("quant.params.backend is required")
    has_selector = (
        quant.method is not None
        or quant.strategy is not None
        or bool(quant.policy)
        or bool(quant.component_policies)
    )
    if not has_selector:
        raise XQTConfigError(
            "quant.params must specify method, strategy, policy, or component_policies"
        )
    try:
        if quant.component_policies:
            for component in quant.component_policies:
                if not component.enabled:
                    continue
                policy = dict(quant.policy)
                policy.update(component.policy)
                describe_quant_backend_capability(
                    component.backend or quant.backend,
                    method=component.method or quant.method,
                    strategy=component.strategy or quant.strategy,
                    policy=policy,
                )
        else:
            describe_quant_backend_capability(
                quant.backend,
                method=quant.method,
                strategy=quant.strategy,
                policy=quant.policy,
            )
    except ValueError as exc:
        raise XQTConfigError(f"invalid quant.params: {exc}") from exc


def _quant_component_policy_config(value: Any) -> QuantComponentPolicyConfig:
    if isinstance(value, QuantComponentPolicyConfig):
        return value
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(QuantComponentPolicyConfig),
            OmegaConf.create(dict(value)),
        )
        return cast(QuantComponentPolicyConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise XQTConfigError(f"failed to load quant component policy: {exc}") from exc


def _validate_prune_stage_spec(spec: PruneStageSpec) -> None:
    PruneConfig(enabled=True, **stage_spec_to_params(spec))


def _validate_operator_stage_spec(spec: OperatorStageSpec) -> None:
    if spec.benchmark is not None:
        _validate_benchmark_stage_spec(
            spec.benchmark,
            location="operator.params.benchmark",
        )


def _validate_export_stage_spec(spec: ExportStageSpec, *, location: str) -> None:
    if spec.validate is not None:
        _validate_output_diff_config(spec.validate, location=location)


def _validate_deploy_stage_spec(spec: DeployStageSpec, *, location: str) -> None:
    if spec.validate is not None:
        _validate_output_diff_config(spec.validate, location=f"{location}.validate")
    if spec.runtime_handle is not None and spec.runtime_handle.materialize:
        if not spec.runtime_handle.runtime:
            raise XQTConfigError(
                f"{location}.runtime_handle.runtime is required when materialize=true"
            )


def _validate_analyze_stage_spec(spec: AnalyzeStageSpec) -> None:
    if not spec.metrics:
        raise XQTConfigError("analyze.params.metrics must not be empty")
    if spec.top_k is not None and spec.top_k <= 0:
        raise XQTConfigError("analyze.params.top_k must be positive when provided")


def _validate_benchmark_stage_spec(
    spec: BenchmarkStageSpec,
    *,
    location: str,
) -> None:
    if spec.warmup is not None and spec.warmup < 0:
        raise XQTConfigError(f"{location}.warmup must be non-negative")
    if spec.iterations is not None and spec.iterations <= 0:
        raise XQTConfigError(f"{location}.iterations must be positive")


def _validate_output_diff_config(
    spec: OutputDiffConfig,
    *,
    location: str,
) -> None:
    if spec.atol < 0:
        raise XQTConfigError(f"{location}.atol must be non-negative")
    if spec.rtol < 0:
        raise XQTConfigError(f"{location}.rtol must be non-negative")


__all__ = [
    "AnalyzeStageSpec",
    "BenchmarkStageSpec",
    "DeployRuntimeHandleSpec",
    "DeployStageSpec",
    "ExportStageSpec",
    "OperatorStageSpec",
    "PruneStageSpec",
    "QuantStageSpec",
    "StageSpec",
    "build_stage_spec",
    "ensure_stage_spec",
    "stage_params",
    "stage_spec_to_params",
]
