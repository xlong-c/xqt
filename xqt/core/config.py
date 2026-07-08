"""OmegaConf-based XQT config loading."""

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Union, cast

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from xdl.config.resolver import register_default_resolvers

from .errors import XQTConfigError
from .schema import (
    COMPRESSION_AXES,
    CUTILE_PASS_CONFIG_KEYS,
    CUTLASS_PASS_CONFIG_KEYS,
    OPERATOR_OPT_ENGINES,
    PRUNE_GRANULARITIES,
    PRUNE_SCOPES,
    QuantComponentPolicyConfig,
    QuantConfig,
    SUPPORTED_QUANT_STRATEGIES,
    TASK_TYPES,
    TILELANG_PASS_CONFIG_KEYS,
    XQTConfig,
    XQT_CONFIG_VERSION,
    normalize_quant_strategy,
)

_STRUCTURED_IMPORTANCE_TYPES = {"l1", "l2", "bn_gamma", "usage"}
_AVAILABLE_QUANT_BACKENDS = {"torchao", "onnxruntime_qdq", "pytorch"}
_PLANNED_QUANT_BACKENDS = {"tilelang", "transformers", "bitsandbytes"}
_SUPPORTED_QUANT_BACKENDS = _AVAILABLE_QUANT_BACKENDS | _PLANNED_QUANT_BACKENDS
_AVAILABLE_QUANT_METHODS = {
    "dynamic_int8",
    "fp8_dynamic",
    "fp8_weight_only",
    "weight_only_int4",
    "weight_only_int8",
    "static_qdq_int8",
}
_PLANNED_QUANT_METHODS = {"awq", "gptq"}
_LEGACY_QUANT_METHODS = {
    "int4_weight_only",
    "int8_weight_only",
    "static_int8",
}
_SUPPORTED_QUANT_METHODS = (
    _AVAILABLE_QUANT_METHODS
    | _PLANNED_QUANT_METHODS
    | _LEGACY_QUANT_METHODS
)
_QUANT_POLICY_SELECTOR_KEYS = {
    "include_module_types",
    "exclude_module_types",
    "include_name_patterns",
    "exclude_name_patterns",
    "include_module_names",
    "exclude_module_names",
}

ConfigInput = Union[str, Path, Mapping[str, Any]]


def _strip_local_recipe_extensions(config_value: Any) -> Any:
    if not OmegaConf.is_config(config_value):
        return config_value
    container = OmegaConf.to_container(
        config_value,
        resolve=False,
        enum_to_str=True,
    )
    if isinstance(container, dict):
        container.pop("report", None)
    return OmegaConf.create(container)


def _load_raw_config(config: ConfigInput) -> Any:
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if not path.exists():
            raise XQTConfigError(f"Config file not found: {path}")
        return _strip_local_recipe_extensions(OmegaConf.load(path))
    if isinstance(config, Mapping):
        return _strip_local_recipe_extensions(OmegaConf.create(dict(config)))
    raise XQTConfigError(f"Unsupported config input type: {type(config).__name__}")


def _plain_container(config_value: Any) -> Any:
    if OmegaConf.is_config(config_value):
        return OmegaConf.to_container(
            config_value,
            resolve=False,
            enum_to_str=True,
        )
    return config_value


def _mapping_at(root: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    value: Any = root
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value if isinstance(value, Mapping) else None


def _validate_raw_quant_config(config_value: Any) -> None:
    root = _plain_container(config_value)
    if not isinstance(root, Mapping):
        return
    quant = _mapping_at(root, "compression", "quant")
    if quant is None or not bool(quant.get("enabled", False)):
        return
    if "backend" not in quant or quant.get("backend") in {None, ""}:
        raise XQTConfigError(
            "compression.quant.backend is required when compression.quant.enabled=true"
        )
    has_selector = (
        quant.get("method") is not None
        or quant.get("strategy") is not None
        or bool(quant.get("policy") or {})
    )
    if not has_selector:
        raise XQTConfigError(
            "compression.quant must explicitly set method, strategy, or policy "
            "when enabled=true"
        )


def _validate_pre_export_fusion(config_value: Any, location: str) -> None:
    if config_value is None:
        return
    if not isinstance(config_value, Mapping):
        raise XQTConfigError(f"{location} must be a mapping")
    mode = str(config_value.get("mode", "eager"))
    if mode not in {"eager", "fx"}:
        raise XQTConfigError(f"{location}.mode must be eager or fx")
    if mode == "eager" and bool(config_value.get("enabled", False)):
        groups = config_value.get("modules_to_fuse")
        if not isinstance(groups, list) or not groups:
            raise XQTConfigError(
                f"{location}.modules_to_fuse must be a non-empty list "
                "when mode=eager and enabled=true"
            )


def _validate_quant_string_list(values: list[str], location: str) -> None:
    if not isinstance(values, list):
        raise XQTConfigError(f"{location} must be a list of strings")
    if any(not isinstance(value, str) or not value for value in values):
        raise XQTConfigError(f"{location} must contain only non-empty strings")


def _validate_quant_component_lists(
    component: QuantConfig | QuantComponentPolicyConfig,
    location: str,
) -> None:
    _validate_quant_string_list(component.keep_high_precision, f"{location}.keep_high_precision")
    _validate_quant_string_list(component.skip_quantize, f"{location}.skip_quantize")
    _validate_quant_string_list(component.force_quantize, f"{location}.force_quantize")
    overlap = sorted(set(component.skip_quantize) & set(component.force_quantize))
    if overlap:
        raise XQTConfigError(
            f"{location}.skip_quantize and {location}.force_quantize overlap: {overlap}"
        )


def _validate_quant_policy(
    policy: Mapping[str, Any],
    location: str,
) -> None:
    for key in _QUANT_POLICY_SELECTOR_KEYS:
        if key not in policy:
            continue
        value = policy[key]
        if not isinstance(value, list):
            raise XQTConfigError(f"{location}.{key} must be a list of strings")
        if any(not isinstance(item, str) or not item for item in value):
            raise XQTConfigError(f"{location}.{key} must contain only non-empty strings")
        if key.endswith("_patterns"):
            for pattern in value:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise XQTConfigError(
                        f"{location}.{key} contains invalid regex {pattern!r}: {exc}"
                    ) from exc
    strategy = normalize_quant_strategy(policy.get("strategy"), policy)
    if strategy is not None and strategy not in SUPPORTED_QUANT_STRATEGIES:
        allowed = ", ".join(SUPPORTED_QUANT_STRATEGIES)
        raise XQTConfigError(f"{location}.strategy must be one of: {allowed}")


def _validate_quant_config(quant_config: QuantConfig) -> None:
    if not quant_config.enabled:
        return
    if quant_config.backend not in _SUPPORTED_QUANT_BACKENDS:
        allowed = ", ".join(sorted(_SUPPORTED_QUANT_BACKENDS))
        raise XQTConfigError(f"compression.quant.backend must be one of: {allowed}")
    if quant_config.method is not None and quant_config.method not in _SUPPORTED_QUANT_METHODS:
        allowed = ", ".join(sorted(_SUPPORTED_QUANT_METHODS))
        raise XQTConfigError(f"compression.quant.method must be one of: {allowed}")
    if quant_config.strategy is not None:
        strategy = normalize_quant_strategy(quant_config.strategy, quant_config.policy)
        if strategy not in SUPPORTED_QUANT_STRATEGIES:
            allowed = ", ".join(SUPPORTED_QUANT_STRATEGIES)
            raise XQTConfigError(f"compression.quant.strategy must be one of: {allowed}")
    if (
        quant_config.method is None
        and normalize_quant_strategy(quant_config.strategy, quant_config.policy) is None
        and not quant_config.policy
    ):
        raise XQTConfigError(
            "compression.quant must explicitly set method, strategy, or policy "
            "when enabled=true"
        )
    _validate_quant_component_lists(quant_config, "compression.quant")
    _validate_quant_policy(quant_config.policy, "compression.quant.policy")
    _validate_quant_string_list(
        quant_config.analysis_only_modules,
        "compression.quant.analysis_only_modules",
    )

    seen_component_names: set[str] = set()
    for index, component in enumerate(quant_config.component_policies):
        location = f"compression.quant.component_policies.{index}"
        if not component.name:
            raise XQTConfigError(f"{location}.name must be a non-empty string")
        if component.name in seen_component_names:
            raise XQTConfigError(
                f"compression.quant.component_policies[*].name must be unique: {component.name}"
            )
        seen_component_names.add(component.name)
        if component.backend is not None and component.backend not in _SUPPORTED_QUANT_BACKENDS:
            allowed = ", ".join(sorted(_SUPPORTED_QUANT_BACKENDS))
            raise XQTConfigError(f"{location}.backend must be one of: {allowed}")
        if component.method is not None and component.method not in _SUPPORTED_QUANT_METHODS:
            allowed = ", ".join(sorted(_SUPPORTED_QUANT_METHODS))
            raise XQTConfigError(f"{location}.method must be one of: {allowed}")
        if component.strategy is not None:
            strategy = normalize_quant_strategy(component.strategy, component.policy)
            if strategy not in SUPPORTED_QUANT_STRATEGIES:
                allowed = ", ".join(SUPPORTED_QUANT_STRATEGIES)
                raise XQTConfigError(f"{location}.strategy must be one of: {allowed}")
        _validate_quant_component_lists(component, location)
        _validate_quant_policy(component.policy, f"{location}.policy")
        _validate_pre_export_fusion(
            component.policy.get("pre_export_fusion"),
            f"{location}.policy.pre_export_fusion",
        )


def _validate_operator_optimization_config(config: XQTConfig) -> None:
    operator_config = config.operator_optimization
    if operator_config.default_engine not in OPERATOR_OPT_ENGINES:
        allowed = ", ".join(OPERATOR_OPT_ENGINES)
        raise XQTConfigError(
            "operator_optimization.default_engine must be one of: "
            f"{allowed}"
        )
    if operator_config.stage not in {"after_compression"}:
        raise XQTConfigError(
            "operator_optimization.stage must be after_compression"
        )

    seen_names: set[str] = set()
    for index, target in enumerate(operator_config.targets):
        location = f"operator_optimization.targets.{index}"
        if not target.name:
            raise XQTConfigError(f"{location}.name must be a non-empty string")
        if target.name in seen_names:
            raise XQTConfigError(
                f"operator_optimization.targets[*].name must be unique: {target.name}"
            )
        seen_names.add(target.name)
        if target.engine is None:
            raise XQTConfigError(f"{location}.engine is required")
        if target.engine not in OPERATOR_OPT_ENGINES:
            allowed = ", ".join(OPERATOR_OPT_ENGINES)
            raise XQTConfigError(f"{location}.engine must be one of: {allowed}")
        if (
            target.target is None
            and target.name != "model"
            and target.engine != "deployment_engine"
        ):
            raise XQTConfigError(
                f"{location}.target is required unless {location}.name=model "
                "or engine=deployment_engine"
            )
        if target.fallback not in {"eager"}:
            raise XQTConfigError(f"{location}.fallback must be eager")
        if target.min_speedup <= 1.0:
            raise XQTConfigError(f"{location}.min_speedup must be greater than 1.0")
        if target.validate.atol < 0 or target.validate.rtol < 0:
            raise XQTConfigError(
                f"{location}.validate.atol and {location}.validate.rtol must be non-negative"
            )
        if not isinstance(target.patterns, list):
            raise XQTConfigError(f"{location}.patterns must be a list of strings")
        if any(not isinstance(pattern, str) or not pattern for pattern in target.patterns):
            raise XQTConfigError(f"{location}.patterns must contain only non-empty strings")
        if target.engine == "torch_compile":
            if target.mode is not None and target.mode not in {
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            }:
                raise XQTConfigError(
                    f"{location}.mode is not a supported torch.compile mode"
                )
        if target.engine == "tilelang":
            if target.tilelang.target != "cuda":
                raise XQTConfigError(f"{location}.tilelang.target must be cuda")
            unknown_keys = sorted(
                set(target.tilelang.pass_configs) - set(TILELANG_PASS_CONFIG_KEYS)
            )
            if unknown_keys:
                allowed = ", ".join(TILELANG_PASS_CONFIG_KEYS)
                raise XQTConfigError(
                    f"{location}.tilelang.pass_configs contains unknown keys {unknown_keys}. "
                    f"Allowed: {allowed}"
                )
        if target.engine == "cutile":
            if target.cutile.target != "cuda":
                raise XQTConfigError(f"{location}.cutile.target must be cuda")
            unknown_keys = sorted(
                set(target.cutile.pass_configs) - set(CUTILE_PASS_CONFIG_KEYS)
            )
            if unknown_keys:
                allowed = ", ".join(CUTILE_PASS_CONFIG_KEYS)
                raise XQTConfigError(
                    f"{location}.cutile.pass_configs contains unknown keys {unknown_keys}. "
                    f"Allowed: {allowed}"
                )
        if target.engine == "cutlass":
            if len(target.cutlass.tile_shape) != 3:
                raise XQTConfigError(f"{location}.cutlass.tile_shape must contain three integers")
            if any(not isinstance(value, int) or value <= 0 for value in target.cutlass.tile_shape):
                raise XQTConfigError(f"{location}.cutlass.tile_shape must contain positive integers")
            if target.cutlass.cluster_shape is not None:
                if len(target.cutlass.cluster_shape) != 3:
                    raise XQTConfigError(f"{location}.cutlass.cluster_shape must contain three integers")
                if any(not isinstance(value, int) or value <= 0 for value in target.cutlass.cluster_shape):
                    raise XQTConfigError(f"{location}.cutlass.cluster_shape must contain positive integers")
            unknown_keys = sorted(
                set(target.cutlass.pass_configs) - set(CUTLASS_PASS_CONFIG_KEYS)
            )
            if unknown_keys:
                allowed = ", ".join(CUTLASS_PASS_CONFIG_KEYS)
                raise XQTConfigError(
                    f"{location}.cutlass.pass_configs contains unknown keys {unknown_keys}. "
                    f"Allowed: {allowed}"
                )


def _validate_config(config: XQTConfig) -> None:
    if config.config_version != XQT_CONFIG_VERSION:
        raise XQTConfigError(
            f"Unsupported config version {config.config_version}. "
            f"Expected {XQT_CONFIG_VERSION}."
        )

    invalid_axes = [axis for axis in config.compression.axes if axis not in COMPRESSION_AXES]
    if invalid_axes:
        allowed = ", ".join(COMPRESSION_AXES)
        raise XQTConfigError(f"Unsupported compression axes {invalid_axes}. Allowed: {allowed}")

    if config.benchmark.warmup < 0:
        raise XQTConfigError("benchmark.warmup must be non-negative")
    if config.benchmark.iterations <= 0:
        raise XQTConfigError("benchmark.iterations must be positive")
    if config.analysis.compare_to not in {"baseline"}:
        raise XQTConfigError("analysis.compare_to must be baseline")
    if config.analysis.top_k is not None and config.analysis.top_k <= 0:
        raise XQTConfigError("analysis.top_k must be positive when provided")
    if not config.analysis.metrics:
        raise XQTConfigError("analysis.metrics must not be empty")
    if config.task.type not in TASK_TYPES:
        allowed = ", ".join(TASK_TYPES)
        raise XQTConfigError(f"task.type must be one of: {allowed}")
    postprocess = config.task.detection_postprocess
    if postprocess.score_threshold < 0 or postprocess.score_threshold > 1:
        raise XQTConfigError("task.detection_postprocess.score_threshold must be in [0, 1]")
    if postprocess.iou_threshold < 0 or postprocess.iou_threshold > 1:
        raise XQTConfigError("task.detection_postprocess.iou_threshold must be in [0, 1]")
    if postprocess.max_detections <= 0:
        raise XQTConfigError("task.detection_postprocess.max_detections must be positive")
    if postprocess.format not in {"auto", "end2end", "yolo_raw"}:
        raise XQTConfigError(
            "task.detection_postprocess.format must be auto, end2end, or yolo_raw"
        )
    if postprocess.box_format not in {"xyxy", "cxcywh"}:
        raise XQTConfigError(
            "task.detection_postprocess.box_format must be xyxy or cxcywh"
        )
    if postprocess.score_activation not in {"identity", "sigmoid", "softmax"}:
        raise XQTConfigError(
            "task.detection_postprocess.score_activation must be identity, sigmoid, or softmax"
        )
    _validate_quant_config(config.compression.quant)
    _validate_operator_optimization_config(config)
    if config.compression.prune.target_sparsity < 0 or config.compression.prune.target_sparsity > 1:
        raise XQTConfigError("compression.prune.target_sparsity must be in [0, 1]")
    prune_config = config.compression.prune
    if prune_config.granularity is not None and prune_config.granularity not in PRUNE_GRANULARITIES:
        allowed = ", ".join(PRUNE_GRANULARITIES)
        raise XQTConfigError(
            "compression.prune.granularity must be one of: "
            f"{allowed}"
        )
    if prune_config.scope not in PRUNE_SCOPES:
        allowed = ", ".join(PRUNE_SCOPES)
        raise XQTConfigError(f"compression.prune.scope must be one of: {allowed}")
    if prune_config.method == "structured" and prune_config.granularity is None:
        raise XQTConfigError(
            "compression.prune.granularity is required when "
            "compression.prune.method=structured"
        )
    importance_type = prune_config.importance.get(
        "type",
        prune_config.importance.get("metric"),
    )
    if importance_type is not None and str(importance_type) not in _STRUCTURED_IMPORTANCE_TYPES:
        allowed = ", ".join(sorted(_STRUCTURED_IMPORTANCE_TYPES))
        raise XQTConfigError(
            "compression.prune.importance.type or compression.prune.importance.metric "
            f"must be one of: {allowed}"
        )
    keep_indices = prune_config.selection.get("keep_indices")
    if keep_indices is not None:
        if not isinstance(keep_indices, Mapping):
            raise XQTConfigError("compression.prune.selection.keep_indices must be a mapping")
        for module_name, indices in keep_indices.items():
            if not isinstance(module_name, str):
                raise XQTConfigError(
                    "compression.prune.selection.keep_indices keys must be strings"
                )
            if not isinstance(indices, list) or not indices:
                raise XQTConfigError(
                    "compression.prune.selection.keep_indices values must be non-empty lists"
                )
            if any(not isinstance(index, int) for index in indices):
                raise XQTConfigError(
                    "compression.prune.selection.keep_indices values must contain only integers"
                )
    for index, target in enumerate(config.export.targets):
        _validate_pre_export_fusion(
            target.params.get("pre_export_fusion"),
            f"export.targets.{index}.params.pre_export_fusion",
        )

    _validate_pre_export_fusion(
        config.compression.quant.policy.get("pre_export_fusion"),
        "compression.quant.policy.pre_export_fusion",
    )


def load_xqt_config(
    config: ConfigInput,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    resolve: bool = True,
) -> XQTConfig:
    """Load an XQT recipe using structured defaults then YAML or mapping overrides."""

    register_default_resolvers()
    raw_config = _load_raw_config(config)
    raw_nodes = [raw_config]
    if overrides:
        raw_nodes.append(OmegaConf.create(dict(overrides)))
    raw_user_config = OmegaConf.merge(*raw_nodes)
    _validate_raw_quant_config(raw_user_config)
    nodes = [OmegaConf.structured(XQTConfig), raw_user_config]

    try:
        merged = OmegaConf.merge(*nodes)
        if resolve:
            OmegaConf.resolve(merged)
        loaded = cast(XQTConfig, OmegaConf.to_object(merged))
    except OmegaConfBaseException as exc:
        raise XQTConfigError(f"Failed to load XQT config: {exc}") from exc

    _validate_config(loaded)
    return loaded


def xqt_config_to_dict(config: XQTConfig) -> dict[str, Any]:
    """Convert an XQT config dataclass to a plain dictionary."""

    return asdict(config)


__all__ = [
    "ConfigInput",
    "load_xqt_config",
    "xqt_config_to_dict",
]
