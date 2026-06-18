"""OmegaConf-based XQT config loading."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Union, cast

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from xdl.config.resolver import register_default_resolvers

from .errors import XQTConfigError
from .schema import COMPRESSION_AXES, XQTConfig, XQT_CONFIG_VERSION

ConfigInput = Union[str, Path, Mapping[str, Any]]


def _load_raw_config(config: ConfigInput) -> Any:
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if not path.exists():
            raise XQTConfigError(f"Config file not found: {path}")
        return OmegaConf.load(path)
    if isinstance(config, Mapping):
        return OmegaConf.create(dict(config))
    raise XQTConfigError(f"Unsupported config input type: {type(config).__name__}")


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
    if config.compression.prune.target_sparsity < 0 or config.compression.prune.target_sparsity > 1:
        raise XQTConfigError("compression.prune.target_sparsity must be in [0, 1]")
    if config.compression.diffusion_distill.teacher_steps <= 0:
        raise XQTConfigError("compression.diffusion_distill.teacher_steps must be positive")
    if config.compression.diffusion_distill.student_steps <= 0:
        raise XQTConfigError("compression.diffusion_distill.student_steps must be positive")
    def validate_pre_export_fusion(config_value: Any, location: str) -> None:
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

    for index, target in enumerate(config.export.targets):
        validate_pre_export_fusion(
            target.params.get("pre_export_fusion"),
            f"export.targets.{index}.params.pre_export_fusion",
        )

    validate_pre_export_fusion(
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
    nodes = [OmegaConf.structured(XQTConfig), _load_raw_config(config)]
    if overrides:
        nodes.append(OmegaConf.create(dict(overrides)))

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
