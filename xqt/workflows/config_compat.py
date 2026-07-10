"""Shared config-input compatibility helpers for workflow migration."""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from omegaconf import OmegaConf

from xqt.core.config import ConfigInput
from xqt.core.errors import XQTConfigError

if TYPE_CHECKING:
    from xqt.workflows.optimization import OptimizationConfig


_LEGACY_RECIPE_TOP_LEVEL_KEYS = {
    "analysis",
    "compression",
    "config_version",
    "export",
    "operator_optimization",
    "validation",
}

_WORKFLOW_ONLY_TOP_LEVEL_KEYS = {
    "compression_axes",
    "device",
    "hardware",
    "stages",
}


def _top_level_keys(config: ConfigInput | OptimizationConfig | Any) -> set[str] | None:
    """Return raw top-level keys for mapping or path inputs."""

    if is_dataclass(config):
        return None
    raw: Any
    if isinstance(config, Mapping):
        raw = config
    elif isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if not path.exists():
            raise XQTConfigError(f"Config file not found: {path}")
        raw = OmegaConf.load(path)
    else:
        return None
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=False, enum_to_str=True)
    return set(raw) if isinstance(raw, Mapping) else None


def is_optimization_workflow_source(
    config: ConfigInput | OptimizationConfig | Any,
) -> bool:
    """Detect stage-workflow inputs without conflating them with removed recipes."""

    if is_dataclass(config):
        from xqt.workflows.optimization import OptimizationConfig

        return isinstance(config, OptimizationConfig)
    keys = _top_level_keys(config)
    if keys is None:
        return False
    if keys & _LEGACY_RECIPE_TOP_LEVEL_KEYS:
        return False
    return bool(keys & _WORKFLOW_ONLY_TOP_LEVEL_KEYS)


def ensure_optimization_workflow_config(
    config: ConfigInput | OptimizationConfig | Any,
    *,
    caller: str,
) -> OptimizationConfig:
    """Load workflow config inputs."""

    from xqt.workflows.optimization import OptimizationConfig, load_optimization_config

    if is_dataclass(config) and isinstance(config, OptimizationConfig):
        return config
    del caller
    return load_optimization_config(config)


__all__ = [
    "ensure_optimization_workflow_config",
    "is_optimization_workflow_source",
]
