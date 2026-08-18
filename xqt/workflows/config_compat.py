"""Shared config-input helpers for workflow entrypoints."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any

from xqt.core.config import ConfigInput

if TYPE_CHECKING:
    from xqt.workflows.optimization import OptimizationConfig


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
]
