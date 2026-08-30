"""Stage workflow loader for XQT (core-owned, no pipeline/workflows cycle)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf

from xqt.core.config import ConfigInput, register_default_resolvers
from xqt.core.base.errors import XQTConfigError
from xqt.core.workflow_schema import OptimizationConfig

STAGE_KINDS = {
    "benchmark",
    "prune",
    "quant",
    "operator",
    "export",
    "deploy",
    "analyze",
}

_REMOVED_WORKFLOW_TOP_LEVEL_KEYS = {
    "analysis",
    "compression",
    "config_version",
    "export",
    "operator_optimization",
    "validation",
}

_REMOVED_KEY_MIGRATIONS = {
    "compression": "split into stages[*].params for quant / prune stages",
    "operator_optimization": "move under stages[*].params for operator stages",
    "export": "move under stages[*].params.targets for export / deploy stages",
    "analysis": "move under stages[*].params for analyze stages",
    "validation": "move output_diff thresholds under stages[*].params.validate",
    "config_version": "drop it; OptimizationConfig has no public version field",
}


def _validate_raw_optimization_config(raw_config: Any) -> None:
    raw = OmegaConf.to_container(raw_config, resolve=False, enum_to_str=True)
    if not isinstance(raw, Mapping):
        return
    removed = sorted(set(raw) & _REMOVED_WORKFLOW_TOP_LEVEL_KEYS)
    if removed:
        migration = ", ".join(
            f"{key} -> {_REMOVED_KEY_MIGRATIONS[key]}" for key in removed
        )
        raise XQTConfigError(
            "OptimizationConfig does not accept removed recipe top-level keys: "
            f"{removed}. Migration: {migration}."
        )


def _attach_stage_specs(config: OptimizationConfig) -> None:
    from xqt.workflows.stage_specs import ensure_stage_spec

    seen: set[str] = set()
    for index, stage in enumerate(config.stages):
        if not stage.name:
            raise XQTConfigError(f"stages.{index}.name is required")
        if stage.name in seen:
            raise XQTConfigError(f"stage names must be unique: {stage.name}")
        seen.add(stage.name)
        if stage.kind not in STAGE_KINDS:
            allowed = ", ".join(sorted(STAGE_KINDS))
            raise XQTConfigError(
                f"unsupported stage kind {stage.kind}. Allowed: {allowed}"
            )
        ensure_stage_spec(stage, rebuild=True)


def load_optimization_config(
    config: ConfigInput | OptimizationConfig,
) -> OptimizationConfig:
    """Load a stage workflow config using OmegaConf structured defaults."""

    from dataclasses import is_dataclass

    if is_dataclass(config) and isinstance(config, OptimizationConfig):
        _attach_stage_specs(config)
        return config

    register_default_resolvers()
    raw = (
        OmegaConf.load(config)
        if isinstance(config, (str, Path))
        else OmegaConf.create(config)
    )
    try:
        _validate_raw_optimization_config(raw)
        merged = OmegaConf.merge(OmegaConf.structured(OptimizationConfig), raw)
        OmegaConf.resolve(merged)
        loaded = cast(OptimizationConfig, OmegaConf.to_object(merged))
        _attach_stage_specs(loaded)
        return loaded
    except Exception as exc:
        if isinstance(exc, XQTConfigError):
            raise
        raise XQTConfigError(f"failed to load XQT optimization config: {exc}") from exc


__all__ = [
    "STAGE_KINDS",
    "load_optimization_config",
]
