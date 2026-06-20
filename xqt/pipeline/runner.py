"""Minimal YAML runner for XQT recipes."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from xqt.core.artifact import ArtifactManifest, file_sha256
from xqt.core.config import ConfigInput, load_xqt_config, xqt_config_to_dict
from xqt.core.registry import PASS_REGISTRY, XQTRegistry
from xqt.core.schema import XQTConfig
from xqt.core.types import XQTContext

from .pass_manager import SequentialPipeline


DEFAULT_COMPRESSION_PASS_ORDER = (
    "distill",
    "prune",
    "quant",
    "operator_optimization",
    "diffusion_distill",
)

DEFAULT_PASS_ORDER = (
    "load_model",
    "load_data",
    "baseline_eval",
    *DEFAULT_COMPRESSION_PASS_ORDER,
    "analyze",
    "export",
    "benchmark",
    "write_reports",
)


def _ensure_builtin_passes_registered() -> None:
    """Import built-in passes for their registration side effects."""

    import xqt.pipeline.passes  # noqa: F401


def _ensure_config(config: ConfigInput | XQTConfig) -> XQTConfig:
    """Accept either a loaded config object or a config source."""

    if is_dataclass(config) and isinstance(config, XQTConfig):
        return config
    return load_xqt_config(config)


def enabled_pass_names(config: XQTConfig) -> list[str]:
    """Return enabled compression pass names in the default execution order."""

    compression = config.compression
    enabled = {
        "distill": compression.distill.enabled,
        "prune": compression.prune.enabled,
        "quant": compression.quant.enabled,
        "operator_optimization": config.operator_optimization.enabled,
        "diffusion_distill": compression.diffusion_distill.enabled,
    }
    return [name for name in DEFAULT_COMPRESSION_PASS_ORDER if enabled[name]]


def default_pass_names(config: XQTConfig) -> list[str]:
    """Return the default runnable pass list for a recipe."""

    compression_passes = set(enabled_pass_names(config))
    names: list[str] = []
    for name in DEFAULT_PASS_ORDER:
        if name in DEFAULT_COMPRESSION_PASS_ORDER and name not in compression_passes:
            continue
        if name == "analyze" and not config.analysis.enabled:
            continue
        if name == "export" and not config.export.targets:
            continue
        names.append(name)
    return names


def create_manifest(config: XQTConfig) -> ArtifactManifest:
    """Create a manifest initialized from the recipe config."""

    source_checksum: Optional[str] = None
    if config.model.checkpoint:
        checkpoint_path = Path(config.model.checkpoint).expanduser()
        if checkpoint_path.is_file():
            source_checksum = file_sha256(checkpoint_path)

    return ArtifactManifest(
        project_name=config.project.name,
        source_checkpoint=config.model.checkpoint,
        source_checksum=source_checksum,
        compression_axes=list(config.compression.axes),
        config_snapshot=xqt_config_to_dict(config),
    )


def create_context(
    config: ConfigInput | XQTConfig,
    *,
    model: Any = None,
    teacher: Any = None,
    data: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    manifest: Optional[ArtifactManifest] = None,
) -> XQTContext:
    """Build an XQTContext from a config path, mapping, or dataclass."""

    loaded_config = _ensure_config(config)
    return XQTContext(
        config=loaded_config,
        model=model,
        reference_model=copy.deepcopy(model) if model is not None else None,
        teacher=teacher,
        data=dict(data or {}),
        artifacts=dict(artifacts or {}),
        metrics=dict(metrics or {}),
        device=loaded_config.model.device,
        manifest=manifest or create_manifest(loaded_config),
    )


def build_pipeline_from_config(
    config: XQTConfig,
    *,
    pass_names: Optional[Sequence[str]] = None,
    pass_registry: XQTRegistry = PASS_REGISTRY,
    pass_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> SequentialPipeline:
    """Build a sequential pipeline from enabled pass names."""

    _ensure_builtin_passes_registered()
    names = list(pass_names) if pass_names is not None else default_pass_names(config)
    params_by_name = dict(pass_params or {})
    passes = [
        pass_registry.build(name, **dict(params_by_name.get(name, {})))
        for name in names
    ]
    return SequentialPipeline.from_iterable(passes)


def run_xqt_recipe(
    config: ConfigInput | XQTConfig,
    *,
    model: Any = None,
    teacher: Any = None,
    data: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    manifest: Optional[ArtifactManifest] = None,
    pass_names: Optional[Sequence[str]] = None,
    pass_registry: XQTRegistry = PASS_REGISTRY,
    pass_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    write_manifest: bool = True,
    manifest_name: str = "manifest.json",
) -> XQTContext:
    """Load an XQT recipe, run enabled passes, and optionally write manifest JSON."""

    context = create_context(
        config,
        model=model,
        teacher=teacher,
        data=data,
        artifacts=artifacts,
        metrics=metrics,
        manifest=manifest,
    )
    pipeline = build_pipeline_from_config(
        context.config,
        pass_names=pass_names,
        pass_registry=pass_registry,
        pass_params=pass_params,
    )
    output = pipeline.run(context)

    if write_manifest and output.manifest is not None:
        manifest_path = Path(output.config.project.artifact_dir) / manifest_name
        output.manifest.write_json(manifest_path)
        output.artifacts["manifest"] = manifest_path

    return output


__all__ = [
    "DEFAULT_COMPRESSION_PASS_ORDER",
    "DEFAULT_PASS_ORDER",
    "build_pipeline_from_config",
    "create_context",
    "create_manifest",
    "default_pass_names",
    "enabled_pass_names",
    "run_xqt_recipe",
]
