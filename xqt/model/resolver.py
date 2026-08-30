"""Resolve workflow model settings into a concrete compatibility profile."""

from __future__ import annotations

from xqt.core.schema import ModelConfig

from .config import ModelProfile
from .registry import resolve_model_profile


def profile_from_model_config(config: ModelConfig) -> ModelProfile | None:
    """Build the effective profile from a structured ``model`` config.

    A named profile is resolved from the registry and explicit workflow fields
    override its optional declarations. Without ``model.profile`` no registry
    lookup occurs; the explicit model fields still form a useful profile.
    """

    if config.profile:
        base = resolve_model_profile(config.profile)
        profile_id = base.profile_id
        family = config.family or base.family
        loader_target = config.target or base.loader_target
        adapter_target = config.adapter_target or base.adapter_target
        loader_params = dict(base.loader_params)
        loader_params.update(config.params)
        structure_contract = config.structure_contract or base.structure_contract
        inference_adapter = config.inference_adapter or base.inference_adapter
        requirements = dict(base.requirements)
        requirements.update(config.requirements)
        metadata = dict(base.metadata)
        metadata.update(config.metadata)
    else:
        if not any(
            value is not None
            for value in (
                config.family,
                config.target,
                config.structure_contract,
                config.inference_adapter,
                config.adapter_target,
            )
        ):
            return None
        profile_id = config.target or config.family or "inline"
        family = config.family or "unknown"
        loader_target = config.target
        adapter_target = config.adapter_target
        loader_params = dict(config.params)
        structure_contract = config.structure_contract
        inference_adapter = config.inference_adapter
        requirements = dict(config.requirements)
        metadata = dict(config.metadata)

    return ModelProfile(
        profile_id=str(profile_id),
        family=str(family),
        loader_target=loader_target,
        loader_params=loader_params,
        adapter_target=adapter_target,
        structure_contract=structure_contract,
        inference_adapter=inference_adapter,
        requirements=requirements,
        metadata=metadata,
    )


__all__ = ["profile_from_model_config"]
