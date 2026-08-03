"""Adapters from XDL model artifacts to XQT contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.core.config import ConfigInput
from xqt.core.types import XQTContext
from xqt.pipeline.runner import create_context
from xqt.workflows import OptimizationConfig
from xqt.workflows.config_compat import ensure_optimization_workflow_config


def xdl_setup_to_xqt_context(
    setup: Any,
    config: ConfigInput | OptimizationConfig,
) -> XQTContext:
    """Create an XQT context from the model in a TrainSetup-like object."""

    workflow_config = ensure_optimization_workflow_config(
        config,
        caller="xdl_setup_to_xqt_context()",
    )
    context = create_context(
        workflow_config,
        model=getattr(setup, "model", None),
        metrics={
            "xdl_setup": {
                "device": getattr(setup, "device", None),
            }
        },
    )
    return context


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    state_key: Optional[str] = None,
    strict: bool = True,
) -> nn.Module:
    """Load a PyTorch or XDL-style checkpoint into a model."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict: Mapping[str, Any]
    # Support the common checkpoint layouts seen in XDL experiments:
    # explicit state_key, model_state_dict, nested state_dict, or a raw state dict.
    if state_key is not None:
        state_dict = checkpoint[state_key]
    elif isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        xdl_state = checkpoint["state_dict"]
        if isinstance(xdl_state, Mapping) and len(xdl_state) == 1:
            only_state = next(iter(xdl_state.values()))
            state_dict = only_state if isinstance(only_state, Mapping) else xdl_state
        else:
            state_dict = xdl_state
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return model


def xdl_checkpoint_to_xqt_context(
    model: nn.Module,
    checkpoint_path: str | Path,
    config: ConfigInput | OptimizationConfig,
    *,
    map_location: str | torch.device = "cpu",
    state_key: Optional[str] = None,
    strict: bool = True,
) -> XQTContext:
    """Load a checkpoint and create an XQT context."""

    workflow_config = ensure_optimization_workflow_config(
        config,
        caller="xdl_checkpoint_to_xqt_context()",
    )
    load_checkpoint_into_model(
        model,
        checkpoint_path,
        map_location=map_location,
        state_key=state_key,
        strict=strict,
    )
    # Keep the checkpoint path in the manifest so downstream export and
    # reporting can trace the run back to its training artifact.
    context = create_context(
        workflow_config,
        model=model,
    )
    if context.manifest is not None:
        context.manifest.source_checkpoint = str(Path(checkpoint_path))
    return context


__all__ = [
    "load_checkpoint_into_model",
    "xdl_checkpoint_to_xqt_context",
    "xdl_setup_to_xqt_context",
]
