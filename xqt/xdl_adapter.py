"""Adapters between XDL training objects and XQT contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.core.config import ConfigInput
from xqt.core.schema import XQTConfig
from xqt.core.types import XQTContext
from xqt.pipeline.runner import create_context


def xdl_setup_to_xqt_context(
    setup: Any,
    config: ConfigInput | XQTConfig,
    *,
    use_validation_loader: bool = True,
    include_test_loader: bool = False,
    teacher: Optional[nn.Module] = None,
) -> XQTContext:
    """Create an XQT context from an XDL TrainSetup-like object."""

    data: dict[str, Any] = {}
    train_loader = getattr(setup, "train_loader", None)
    if train_loader is not None:
        data["train"] = train_loader
    val_loader = getattr(setup, "val_loader", None)
    if use_validation_loader and val_loader is not None:
        data["validation"] = val_loader
    test_loader = getattr(setup, "test_loader", None)
    if include_test_loader and test_loader is not None:
        data["test"] = test_loader

    context = create_context(
        config,
        model=getattr(setup, "model", None),
        teacher=teacher,
        data=data,
        metrics={
            "xdl_setup": {
                "device": getattr(setup, "device", None),
                "batch_size": getattr(setup, "batch_size", None),
                "num_epochs": getattr(setup, "num_epochs", None),
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
    config: ConfigInput | XQTConfig,
    *,
    map_location: str | torch.device = "cpu",
    state_key: Optional[str] = None,
    strict: bool = True,
    data: Optional[Mapping[str, Any]] = None,
    teacher: Optional[nn.Module] = None,
) -> XQTContext:
    """Load a checkpoint and create an XQT context."""

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
        config,
        model=model,
        teacher=teacher,
        data=data,
    )
    if context.manifest is not None:
        context.manifest.source_checkpoint = str(Path(checkpoint_path))
    return context


__all__ = [
    "load_checkpoint_into_model",
    "xdl_checkpoint_to_xqt_context",
    "xdl_setup_to_xqt_context",
]
