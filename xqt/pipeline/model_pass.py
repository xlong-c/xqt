"""Model-loading pass for XQT stage execution."""

from __future__ import annotations

import copy
from typing import Any

from torch import nn

from xqt.core.imports import build_target
from xqt.core.types import XQTContext
from xqt.model import resolve_model_adapter

from .pass_helpers.context import _context_model_params, _context_model_target


class LoadModelPass:
    """Build the configured PyTorch model and its reference snapshot."""

    name = "load_model"

    def run(self, context: XQTContext) -> XQTContext:
        if context.model is not None:
            return context
        target = _context_model_target(context)
        adapter = (
            resolve_model_adapter(context.model_profile)
            if context.model_profile is not None
            else None
        )
        if adapter is not None:
            model = adapter.load(
                context.model_checkpoint,
                **_context_model_params(context),
            )
            model = adapter.adapt(model, **_context_model_params(context))
            context.structure_contract = adapter.structure_contract(model)
            context.inference_contract = adapter.inference_contract(model)
        else:
            if not target:
                raise ValueError(
                    "model.target or model.adapter_target is required when "
                    "context.model is not set"
                )
            model = build_target(target, _context_model_params(context))
        if not isinstance(model, nn.Module):
            raise TypeError("model.target must build a torch.nn.Module")
        model.eval()
        context.model = model
        if context.reference_model is None:
            context.reference_model = copy.deepcopy(model)
        return context


__all__ = ["LoadModelPass"]
