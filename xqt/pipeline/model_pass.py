"""Model-loading pass for XQT stage execution."""

from __future__ import annotations

import copy
from typing import Any

from torch import nn

from xqt.contracts.model_structure import (
    compute_topology_fingerprint,
    is_structure_contract_valid_for_model,
    resolve_and_validate_structure_contract,
    structure_contract_mismatches,
)
from xqt.core.base import XQTConfigError
from xqt.core.imports import build_target
from xqt.core.types import XQTContext
from xqt.model import resolve_model_adapter

from .pass_helpers.context import _context_model_params, _context_model_target


class LoadModelPass:
    """Build the configured PyTorch model and its reference snapshot."""

    name = "load_model"

    def run(self, context: XQTContext) -> XQTContext:
        adapter = (
            resolve_model_adapter(context.model_profile)
            if context.model_profile is not None
            else None
        )
        if context.model is None:
            target = _context_model_target(context)
            if adapter is not None:
                model = adapter.load(
                    context.model_checkpoint,
                    **_context_model_params(context),
                )
                model = adapter.adapt(model, **_context_model_params(context))
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
        else:
            if not isinstance(context.model, nn.Module):
                raise TypeError("context.model must be a torch.nn.Module")
            context.model.eval()

        if context.reference_model is None:
            context.reference_model = copy.deepcopy(context.model)

        if context.structure_contract is None:
            has_explicit_profile = False
            if context.model_profile is not None:
                has_contract_or_adapter = (
                    getattr(context.model_profile, "structure_contract", None) is not None
                    or getattr(context.model_profile, "adapter_target", None) is not None
                )
                is_standalone_target = (
                    context.model_profile.profile_id == context.model_target
                    or context.model_profile.profile_id == "inline"
                )
                if has_contract_or_adapter or not is_standalone_target:
                    has_explicit_profile = True

            context.structure_contract = resolve_and_validate_structure_contract(
                context.model,
                profile=context.model_profile,
                adapter=adapter,
                strict=has_explicit_profile,
            )
        else:
            if not is_structure_contract_valid_for_model(
                context.structure_contract, context.model
            ):
                mismatches = structure_contract_mismatches(
                    context.model, context.structure_contract
                )
                if not mismatches.is_consistent:
                    raise XQTConfigError(
                        "Supplied ModelStructureContract does not match model topology: "
                        f"{mismatches.to_dict()}"
                    )
                context.structure_contract = (
                    context.structure_contract.with_topology_fingerprint(
                        compute_topology_fingerprint(context.model)
                    )
                )

        if context.inference_contract is None and adapter is not None:
            context.inference_contract = adapter.inference_contract(context.model)

        return context


__all__ = ["LoadModelPass"]
