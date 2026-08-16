"""Materialize multi-branch composite modules for Infer.

Applies compute_config branch contracts onto storage-only shells that implement
materialize_compute(spec). Does not import quantizers or re-run calibration.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Protocol, runtime_checkable

from torch import nn

from xqt.contracts.compute import (
    ComputeConfig,
    ModuleComputeSpec,
    compute_config_from_mapping,
    normalize_compute_contract,
)
from xqt.runtime.composite_branch import replace_submodule
from xqt.contracts.composite import CompositeAddLinear
from xqt.runtime.modules.composite_add import (
    materialize_composite_compute as _materialize_composite_compute,
)


@runtime_checkable
class SupportsCompositeMaterialize(Protocol):
    """Storage shell that can bind compute paths from a ModuleComputeSpec."""

    def materialize_compute(self, spec: ModuleComputeSpec) -> nn.Module:
        """Return self or a replacement module with compute bound."""


def materialize_composite_compute(
    model: nn.Module,
    compute_config: ComputeConfig | Mapping[str, Any] | None,
    *,
    inplace: bool = True,
) -> nn.Module:
    """Materialize dual-branch modules declared in compute_config."""

    config = (
        compute_config
        if isinstance(compute_config, ComputeConfig)
        else compute_config_from_mapping(compute_config)
    )
    target = model if inplace else copy.deepcopy(model)
    if config is None or not config.modules:
        return target
    for spec in config.modules:
        contract = normalize_compute_contract(spec.compute_contract)
        if contract not in {"composite_add", "mix_fp4_int8_mma"} and not spec.branches:
            continue
        try:
            module = target.get_submodule(spec.name)
        except AttributeError:
            continue
        if type(module) is CompositeAddLinear:
            replacement = _materialize_composite_compute(module, spec)
        else:
            materialize = getattr(module, "materialize_compute", None)
            if not callable(materialize):
                continue
            replacement = materialize(spec)
        if replacement is not None and replacement is not module:
            replace_submodule(target, spec.name, replacement)
    return target


__all__ = [
    "SupportsCompositeMaterialize",
    "materialize_composite_compute",
]
