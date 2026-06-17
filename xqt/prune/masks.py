"""Mask and sparsity helpers for pruning recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import nn
from torch.nn.utils import prune


@dataclass
class PruningEntry:
    """Per-module sparsity summary."""

    module_name: str
    parameter_name: str
    total: int
    zero: int
    sparsity: float


@dataclass
class PruningReport:
    """Aggregate pruning summary."""

    amount: float
    total_parameters: int
    zero_parameters: int
    sparsity: float
    entries: list[PruningEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "amount": self.amount,
            "total_parameters": self.total_parameters,
            "zero_parameters": self.zero_parameters,
            "sparsity": self.sparsity,
            "entries": [
                {
                    "module_name": entry.module_name,
                    "parameter_name": entry.parameter_name,
                    "total": entry.total,
                    "zero": entry.zero,
                    "sparsity": entry.sparsity,
                }
                for entry in self.entries
            ],
        }


def tensor_sparsity(tensor: torch.Tensor) -> float:
    """Return the fraction of exact zeros in a tensor."""

    if tensor.numel() == 0:
        return 0.0
    zero = torch.count_nonzero(tensor == 0).item()
    return float(zero) / float(tensor.numel())


def _named_modules_for_pruning(
    model: nn.Module,
    module_types: Sequence[type[nn.Module]],
) -> list[tuple[str, nn.Module]]:
    return [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(module, tuple(module_types))
    ]


def summarize_pruning(
    model: nn.Module,
    *,
    module_types: Sequence[type[nn.Module]] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> PruningReport:
    """Summarize sparsity over the selected modules."""

    entries: list[PruningEntry] = []
    total_parameters = 0
    zero_parameters = 0

    for module_name, module in _named_modules_for_pruning(model, module_types):
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            continue
        total = int(parameter.numel())
        zero = int(torch.count_nonzero(parameter == 0).item())
        sparsity = 0.0 if total == 0 else zero / total
        entries.append(
            PruningEntry(
                module_name=module_name,
                parameter_name=parameter_name,
                total=total,
                zero=zero,
                sparsity=sparsity,
            )
        )
        total_parameters += total
        zero_parameters += zero

    total_sparsity = 0.0 if total_parameters == 0 else zero_parameters / total_parameters
    return PruningReport(
        amount=total_sparsity,
        total_parameters=total_parameters,
        zero_parameters=zero_parameters,
        sparsity=total_sparsity,
        entries=entries,
    )


def apply_global_l1_unstructured_pruning(
    model: nn.Module,
    amount: float,
    *,
    module_types: Sequence[type[nn.Module]] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> PruningReport:
    """Apply global L1 pruning and return a sparsity report."""

    if amount < 0.0 or amount > 1.0:
        raise ValueError("amount must be in [0, 1]")

    parameters_to_prune = [
        (module, parameter_name)
        for _name, module in _named_modules_for_pruning(model, module_types)
        if isinstance(getattr(module, parameter_name, None), torch.Tensor)
    ]
    if parameters_to_prune:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    return summarize_pruning(
        model,
        module_types=module_types,
        parameter_name=parameter_name,
    )


def remove_pruning_reparameterization(
    model: nn.Module,
    *,
    module_types: Sequence[type[nn.Module]] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> None:
    """Make pruning permanent by removing mask parametrizations."""

    for _name, module in _named_modules_for_pruning(model, module_types):
        if hasattr(module, f"{parameter_name}_mask"):
            prune.remove(module, parameter_name)


__all__ = [
    "PruningEntry",
    "PruningReport",
    "apply_global_l1_unstructured_pruning",
    "remove_pruning_reparameterization",
    "summarize_pruning",
    "tensor_sparsity",
]
