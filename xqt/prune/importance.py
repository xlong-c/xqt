"""Prune importance and sensitivity ranking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import nn

DEFAULT_PRUNABLE_TYPES = (nn.Linear, nn.Conv2d)


@dataclass
class ModuleImportanceRecord:
    """Per-module importance statistics."""

    name: str
    module_type: str
    parameter_count: int
    l1_mean: float
    l2_norm: float
    max_abs: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a plain dictionary."""

        return {
            "name": self.name,
            "module_type": self.module_type,
            "parameter_count": self.parameter_count,
            "l1_mean": self.l1_mean,
            "l2_norm": self.l2_norm,
            "max_abs": self.max_abs,
        }


@dataclass
class PruneCandidateRecord:
    """Combined importance + sensitivity ranking result."""

    rank: int
    name: str
    module_type: str
    parameter_count: int
    importance_score: float
    sensitivity_score: float
    importance_normalized: float
    sensitivity_normalized: float
    combined_score: float
    sensitivity_available: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a plain dictionary."""

        return {
            "rank": self.rank,
            "name": self.name,
            "module_type": self.module_type,
            "parameter_count": self.parameter_count,
            "importance_score": self.importance_score,
            "sensitivity_score": self.sensitivity_score,
            "importance_normalized": self.importance_normalized,
            "sensitivity_normalized": self.sensitivity_normalized,
            "combined_score": self.combined_score,
            "sensitivity_available": self.sensitivity_available,
        }


def _materialize_tensor(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if hasattr(tensor, "dequantize"):
        try:
            tensor = tensor.dequantize()
        except (NotImplementedError, RuntimeError, TypeError):
            return None
    return tensor.detach().to(dtype=torch.float32, device="cpu")


def _module_has_dequantized_weight(module: nn.Module) -> bool:
    return callable(getattr(module, "dequantize_weight", None))


def _iter_target_modules(
    model: nn.Module,
    module_names: Optional[Sequence[str]],
    module_types: Sequence[type[nn.Module]],
) -> list[tuple[str, nn.Module]]:
    target_names = set(module_names) if module_names is not None else None
    targets: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        module_name = name or "<root>"
        if target_names is not None and module_name not in target_names:
            continue
        if not isinstance(module, tuple(module_types)) and not _module_has_dequantized_weight(module):
            continue
        targets.append((module_name, module))
    return targets


def collect_module_importance(
    model: nn.Module,
    *,
    module_names: Optional[Sequence[str]] = None,
    module_types: Sequence[type[nn.Module]] = DEFAULT_PRUNABLE_TYPES,
    parameter_name: str = "weight",
) -> list[ModuleImportanceRecord]:
    """Collect simple weight-norm importance statistics for candidate modules."""

    records: list[ModuleImportanceRecord] = []
    for name, module in _iter_target_modules(model, module_names, module_types):
        parameter = getattr(module, parameter_name, None)
        if isinstance(parameter, torch.Tensor):
            tensor = _materialize_tensor(parameter)
        elif _module_has_dequantized_weight(module):
            try:
                weight = module.dequantize_weight()
            except Exception:
                tensor = None
            else:
                tensor = _materialize_tensor(weight) if isinstance(weight, torch.Tensor) else None
        else:
            tensor = None
        if tensor is None:
            continue
        if tensor.numel() == 0:
            continue
        flat = tensor.flatten()
        abs_flat = flat.abs()
        records.append(
            ModuleImportanceRecord(
                name=name,
                module_type=type(module).__name__,
                parameter_count=int(flat.numel()),
                l1_mean=float(abs_flat.mean().item()),
                l2_norm=float(torch.linalg.vector_norm(flat).item()),
                max_abs=float(abs_flat.max().item()),
            )
        )
    return records


def _extract_sensitivity_score(record: object) -> Optional[float]:
    diff = getattr(record, "diff", None)
    if diff is not None and hasattr(diff, "mean_abs"):
        return float(diff.mean_abs)
    value = getattr(record, "mean_abs", None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.0 for _ in values]
    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


def rank_prune_candidates(
    model: nn.Module,
    sensitivity_records: Sequence[object],
    *,
    module_names: Optional[Sequence[str]] = None,
    module_types: Sequence[type[nn.Module]] = DEFAULT_PRUNABLE_TYPES,
    parameter_name: str = "weight",
    top_k: Optional[int] = None,
    importance_weight: float = 1.0,
    sensitivity_weight: float = 1.0,
    require_sensitivity: bool = False,
) -> list[PruneCandidateRecord]:
    """Rank prune candidates using weight importance and external sensitivity scores."""

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if importance_weight < 0.0 or sensitivity_weight < 0.0:
        raise ValueError("importance_weight and sensitivity_weight must be non-negative")
    if importance_weight == 0.0 and sensitivity_weight == 0.0:
        raise ValueError("At least one ranking weight must be positive")

    importance_records = collect_module_importance(
        model,
        module_names=module_names,
        module_types=module_types,
        parameter_name=parameter_name,
    )
    if not importance_records:
        return []

    sensitivity_map = {
        str(getattr(record, "name")): score
        for record in sensitivity_records
        if (score := _extract_sensitivity_score(record)) is not None
    }
    sensitivity_available = bool(sensitivity_map)

    filtered_importance_records: list[ModuleImportanceRecord] = []
    raw_sensitivity: list[float] = []
    for record in importance_records:
        if require_sensitivity and record.name not in sensitivity_map:
            continue
        filtered_importance_records.append(record)
        if sensitivity_available:
            raw_sensitivity.append(sensitivity_map.get(record.name, max(sensitivity_map.values())))

    if not filtered_importance_records:
        return []

    raw_importance = [record.l1_mean for record in filtered_importance_records]
    importance_norm = _normalize(raw_importance)
    if sensitivity_available:
        sensitivity_norm = _normalize(raw_sensitivity)
    else:
        sensitivity_norm = [0.0 for _ in filtered_importance_records]
    combined_weight = importance_weight + (sensitivity_weight if sensitivity_available else 0.0)
    if combined_weight <= 0.0:
        raise ValueError("At least one ranking weight must be positive")
    importance_weight = importance_weight / combined_weight
    sensitivity_weight = (
        sensitivity_weight / combined_weight if sensitivity_available else 0.0
    )

    candidate_records: list[PruneCandidateRecord] = []
    for index, record in enumerate(filtered_importance_records):
        combined_score = (
            importance_weight * importance_norm[index]
            + sensitivity_weight * sensitivity_norm[index]
        )
        candidate_records.append(
            PruneCandidateRecord(
                rank=0,
                name=record.name,
                module_type=record.module_type,
                parameter_count=record.parameter_count,
                importance_score=record.l1_mean,
                sensitivity_score=raw_sensitivity[index] if sensitivity_available else 0.0,
                importance_normalized=importance_norm[index],
                sensitivity_normalized=sensitivity_norm[index],
                combined_score=combined_score,
                sensitivity_available=record.name in sensitivity_map,
            )
        )

    candidate_records.sort(key=lambda record: record.combined_score)
    for index, record in enumerate(candidate_records, start=1):
        record.rank = index

    if top_k is not None:
        candidate_records = candidate_records[:top_k]
    return candidate_records


__all__ = [
    "DEFAULT_PRUNABLE_TYPES",
    "ModuleImportanceRecord",
    "PruneCandidateRecord",
    "collect_module_importance",
    "rank_prune_candidates",
]
