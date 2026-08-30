"""Component and module-path helpers for quantization."""

from __future__ import annotations

from typing import Any, Iterable

from torch import nn


def ordered_unique(values: Iterable[str]) -> list[str]:
    """Return values in first-seen order without duplicates."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def prefix_module_names(names: Iterable[str], prefix: str | None) -> list[str]:
    """Prefix component-local module names with their parent target path."""

    if not prefix:
        return list(names)
    prefixed: list[str] = []
    for name in names:
        prefixed.append(f"{prefix}.{name}" if name else prefix)
    return prefixed


def module_structure_name(example_input: Any) -> str:
    """Return a short name for a model input structure."""

    if isinstance(example_input, dict):
        return "mapping"
    if isinstance(example_input, tuple):
        return "tuple"
    if isinstance(example_input, list):
        return "list"
    return "tensor"


def resolve_component_model(
    model: nn.Module | None,
    target_path: str | None,
) -> nn.Module:
    """Resolve a component target path against a root PyTorch module."""

    if model is None:
        raise ValueError("PyTorch model is required for this quantization component")
    if not target_path:
        return model
    return model.get_submodule(target_path)


def replace_component_model(
    model: nn.Module | None,
    target_path: str | None,
    replacement: nn.Module,
) -> nn.Module:
    """Replace a component in a root model and return the updated root."""

    if model is None:
        raise ValueError("PyTorch model is required to replace a quantized component")
    if not target_path:
        return replacement
    parent_path, _, attribute = target_path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)
    return model


__all__ = [
    "module_structure_name",
    "ordered_unique",
    "prefix_module_names",
    "replace_component_model",
    "resolve_component_model",
]
