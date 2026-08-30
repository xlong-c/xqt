"""Compatibility imports for quantization component helpers."""

from __future__ import annotations

from xqt.compression.quant.component import (
    module_structure_name,
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)


__all__ = [
    "module_structure_name",
    "ordered_unique",
    "prefix_module_names",
    "replace_component_model",
    "resolve_component_model",
]
