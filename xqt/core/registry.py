"""Small registries for XQT passes, recipes, and exporters."""

from __future__ import annotations

import difflib
from typing import Any, Callable, Dict, Iterable, Optional

from .errors import XQTRegistryError


_BANNED_REGISTRY_KINDS = {
    "MODEL",
    "DATASET",
    "LOSS",
    "METRIC",
    "TRAINER",
    "OPTIMIZER",
    "SCHEDULER",
}


class XQTRegistry:
    """Registry for XQT-owned extension points, not task framework objects."""

    def __init__(self, name: str) -> None:
        upper_name = name.upper()
        for banned in _BANNED_REGISTRY_KINDS:
            if banned in upper_name:
                raise XQTRegistryError(
                    f"XQTRegistry must not be used as a {banned.lower()} registry"
                )
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
        """Register an object under an explicit name or its ``__name__``."""

        def decorator(obj: Any) -> Any:
            register_name = name or getattr(obj, "__name__", None)
            if not register_name:
                raise XQTRegistryError(
                    f"Cannot infer registration name for {self.name} registry"
                )
            if register_name in self._items:
                raise XQTRegistryError(
                    f"'{register_name}' already registered in {self.name} registry"
                )
            self._items[register_name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        """Return a registered object by name."""

        if name in self._items:
            return self._items[name]

        available = self.list_available()
        matches = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        message = f"'{name}' not found in {self.name} registry"
        if matches:
            message += f". Did you mean: {matches}?"
        raise XQTRegistryError(message)

    def build(self, name: str, **params: Any) -> Any:
        """Instantiate or call a registered target with keyword params."""

        target = self.get(name)
        if isinstance(target, type):
            return target(**params)
        if callable(target):
            return target(**params)
        if params:
            raise XQTRegistryError(
                f"Registered object '{name}' in {self.name} does not accept params"
            )
        return target

    def list_available(self) -> list[str]:
        """List registered names in insertion order."""

        return list(self._items.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __iter__(self) -> Iterable[str]:
        return iter(self._items)


PASS_REGISTRY = XQTRegistry("XQT_PASS")
RECIPE_REGISTRY = XQTRegistry("XQT_RECIPE")
EXPORTER_REGISTRY = XQTRegistry("XQT_EXPORTER")

register_pass = PASS_REGISTRY.register
register_recipe = RECIPE_REGISTRY.register
register_exporter = EXPORTER_REGISTRY.register


__all__ = [
    "EXPORTER_REGISTRY",
    "PASS_REGISTRY",
    "RECIPE_REGISTRY",
    "XQTRegistry",
    "register_exporter",
    "register_pass",
    "register_recipe",
]
