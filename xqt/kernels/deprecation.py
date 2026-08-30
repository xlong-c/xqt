"""Small helpers for compatibility warnings during namespace migration."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import sys
from types import ModuleType
from typing import Any
import warnings


_SUPPRESS_LEGACY_WARNING: ContextVar[bool] = ContextVar(
    "xqt_suppress_legacy_warning", default=False
)


@contextmanager
def suppress_legacy_warnings():
    """Silence warnings while the unified lazy bridge loads an implementation."""

    token = _SUPPRESS_LEGACY_WARNING.set(True)
    try:
        yield
    finally:
        _SUPPRESS_LEGACY_WARNING.reset(token)


def warn_legacy_import(old: str, new: str) -> None:
    """Warn whenever a legacy namespace is imported."""

    if _SUPPRESS_LEGACY_WARNING.get():
        return
    warnings.warn(
        f"{old} is deprecated; use {new} instead",
        DeprecationWarning,
        stacklevel=3,
    )


def forward(namespace: dict[str, Any], target: str) -> None:
    """Alias a legacy module to its canonical implementation."""

    def resolve() -> ModuleType:
        return importlib.import_module(target)

    def __getattr__(name: str) -> Any:
        return getattr(resolve(), name)

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(dir(resolve())))

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
    module_name = namespace.get("__name__")
    if isinstance(module_name, str) and "__path__" not in namespace:
        sys.modules[module_name] = resolve()
