"""Small dependency-free helpers shared by the JIT build modules."""

from __future__ import annotations

from functools import wraps
from threading import Lock
from typing import Any, Callable, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


def cache_once(function: Callable[P, T]) -> Callable[P, T]:
    """Cache the first successful call, preserving exceptions for retry."""

    lock = Lock()
    missing = object()
    value: object = missing

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal value
        if value is missing:
            with lock:
                if value is missing:
                    value = function(*args, **kwargs)
        return value  # type: ignore[return-value]

    return wrapper


def lazy_register(registry: dict[str, Any], name: str, factory: Callable[[], T]) -> T:
    """Materialize and memoize one named lazy registry item."""

    if name not in registry:
        registry[name] = factory()
    return registry[name]


__all__ = ["cache_once", "lazy_register"]
