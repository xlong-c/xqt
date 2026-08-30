"""Lightweight kernel API logging."""

from __future__ import annotations

import functools
from typing import Any, Callable


def debug_kernel_api(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return wrapper
