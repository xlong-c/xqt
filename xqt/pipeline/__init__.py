"""Internal stage execution helpers for XQT workflows.

User-facing orchestration belongs to :mod:`xqt.workflows`; this package keeps
the context builder and concrete pass implementations used by that layer.
"""

from .runner import create_context

__all__ = [
    "create_context",
]
