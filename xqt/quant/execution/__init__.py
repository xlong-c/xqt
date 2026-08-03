"""Quantization execution helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "execute_quantization_plan",
    "summarize_quantization_reports",
]


def __getattr__(name: str) -> Any:
    if name == "execute_quantization_plan":
        from .executor import execute_quantization_plan

        return execute_quantization_plan
    if name == "summarize_quantization_reports":
        from .reporting import summarize_quantization_reports

        return summarize_quantization_reports
    raise AttributeError(name)
