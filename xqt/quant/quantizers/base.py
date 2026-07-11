"""Lightweight quantizer contracts for XQT model-side quantization algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from torch import nn

from xqt.contracts import QuantizedModel


@dataclass(frozen=True)
class QuantizerOptions:
    """Resolved options passed from a quantization component to a quantizer."""

    backend: str
    method: str | None = None
    strategy: str | None = None
    policy: Mapping[str, Any] = field(default_factory=dict)
    inplace: bool = True


QuantizerResult = QuantizedModel


class Quantizer(Protocol):
    """Protocol implemented by concrete quantizer algorithms."""

    def quantize(self, model: nn.Module, options: QuantizerOptions) -> QuantizedModel:
        """Quantize a model or component and return a normalized result."""


__all__ = [
    "Quantizer",
    "QuantizerOptions",
    "QuantizerResult",
]
