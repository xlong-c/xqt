"""Lightweight quantizer contracts for XQT model-side quantization algorithms.

New executable quantizer PR checklist (U12 / T4):

1. Implement algorithm under ``xqt/quant/quantizers/`` (or backends for adapters).
2. Register via ``register_quant_route(QuantRouteRegistration(...))`` in
   ``quantizers/__init__.py`` (or the package that owns the route).
3. If ``maturity="executable"``, **must** set non-empty ``primary_kernel`` and
   ``reference_kernel`` (scheme x kernel pair). Missing either raises at
   registration time.
4. Prefer attaching ``RuntimeQuantContract`` + ``layout_kernel`` on success
   (see ``layout_apply_report`` / ``build_runtime_quant_contract``).
5. Do not mark cutile/cute_dsl/cutlass as the executable primary for new routes
   unless the engine registration maturity is actually executable and dispatchable.
"""

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
