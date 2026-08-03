"""Quantization route registry (C2).

Single source of truth for "which (backend, method, strategy, compute)
combination executes which quantizer". The executor performs a table lookup
here instead of hardcoding an if-elif chain, and backend capability maturity
is derived from the same table so the two never drift apart.

Registrations are ordered by ``priority`` (lower wins, mirrors the historical
dispatch order). Routes that exist only as capability advertisements register
with ``status="planned"`` and the planned-report handler, keeping the honest
"capability report only, no executable algorithm" semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional, Protocol

from torch import nn

from .types import QuantizationComponentPlan, QuantizationReport, QuantScheme

if TYPE_CHECKING:
    from xqt.core.types import XQTContext


RouteRuntime = Mapping[str, Any]


class ComponentHandler(Protocol):
    """Uniform component execution handler signature."""

    def __call__(
        self,
        context: "XQTContext",
        model: Optional[nn.Module],
        component: QuantizationComponentPlan,
        *,
        runtime: RouteRuntime | None = None,
    ) -> tuple[Optional[nn.Module], QuantizationReport, dict[str, Any]]:
        """Execute one component; return (model, report, artifact_updates)."""


@dataclass(frozen=True)
class RouteQuery:
    """Normalized route lookup key for one quantization component."""

    backend: str
    method: str = "none"
    strategy: str = ""
    compute: str = "dequant_fp16"
    scheme: QuantScheme | None = None


RouteMatcher = Callable[[RouteQuery], bool]


@dataclass(frozen=True)
class QuantRouteRegistration:
    """One registered quantization route.

    Executable routes must declare ``primary_kernel`` and ``reference_kernel``
    (T4 / GUIDE): scheme×kernel pairs stay explicit; missing reference is not
    allowed for ``maturity="executable"``.
    """

    name: str
    backend: str
    handler: ComponentHandler
    matcher: RouteMatcher
    priority: int = 100
    maturity: str = "executable"
    status: str = "available"
    methods: tuple[str, ...] = ()
    strategies: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    primary_kernel: str | None = None
    reference_kernel: str | None = None


_ROUTE_REGISTRY: list[QuantRouteRegistration] = []


def _validate_kernel_pair(registration: QuantRouteRegistration) -> None:
    """Reject executable routes that omit primary/reference kernels."""

    if registration.maturity != "executable":
        return
    if not registration.primary_kernel or not str(registration.primary_kernel).strip():
        raise ValueError(
            f"quant route {registration.name!r} is maturity=executable but "
            "primary_kernel is missing; declare a primary kernel or lower maturity"
        )
    if not registration.reference_kernel or not str(registration.reference_kernel).strip():
        raise ValueError(
            f"quant route {registration.name!r} is maturity=executable but "
            "reference_kernel is missing; declare a reference/fallback kernel "
            "or lower maturity to planned/reference_guarded/metadata_only"
        )


def register_quant_route(registration: QuantRouteRegistration) -> QuantRouteRegistration:
    """Append one route registration and return it."""

    _validate_kernel_pair(registration)
    _ROUTE_REGISTRY.append(registration)
    return registration


def clear_quant_routes() -> None:
    """Drop all route registrations. Test support only."""

    _ROUTE_REGISTRY.clear()


def _ordered_routes() -> list[QuantRouteRegistration]:
    return sorted(_ROUTE_REGISTRY, key=lambda item: (item.priority, item.name))


def resolve_quant_route(query: RouteQuery) -> QuantRouteRegistration | None:
    """Return the highest-priority registration whose matcher accepts the query."""

    for registration in _ordered_routes():
        if registration.matcher(query):
            return registration
    return None


def iter_quant_routes() -> Iterator[QuantRouteRegistration]:
    """Iterate registrations in dispatch order."""

    return iter(_ordered_routes())


def scheme_kernel_matrix() -> list[dict[str, Any]]:
    """Export executable route scheme×kernel pairs from the registry (V5).

    Reads the single route table; does not invent a second matrix. Each row is
    one registration with its strategies/methods and primary/reference kernels.
    """

    rows: list[dict[str, Any]] = []
    for registration in _ordered_routes():
        if registration.maturity != "executable":
            continue
        rows.append(
            {
                "name": registration.name,
                "backend": registration.backend,
                "methods": list(registration.methods),
                "strategies": list(registration.strategies),
                "primary_kernel": registration.primary_kernel,
                "reference_kernel": registration.reference_kernel,
                "priority": registration.priority,
                "maturity": registration.maturity,
            }
        )
    return rows


def route_matcher(
    *,
    backend: str = "pytorch",
    methods: tuple[str, ...] | None = None,
    strategies: tuple[str, ...] | None = None,
    computes: tuple[str, ...] | None = None,
) -> RouteMatcher:
    """Build a predicate matcher over the normalized route key fields."""

    def _matches(query: RouteQuery) -> bool:
        if query.backend != backend:
            return False
        if methods is not None and query.method not in methods:
            return False
        if strategies is not None and query.strategy not in strategies:
            return False
        if computes is not None and query.compute not in computes:
            return False
        return True

    return _matches


__all__ = [
    "ComponentHandler",
    "QuantRouteRegistration",
    "RouteMatcher",
    "RouteQuery",
    "RouteRuntime",
    "clear_quant_routes",
    "iter_quant_routes",
    "register_quant_route",
    "resolve_quant_route",
    "route_matcher",
    "scheme_kernel_matrix",
]
