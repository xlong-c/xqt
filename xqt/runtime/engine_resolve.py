"""Capability-based operator engine resolution for Infer (DEBT-003 / G1).

Quant stages declare required_capabilities (+ optional preferred_engines hints).
This module selects an engine candidate; it does not run quantizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Default preference when preferred_engines is empty or contains only "auto".
# Order is capability-family aware; unknown engines are ignored.
_DEFAULT_INT8_MMA_ORDER: tuple[str, ...] = (
    "ptx_sm89",
    "tilelang",
    "torch_int_mm",
)

_ENGINE_PROVIDES: dict[str, frozenset[str]] = {
    "ptx_sm89": frozenset({"int8_mma", "true_int8_mma"}),
    "native_sm89": frozenset({"int8_mma", "true_int8_mma"}),
    "tilelang": frozenset(
        {
            "int8_mma",
            "true_int8_mma",
            "fp4_mma",
            "dequant_gemm_epilogue",
            "w4_storage_int8_mma",
        }
    ),
    "torch_int_mm": frozenset({"int8_mma", "true_int8_mma"}),
    "torch": frozenset({"generic", "fp16_mma", "fp8_mma"}),
    "triton": frozenset({"generic", "fp16_mma", "fp4_mma", "dequant_gemm_epilogue"}),
    "cutlass": frozenset({"int8_mma", "fp4_mma", "int4_mma", "fp16_mma"}),
    "cute_dsl": frozenset({"int8_mma", "fp4_mma"}),
    "cutile": frozenset({"generic", "fp16_mma"}),
    "reference": frozenset({"int8_mma", "generic"}),
}

_ALIASES: dict[str, str] = {
    "native_sm89": "ptx_sm89",
    "auto": "auto",
}


@dataclass(frozen=True, kw_only=True)
class EngineResolveResult:
    """Outcome of one capability-based engine resolve."""

    engine: str
    required_capabilities: tuple[str, ...] = ()
    preferred_engines: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    reason: str = "resolved"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "required_capabilities": list(self.required_capabilities),
            "preferred_engines": list(self.preferred_engines),
            "candidates": list(self.candidates),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


def normalize_engine_name(engine: str | None) -> str:
    """Normalize engine token; ``None`` / empty → ``auto``."""
    if engine is None:
        return "auto"
    name = str(engine).strip().lower()
    if not name:
        return "auto"
    return _ALIASES.get(name, name)


def engines_providing(capability: str) -> list[str]:
    """List engines that declare a single capability."""
    cap = str(capability).strip()
    return sorted(
        engine
        for engine, caps in _ENGINE_PROVIDES.items()
        if cap in caps and engine not in {"native_sm89"}
    )


def engines_providing_all(capabilities: Sequence[str]) -> list[str]:
    """List engines that provide every required capability."""
    required = [str(item).strip() for item in capabilities if str(item).strip()]
    if not required:
        return sorted(
            engine
            for engine in _ENGINE_PROVIDES
            if engine not in {"native_sm89"}
        )
    result: list[str] = []
    for engine, caps in _ENGINE_PROVIDES.items():
        if engine in {"native_sm89"}:
            continue
        if all(cap in caps for cap in required):
            result.append(engine)
    return sorted(result)


def _stable_prefer(
    candidates: Sequence[str],
    preferred: Sequence[str],
) -> list[str]:
    preferred_norm = [normalize_engine_name(item) for item in preferred]
    preferred_norm = [item for item in preferred_norm if item != "auto"]
    ordered: list[str] = []
    seen: set[str] = set()
    for name in preferred_norm:
        if name in candidates and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in candidates:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def resolve_engine(
    *,
    required_capabilities: Sequence[str] | None = None,
    preferred_engines: Sequence[str] | None = None,
    default_order: Sequence[str] | None = None,
    fallback: str = "torch_int_mm",
) -> EngineResolveResult:
    """Resolve one operator engine from capabilities + optional preference hints.

    Never treats preferred_engines as a hard required_engine constraint:
    engines that lack capabilities are skipped even if preferred.
    """
    caps = tuple(
        str(item).strip()
        for item in (required_capabilities or ())
        if str(item).strip()
    )
    preferred_raw = [
        normalize_engine_name(item) for item in (preferred_engines or ())
    ]
    preferred = tuple(item for item in preferred_raw if item != "auto")
    providing = engines_providing_all(caps)
    if not providing:
        # No matrix hit: still allow explicit preferred if caller only passed hints.
        providing = list(preferred) if preferred else [normalize_engine_name(fallback)]
    order = tuple(default_order) if default_order is not None else _DEFAULT_INT8_MMA_ORDER
    if preferred:
        ranked = _stable_prefer(providing, preferred)
    else:
        ranked = _stable_prefer(providing, order)
    if not ranked:
        ranked = [normalize_engine_name(fallback)]
    chosen = ranked[0]
    return EngineResolveResult(
        engine=chosen,
        required_capabilities=caps,
        preferred_engines=preferred,
        candidates=tuple(ranked),
        reason="preferred" if preferred and chosen in preferred else "default_order",
        metadata={"fallback": normalize_engine_name(fallback)},
    )


def resolve_int8_mma_engine(
    engine: str | None = "auto",
    *,
    preferred_engines: Sequence[str] | None = None,
    fallback: str = "torch_int_mm",
) -> EngineResolveResult:
    """Resolve engine for int8_mma modules (quantizer / forward shared path)."""
    normalized = normalize_engine_name(engine)
    hints: list[str] = list(preferred_engines or [])
    if normalized != "auto":
        # Explicit request is a strong preference, not a schema required_engine key.
        hints = [normalized, *hints]
    return resolve_engine(
        required_capabilities=["int8_mma"],
        preferred_engines=hints,
        default_order=_DEFAULT_INT8_MMA_ORDER,
        fallback=fallback,
    )


def engine_resolve_from_compute_config(
    compute_config: Mapping[str, Any] | Any | None,
    *,
    fallback: str = "torch_int_mm",
) -> EngineResolveResult:
    """Resolve from a ComputeConfig object or mapping."""
    if compute_config is None:
        return resolve_engine(fallback=fallback)
    modules: Sequence[Any]
    if hasattr(compute_config, "modules"):
        modules = list(getattr(compute_config, "modules") or [])
        caps = list(getattr(compute_config, "all_required_capabilities")())
        preferred: list[str] = []
        for module in modules:
            preferred.extend(list(getattr(module, "preferred_engines", []) or []))
    elif isinstance(compute_config, Mapping):
        from xqt.contracts.compute import ComputeConfig

        parsed = ComputeConfig.from_mapping(compute_config)
        if parsed is None:
            return resolve_engine(fallback=fallback)
        caps = parsed.all_required_capabilities()
        preferred = []
        for module in parsed.modules:
            preferred.extend(module.preferred_engines)
    else:
        return resolve_engine(fallback=fallback)
    if not caps:
        caps = ["int8_mma"]
    return resolve_engine(
        required_capabilities=caps,
        preferred_engines=preferred,
        fallback=fallback,
    )


__all__ = [
    "EngineResolveResult",
    "engine_resolve_from_compute_config",
    "engines_providing",
    "engines_providing_all",
    "normalize_engine_name",
    "resolve_engine",
    "resolve_int8_mma_engine",
]
