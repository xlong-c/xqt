"""Registry for graph-level quantization transforms (XQT-012)."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from xqt.core.base import XQTConfigError
from .base import GraphQuantTransform

_TRANSFORM_FACTORIES: dict[str, Callable[..., GraphQuantTransform]] = {}
_TRANSFORM_ALIASES: dict[str, str] = {
    "quarot": "rotation_absorb",
    "rotation": "rotation_absorb",
}


def register_graph_transform(
    name: str,
    factory: Callable[..., GraphQuantTransform],
    *,
    aliases: tuple[str, ...] = (),
) -> None:
    """Register a graph transform factory by canonical name and optional aliases."""
    canonical = name.strip().lower()
    if not canonical:
        raise ValueError("transform name must be non-empty")
    _TRANSFORM_FACTORIES[canonical] = factory
    for alias in aliases:
        _TRANSFORM_ALIASES[alias.strip().lower()] = canonical


def is_known_graph_transform(name: str) -> bool:
    """Check whether a transform name or alias is registered."""
    key = str(name).strip().lower()
    return key in _TRANSFORM_FACTORIES or key in _TRANSFORM_ALIASES


def canonical_transform_name(name: str) -> str:
    """Resolve an alias to its canonical transform name."""
    key = str(name).strip().lower()
    return _TRANSFORM_ALIASES.get(key, key)


def available_graph_transforms() -> tuple[str, ...]:
    """Return sorted list of canonical registered transform names."""
    return tuple(sorted(_TRANSFORM_FACTORIES.keys()))


def build_graph_transform(
    name: str,
    params: Mapping[str, Any] | None = None,
) -> GraphQuantTransform:
    """Instantiate a transform by name and parameters."""
    canonical = canonical_transform_name(name)
    factory = _TRANSFORM_FACTORIES.get(canonical)
    if factory is None:
        supported = ", ".join(available_graph_transforms())
        raise XQTConfigError(
            f"Unknown graph transform {name!r}; registered transforms: [{supported}]"
        )
    return factory(**(dict(params) if params else {}))


def _register_builtins() -> None:
    from .rotation import RotationAbsorbTransform
    from .patterns.dequant_gemm import DequantGemmTransform
    from .patterns.norm_quant import NormQuantTransform
    from .patterns.activation_quant import ActivationQuantTransform

    register_graph_transform(
        "rotation_absorb",
        lambda rot_size=32, **kwargs: RotationAbsorbTransform(rot_size=int(rot_size)),
        aliases=("quarot", "rotation"),
    )
    register_graph_transform(
        "dequant_gemm",
        lambda **kwargs: DequantGemmTransform(**kwargs),
    )
    register_graph_transform(
        "norm_quant",
        lambda **kwargs: NormQuantTransform(**kwargs),
    )
    register_graph_transform(
        "activation_quant",
        lambda **kwargs: ActivationQuantTransform(**kwargs),
    )


_register_builtins()

__all__ = [
    "available_graph_transforms",
    "build_graph_transform",
    "canonical_transform_name",
    "is_known_graph_transform",
    "register_graph_transform",
]
