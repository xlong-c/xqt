"""In-memory registry of :class:`KernelSpec` entries."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping

from xqt.kernels.spec import FormatSignature, KernelBackend, KernelSpec


class KernelRegistry:
    def __init__(self) -> None:
        self._by_op: Dict[str, List[KernelSpec]] = defaultdict(list)

    def register(self, spec: KernelSpec) -> KernelSpec:
        existing = self._by_op[spec.op]
        for other in existing:
            if other.backend == spec.backend:
                if other != spec:
                    raise ValueError(
                        f"Conflicting kernel registration for op {spec.op!r}, "
                        f"backend {spec.backend.value!r}: "
                        f"{other.target!r} != {spec.target!r}"
                    )
                return spec
        existing.append(spec)
        return spec

    def get(self, op: str) -> List[KernelSpec]:
        return list(self._by_op.get(op, ()))

    def get_backend(self, op: str, backend: KernelBackend) -> KernelSpec:
        for spec in self._by_op.get(op, ()):
            if spec.backend == backend:
                return spec
        raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")

    def has(self, op: str) -> bool:
        return bool(self._by_op.get(op))

    def ops(self) -> List[str]:
        return sorted(self._by_op.keys())

    def all_specs(self) -> List[KernelSpec]:
        specs: List[KernelSpec] = []
        for op in self.ops():
            specs.extend(self._by_op[op])
        return specs

    def clear(self) -> None:
        self._by_op.clear()


registry = KernelRegistry()

# Engine metadata is a separate compatibility surface from kernel specs.  It
# lives here during migration so callers can observe one inventory root.
engine_registry: dict[str, object] = {}


def register_kernel(spec: KernelSpec) -> KernelSpec:
    return registry.register(spec)


_LEGACY_BACKENDS = {
    "torch": KernelBackend.TORCH,
    "torch_compile": KernelBackend.TORCH_COMPILE,
    "triton": KernelBackend.TRITON,
    "tilelang": KernelBackend.TILELANG,
    "cutile": KernelBackend.CUTILE,
    "cutlass": KernelBackend.CUTLASS,
    "cute_dsl": KernelBackend.CUTE_DSL,
    "custom_cuda": KernelBackend.CUSTOM_CUDA,
    "flashinfer": KernelBackend.FLASHINFER,
}


def mirror_legacy_gemm_entry(entry: object) -> KernelSpec | None:
    """Mirror a legacy ``GemmKernelRegistration`` into the unified table.

    The import is deliberately duck-typed so this torch-free registry does not
    depend on the GEMM package or create an import cycle.
    """

    name = getattr(entry, "name", None)
    backend_name = str(getattr(entry, "backend", "")).strip().lower()
    if not name or backend_name not in _LEGACY_BACKENDS:
        return None
    maturity = getattr(entry, "maturity", "reference_guarded")
    return register_kernel(
        KernelSpec(
            op=f"gemm.{name}",
            backend=_LEGACY_BACKENDS[backend_name],
            target="xqt.kernels.ops.gemm.dispatch:dispatch_gemm",
            format_signature=FormatSignature(
                description=f"legacy gemm {name} ({maturity})"
            ),
        )
    )


def mirror_legacy_operator_registry(
    backend: str,
    entries: Mapping[str, object] | Iterable[str],
    *,
    target: str,
) -> tuple[KernelSpec, ...]:
    """Mirror one legacy backend pattern table into ``registry``."""

    backend_enum = _LEGACY_BACKENDS.get(str(backend).strip().lower())
    if backend_enum is None:
        return ()
    patterns = entries.keys() if isinstance(entries, Mapping) else entries
    mirrored: list[KernelSpec] = []
    for pattern in patterns:
        mirrored.append(
            register_kernel(
                KernelSpec(
                    op=f"{backend}.{pattern}",
                    backend=backend_enum,
                    target=target,
                    format_signature=FormatSignature(
                        description=f"legacy operator pattern {pattern}"
                    ),
                )
            )
        )
    return tuple(mirrored)


def mirror_engine_registration(registration: object) -> object:
    """Mirror one engine registration into the unified inventory."""

    name = getattr(registration, "name", None)
    if name:
        engine_registry[str(name)] = registration
    return registration
