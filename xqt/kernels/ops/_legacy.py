"""Auto-inventory of legacy registries into xqt.kernels.registry.

Phase 2 bridge: every entry in the legacy registries gets a KernelSpec pointing
to a real callable, so old surfaces are inventoried and remotely comparable.
"""

from __future__ import annotations

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import FormatSignature, KernelBackend, KernelSpec

_BACKEND_MAP = {
    "torch": KernelBackend.TORCH,
    "triton": KernelBackend.TRITON,
    "tilelang": KernelBackend.TILELANG,
    "cutile": KernelBackend.CUTILE,
    "cutlass": KernelBackend.CUTLASS,
    "cute_dsl": KernelBackend.CUTE_DSL,
    "custom_cuda": KernelBackend.CUSTOM_CUDA,
}

_GROUP_TARGET = {
    "triton": "xqt.kernels.ops._impl.engines.triton:run_triton_kernel",
    "tilelang": "xqt.kernels.ops._impl.engines.tilelang:run_tilelang_kernel",
    "cutile": "xqt.kernels.ops._impl.engines.cutile:run_cutile_kernel",
    "cutlass": "xqt.kernels.ops._impl.engines.cutlass:run_cutlass_kernel",
    "cute_dsl": "xqt.kernels.ops._impl.engines.cute_dsl:run_cute_dsl_kernel",
}


def register_legacy_gemm() -> None:
    try:
        from xqt.kernels.ops.gemm.registry import default_registry
    except Exception:
        return
    registry = default_registry()
    for entry in registry.entries():
        backend = _BACKEND_MAP.get(entry.backend)
        if backend is None:
            continue
        try:
            register_kernel(
                KernelSpec(
                    op=f"gemm.{entry.name}",
                    backend=backend,
                    target="xqt.kernels.ops.gemm.dispatch:dispatch_gemm",
                    format_signature=FormatSignature(
                        description=f"legacy gemm {entry.name} ({entry.maturity})"
                    ),
                )
            )
        except ValueError:
            continue


def register_legacy_operator_patterns() -> None:
    for bk, target in _GROUP_TARGET.items():
        try:
            import importlib

            m = importlib.import_module(f"xqt.kernels.ops._impl.engines.{bk}")
        except Exception:
            continue
        backend = KernelBackend(bk)
        for attr in dir(m):
            if "REGISTRY" not in attr:
                continue
            reg = getattr(m, attr)
            if not isinstance(reg, dict):
                continue
            for pattern in reg:
                try:
                    register_kernel(
                        KernelSpec(
                            op=f"{bk}.{pattern}",
                            backend=backend,
                            target=target,
                            format_signature=FormatSignature(
                                description=f"legacy operator pattern {pattern}"
                            ),
                        )
                    )
                except ValueError:
                    continue
            break


register_legacy_gemm()
register_legacy_operator_patterns()
