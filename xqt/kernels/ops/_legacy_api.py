"""Lazy bridge from unified op groups to the existing kernel implementations.

The bridge is intentionally private.  It keeps migration changes small while
ensuring that optional CUDA extensions are imported only when an operation is
actually executed.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from xqt.kernels.deprecation import suppress_legacy_warnings


_GROUP_MODULES: dict[str, tuple[str, ...]] = {
    "gemm": (
        "xqt.kernels.ops._impl.triton.gemm",
        "xqt.kernels.ops._impl.tilelang.int8_mma",
        "xqt.kernels.ops._impl.cute.int8mma_binding",
        "xqt.kernels.ops._impl.gemm_backends.sm89.awq_w4a16_decode_sm89",
        "xqt.kernels.ops._impl.cuda.awq_w4a16_sm89",
        "xqt.kernels.ops._impl.gemm_precision",
        "xqt.kernels.ops._impl.fp4_quant_common",
    ),
    "attention": (
        "xqt.kernels.ops._impl.triton.attention",
        "xqt.kernels.ops._impl.tilelang.kv_int8_attention",
        "xqt.kernels.ops._impl.tilelang._common",
        "xqt.kernels.ops._impl.cute.svdq_w4a4_sm89",
    ),
    "quantization": (
        "xqt.kernels.ops._impl.fp4_quant_common",
        "xqt.kernels.ops._impl.cute.convrot_w4a4_rowwise_sm89",
        "xqt.kernels.ops._impl.cute.convrot_w8a8_sm89",
        "xqt.kernels.ops._impl.cute.svdq_w4a4_sm89",
        "xqt.kernels.ops._impl.cute.svdq_w8a8_sm89",
        "xqt.kernels.ops._impl.cute.int8mma_binding",
        "xqt.kernels.ops._impl.tilelang",
        "xqt.kernels.ops._impl.tilelang.int8_mma",
        "xqt.kernels.ops._impl.tilelang.svd_fused",
        "xqt.kernels.ops._impl.tilelang._common",
        "xqt.kernels.ops._impl.triton",
        "xqt.kernels.ops._impl.triton.convrot",
        "xqt.kernels.ops._impl.cuda.awq_w4a16_sm89",
    ),
}


@lru_cache(maxsize=None)
def load_legacy(group: str, name: str) -> Any:
    """Resolve one legacy symbol on first use.

    ``AttributeError`` is raised instead of leaking an optional dependency's
    import error so normal module attribute lookup remains predictable.
    """

    for module_name in _GROUP_MODULES.get(group, ()):
        with suppress_legacy_warnings():
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'xqt.kernels.ops.{group}' has no attribute {name!r}")


def group_dir(group: str) -> tuple[str, ...]:
    """Return already-known compatibility names for introspection."""

    names: set[str] = set()
    for module_name in _GROUP_MODULES.get(group, ()):
        with suppress_legacy_warnings():
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
        names.update(name for name in dir(module) if not name.startswith("__"))
    return tuple(sorted(names))


def clear_legacy_cache() -> None:
    """Clear cached legacy module attribute resolutions."""
    load_legacy.cache_clear()

