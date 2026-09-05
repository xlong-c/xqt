"""Unified public kernel namespace for XQT."""

from __future__ import annotations

import importlib
from typing import Any

from xqt.kernels.registry import KernelRegistry, register_kernel, registry
from xqt.kernels.selector import (
    clear_cache,
    get_fastest_kernel,
    get_kernel,
    select_fastest_kernel,
    select_kernel,
)
from xqt.kernels.spec import (
    CapabilityRequirement,
    DeviceType,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
    capabilities_satisfied,
)

__all__ = [
    "CapabilityRequirement",
    "DeviceType",
    "FormatSignature",
    "KernelBackend",
    "KernelSpec",
    "KernelRegistry",
    "PlatformInfo",
    "BaseFusedOp",
    "capabilities_satisfied",
    "clear_cache",
    "get_kernel",
    "get_fastest_kernel",
    "register_fused_op",
    "register_kernel",
    "registry",
    "select_kernel",
    "select_fastest_kernel",
    "AutoTunedOperator",
    "get_fused_op_backend",
    "set_fused_op_backend",
    "enable_kernel_trace",
    "get_kernel_trace",
    "clear_kernel_trace",
    "disable_kernel_trace",
]

# Torch-dependent symbols are lazily re-exported so that importing this package
# (and spec/registry/selector) keeps working on a CPU-only box.
_LAZY_EXPORTS = {
    "AutoTunedOperator": ("xqt.kernels.dispatcher", "AutoTunedOperator"),
    "BaseFusedOp": ("xqt.kernels.fused_op", "BaseFusedOp"),
    "register_fused_op": ("xqt.kernels.fused_op", "register_fused_op"),
    "get_fused_op_backend": ("xqt.kernels.fused_op", "get_fused_op_backend"),
    "set_fused_op_backend": ("xqt.kernels.fused_op", "set_fused_op_backend"),
    "enable_kernel_trace": ("xqt.kernels.fused_op", "enable_kernel_trace"),
    "get_kernel_trace": ("xqt.kernels.fused_op", "get_kernel_trace"),
    "clear_kernel_trace": ("xqt.kernels.fused_op", "clear_kernel_trace"),
    "disable_kernel_trace": ("xqt.kernels.fused_op", "disable_kernel_trace"),
    "enable_fused_op_trace": ("xqt.kernels.fused_op", "enable_fused_op_trace"),
    "get_fused_op_trace": ("xqt.kernels.fused_op", "get_fused_op_trace"),
    "clear_fused_op_trace": ("xqt.kernels.fused_op", "clear_fused_op_trace"),
    "disable_fused_op_trace": ("xqt.kernels.fused_op", "disable_fused_op_trace"),
}

# Core is torch-free: importing spec/registry/selector never pulls torch.
_ops_loaded = False


def _ensure_ops() -> None:
    global _ops_loaded
    if _ops_loaded:
        return
    importlib.import_module("xqt.kernels.ops")
    _ops_loaded = True


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attr = target
        mod = importlib.import_module(module_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    if name == "ops":
        _ensure_ops()
        return importlib.import_module("xqt.kernels.ops")
    if name in {"wrappers", "nn", "engine_resolve", "precision"}:
        module = importlib.import_module(f"xqt.kernels.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'xqt.kernels' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(
            __all__
            + list(_LAZY_EXPORTS.keys())
            + ["ops", "wrappers", "nn", "engine_resolve", "precision"]
        )
    )
