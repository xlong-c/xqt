"""Unified multi-backend / multi-platform operator contract.

Adapted from ``sglang.kernels.fused_op`` for ``xqt.kernels``.
This module is lazily imported via ``xqt.kernels`` module ``__getattr__`` so
that ``import xqt.kernels`` stays torch-free (CPU-only inventory works).
"""

from __future__ import annotations

import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
    capabilities_satisfied,
)

logger = logging.getLogger(__name__)

BACKEND_METHODS: Dict[KernelBackend, str] = {
    KernelBackend.TORCH: "forward_native",
    KernelBackend.TORCH_COMPILE: "forward_torch_compile",
    KernelBackend.TRITON: "forward_triton",
    KernelBackend.TILELANG: "forward_tilelang",
    KernelBackend.CUTILE: "forward_cutile",
    KernelBackend.CUTLASS: "forward_cutlass",
    KernelBackend.CUTE_DSL: "forward_cute_dsl",
    KernelBackend.CUSTOM_CUDA: "forward_custom_cuda",
    KernelBackend.FLASHINFER: "forward_flashinfer",
}

_METHOD_BACKEND_LABELS: Dict[str, str] = {v: k.value for k, v in BACKEND_METHODS.items()}

DEFAULT_PRIORITY: Tuple[KernelBackend, ...] = (
    KernelBackend.FLASHINFER,
    KernelBackend.CUTLASS,
    KernelBackend.CUTE_DSL,
    KernelBackend.TILELANG,
    KernelBackend.CUTILE,
    KernelBackend.CUSTOM_CUDA,
    KernelBackend.TRITON,
    KernelBackend.TORCH,
)

_ALWAYS_AVAILABLE = (KernelBackend.TORCH, KernelBackend.TORCH_COMPILE)

_PLATFORM_METHODS: Dict[str, Tuple[str, ...]] = {
    "cuda": ("forward_cuda",),
    "hip": ("forward_hip", "forward_cuda"),
    "npu": ("forward_npu",),
    "cpu": ("forward_cpu",),
}

_FORCED_BACKEND_ENV = "XQT_FORCE_KERNEL_BACKEND"
_forced_backend: Optional[KernelBackend] = None
_forced_backend_initialized = False
_forced_backend_warned: set[str] = set()

_trace_enabled = False
_trace_records: list[dict[str, Any]] = []


@functools.lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


def _platform_key() -> str:
    info = _platform()
    if info.is_cuda:
        return "cuda"
    if info.is_hip:
        return "hip"
    if info.device_type == "npu":
        return "npu"
    if info.device_type == "cpu":
        return "cpu"
    return ""


def get_fused_op_backend() -> Optional[KernelBackend]:
    global _forced_backend, _forced_backend_initialized
    if _forced_backend_initialized:
        return _forced_backend
    raw = os.environ.get(_FORCED_BACKEND_ENV)
    if raw is not None and raw.strip():
        try:
            _forced_backend = KernelBackend(raw.strip())
        except ValueError:
            logger.warning("Unknown %s=%r", _FORCED_BACKEND_ENV, raw)
            _forced_backend = None
    _forced_backend_initialized = True
    return _forced_backend


def set_fused_op_backend(backend: Optional[KernelBackend]) -> None:
    global _forced_backend, _forced_backend_initialized
    _forced_backend = backend
    _forced_backend_initialized = True


def clear_fused_op_backend_cache() -> None:
    global _forced_backend, _forced_backend_initialized
    _forced_backend = None
    _forced_backend_initialized = False
    _forced_backend_warned.clear()


def enable_kernel_trace() -> None:
    global _trace_enabled
    _trace_enabled = True


def get_kernel_trace() -> list[dict[str, Any]]:
    return list(_trace_records)


def clear_kernel_trace() -> None:
    _trace_records.clear()


def disable_kernel_trace() -> None:
    global _trace_enabled
    _trace_enabled = False
    _trace_records.clear()


# Back-compat aliases
enable_fused_op_trace = enable_kernel_trace
get_fused_op_trace = get_kernel_trace
clear_fused_op_trace = clear_kernel_trace
disable_fused_op_trace = disable_kernel_trace


class BaseFusedOp(nn.Module, ABC):
    op: ClassVar[str] = ""
    priority: ClassVar[Tuple[KernelBackend, ...]] = DEFAULT_PRIORITY
    capabilities: ClassVar[Mapping[KernelBackend, Any]] = {}
    format_signature: ClassVar[FormatSignature] = FormatSignature()
    descriptions: ClassVar[Mapping[KernelBackend, str]] = {}

    def __init__(self) -> None:
        super().__init__()
        self._forward_method: Optional[Callable[..., Any]] = None
        self._in_torch_compile = False
        self._saved_forward_method: Optional[Callable[..., Any]] = None

    @abstractmethod
    def forward_native(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_torch_compile(self, *args: Any, **kwargs: Any) -> Any:
        return torch.compile(self.forward_native)(*args, **kwargs)

    def forward_triton(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_tilelang(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_cutile(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_cutlass(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_cute_dsl(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_custom_cuda(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_flashinfer(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # Platform forwards (optional)
    def forward_cuda(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_hip(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_npu(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_cpu(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def available_backends(self) -> list[KernelBackend]:
        out: list[KernelBackend] = []
        for backend, method_name in BACKEND_METHODS.items():
            if backend in _ALWAYS_AVAILABLE:
                out.append(backend)
                continue
            cls_method = getattr(type(self), method_name, None)
            base_method = getattr(BaseFusedOp, method_name, None)
            if cls_method is not None and cls_method is not base_method:
                out.append(backend)
        return out

    def backend_eligible(self, backend: KernelBackend, *args: Any, **kwargs: Any) -> bool:
        caps = self.capabilities.get(backend)
        if caps is None:
            return False
        return capabilities_satisfied(caps, _platform())

    def _resolve_forward_method(self) -> Callable[..., Any]:
        if type(self).backend_eligible is not BaseFusedOp.backend_eligible:  # type: ignore[comparison-overlap]
            return self._forward_backend_dynamic
        for backend in self.priority:
            if backend in _ALWAYS_AVAILABLE:
                continue
            method_name = BACKEND_METHODS.get(backend)
            if method_name is None:
                continue
            cls_method = getattr(type(self), method_name, None)
            base_method = getattr(BaseFusedOp, method_name, None)
            if cls_method is None or cls_method is base_method:
                continue
            if backend not in self.capabilities:
                continue
            if not self.backend_eligible(backend):
                continue
            return getattr(self, method_name)
        key = _platform_key()
        for method_name in _PLATFORM_METHODS.get(key, ()):
            cls_method = getattr(type(self), method_name, None)
            base_method = getattr(BaseFusedOp, method_name, None)
            if cls_method is not None and cls_method is not base_method:
                return getattr(self, method_name)
        return self.forward_native

    def _forward_backend_dynamic(self, *args: Any, **kwargs: Any) -> Any:
        for backend in self.priority:
            if backend in _ALWAYS_AVAILABLE:
                continue
            method_name = BACKEND_METHODS.get(backend)
            if method_name is None:
                continue
            cls_method = getattr(type(self), method_name, None)
            base_method = getattr(BaseFusedOp, method_name, None)
            if cls_method is None or cls_method is base_method:
                continue
            if backend not in self.capabilities:
                continue
            if not self.backend_eligible(backend, *args, **kwargs):
                continue
            return getattr(self, method_name)(*args, **kwargs)
        key = _platform_key()
        for method_name in _PLATFORM_METHODS.get(key, ()):
            cls_method = getattr(type(self), method_name, None)
            base_method = getattr(BaseFusedOp, method_name, None)
            if cls_method is not None and cls_method is not base_method:
                return getattr(self, method_name)(*args, **kwargs)
        return self.forward_native(*args, **kwargs)

    def _call_with_trace(self, backend_label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if _trace_enabled:
            _trace_records.append({"op": self.op, "backend": backend_label, "args_types": [type(a).__name__ for a in args]})
        return fn(*args, **kwargs)

    def forward(self, *args: Any, backend: Optional[KernelBackend] = None, **kwargs: Any) -> Any:
        if backend is not None:
            method_name = BACKEND_METHODS.get(backend)
            if method_name is None:
                raise ValueError(f"Unknown backend {backend!r}")
            fn = getattr(self, method_name, None)
            if fn is None:
                raise ValueError(f"Backend {backend.value!r} not available for op {self.op!r}")
            base_fn = getattr(BaseFusedOp, method_name, None)
            cls_fn = getattr(type(self), method_name, None)
            if backend not in _ALWAYS_AVAILABLE and cls_fn is base_fn:
                raise ValueError(f"Backend {backend.value!r} not implemented for op {self.op!r}")
            return self._call_with_trace(backend.value, fn, *args, **kwargs)

        forced = get_fused_op_backend()
        if forced is not None:
            method_name = BACKEND_METHODS.get(forced)
            if method_name is not None:
                fn = getattr(self, method_name, None)
                base_fn = getattr(BaseFusedOp, method_name, None)
                cls_fn = getattr(type(self), method_name, None)
                is_available = forced in _ALWAYS_AVAILABLE or (cls_fn is not None and cls_fn is not base_fn)
                if is_available:
                    return self._call_with_trace(forced.value, fn, *args, **kwargs)
                key = f"{self.op}:{forced.value}"
                if key not in _forced_backend_warned:
                    logger.warning("Forced backend %r not available for op %r, falling back", forced.value, self.op)
                    _forced_backend_warned.add(key)

        if self._in_torch_compile:
            return self._call_with_trace("torch", self.forward_native, *args, **kwargs)

        if self._forward_method is None:
            resolved = self._resolve_forward_method()
            if resolved is self._forward_backend_dynamic:
                self._forward_method = resolved
                result = self._forward_method(*args, **kwargs)
                if _trace_enabled:
                    _trace_records.append({"op": self.op, "backend": "dynamic", "args_types": [type(a).__name__ for a in args]})
                return result
            self._forward_method = resolved

        label = "native"
        for b, m in BACKEND_METHODS.items():
            if getattr(self, m, None) is self._forward_method:
                label = b.value
                break
        if self._forward_method is self.forward_native:
            label = "torch"
        elif self._forward_method is getattr(self, "forward_cuda", None):
            label = "cuda"
        elif self._forward_method is getattr(self, "forward_hip", None):
            label = "hip"
        return self._call_with_trace(label, self._forward_method, *args, **kwargs)

    def _torch_compile_forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_native(*args, **kwargs)

    def enter_torch_compile(self, *args: Any, **kwargs: Any) -> None:
        if self._in_torch_compile:
            return
        self._saved_forward_method = self._forward_method
        self._forward_method = self.forward_native
        self._in_torch_compile = True

    def leave_torch_compile(self) -> None:
        if not self._in_torch_compile:
            return
        self._forward_method = self._saved_forward_method
        self._saved_forward_method = None
        self._in_torch_compile = False

    def register_oot_forward(self, key: str, fn: Callable[..., Any]) -> None:
        setattr(self, f"forward_{key}", fn)


def register_fused_op(instance: BaseFusedOp, module: str, attr: str) -> BaseFusedOp:
    for backend in instance.available_backends():
        method_name = BACKEND_METHODS.get(backend)
        if method_name is None:
            continue
        caps = instance.capabilities.get(backend, frozenset())
        if isinstance(caps, set):
            caps = frozenset(caps)
        elif isinstance(caps, tuple):
            caps = frozenset(caps)
        spec = KernelSpec(
            op=instance.op,
            backend=backend,
            target=f"{module}:{attr}.{method_name}",
            capabilities=caps if isinstance(caps, frozenset) else frozenset(),
            format_signature=instance.format_signature,
            description=instance.descriptions.get(backend, ""),
        )
        try:
            register_kernel(spec)
        except ValueError as exc:
            logger.warning("register_fused_op skipped duplicate %r backend %r: %s", spec.op, spec.backend.value, exc)
    return instance
