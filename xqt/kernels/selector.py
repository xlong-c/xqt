"""Device-aware fixed-path kernel resolution over the :data:`registry`."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from xqt.kernels.registry import registry
from xqt.kernels.spec import DeviceType, KernelBackend, KernelSpec, PlatformInfo

if TYPE_CHECKING:
    from xqt.kernels.timing.cache import UnifiedKernelTimingCache

# Default backend precedence when auto-tuning has no timing measurements
_DEFAULT_BACKEND_PRECEDENCE: tuple[KernelBackend, ...] = (
    KernelBackend.TILELANG,
    KernelBackend.TRITON,
    KernelBackend.CUTLASS,
    KernelBackend.CUTE_DSL,
    KernelBackend.FLASHINFER,
    KernelBackend.TORCH_COMPILE,
    KernelBackend.CUSTOM_CUDA,
    KernelBackend.TORCH,
)


@lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


def _get_platform_arch_string(platform: PlatformInfo) -> str:
    if platform.device == DeviceType.CUDA and platform.cuda_arch_major:
        return f"sm_{platform.cuda_arch_major}{platform.cuda_arch_minor or 0}"
    return platform.device.value


def select_kernel(
    op: str,
    backend: Optional[KernelBackend] = None,
    shape: Optional[tuple[int, ...] | Sequence[int]] = None,
    dtype: Optional[str] = None,
    extra: str = "",
    arch: Optional[str] = None,
    auto_tune: bool = False,
) -> KernelSpec:
    specs = registry.get(op)
    if not specs:
        raise KeyError(f"No kernels registered for op {op!r}")

    # Check environment variable override
    env_force = os.environ.get("XQT_FORCE_BACKEND", "").strip().lower()
    if backend is None and env_force:
        for spec in specs:
            if spec.backend.value.lower() == env_force:
                backend = spec.backend
                break

    if backend is not None:
        for spec in specs:
            if spec.backend == backend:
                return spec
        raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")

    if len(specs) == 1:
        return specs[0]

    platform = _platform()
    eligible = [s for s in specs if s.is_available(platform)]
    if len(eligible) == 1:
        return eligible[0]
    if not eligible:
        raise ValueError(
            f"op {op!r} has no backend usable on device {platform.device.value!r} "
            f"(registered: {[s.backend.value for s in specs]})"
        )

    # When auto-tune or shape is requested, arbitrate dynamically
    if auto_tune or shape is not None:
        return _arbitrate_eligible(
            op=op,
            eligible=eligible,
            platform=platform,
            shape=shape,
            dtype=dtype,
            extra=extra,
            arch=arch,
        )

    raise ValueError(
        f"op {op!r} has multiple backends usable on device "
        f"{platform.device.value!r} ({[s.backend.value for s in eligible]}); "
        f"pass backend=... to choose one"
    )


def _arbitrate_eligible(
    op: str,
    eligible: list[KernelSpec],
    platform: PlatformInfo,
    shape: Optional[tuple[int, ...] | Sequence[int]] = None,
    dtype: Optional[str] = None,
    extra: str = "",
    arch: Optional[str] = None,
    preferred_backends: Optional[Sequence[KernelBackend]] = None,
    timing_cache: Optional[UnifiedKernelTimingCache] = None,
) -> KernelSpec:
    """Resolve the optimal KernelSpec from multiple eligible candidates."""
    eligible_backends = {s.backend.value.lower(): s for s in eligible}

    # 1. Query timing cache if shape & dtype are provided
    if shape is not None and dtype is not None:
        if timing_cache is not None:
            cache = timing_cache
        else:
            from xqt.kernels.timing.cache import get_timing_cache

            cache = get_timing_cache()

        arch_candidates: list[str] = []
        if arch is not None:
            arch_candidates.append(arch.lower())
        detected_arch = _get_platform_arch_string(platform)
        if detected_arch.lower() not in arch_candidates:
            arch_candidates.append(detected_arch.lower())
        if "cpu" not in arch_candidates:
            arch_candidates.append("cpu")

        for arch_str in arch_candidates:
            res = cache.query_fastest(
                op=op,
                arch=arch_str,
                dtype=str(dtype).lower(),
                shape=tuple(shape),
                extra=extra,
                eligible_backends=list(eligible_backends.keys()),
            )
            if res is not None and res.fastest_backend in eligible_backends:
                return eligible_backends[res.fastest_backend]

    # 2. Fall back to precedence order
    precedence = preferred_backends or _DEFAULT_BACKEND_PRECEDENCE
    for candidate_backend in precedence:
        for s in eligible:
            if s.backend == candidate_backend:
                return s

    return eligible[0]


def select_fastest_kernel(
    op: str,
    shape: Optional[tuple[int, ...] | Sequence[int]] = None,
    dtype: Optional[str] = None,
    extra: str = "",
    arch: Optional[str] = None,
    preferred_backends: Optional[Sequence[KernelBackend]] = None,
    timing_cache: Optional[UnifiedKernelTimingCache] = None,
) -> KernelSpec:
    """Select the empirically fastest or highest-priority backend for an op."""
    specs = registry.get(op)
    if not specs:
        raise KeyError(f"No kernels registered for op {op!r}")

    # Environment override
    env_force = os.environ.get("XQT_FORCE_BACKEND", "").strip().lower()
    if env_force:
        for spec in specs:
            if spec.backend.value.lower() == env_force:
                return spec

    platform = _platform()
    eligible = [s for s in specs if s.is_available(platform)]
    if not eligible:
        raise ValueError(
            f"op {op!r} has no backend usable on device {platform.device.value!r} "
            f"(registered: {[s.backend.value for s in specs]})"
        )
    if len(eligible) == 1:
        return eligible[0]

    return _arbitrate_eligible(
        op=op,
        eligible=eligible,
        platform=platform,
        shape=shape,
        dtype=dtype,
        extra=extra,
        arch=arch,
        preferred_backends=preferred_backends,
        timing_cache=timing_cache,
    )


@lru_cache(maxsize=None)
def _resolve(op: str, backend: Optional[KernelBackend]) -> Callable[..., object]:
    return select_kernel(op, backend=backend).load()


def get_kernel(op: str, backend: Optional[KernelBackend] = None) -> Callable[..., object]:
    return _resolve(op, backend)


def get_fastest_kernel(
    op: str,
    shape: Optional[tuple[int, ...] | Sequence[int]] = None,
    dtype: Optional[str] = None,
    extra: str = "",
) -> Callable[..., object]:
    """Resolve and load the callable for the fastest kernel specification."""
    spec = select_fastest_kernel(op, shape=shape, dtype=dtype, extra=extra)
    return spec.load()


def clear_cache() -> None:
    _resolve.cache_clear()
    _platform.cache_clear()
    try:
        from xqt.kernels.ops._legacy_api import clear_legacy_cache

        clear_legacy_cache()
    except ImportError:
        pass


