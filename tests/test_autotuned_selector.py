"""Tests for data-driven auto-tuned kernel selection and dynamic operator dispatch."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from xqt.kernels.dispatcher import AutoTunedOperator
from xqt.kernels.registry import registry
from xqt.kernels.selector import (
    clear_cache,
    get_fastest_kernel,
    select_fastest_kernel,
    select_kernel,
)
from xqt.kernels.spec import KernelBackend, KernelSpec
from xqt.kernels.timing.cache import UnifiedKernelTimingCache


def _mock_op(name: str) -> str:
    return f"test.autotuned.{name}"


def test_select_fastest_kernel_with_timing_cache(tmp_path: Path) -> None:
    op = _mock_op("competing_ops")
    s_triton = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:abs")
    s_tilelang = KernelSpec(op=op, backend=KernelBackend.TILELANG, target="builtins:len")
    registry.register(s_triton)
    registry.register(s_tilelang)

    cache = UnifiedKernelTimingCache(cache_path=tmp_path / "timing.json")
    shape = (1, 1024)
    # Record triton as 50us, tilelang as 20us
    cache.record_measurement(
        op=op,
        backend="triton",
        arch="cpu",
        dtype="float32",
        shape=shape,
        median_latency_us=50.0,
    )
    cache.record_measurement(
        op=op,
        backend="tilelang",
        arch="cpu",
        dtype="float32",
        shape=shape,
        median_latency_us=20.0,
    )

    try:
        # Should pick tilelang as fastest
        spec = select_fastest_kernel(
            op=op,
            shape=shape,
            dtype="float32",
            timing_cache=cache,
        )
        assert spec.backend == KernelBackend.TILELANG
        assert spec.target == "builtins:len"

        # If triton is recorded faster (e.g. 10us) for another shape
        shape2 = (64, 1024)
        cache.record_measurement(
            op=op,
            backend="triton",
            arch="cpu",
            dtype="float32",
            shape=shape2,
            median_latency_us=10.0,
        )
        cache.record_measurement(
            op=op,
            backend="tilelang",
            arch="cpu",
            dtype="float32",
            shape=shape2,
            median_latency_us=40.0,
        )
        spec2 = select_fastest_kernel(
            op=op,
            shape=shape2,
            dtype="float32",
            timing_cache=cache,
        )
        assert spec2.backend == KernelBackend.TRITON
        assert spec2.target == "builtins:abs"
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_select_fastest_kernel_precedence_fallback() -> None:
    op = _mock_op("precedence")
    s_torch = KernelSpec(op=op, backend=KernelBackend.TORCH, target="builtins:int")
    s_triton = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:float")
    registry.register(s_torch)
    registry.register(s_triton)

    try:
        # Without timing cache data, triton has higher precedence than torch
        spec = select_fastest_kernel(op)
        assert spec.backend == KernelBackend.TRITON

        # With custom preferred backends precedence
        spec_custom = select_fastest_kernel(
            op,
            preferred_backends=[KernelBackend.TORCH, KernelBackend.TRITON],
        )
        assert spec_custom.backend == KernelBackend.TORCH
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_force_backend_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _mock_op("env_override")
    s_torch = KernelSpec(op=op, backend=KernelBackend.TORCH, target="builtins:int")
    s_tilelang = KernelSpec(op=op, backend=KernelBackend.TILELANG, target="builtins:len")
    registry.register(s_torch)
    registry.register(s_tilelang)

    try:
        monkeypatch.setenv("XQT_FORCE_BACKEND", "torch")
        spec = select_fastest_kernel(op)
        assert spec.backend == KernelBackend.TORCH

        # Even select_kernel respects the environment override
        spec2 = select_kernel(op)
        assert spec2.backend == KernelBackend.TORCH
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_legacy_select_kernel_multiple_backends_raises() -> None:
    op = _mock_op("strict_legacy")
    s_torch = KernelSpec(op=op, backend=KernelBackend.TORCH, target="builtins:int")
    s_triton = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:abs")
    registry.register(s_torch)
    registry.register(s_triton)

    try:
        # Standard select_kernel without shape or auto_tune raises ValueError
        with pytest.raises(ValueError, match="multiple backends"):
            select_kernel(op)

        # But passing auto_tune=True resolves dynamically
        spec = select_kernel(op, auto_tune=True)
        assert spec.backend in {KernelBackend.TORCH, KernelBackend.TRITON}
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_autotuned_operator_dispatch_and_fallback() -> None:
    op = _mock_op("dispatcher_test")
    # Target that will fail when called with tensor
    s_fail = KernelSpec(op=op, backend=KernelBackend.TILELANG, target="builtins:len")
    registry.register(s_fail)

    fallback_called = False

    def fallback_implementation(x: torch.Tensor) -> torch.Tensor:
        nonlocal fallback_called
        fallback_called = True
        return x * 2.0

    operator = AutoTunedOperator(op=op, fallback_fn=fallback_implementation)

    try:
        # len(torch.randn(2, 4)) returns int 2, but if an error occurs it should fall back
        t = torch.tensor([1.0, 2.0, 3.0])
        # When calling with fallback
        res = operator(t)
        # Builtin len on 1D tensor returns 3, which succeeds without exception
        assert res == 3

        # Now register a kernel that raises TypeError
        def broken_fn(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated kernel failure")

        operator.fallback_fn = fallback_implementation
        # Force fallback trigger
        s_broken = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:dict")
        # monkeypatch get_fastest_kernel
        import xqt.kernels.dispatcher as disp_mod

        old_fn = disp_mod.get_fastest_kernel
        disp_mod.get_fastest_kernel = lambda *a, **kw: broken_fn  # type: ignore[assignment]
        try:
            out = operator(t)
            assert fallback_called is True
            assert torch.allclose(out, torch.tensor([2.0, 4.0, 6.0]))
        finally:
            disp_mod.get_fastest_kernel = old_fn  # type: ignore[assignment]
    finally:
        registry._by_op.pop(op, None)
        clear_cache()
