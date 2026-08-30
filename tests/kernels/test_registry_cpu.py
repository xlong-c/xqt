"""CPU-only registry contract — S1/S2/S4/S5."""

from __future__ import annotations

import subprocess
import sys

import pytest

from xqt.kernels.registry import registry
from xqt.kernels.selector import clear_cache, get_kernel, select_kernel
from xqt.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend, KernelSpec


def _unique_op(prefix: str = "test.kernels.cpu") -> str:
    import uuid

    return f"{prefix}.{uuid.uuid4().hex[:8]}"


def test_s1_core_import_does_not_pull_torch() -> None:
    code = (
        "import sys; "
        "import xqt.kernels.spec, xqt.kernels.registry, xqt.kernels.selector"
        "; "
        "assert 'torch' not in sys.modules, 'torch should not be imported'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_s1_registry_ops_sorted() -> None:
    ops = registry.ops()
    assert ops == sorted(ops)


def test_s2_single_backend_select_without_backend() -> None:
    op = _unique_op("test.s2.single")
    spec = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:print")
    registry.register(spec)
    try:
        selected = select_kernel(op)
        assert selected.backend == KernelBackend.TRITON
        assert selected.op == op
    finally:
        # cleanup
        registry._by_op.pop(op, None)
        clear_cache()


def test_s4_multi_backend_requires_explicit_backend() -> None:
    op = _unique_op("test.s4.multi")
    s1 = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:print")
    s2 = KernelSpec(op=op, backend=KernelBackend.TILELANG, target="builtins:len")
    registry.register(s1)
    registry.register(s2)
    try:
        with pytest.raises(ValueError, match="multiple backends"):
            select_kernel(op)
        assert select_kernel(op, backend=KernelBackend.TRITON).target == "builtins:print"
        assert select_kernel(op, backend=KernelBackend.TILELANG).target == "builtins:len"
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_s4_unknown_op_raises() -> None:
    with pytest.raises(KeyError):
        select_kernel("test.kernels.nonexistent_xyz_999")


def test_s4_explicit_backend_not_registered_raises() -> None:
    op = _unique_op("test.s4.missing")
    spec = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:print")
    registry.register(spec)
    try:
        with pytest.raises(KeyError, match="No '.*' backend"):
            select_kernel(op, backend=KernelBackend.FLASHINFER)
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_s4_no_available_backend_raises() -> None:
    from xqt.kernels.selector import _platform as _get_platform

    platform = _get_platform()
    # Pick a device opposite to current platform so both are ineligible.
    opposite = CapabilityRequirement.HIP if platform.device.value != "hip" else CapabilityRequirement.CUDA
    op = _unique_op("test.s4.noavail")
    s = KernelSpec(
        op=op,
        backend=KernelBackend.TRITON,
        target="builtins:print",
        capabilities=frozenset({opposite}),
    )
    registry.register(s)
    s2 = KernelSpec(
        op=op,
        backend=KernelBackend.FLASHINFER,
        target="builtins:len",
        capabilities=frozenset({opposite}),
    )
    registry.register(s2)
    try:
        with pytest.raises(ValueError, match="no backend usable"):
            select_kernel(op)
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_s4_capability_hard_filter_single_remains() -> None:
    from xqt.kernels.selector import _platform as _get_platform

    platform = _get_platform()
    opposite = CapabilityRequirement.HIP if platform.device.value != "hip" else CapabilityRequirement.CUDA
    # One eligible (empty caps), one ineligible (opposite device)
    op = _unique_op("test.s4.filter")
    s_bad = KernelSpec(
        op=op,
        backend=KernelBackend.TRITON,
        target="builtins:print",
        capabilities=frozenset({opposite}),
    )
    s_good = KernelSpec(op=op, backend=KernelBackend.TORCH, target="builtins:len")
    registry.register(s_bad)
    registry.register(s_good)
    try:
        selected = select_kernel(op)
        assert selected.backend == KernelBackend.TORCH
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_s5_idempotent_register_no_error() -> None:
    op = _unique_op("test.s5.idem")
    spec = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:print")
    registry.register(spec)
    try:
        # same spec again -> no error, returns same
        registry.register(spec)
        assert len(registry.get(op)) == 1
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_s5_conflicting_register_raises() -> None:
    op = _unique_op("test.s5.conflict")
    s1 = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:print")
    s2 = KernelSpec(op=op, backend=KernelBackend.TRITON, target="builtins:len")
    registry.register(s1)
    try:
        with pytest.raises(ValueError, match="Conflicting kernel registration"):
            registry.register(s2)
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_s5_get_kernel_caches_and_loads() -> None:
    op = _unique_op("test.s5.cache")
    spec = KernelSpec(op=op, backend=KernelBackend.TORCH, target="builtins:len")
    registry.register(spec)
    try:
        fn1 = get_kernel(op)
        fn2 = get_kernel(op)
        assert fn1 is fn2
        assert fn1 is len
    finally:
        registry._by_op.pop(op, None)
        clear_cache()


def test_format_signature_and_capabilities_satisfied() -> None:
    from xqt.kernels.spec import PlatformInfo, capabilities_satisfied

    cpu = PlatformInfo(device_type="cpu")
    assert capabilities_satisfied(frozenset(), cpu) is True
    assert capabilities_satisfied(frozenset({CapabilityRequirement.CUDA}), cpu) is False
    spec = KernelSpec(
        op=_unique_op("test.misc"),
        backend=KernelBackend.TORCH,
        target="builtins:print",
        format_signature=FormatSignature(supported_dtypes=("float16",), description="test"),
    )
    assert spec.format_signature.supported_dtypes == ("float16",)
