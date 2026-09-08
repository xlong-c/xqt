from __future__ import annotations

import importlib
import sys


def test_resolve_int8_mma_auto_prefers_capability_order() -> None:
    from xqt.kernels.engine_resolve import (
        default_auto_engine_order,
        resolve_int8_mma_engine,
    )

    result = resolve_int8_mma_engine("auto")
    assert result.engine == "tilelang"
    assert result.engine == default_auto_engine_order()[0]
    assert "int8_mma" in result.required_capabilities
    assert result.candidates
    assert result.candidates[0] == "tilelang"


def test_resolve_engine_preferred_is_hint_not_hard_requirement() -> None:
    from xqt.kernels.engine_resolve import resolve_engine

    # preferred engine that cannot provide int8_mma is skipped
    result = resolve_engine(
        required_capabilities=["int8_mma"],
        preferred_engines=["torch", "tilelang"],
    )
    assert result.engine == "tilelang"
    assert "torch" not in result.candidates or result.engine != "torch"


def test_resolve_engine_fp4_mma_prefers_supported_engine() -> None:
    from xqt.kernels.engine_resolve import resolve_engine

    result = resolve_engine(
        required_capabilities=["fp4_mma"],
        preferred_engines=["triton"],
        fallback="torch",
    )

    assert result.engine == "triton"
    assert "fp4_mma" in result.required_capabilities
    assert "triton" in result.candidates


def test_int8_mma_quantizer_import_does_not_load_tilelang_kernels() -> None:
    banned = [
        "xqt.kernels.ops._impl.tilelang.int8_mma",
        "xqt.kernels.ops._impl.cute.int8mma_binding",
    ]
    for name in banned:
        sys.modules.pop(name, None)
    # Ensure fresh import of quantizer module path side effects
    sys.modules.pop("xqt.compression.quant.quantizers.int8_mma", None)
    importlib.import_module("xqt.compression.quant.quantizers.int8_mma")
    for name in banned:
        assert name not in sys.modules, f"{name} was imported at quantizer import time"


def test_query_engine_capabilities_returns_static_registrations() -> None:
    from xqt.kernels.engine_resolve import query_engine_capabilities

    results = query_engine_capabilities(["int8_mma"])
    engine_names = [item.name for item in results]
    # Includes planned, metadata_only, and non-dispatchable engines
    assert "cutlass" in engine_names
    assert "ptx_sm89" in engine_names
    assert "tilelang" in engine_names
    assert "torch_int_mm" in engine_names


def test_resolve_executable_engine_ignores_metadata_only_preferred_hint() -> None:
    from xqt.kernels.engine_resolve import resolve_executable_engine

    # cutlass declares int8_mma capability, but is metadata_only & dispatchable=False
    result = resolve_executable_engine(
        required_capabilities=["int8_mma"],
        preferred_engines=["cutlass"],
    )
    assert result.engine != "cutlass"
    assert "cutlass" not in result.candidates
    assert result.engine in {"tilelang", "torch_int_mm"}


def test_resolve_executable_engine_excludes_missing_package(monkeypatch) -> None:
    from xqt.core.errors import XQTBackendError
    import xqt.kernels.engine_resolve as er

    # Mock package availability for triton to False
    real_is_pkg = er._is_package_available

    def mock_is_pkg(pkg_name: str) -> bool:
        if pkg_name == "triton":
            return False
        return real_is_pkg(pkg_name)

    monkeypatch.setattr(er, "_is_package_available", mock_is_pkg)

    # triton should not be executable
    ok, reason = er.is_engine_executable("triton")
    assert not ok
    assert "not installed" in (reason or "")

    # If triton is preferred, it should be ignored
    result = er.resolve_executable_engine(
        required_capabilities=["int8_mma"],
        preferred_engines=["triton"],
    )
    assert result.engine != "triton"
    assert "triton" not in result.candidates


def test_resolve_executable_engine_fail_closed_on_unsupported() -> None:
    import pytest
    from xqt.core.errors import XQTBackendError
    from xqt.kernels.engine_resolve import resolve_executable_engine

    with pytest.raises(XQTBackendError) as exc_info:
        resolve_executable_engine(
            required_capabilities=["non_existent_capability_xyz_123"],
            fallback="torch_int_mm",
            allow_fallback=False,
        )
    assert "No executable engine satisfies required capabilities" in str(exc_info.value)


def test_resolve_executable_engine_device_cpu_excludes_cuda() -> None:
    from xqt.kernels.engine_resolve import resolve_executable_engine

    # tilelang requires_cuda=True; on CPU it must be excluded
    result = resolve_executable_engine(
        required_capabilities=["int8_mma"],
        preferred_engines=["tilelang"],
        device="cpu",
    )
    # torch_int_mm runs on CPU, so it is selected
    assert result.engine == "torch_int_mm"
    assert "tilelang" not in result.candidates
