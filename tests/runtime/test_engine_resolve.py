from __future__ import annotations

import importlib
import sys


def test_resolve_int8_mma_auto_prefers_capability_order() -> None:
    from xqt.contracts.engine_resolve import (
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
    from xqt.contracts.engine_resolve import resolve_engine

    # preferred engine that cannot provide int8_mma is skipped
    result = resolve_engine(
        required_capabilities=["int8_mma"],
        preferred_engines=["torch", "tilelang"],
    )
    assert result.engine == "tilelang"
    assert "torch" not in result.candidates or result.engine != "torch"


def test_resolve_engine_fp4_mma_prefers_supported_engine() -> None:
    from xqt.contracts.engine_resolve import resolve_engine

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
        "xqt.operator_opt.kernels.tilelang.int8_mma",
        "xqt.operator_opt.kernels.cute.int8mma_binding",
    ]
    for name in banned:
        sys.modules.pop(name, None)
    # Ensure fresh import of quantizer module path side effects
    sys.modules.pop("xqt.quant.quantizers.int8_mma", None)
    importlib.import_module("xqt.quant.quantizers.int8_mma")
    for name in banned:
        assert name not in sys.modules, f"{name} was imported at quantizer import time"
