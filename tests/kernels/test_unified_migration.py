"""Regression tests for the unified kernel migration bridge."""

from __future__ import annotations

import subprocess
import sys


def test_legacy_registry_entries_are_mirrored() -> None:
    import xqt.kernels.engine_resolve  # noqa: F401  # populate engine_registry
    import xqt.kernels.ops  # noqa: F401  # trigger registration

    from xqt.kernels.ops.gemm.registry import default_registry
    from xqt.kernels.registry import engine_registry, registry

    assert len(engine_registry) >= 1
    for entry in default_registry().entries():
        if entry.backend in {"torch", "triton", "tilelang", "cutile", "cutlass", "cute_dsl"}:
            assert registry.has(f"gemm.{entry.name}")


def test_operator_opt_package_is_removed() -> None:
    code = "import importlib; importlib.import_module('xqt.operator_opt')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


def test_benchmark_package_is_removed() -> None:
    code = "import importlib; importlib.import_module('xqt.benchmark')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr




def test_nn_package_is_removed() -> None:
    code = "import importlib; importlib.import_module('xqt.nn')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr

