"""Boundary tests for the three-folder xqt.kernels layout."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "xqt" / "kernels"

_FORBIDDEN_FROM_OPS_IMPL = (
    "xqt.kernels.nn",
    "xqt.kernels.wrappers",
    "xqt.nn",
    "xqt.operator_opt",
    "xqt.conversion",
    "xqt.model",
    "xqt.benchmark",
)
_FORBIDDEN_FROM_GEMM_CONTRACTS = (
    "xqt.kernels.nn",
    "xqt.kernels.wrappers",
    "xqt.nn",
    "xqt.operator_opt",
    "xqt.model",
    "xqt.benchmark",
)
_FORBIDDEN_FROM_NN = (
    "xqt.kernels.jit",
)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _assert_no_forbidden(paths: list[Path], forbidden: tuple[str, ...]) -> None:
    for path in paths:
        for name in _imported_modules(path):
            for prefix in forbidden:
                assert name != prefix and not name.startswith(prefix + "."), (
                    f"{path.relative_to(ROOT)} imports {name}"
                )


def test_ops_impl_and_jit_do_not_import_nn_or_wrappers() -> None:
    paths = [
        *sorted((KERNELS / "ops" / "_impl").rglob("*.py")),
        *sorted((KERNELS / "jit").rglob("*.py")),
    ]
    _assert_no_forbidden(paths, _FORBIDDEN_FROM_OPS_IMPL)


def test_gemm_contracts_do_not_import_nn_or_wrappers() -> None:
    paths = sorted((KERNELS / "ops" / "gemm").glob("*.py"))
    _assert_no_forbidden(paths, _FORBIDDEN_FROM_GEMM_CONTRACTS)


def test_nn_does_not_import_jit() -> None:
    paths = sorted((KERNELS / "nn").rglob("*.py"))
    _assert_no_forbidden(paths, _FORBIDDEN_FROM_NN)


def test_three_kernel_packages_are_importable() -> None:
    import xqt.kernels.nn as nn_pkg
    import xqt.kernels.ops.gemm.contracts as gemm_contracts
    import xqt.kernels.wrappers as wrappers

    assert nn_pkg.Linear is not None
    assert gemm_contracts.GemmProblem is not None
    assert wrappers.materialize_module is not None


def test_public_aliases_resolve_to_kernel_packages() -> None:
    import xqt
    import xqt.conversion
    import xqt.kernels.ops.gemm as gemm_pkg
    import xqt.kernels.nn.convert as convert_mod
    import xqt.kernels.nn as nn_pkg
    import xqt.kernels.nn.fixtures as fixtures
    import xqt.kernels.ops.gemm.contracts as gemm_contracts
    import xqt.kernels.wrappers.bench as bench

    assert xqt.nn.Linear is nn_pkg.Linear
    assert xqt.conversion.convert is convert_mod.convert
    assert gemm_pkg.GemmProblem is gemm_contracts.GemmProblem
    assert bench.benchmark_callable is not None
    assert fixtures.build_smoke_llm is not None


def test_conversion_impl_is_removed() -> None:
    import importlib

    assert not (ROOT / "xqt" / "conversion_impl").exists()
    try:
        importlib.import_module("xqt.conversion_impl")
    except ModuleNotFoundError:
        return
    raise AssertionError("xqt.conversion_impl should not exist")


def test_gemm_package_is_removed() -> None:
    import importlib

    assert not (ROOT / "xqt" / "gemm").exists()
    try:
        importlib.import_module("xqt.gemm")
    except ModuleNotFoundError:
        return
    raise AssertionError("xqt.gemm should not exist")


def test_operator_opt_package_is_removed() -> None:
    import importlib

    assert not (ROOT / "xqt" / "operator_opt").exists()
    try:
        importlib.import_module("xqt.operator_opt")
    except ModuleNotFoundError:
        return
    raise AssertionError("xqt.operator_opt should not exist")


def test_benchmark_package_is_removed() -> None:
    import importlib

    assert not (ROOT / "xqt" / "benchmark").exists()
    try:
        importlib.import_module("xqt.benchmark")
    except ModuleNotFoundError:
        return
    raise AssertionError("xqt.benchmark should not exist")


def test_model_package_is_removed() -> None:
    import importlib

    assert not (ROOT / "xqt" / "model").exists()
    try:
        importlib.import_module("xqt.model")
    except ModuleNotFoundError:
        return
    raise AssertionError("xqt.model should not exist")


def test_nn_package_is_removed() -> None:
    import importlib

    assert not (ROOT / "xqt" / "nn").exists()
    try:
        importlib.import_module("xqt.nn")
    except ModuleNotFoundError:
        return
    raise AssertionError("xqt.nn should not exist")


def test_engine_resolve_and_nvfp4_left_contracts() -> None:
    import importlib

    assert not (ROOT / "xqt" / "contracts" / "engine_resolve.py").exists()
    assert not (ROOT / "xqt" / "contracts" / "nvfp4.py").exists()
    for name in ("xqt.contracts.engine_resolve", "xqt.contracts.nvfp4"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{name} should not exist")


def test_precision_and_engine_resolve_live_in_kernels() -> None:
    import xqt.contracts as contracts
    import xqt.kernels.engine_resolve as engine_resolve
    import xqt.kernels.precision as precision
    import xqt.kernels.ops.quantization.nvfp4 as nvfp4_ops
    import xqt.kernels.wrappers.nvfp4 as nvfp4_wrappers

    assert not hasattr(contracts, "PrecisionPolicy")
    assert not hasattr(contracts, "EngineRegistration")
    assert not hasattr(contracts, "unpack_nvfp4e2m1")
    assert precision.PrecisionPolicy is not None
    assert precision.OperatorContract is precision.ModuleContract
    assert engine_resolve.EngineRegistration is not None
    assert callable(nvfp4_ops.unpack_nvfp4e2m1)
    assert nvfp4_wrappers.NVFP4LinearBridge is not None
