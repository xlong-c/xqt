"""Tests for Registry authoritative layering, one-way data flow, and idempotency (XQT-011)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from xqt.kernels.ops.gemm.registry import (
    GemmCapability,
    GemmKernelRegistration,
    GemmKernelRegistry,
    default_registry,
)
from xqt.kernels.registry import registry, sync_gemm_inventory


def test_gemm_registration_idempotency_and_conflict() -> None:
    reg = GemmKernelRegistry()
    entry1 = GemmKernelRegistration(
        name="test_op_layering",
        backend="torch",
        maturity="executable",
        capability=GemmCapability(architectures=("any",), weight_dtypes=("fp16",)),
        kernel_family="gemm",
        executor=lambda a, b: a @ b,
    )
    # 第一次注册
    reg.register(entry1)
    assert reg.get("test_op_layering") == entry1

    # 重复注册完全相同内容：必须保持幂等
    reg.register(entry1)
    assert len(reg.entries()) == 1

    # 重复注册同名但不同内容：必须显式抛出包含冲突详情的 ValueError
    conflict_entry = GemmKernelRegistration(
        name="test_op_layering",
        backend="tilelang",
        maturity="executable",
        capability=GemmCapability(architectures=("sm_89",), weight_dtypes=("fp16",)),
        kernel_family="gemm",
        executor=lambda a, b: a @ b,
    )
    with pytest.raises(ValueError, match="Conflicting GEMM kernel registration"):
        reg.register(conflict_entry)


def test_ops_gemm_registry_has_no_dependency_on_kernels_registry() -> None:
    import ast
    import inspect
    import xqt.kernels.ops.gemm.registry as gr

    source = inspect.getsource(gr)
    tree = ast.parse(source)

    # 检查模块级与函数内局部的所有 import 语句
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)

    assert not any("kernels.registry" in mod for mod in imported_modules), (
        f"ops.gemm.registry must not import kernels.registry, found: {imported_modules}"
    )
    assert "mirror_legacy_gemm_entry" not in source


def test_single_authoritative_source_consistency() -> None:
    sync_gemm_inventory(force=True)
    gemm_entries = default_registry().entries()
    assert len(gemm_entries) > 0

    # 验证每一个具备标准 backend 的 GEMM entry 在全局 registry 中均可查到
    for entry in gemm_entries:
        if entry.backend in {"torch", "triton", "tilelang", "cutile", "cutlass", "cute_dsl"}:
            op_name = f"gemm.{entry.name}"
            assert registry.has(op_name), f"Global registry missing {op_name}"
            specs = registry.get(op_name)
            assert any(spec.backend.value == entry.backend for spec in specs)


def test_optional_backend_registration_no_cuda_init() -> None:
    code = (
        "import sys; "
        "import torch; "
        "import xqt.kernels.ops.gemm.registry as gr; "
        "reg = gr.default_registry(); "
        "assert not torch.cuda.is_initialized(), 'CUDA must not be initialized during registry setup'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
