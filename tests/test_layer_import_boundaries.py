"""Guard tests for XQT layer import boundaries.

Ensures transform (quant) and inference (runtime) layers remain decoupled:
- xqt.quant must not import from xqt.runtime
- xqt.runtime must not import from xqt.quant
- xqt.runtime must not import from xqt.export
- xqt.contracts must not import from xqt.quant / xqt.runtime / xqt.export

Quantizers produce contracts-layer storage artifacts. Runtime modules subclass
those storage shells and add backend-specific execution views.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, Sequence

import pytest

XQT_ROOT = Path(__file__).resolve().parent.parent.parent / "xqt"

# --- AST scan ------------------------------------------------------------------

def _scan_imports_ast(
    py_files: Iterable[Path], forbidden_prefixes: Sequence[str],
) -> list[str]:
    """Scan Python source files via AST for Import/ImportFrom violations.

    Returns a list of human-readable violation strings like
    ``quant/quantizers/int8_mma.py:19: from xqt.runtime.modules import ...``.
    """
    violations: list[str] = []
    for fp in sorted(py_files):
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = None
                # check each alias
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        if alias.name == prefix or alias.name.startswith(prefix + "."):
                            rel = fp.relative_to(XQT_ROOT)
                            violations.append(
                                f"{rel}:{node.lineno}: import {alias.name}"
                            )
                continue  # already handled aliases
            else:
                continue

            for prefix in forbidden_prefixes:
                if module == prefix or module.startswith(prefix + "."):
                    rel = fp.relative_to(XQT_ROOT)
                    violations.append(
                        f"{rel}:{node.lineno}: from {module} import ..."
                    )

    return violations

# --- String scan (importlib / __import__) --------------------------------------

_STRING_IMPORT_RE = re.compile(
    r"""(?:
        importlib\.import_module\s*\(\s*(['"])(.+?)\1
        |
        __import__\s*\(\s*(['"])(.+?)\3
    )""",
    re.VERBOSE,
)


def _scan_imports_string(
    py_files: Iterable[Path], forbidden_prefixes: Sequence[str],
) -> list[str]:
    """String-scan for ``importlib.import_module()`` / ``__import__()``
    targeting forbidden prefixes."""
    violations: list[str] = []
    for fp in sorted(py_files):
        text = fp.read_text(encoding="utf-8")
        for m in _STRING_IMPORT_RE.finditer(text):
            target = m.group(2) or m.group(4)
            if not target:
                continue
            for prefix in forbidden_prefixes:
                if target == prefix or target.startswith(prefix + "."):
                    lineno = text[: m.start()].count("\n") + 1
                    rel = fp.relative_to(XQT_ROOT)
                    violations.append(
                        f"{rel}:{lineno}: dynamic import {target!r}"
                    )
    return violations


# --- Test helpers --------------------------------------------------------------

def _quant_py_files() -> list[Path]:
    return sorted((XQT_ROOT / "quant").rglob("*.py"))


def _runtime_py_files() -> list[Path]:
    return sorted((XQT_ROOT / "runtime").rglob("*.py"))


def _contracts_py_files() -> list[Path]:
    return sorted((XQT_ROOT / "contracts").rglob("*.py"))


# ---------------------------------------------------------------------------
# quant → runtime
# ---------------------------------------------------------------------------

def test_quant_no_runtime_imports_ast() -> None:
    """xqt.quant must not import from xqt.runtime."""
    violations = _scan_imports_ast(_quant_py_files(), ["xqt.runtime"])
    assert violations == [], (
        f"quant → runtime violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_quant_no_runtime_imports_string() -> None:
    """No dynamic imports into xqt.runtime from xqt.quant."""
    violations = _scan_imports_string(_quant_py_files(), ["xqt.runtime"])
    assert violations == [], (
        f"dynamic quant → runtime violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# runtime → quant  (expected PASS — no current violations)
# ---------------------------------------------------------------------------

def test_runtime_no_quant_imports_ast() -> None:
    """xqt.runtime must not import from xqt.quant."""
    violations = _scan_imports_ast(_runtime_py_files(), ["xqt.quant"])
    assert violations == [], (
        f"runtime → quant violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_runtime_no_quant_imports_string() -> None:
    """No dynamic imports into xqt.quant from xqt.runtime."""
    violations = _scan_imports_string(_runtime_py_files(), ["xqt.quant"])
    assert violations == [], (
        f"dynamic runtime → quant violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# runtime → export  (expected PASS — no current violations)
# ---------------------------------------------------------------------------

def test_runtime_no_export_imports_ast() -> None:
    """xqt.runtime must not import from xqt.export."""
    violations = _scan_imports_ast(_runtime_py_files(), ["xqt.export"])
    assert violations == [], (
        f"runtime → export violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_runtime_no_export_imports_string() -> None:
    """No dynamic imports into xqt.export from xqt.runtime."""
    violations = _scan_imports_string(_runtime_py_files(), ["xqt.export"])
    assert violations == [], (
        f"dynamic runtime → export violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# contracts  (expected PASS — no current violations)
# ---------------------------------------------------------------------------

_CONTRACTS_FORBIDDEN = ["xqt.quant", "xqt.runtime", "xqt.export"]


_CONVROT_QUANTIZER_FILES = {
    XQT_ROOT / "quant/quantizers/convrot_int8.py",
    XQT_ROOT / "quant/quantizers/convrot_4bit.py",
}


def _scan_top_level_kernel_imports(py_file: Path) -> list[str]:
    """Find eager operator-kernel imports in the ConvRot compatibility files."""

    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    violations: list[str] = []

    class _Visitor(ast.NodeVisitor):
        function_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node: ast.Import) -> None:
            if self.function_depth == 0:
                for alias in node.names:
                    if alias.name.startswith("xqt.operator_opt.kernels"):
                        rel = py_file.relative_to(XQT_ROOT)
                        violations.append(
                            f"{rel}:{node.lineno}: import {alias.name}"
                        )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if self.function_depth == 0 and (node.module or "").startswith(
                "xqt.operator_opt.kernels"
            ):
                rel = py_file.relative_to(XQT_ROOT)
                violations.append(
                    f"{rel}:{node.lineno}: from {node.module} import ..."
                )

    _Visitor().visit(tree)
    return violations


def test_convrot_kernel_compat_imports_are_lazy() -> None:
    """Historical ConvRot kernel imports must stay out of module import time."""

    violations = [
        violation
        for py_file in sorted(_CONVROT_QUANTIZER_FILES)
        for violation in _scan_top_level_kernel_imports(py_file)
    ]
    assert violations == [], (
        "ConvRot quantizers must keep optional kernel imports lazy:\n"
        + "\n".join(f"  {violation}" for violation in violations)
    )


def test_contracts_no_forbidden_imports_ast() -> None:
    """xqt.contracts must not import from quant/runtime/export layers."""
    violations = _scan_imports_ast(
        _contracts_py_files(), _CONTRACTS_FORBIDDEN,
    )
    assert violations == [], (
        f"contracts forbidden violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_contracts_no_forbidden_imports_string() -> None:
    """No dynamic imports into forbidden layers from xqt.contracts."""
    violations = _scan_imports_string(
        _contracts_py_files(), _CONTRACTS_FORBIDDEN,
    )
    assert violations == [], (
        f"contracts dynamic forbidden violations ({len(violations)}):\n"
        + "\n".join(f"  {v}" for v in violations)
    )
