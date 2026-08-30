"""Parity check: xqt.kernels.registry vs legacy registries."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xqt.kernels  # noqa: F401
import xqt.kernels.ops  # noqa: F401  # trigger group registration
from xqt.kernels.registry import registry
from xqt.kernels.spec import KernelBackend

STRICT = os.environ.get("XQT_KERNEL_PARITY_STRICT", "0") == "1" or "--strict" in sys.argv[1:]

_BACKEND_MAP = {
    "torch": KernelBackend.TORCH,
    "triton": KernelBackend.TRITON,
    "tilelang": KernelBackend.TILELANG,
    "cutile": KernelBackend.CUTILE,
    "cutlass": KernelBackend.CUTLASS,
    "cute_dsl": KernelBackend.CUTE_DSL,
    "custom_cuda": KernelBackend.CUSTOM_CUDA,
    "flashinfer": KernelBackend.FLASHINFER,
}


def _legacy_gemm_ops() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    try:
        mod = importlib.import_module("xqt.kernels.ops.gemm.registry")
        r = mod.default_registry()
        for e in r.entries():
            backend = _BACKEND_MAP.get(e.backend)
            if backend is not None:
                out.add((f"gemm.{e.name}", backend.value))
    except Exception:
        pass
    return out


def _legacy_operator_ops() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for bk, group in [
        ("triton", "triton"),
        ("tilelang", "tilelang"),
        ("cutile", "cutile"),
        ("cutlass", "cutlass"),
        ("cute_dsl", "cute_dsl"),
    ]:
        try:
            m = importlib.import_module(f"xqt.kernels.ops._impl.engines.{bk}")
            for attr in dir(m):
                if "REGISTRY" in attr:
                    reg = getattr(m, attr)
                    if isinstance(reg, dict):
                        for pattern in reg:
                            out.add((f"{group}.{pattern}", group))
                        break
        except Exception:
            pass
    return out


def main() -> int:
    ops = set(registry.ops())
    new_pairs = {(s.op, s.backend.value) for s in registry.all_specs()}
    print(f"xqt.kernels ops: {sorted(ops)}")
    print(f"total specs: {len(registry.all_specs())}")

    legacy_gemm = _legacy_gemm_ops()
    legacy_op = _legacy_operator_ops()
    print(f"legacy gemm entries: {len(legacy_gemm)}")
    print(f"legacy operator patterns: {len(legacy_op)}")

    expected = {"gemm.bmm_fp8", "attention.fused_attention", "layernorm.rmsnorm"}
    missing = expected - ops
    if missing:
        print(f"Missing expected ops: {missing}", file=sys.stderr)
        return 1

    flash = [s for s in registry.all_specs() if s.backend == KernelBackend.FLASHINFER]
    if not flash:
        print("No FLASHINFER specs found", file=sys.stderr)
        return 1
    print(f"FLASHINFER specs: {len(flash)} ok")

    seen: set[tuple[str, str]] = set()
    for s in registry.all_specs():
        key = (s.op, s.backend.value)
        if key in seen:
            print(f"Duplicate (op, backend): {key}", file=sys.stderr)
            return 1
        seen.add(key)

    gemm_gap = legacy_gemm - new_pairs
    op_gap = legacy_op - new_pairs
    print(f"gemm entries not in xqt.kernels: {len(gemm_gap)}")
    print(f"operator patterns not in xqt.kernels: {len(op_gap)}")
    if STRICT and (gemm_gap or op_gap):
        print("Parity FAILED: legacy entries missing from unified registry", file=sys.stderr)
        return 1

    print("Parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
