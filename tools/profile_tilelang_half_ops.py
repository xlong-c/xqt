"""Minimal profiler entry point for TileLang half-op steady-state analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.operator_opt.kernels.tilelang.attention import (
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)
from xqt.operator_opt.kernels.tilelang.linear import (
    half_linear_reference,
    half_linear_tilelang,
)
from xqt.operator_opt.kernels.tilelang.norm import (
    layer_norm_reference,
    layer_norm_tilelang,
)


DEFAULT_WARMUP = 20
DEFAULT_ITERS = 200


def _require_cuda_tilelang() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TileLang profiler entry")
    if importlib.util.find_spec("tilelang") is None:
        raise RuntimeError("tilelang package is required for TileLang profiler entry")


def _cuda_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _run_attention(*, backend: str, warmup: int, iterations: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)
    if backend == "tilelang":
        fn = lambda: fused_attention_forward_tilelang(
            q,
            k,
            v,
            causal=False,
            dropout_p=0.0,
            block_m=64,
            block_n=64,
            threads=128,
            num_stages=2,
        )
    elif backend == "torch":
        fn = lambda: fused_attention_forward_reference(
            q,
            k,
            v,
            causal=False,
            dropout_p=0.0,
        )
    else:
        raise ValueError(f"unsupported attention backend: {backend}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


def _run_linear(*, backend: str, warmup: int, iterations: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    target_arch = _cuda_arch()
    if backend == "tilelang":
        fn = lambda: half_linear_tilelang(
            x,
            weight,
            bias,
            block_m=64,
            block_n=64,
            block_k=32,
            threads=128,
            num_stages=2,
            target_arch=target_arch,
        )
    elif backend == "torch":
        fn = lambda: half_linear_reference(
            x,
            weight,
            bias,
        )
    else:
        raise ValueError(f"unsupported linear backend: {backend}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


def _run_norm(*, backend: str, warmup: int, iterations: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    if backend == "tilelang":
        fn = lambda: layer_norm_tilelang(
            x,
            weight,
            bias,
            eps=1e-5,
            threads=64,
        )
    elif backend == "torch":
        fn = lambda: layer_norm_reference(
            x,
            weight,
            bias,
            eps=1e-5,
        )
    else:
        raise ValueError(f"unsupported norm backend: {backend}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


def main() -> None:
    _require_cuda_tilelang()
    operator = "attention"
    backend = "tilelang"
    if len(sys.argv) >= 2:
        operator = str(sys.argv[1]).strip().lower()
    if len(sys.argv) >= 3:
        backend = str(sys.argv[2]).strip().lower()

    torch.manual_seed(0)
    print(
        {
            "operator": operator,
            "backend": backend,
            "device": torch.cuda.get_device_name(0),
            "target_arch": _cuda_arch(),
            "warmup": DEFAULT_WARMUP,
            "iterations": DEFAULT_ITERS,
        }
    )
    if operator == "attention":
        _run_attention(backend=backend, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERS)
        return
    if operator == "linear":
        _run_linear(backend=backend, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERS)
        return
    if operator == "norm":
        _run_norm(backend=backend, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERS)
        return
    raise ValueError(f"unsupported operator: {operator}")


if __name__ == "__main__":
    main()
