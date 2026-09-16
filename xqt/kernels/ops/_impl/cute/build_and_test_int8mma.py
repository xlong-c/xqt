#!/usr/bin/env python3
"""Build + correctness + microbench for int8mma Ada kernel (sm_89)."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from xqt.kernels.jit.utils.compile import csrc_path

ROOT = Path(__file__).resolve().parent
SRC = csrc_path("quantization", "int8mma_kernel.cu")
SO = ROOT / "build" / "int8mma_sm89.so"


def _cutlass_include_dir() -> Path:
    """Resolve CUTLASS headers for both the monorepo and standalone checkouts."""

    override = os.environ.get("XQT_CUTLASS_INCLUDE")
    if override:
        return Path(override).expanduser().resolve()
    for ancestor in ROOT.parents[3:5]:
        candidate = ancestor / "third_party" / "cutlass" / "include"
        if (candidate / "cutlass").is_dir():
            return candidate
    raise RuntimeError(
        "CUTLASS headers not found; set XQT_CUTLASS_INCLUDE to an include directory"
    )


def build() -> Path:
    SO.parent.mkdir(parents=True, exist_ok=True)
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-use_fast_math",
        "-lineinfo",
        "-gencode",
        "arch=compute_89,code=sm_89",
        "-gencode",
        "arch=compute_89,code=compute_89",
        "-I",
        str(_cutlass_include_dir()),
        str(SRC),
        "-o",
        str(SO),
        "-lcublasLt",
        "-lcublas",
        "-lcudart",
    ]
    print("BUILD:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("SO:", SO, "size=", SO.stat().st_size)
    return SO


def load_lib(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.int8mma_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.int8mma_run.restype = ctypes.c_int
    lib.int8mma_smem_bytes.argtypes = []
    lib.int8mma_smem_bytes.restype = ctypes.c_int
    lib.int8mma_version.argtypes = []
    lib.int8mma_version.restype = ctypes.c_char_p
    if hasattr(lib, "int8mma_run_prepacked_b"):
        lib.int8mma_run_prepacked_b.argtypes = lib.int8mma_run.argtypes
        lib.int8mma_run_prepacked_b.restype = ctypes.c_int
    if hasattr(lib, "int8mma_run_cutlass_64x128_prepacked_b"):
        lib.int8mma_run_cutlass_64x128_prepacked_b.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.int8mma_run_cutlass_64x128_prepacked_b.restype = ctypes.c_int
    return lib


def ref_mm(a: torch.Tensor, b: torch.Tensor, sa: float, sw: torch.Tensor) -> torch.Tensor:
    if a.shape[0] % 8 == 0 and b.shape[1] % 8 == 0 and a.shape[1] % 8 == 0:
        acc = torch._int_mm(a, b).float()
    else:
        acc = a.float() @ b.float()
    return (acc * sa * sw.view(1, -1)).half()


def check_correct(lib: ctypes.CDLL, shapes: list[tuple[int, int, int]]) -> None:
    for M, N, K in shapes:
        a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
        b = torch.randint(-8, 8, (K, N), device="cuda", dtype=torch.int8)
        sw = torch.randn(N, device="cuda", dtype=torch.float32).abs() * 0.01 + 0.001
        sa = 0.02
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)
        err = lib.int8mma_run(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            float(sa),
            sw.data_ptr(),
            M,
            N,
            K,
        )
        torch.cuda.synchronize()
        if err != 0:
            raise RuntimeError(f"int8mma_run failed err={err} shape={(M,N,K)}")
        ref = ref_mm(a, b, sa, sw)
        max_abs = (c.float() - ref.float()).abs().max().item()
        ok = max_abs < 1e-2 or max_abs < (ref.float().abs().max().item() * 1e-3 + 1e-2)
        print(f"CHECK M={M} N={N} K={K} max_abs_diff={max_abs:.6g} ok={ok}")
        if not ok:
            diff = (c.float() - ref.float()).abs()
            idx = diff.argmax().item()
            mi, ni = divmod(idx, N)
            print("  worst at", mi, ni, "got", c[mi, ni].item(), "ref", ref[mi, ni].item())
            raise AssertionError("correctness failed")



def check_prepacked(lib: ctypes.CDLL) -> None:
    M, N, K = 256, 256, 256
    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b_kn = torch.randint(-8, 8, (K, N), device="cuda", dtype=torch.int8)
    sw = torch.ones(N, device="cuda", dtype=torch.float32) * 0.01
    sa = 0.02
    packed = b_kn.transpose(0, 1).contiguous()
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    err = lib.int8mma_run_prepacked_b(
        a.data_ptr(), packed.data_ptr(), c.data_ptr(), float(sa), sw.data_ptr(), M, N, K
    )
    torch.cuda.synchronize()
    if err != 0:
        raise RuntimeError(err)
    ref = ref_mm(a, b_kn, sa, sw)
    max_abs = (c.float() - ref.float()).abs().max().item()
    print(f"CHECK prepacked_b M={M} max_abs_diff={max_abs:.6g}")
    if max_abs >= 1e-2:
        raise AssertionError("prepacked correctness failed")


def check_cutlass_fused_scale_bias(lib: ctypes.CDLL) -> None:
    if not hasattr(lib, "int8mma_run_cutlass_64x128_prepacked_b"):
        raise RuntimeError("CUTLASS fused scale/bias entry point is missing")
    m, n, k = 256, 256, 256
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b_kn = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    packed = b_kn.transpose(0, 1).contiguous()
    activation_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    weight_scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    scale_bias = torch.stack((activation_scale * weight_scale, bias), dim=1).contiguous()
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    err = lib.int8mma_run_cutlass_64x128_prepacked_b(
        a.data_ptr(),
        packed.data_ptr(),
        c.data_ptr(),
        scale_bias.data_ptr(),
        m,
        n,
        k,
    )
    torch.cuda.synchronize()
    if err != 0:
        raise RuntimeError(f"CUTLASS fused scale/bias failed err={err}")
    reference = (
        torch._int_mm(a, b_kn).float()
        * (activation_scale * weight_scale).view(1, -1)
        + bias.view(1, -1)
    ).half()
    max_abs = (c.float() - reference.float()).abs().max().item()
    print(f"CHECK CUTLASS fused_scale_bias M={m} max_abs_diff={max_abs:.6g}")
    if max_abs >= 2e-3:
        raise AssertionError("CUTLASS fused scale/bias correctness failed")


def bench(lib: ctypes.CDLL, M: int, N: int, K: int, warmup: int = 10, iters: int = 50) -> None:
    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (K, N), device="cuda", dtype=torch.int8)
    sw = torch.ones(N, device="cuda", dtype=torch.float32) * 0.01
    sa = 0.02
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    def run_once() -> None:
        err = lib.int8mma_run(a.data_ptr(), b.data_ptr(), c.data_ptr(), float(sa), sw.data_ptr(), M, N, K)
        if err != 0:
            raise RuntimeError(err)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    tops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    print(f"BENCH M={M} N={N} K={K}  {ms:.3f} ms  {tops:.2f} TOPS (int8 theoretical ops)")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1
    major, minor = torch.cuda.get_device_capability()
    print("GPU:", torch.cuda.get_device_name(0), f"sm_{major}{minor}")
    so = build()
    lib = load_lib(so)
    print("version:", lib.int8mma_version().decode())
    print("smem_bytes:", lib.int8mma_smem_bytes())

    shapes = [
        (128, 128, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (127, 129, 96),
    ]
    check_correct(lib, shapes)
    check_prepacked(lib)
    check_cutlass_fused_scale_bias(lib)
    for M, N, K in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
        bench(lib, M, N, K)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
