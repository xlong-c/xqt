"""ctypes binding for Ada sm_89 INT8 MMA (math, prepacked B, fused static act)."""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import Any

import torch

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.kernels.prepack import INT8_SM89_B_NK, prepack_weight

_SO_NAME = "int8mma_sm89.so"
_ROOT = Path(__file__).resolve().parent
_DEFAULT_SO = _ROOT / "build" / _SO_NAME


@functools.lru_cache(maxsize=1)
def _load_lib(so_path: str | None = None) -> Any:
    path = Path(so_path) if so_path else _DEFAULT_SO
    if not path.is_file():
        raise XQTBackendError(
            f"ptx_sm89 INT8 MMA library not found at {path}. "
            f"Build with: python {_ROOT / 'build_and_test_int8mma.py'}"
        )
    lib = ctypes.CDLL(str(path))
    run_names = (
        "int8mma_run",
        "int8mma_run_prepacked_b",
        "int8mma_run_fused_static_prepacked_b",
    )
    if hasattr(lib, "int8mma_quantize_static_half"):
        lib.int8mma_quantize_static_half.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int
        ]
        lib.int8mma_quantize_static_half.restype = ctypes.c_int
    for name in run_names:
        if not hasattr(lib, name):
            continue
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_int
    lib.int8mma_version.argtypes = []
    lib.int8mma_version.restype = ctypes.c_char_p
    lib.int8mma_smem_bytes.argtypes = []
    lib.int8mma_smem_bytes.restype = ctypes.c_int
    return lib


def int8mma_available() -> bool:
    try:
        _load_lib()
        return True
    except XQTBackendError:
        return False


def int8mma_version() -> str:
    return _load_lib().int8mma_version().decode()


def prepack_qweight_t_for_ptx_sm89(qweight_t: torch.Tensor) -> torch.Tensor:
    result = prepack_weight(qweight_t, INT8_SM89_B_NK)
    return result.packed


def int8_linear_ptx_sm89(
    qactivation: torch.Tensor,
    qweight_t: torch.Tensor,
    activation_scale: torch.Tensor | float,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    output_dtype: torch.dtype = torch.float16,
    prepacked_b: torch.Tensor | None = None,
) -> torch.Tensor:
    if not qactivation.is_cuda or not qweight_t.is_cuda:
        raise XQTBackendError("ptx_sm89 INT8 MMA requires CUDA tensors")
    if qactivation.dtype != torch.int8 or qweight_t.dtype != torch.int8:
        raise XQTBackendError("ptx_sm89 INT8 MMA expects int8 inputs")
    if qactivation.ndim != 2 or qweight_t.ndim != 2:
        raise XQTBackendError("ptx_sm89 INT8 MMA expects 2D tensors")
    if qactivation.shape[1] != qweight_t.shape[0]:
        raise XQTBackendError("ptx_sm89 INT8 MMA requires A.shape[1] == B.shape[0]")

    m, k = int(qactivation.shape[0]), int(qactivation.shape[1])
    n = int(qweight_t.shape[1])
    sa = float(activation_scale)
    sw = weight_scale.detach().to(device=qactivation.device, dtype=torch.float32).contiguous()
    if sw.numel() != n:
        raise XQTBackendError("weight_scale must have N elements")

    c_half = torch.empty(m, n, device=qactivation.device, dtype=torch.float16)
    a = qactivation.contiguous()
    lib = _load_lib()
    if prepacked_b is not None:
        if prepacked_b.shape != (n, k) or prepacked_b.dtype != torch.int8:
            raise XQTBackendError(
                f"prepacked_b must be int8 [N,K]=[{n},{k}], got {prepacked_b.dtype} {tuple(prepacked_b.shape)}"
            )
        b = prepacked_b.contiguous()
        err = lib.int8mma_run_prepacked_b(
            a.data_ptr(), b.data_ptr(), c_half.data_ptr(), sa, sw.data_ptr(), m, n, k
        )
    else:
        b = qweight_t.contiguous()
        err = lib.int8mma_run(
            a.data_ptr(), b.data_ptr(), c_half.data_ptr(), sa, sw.data_ptr(), m, n, k
        )
    if err != 0:
        raise XQTBackendError(f"int8mma_run failed with cuda error code {err}")

    out = c_half if output_dtype == torch.float16 else c_half.to(output_dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


def int8_linear_fused_static_ptx_sm89(
    activations: torch.Tensor,
    qweight_t: torch.Tensor,
    activation_scale: torch.Tensor | float,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    output_dtype: torch.dtype = torch.float16,
    prepacked_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Static act quant (vector CUDA) + prepacked-B INT8 MMA. activations: half [M,K]."""

    if not activations.is_cuda or not qweight_t.is_cuda:
        raise XQTBackendError("ptx_sm89 fused path requires CUDA tensors")
    if activations.dtype != torch.float16:
        raise XQTBackendError("ptx_sm89 fused static path expects float16 activations")
    if qweight_t.dtype != torch.int8 or activations.ndim != 2 or qweight_t.ndim != 2:
        raise XQTBackendError("ptx_sm89 fused path expects half[M,K] x int8[K,N]")
    if activations.shape[1] != qweight_t.shape[0]:
        raise XQTBackendError("ptx_sm89 fused path requires A.shape[1] == B.shape[0]")

    m, k = int(activations.shape[0]), int(activations.shape[1])
    n = int(qweight_t.shape[1])
    sa = float(activation_scale)
    if sa <= 0.0:
        raise XQTBackendError("activation_scale must be positive")
    sw = weight_scale.detach().to(device=activations.device, dtype=torch.float32).contiguous()
    if sw.numel() != n:
        raise XQTBackendError("weight_scale must have N elements")

    if prepacked_b is None:
        prepacked_b = prepack_qweight_t_for_ptx_sm89(qweight_t)
    if prepacked_b.shape != (n, k) or prepacked_b.dtype != torch.int8:
        raise XQTBackendError(
            f"prepacked_b must be int8 [N,K]=[{n},{k}], got {prepacked_b.dtype} {tuple(prepacked_b.shape)}"
        )

    tile_m = 128
    m_pad = ((m + tile_m - 1) // tile_m) * tile_m
    x = activations.contiguous()
    if m_pad != m:
        x_pad = torch.zeros(m_pad, k, device=x.device, dtype=torch.float16)
        x_pad[:m].copy_(x)
        x = x_pad
    q = torch.empty(m_pad, k, device=activations.device, dtype=torch.int8)
    lib = _load_lib()
    if hasattr(lib, "int8mma_quantize_static_half"):
        err_q = lib.int8mma_quantize_static_half(x.data_ptr(), q.data_ptr(), sa, m_pad, k)
        if err_q != 0:
            raise XQTBackendError(f"quantize failed cuda error {err_q}")
    else:
        q = torch.round(x.float() / sa).clamp(-127, 127).to(torch.int8).contiguous()

    c_half = torch.empty(m_pad, n, device=activations.device, dtype=torch.float16)
    b = prepacked_b.contiguous()
    err = lib.int8mma_run_prepacked_b(
        q.data_ptr(), b.data_ptr(), c_half.data_ptr(), sa, sw.data_ptr(), m_pad, n, k
    )
    if err != 0:
        raise XQTBackendError(f"int8mma prepacked run failed with cuda error code {err}")

    out = c_half[:m]
    out = out if output_dtype == torch.float16 else out.to(output_dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


__all__ = [
    "int8mma_available",
    "int8mma_version",
    "int8_linear_ptx_sm89",
    "int8_linear_fused_static_ptx_sm89",
    "prepack_qweight_t_for_ptx_sm89",
]
