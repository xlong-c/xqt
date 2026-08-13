"""Native SM89 AWQ W4A16 decode kernels.

This module owns only the low-level CUDA ABI. Canonical XQT weight validation,
backend prepacking and runtime-module policy live in ``xqt.gemm`` and
``xqt.runtime`` respectively.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from xqt.core.errors import XQTBackendError


_HERE = Path(__file__).resolve().parent
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    if os.environ.get("XQT_DISABLE_AWQ_W4A16_SM89", "0") == "1":
        raise XQTBackendError("native SM89 AWQ W4A16 backend is disabled")
    if not torch.cuda.is_available():
        raise XQTBackendError("native SM89 AWQ W4A16 backend requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"native AWQ W4A16 backend targets sm_89, got sm_{major}{minor}"
        )

    from torch.utils.cpp_extension import load

    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_max_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ.setdefault("MAX_JOBS", "1")
    try:
        return load(
            name="xqt_awq_w4a16_sm89_v2",
            sources=[
                str(_HERE / "awq_w4a16_sm89_binding.cpp"),
                str(_HERE / "awq_w4a16_sm89_kernel.cu"),
            ],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-DENABLE_BF16=1",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--generate-line-info",
            ],
            with_cuda=True,
            verbose=False,
        )
    except Exception as exc:
        raise XQTBackendError(
            f"failed to build native SM89 AWQ W4A16 extension: {exc}"
        ) from exc
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list
        if old_max_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = old_max_jobs


def native_awq_w4a16_available(*, build: bool = False) -> bool:
    """Return whether the SM89 AWQ decode backend can run on this device."""

    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (8, 9):
        return False
    if os.environ.get("XQT_DISABLE_AWQ_W4A16_SM89", "0") == "1":
        return False
    sources_exist = all(
        (_HERE / name).is_file()
        for name in ("awq_w4a16_sm89_binding.cpp", "awq_w4a16_sm89_kernel.cu")
    )
    if not sources_exist:
        return False
    if not build:
        return True
    try:
        _load_extension()
    except XQTBackendError:
        return False
    return True


def pack_awq_w4a16_interleaved(canonical_qweight: torch.Tensor) -> torch.Tensor:
    """Pack canonical low-nibble-first ``uint8 [N,K/2]`` for SM89 GEMV."""

    if canonical_qweight.ndim != 2 or canonical_qweight.dtype != torch.uint8:
        raise XQTBackendError(
            "AWQ W4A16 prepack expects canonical uint8 qweight [N,K/2]"
        )
    if not canonical_qweight.is_cuda:
        raise XQTBackendError("AWQ W4A16 prepack requires a CUDA qweight")
    n = int(canonical_qweight.shape[0])
    k = int(canonical_qweight.shape[1]) * 2
    if n % 8 != 0 or k % 64 != 0:
        raise XQTBackendError("AWQ W4A16 prepack requires N % 8 == 0 and K % 64 == 0")
    return _load_extension().pack(canonical_qweight.contiguous())


def awq_w4a16_decode(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one native ``M<=8`` AWQ W4A16 decode projection."""

    if inputs.ndim != 2 or inputs.dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError("AWQ W4A16 inputs must be FP16/BF16 rank-2")
    if not inputs.is_cuda:
        raise XQTBackendError("AWQ W4A16 inputs must be CUDA")
    if output is None:
        return _load_extension().decode(
            inputs.contiguous(),
            qweight.contiguous(),
            scales.contiguous(),
            scaled_zeros.contiguous(),
        )
    _load_extension().decode_out(
        inputs.contiguous(),
        qweight.contiguous(),
        scales.contiguous(),
        scaled_zeros.contiguous(),
        output,
    )
    return output


def awq_w4a16_decode_bias(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    bias: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run native AWQ decode with a fused output bias."""

    if inputs.ndim != 2 or inputs.dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError("AWQ W4A16 inputs must be FP16/BF16 rank-2")
    if not inputs.is_cuda:
        raise XQTBackendError("AWQ W4A16 inputs must be CUDA")
    bias_exec = bias.to(device=inputs.device, dtype=inputs.dtype).reshape(-1).contiguous()
    if output is None:
        return _load_extension().decode_bias(
            inputs.contiguous(),
            qweight.contiguous(),
            scales.contiguous(),
            scaled_zeros.contiguous(),
            bias_exec,
        )
    _load_extension().decode_bias_out(
        inputs.contiguous(),
        qweight.contiguous(),
        scales.contiguous(),
        scaled_zeros.contiguous(),
        bias_exec,
        output,
    )
    return output


def bind_awq_w4a16_decode(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    *,
    rows: int,
    input_features: int,
    output_features: int,
    dtype: torch.dtype,
    device: torch.device,
    bias: torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind immutable AWQ execution tensors to the compiled CUDA entry.

    Static tensors are validated once here. The returned callable deliberately
    assumes that its dynamic input already matches ``[rows,input_features]``;
    runtime modules retain that public contract check without rebuilding GEMM
    specifications or packed-weight objects on every inference invocation.
    """

    rows = int(rows)
    input_features = int(input_features)
    output_features = int(output_features)
    device = torch.device(device)
    if not 1 <= rows <= 8:
        raise XQTBackendError("bound AWQ W4A16 decode requires 1 <= rows <= 8")
    if dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError("bound AWQ W4A16 decode requires FP16 or BF16")
    if device.type != "cuda":
        raise XQTBackendError("bound AWQ W4A16 decode requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if input_features % 64 != 0 or output_features % 8 != 0:
        raise XQTBackendError(
            "bound AWQ W4A16 decode requires K % 64 == 0 and N % 8 == 0"
        )
    expected_group_output = (input_features // 64, output_features)
    static_tensors = (
        ("qweight", qweight, torch.int32, (output_features // 4, input_features // 2)),
        ("scales", scales, dtype, expected_group_output),
        ("scaled_zeros", scaled_zeros, dtype, expected_group_output),
    )
    for name, tensor, expected_dtype, expected_shape in static_tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != device
            or tensor.dtype != expected_dtype
            or tuple(tensor.shape) != expected_shape
            or not tensor.is_contiguous()
        ):
            raise XQTBackendError(
                f"bound AWQ W4A16 {name} does not match the static execution contract"
            )

    extension = _load_extension()
    if bias is None:
        decode = extension.decode

        def bound(inputs: torch.Tensor) -> torch.Tensor:
            return decode(inputs, qweight, scales, scaled_zeros)

        return bound

    if (
        bias.device != device
        or bias.dtype != dtype
        or tuple(bias.shape) != (output_features,)
        or not bias.is_contiguous()
    ):
        raise XQTBackendError(
            "bound AWQ W4A16 bias does not match the static execution contract"
        )
    decode_bias = extension.decode_bias

    def bound_bias(inputs: torch.Tensor) -> torch.Tensor:
        return decode_bias(inputs, qweight, scales, scaled_zeros, bias)

    return bound_bias


def native_awq_w4a16_version() -> str:
    """Return the compiled extension ABI version."""

    return str(_load_extension().version())


__all__ = [
    "awq_w4a16_decode",
    "awq_w4a16_decode_bias",
    "bind_awq_w4a16_decode",
    "native_awq_w4a16_available",
    "native_awq_w4a16_version",
    "pack_awq_w4a16_interleaved",
]
