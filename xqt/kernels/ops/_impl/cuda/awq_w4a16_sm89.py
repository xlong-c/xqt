"""Native SM89 AWQ W4A16 decode kernels.

This module owns only the low-level CUDA ABI. Canonical XQT weight validation,
backend prepacking and runtime-module policy live in ``xqt.kernels.ops.gemm`` and
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
from xqt.kernels.jit.utils.compile import CompileSpec, csrc_path, load_extension


_HERE = Path(__file__).resolve().parent
_BINDING_SOURCE = csrc_path("quantization", "awq_w4a16_sm89_binding.cpp")
_TVM_BINDING_SOURCE = csrc_path("quantization", "awq_w4a16_sm89_tvm_binding.cpp")
_CUDA_SOURCE = csrc_path("quantization", "awq_w4a16_sm89_kernel.cu")
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


@lru_cache(maxsize=2)
def _load_extension(backend: str | None = None) -> Any:
    selected_backend = backend or os.environ.get("XQT_JIT_BACKEND", "tvm_ffi")
    if os.environ.get("XQT_DISABLE_AWQ_W4A16_SM89", "0") == "1":
        raise XQTBackendError("native SM89 AWQ W4A16 backend is disabled")
    if not torch.cuda.is_available():
        raise XQTBackendError("native SM89 AWQ W4A16 backend requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"native AWQ W4A16 backend targets sm_89, got sm_{major}{minor}"
        )

    binding_source = (
        _TVM_BINDING_SOURCE if selected_backend == "tvm_ffi" else _BINDING_SOURCE
    )
    ext_suffix = "_tvm_v3" if selected_backend == "tvm_ffi" else "_v3"

    if selected_backend == "tvm_ffi":
        return load_extension(
            CompileSpec(
                name=f"xqt_awq_w4a16_sm89{ext_suffix}",
                sources=(binding_source, _CUDA_SOURCE),
                cxx_flags=("-O3", "-std=c++20"),
                cuda_flags=(
                    "-O3",
                    "-std=c++20",
                    "-DENABLE_BF16=1",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--generate-line-info",
                ),
                target_arch="sm_89",
                backend=selected_backend,
            ),
            verbose=False,
        )

    from torch.utils.cpp_extension import load

    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_max_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ.setdefault("MAX_JOBS", "1")
    try:
        return load(
            name=f"xqt_awq_w4a16_sm89{ext_suffix}",
            sources=[
                str(binding_source),
                str(_CUDA_SOURCE),
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
    sources_exist = all(path.is_file() for path in (_BINDING_SOURCE, _CUDA_SOURCE))
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
    output = torch.empty(
        (n // 4, k // 2), dtype=torch.int32, device=canonical_qweight.device
    )
    _load_extension().pack(canonical_qweight.contiguous(), output)
    return output


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
        output = torch.empty(
            (inputs.shape[0], qweight.shape[0] * 4),
            dtype=inputs.dtype,
            device=inputs.device,
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
    bias_exec = (
        bias.to(device=inputs.device, dtype=inputs.dtype).reshape(-1).contiguous()
    )
    if output is None:
        output = torch.empty(
            (inputs.shape[0], qweight.shape[0] * 4),
            dtype=inputs.dtype,
            device=inputs.device,
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
        decode_out = extension.decode_out

        def bound(inputs: torch.Tensor) -> torch.Tensor:
            output = torch.empty(
                (inputs.shape[0], output_features), dtype=dtype, device=device
            )
            decode_out(inputs, qweight, scales, scaled_zeros, output)
            return output

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
    decode_bias_out = extension.decode_bias_out

    def bound_bias(inputs: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (inputs.shape[0], output_features), dtype=dtype, device=device
        )
        decode_bias_out(inputs, qweight, scales, scaled_zeros, bias, output)
        return output

    return bound_bias


def awq_w4a8_decode(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scale_a: torch.Tensor | float,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one native SM89 DP4A AWQ W4A8 decode projection."""
    if inputs.ndim != 2 or inputs.dtype != torch.int8:
        raise XQTBackendError("AWQ W4A8 inputs must be INT8 rank-2")
    if not inputs.is_cuda:
        raise XQTBackendError("AWQ W4A8 inputs must be CUDA")
    if output is None:
        output = torch.empty(
            (inputs.shape[0], qweight.shape[0] * 4),
            dtype=scales.dtype,
            device=inputs.device,
        )
    ext = _load_extension()
    if isinstance(scale_a, torch.Tensor):
        ext.decode_w4a8_out(
            inputs.contiguous(),
            qweight.contiguous(),
            scales.contiguous(),
            scale_a.contiguous(),
            output,
        )
    else:
        ext.decode_w4a8_scalar_out(
            inputs.contiguous(),
            qweight.contiguous(),
            scales.contiguous(),
            float(scale_a),
            output,
        )
    return output


def awq_w4a8_decode_bias(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scale_a: torch.Tensor | float,
    residual: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run native AWQ W4A8 decode with fused residual epilogue."""
    if inputs.ndim != 2 or inputs.dtype != torch.int8:
        raise XQTBackendError("AWQ W4A8 inputs must be INT8 rank-2")
    if not inputs.is_cuda:
        raise XQTBackendError("AWQ W4A8 inputs must be CUDA")
    if output is None:
        output = torch.empty(
            (inputs.shape[0], qweight.shape[0] * 4),
            dtype=scales.dtype,
            device=inputs.device,
        )
    ext = _load_extension()
    if isinstance(scale_a, torch.Tensor):
        ext.decode_w4a8_bias_out(
            inputs.contiguous(),
            qweight.contiguous(),
            scales.contiguous(),
            scale_a.contiguous(),
            residual.contiguous(),
            output,
        )
    else:
        ext.decode_w4a8_scalar_bias_out(
            inputs.contiguous(),
            qweight.contiguous(),
            scales.contiguous(),
            float(scale_a),
            residual.contiguous(),
            output,
        )
    return output


def bind_awq_w4a8_decode(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    rows: int,
    input_features: int,
    output_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor]:
    """Bind immutable AWQ execution tensors for W4A8 decode."""
    rows = int(rows)
    input_features = int(input_features)
    output_features = int(output_features)
    device = torch.device(device)
    if not 1 <= rows <= 8:
        raise XQTBackendError("bound AWQ W4A8 decode requires 1 <= rows <= 8")
    if dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError(
            "bound AWQ W4A8 decode requires FP16 or BF16 scales/output"
        )
    if device.type != "cuda":
        raise XQTBackendError("bound AWQ W4A8 decode requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if input_features % 64 != 0 or output_features % 8 != 0:
        raise XQTBackendError(
            "bound AWQ W4A8 decode requires K % 64 == 0 and N % 8 == 0"
        )

    extension = _load_extension()
    decode_w4a8_out = extension.decode_w4a8_out
    decode_w4a8_scalar_out = extension.decode_w4a8_scalar_out

    def bound(inputs: torch.Tensor, scale_a: torch.Tensor | float) -> torch.Tensor:
        output = torch.empty(
            (inputs.shape[0], output_features), dtype=dtype, device=device
        )
        if isinstance(scale_a, torch.Tensor):
            decode_w4a8_out(inputs, qweight, scales, scale_a, output)
        else:
            decode_w4a8_scalar_out(inputs, qweight, scales, float(scale_a), output)
        return output

    return bound


def bind_awq_w4a8_decode_bias(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    rows: int,
    input_features: int,
    output_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor | float, torch.Tensor], None]:
    """Bind immutable AWQ execution tensors for W4A8 decode with fused residual add."""
    rows = int(rows)
    input_features = int(input_features)
    output_features = int(output_features)
    device = torch.device(device)
    if not 1 <= rows <= 8:
        raise XQTBackendError("bound AWQ W4A8 residual requires 1 <= rows <= 8")
    if dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError(
            "bound AWQ W4A8 residual requires FP16 or BF16 scales/output"
        )
    if device.type != "cuda":
        raise XQTBackendError("bound AWQ W4A8 residual requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if input_features % 64 != 0 or output_features % 8 != 0:
        raise XQTBackendError(
            "bound AWQ W4A8 residual requires K % 64 == 0 and N % 8 == 0"
        )

    extension = _load_extension()
    decode_bias_out = extension.decode_w4a8_bias_out
    decode_scalar_bias_out = extension.decode_w4a8_scalar_bias_out

    def bound_bias(
        inputs: torch.Tensor, scale_a: torch.Tensor | float, residual: torch.Tensor
    ) -> None:
        if isinstance(scale_a, torch.Tensor):
            decode_bias_out(
                inputs, qweight, scales, scale_a, residual.reshape(-1), residual
            )
        else:
            decode_scalar_bias_out(
                inputs, qweight, scales, float(scale_a), residual.reshape(-1), residual
            )

    return bound_bias


def native_awq_w4a16_version() -> str:
    """Return the compiled extension ABI version."""

    return str(_load_extension().version())


__all__ = [
    "awq_w4a16_decode",
    "awq_w4a16_decode_bias",
    "awq_w4a8_decode",
    "awq_w4a8_decode_bias",
    "bind_awq_w4a16_decode",
    "bind_awq_w4a8_decode",
    "bind_awq_w4a8_decode_bias",
    "native_awq_w4a16_available",
    "native_awq_w4a16_version",
    "pack_awq_w4a16_interleaved",
]
