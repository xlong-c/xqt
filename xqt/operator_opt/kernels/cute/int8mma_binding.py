"""ctypes bindings for Ada sm_89 INT8 MMA CUDA kernels."""

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
_CUBLASLT_WORKSPACE_BYTES = 32 * 1024 * 1024
_CUBLASLT_WORKSPACES: dict[tuple[str, int | None], torch.Tensor] = {}


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
    cutlass_names = (
        "int8mma_run_cutlass_prepacked_b",
        "int8mma_run_cutlass_64x128_prepacked_b",
    )
    for name in cutlass_names:
        if not hasattr(lib, name):
            continue
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_int
    if hasattr(lib, "int8mma_run_cublaslt_i32"):
        lib.int8mma_run_cublaslt_i32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.int8mma_run_cublaslt_i32.restype = ctypes.c_int
    if hasattr(lib, "int8mma_dequant_i32"):
        lib.int8mma_dequant_i32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.int8mma_dequant_i32.restype = ctypes.c_int
    if hasattr(lib, "int8mma_run_cutlass_i32_64x128_prepacked_b"):
        lib.int8mma_run_cutlass_i32_64x128_prepacked_b.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.int8mma_run_cutlass_i32_64x128_prepacked_b.restype = ctypes.c_int
    if hasattr(lib, "int8mma_run_cutlass_i32_128x256_prepacked_b"):
        lib.int8mma_run_cutlass_i32_128x256_prepacked_b.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.int8mma_run_cutlass_i32_128x256_prepacked_b.restype = ctypes.c_int
    for name in (
        "int8mma_run_cutlass_scale_64x128_prepacked_b_stream",
        "int8mma_run_cutlass_scale_128x256_prepacked_b_stream",
        "int8mma_run_cutlass_scale_bf16_64x128_prepacked_b_stream",
        "int8mma_run_cutlass_scale_bf16_128x256_prepacked_b_stream",
    ):
        if not hasattr(lib, name):
            continue
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
    for name in (
        "int8mma_run_cutlass_visitor_bf16_64x128_prepacked_b_stream",
        "int8mma_run_cutlass_visitor_bf16_128x256_prepacked_b_stream",
        "int8mma_run_cutlass_visitor_half_64x128_prepacked_b_stream",
        "int8mma_run_cutlass_visitor_half_128x256_prepacked_b_stream",
    ):
        if not hasattr(lib, name):
            continue
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
    for name in (
        "int8mma_run_cutlass_visitor_convrot_bf16_prepacked_b_stream",
        "int8mma_run_cutlass_visitor_convrot_half_prepacked_b_stream",
    ):
        if not hasattr(lib, name):
            continue
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
    if hasattr(lib, "int8mma_convrot_quantize_rows"):
        lib.int8mma_convrot_quantize_rows.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.int8mma_convrot_quantize_rows.restype = ctypes.c_int
    if hasattr(lib, "int8mma_dequant_i32_rowwise"):
        lib.int8mma_dequant_i32_rowwise.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.int8mma_dequant_i32_rowwise.restype = ctypes.c_int
    if hasattr(lib, "int8mma_apply_half_rowwise_scale_bias"):
        lib.int8mma_apply_half_rowwise_scale_bias.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.int8mma_apply_half_rowwise_scale_bias.restype = ctypes.c_int
    if hasattr(lib, "int8mma_apply_bf16_rowwise_scale_bias"):
        lib.int8mma_apply_bf16_rowwise_scale_bias.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.int8mma_apply_bf16_rowwise_scale_bias.restype = ctypes.c_int
    lib.int8mma_version.argtypes = []
    lib.int8mma_version.restype = ctypes.c_char_p
    lib.int8mma_smem_bytes.argtypes = []
    lib.int8mma_smem_bytes.restype = ctypes.c_int
    gemv_names = (
        "int8_gemv_m1_run",
        "int8_gemv_m1_run_prepacked_b",
        "int8_gemv_m1_run_fused_static",
        "int8_gemv_m1_run_fused_static_prepacked_b",
    )
    for name in gemv_names:
        if not hasattr(lib, name):
            continue
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_int
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


def int8_linear_cutlass_sm89(
    qactivation: torch.Tensor,
    qweight_t: torch.Tensor,
    activation_scale: torch.Tensor | float,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    output_dtype: torch.dtype = torch.float16,
    prepacked_b: torch.Tensor | None = None,
    scale_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused-per-channel CUTLASS W8A8 kernel on Ada ``sm_89``.

    ``prepacked_b`` is the cached contiguous ``[N, K]`` transpose of
    ``qweight_t``. ``scale_bias`` is an optional cached ``float32[N, 2]``
    tensor where every row stores ``activation_scale * weight_scale[n]`` and
    ``bias[n]``. Its layout matches the CUDA epilogue's 8-byte vector type.
    """

    if not qactivation.is_cuda or not qweight_t.is_cuda:
        raise XQTBackendError("cuda_sm89 W8A8 path requires CUDA tensors")
    if qactivation.dtype != torch.int8 or qweight_t.dtype != torch.int8:
        raise XQTBackendError("cuda_sm89 W8A8 path expects int8 inputs")
    if qactivation.ndim != 2 or qweight_t.ndim != 2:
        raise XQTBackendError("cuda_sm89 W8A8 path expects 2D tensors")
    if qactivation.shape[1] != qweight_t.shape[0]:
        raise XQTBackendError("cuda_sm89 W8A8 path requires A.shape[1] == B.shape[0]")
    if output_dtype != torch.float16:
        raise XQTBackendError("cuda_sm89 W8A8 fast path supports float16 output only")
    major, minor = torch.cuda.get_device_capability(qactivation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"cuda_sm89 W8A8 fast path is tuned for sm_89, got sm_{major}{minor}"
        )

    m, k = int(qactivation.shape[0]), int(qactivation.shape[1])
    n = int(qweight_t.shape[1])
    if k % 32 != 0 or n % 8 != 0:
        raise XQTBackendError("cuda_sm89 W8A8 requires K % 32 == 0 and N % 8 == 0")
    if prepacked_b is None:
        prepacked_b = prepack_qweight_t_for_ptx_sm89(qweight_t)
    if (
        prepacked_b.shape != (n, k)
        or prepacked_b.dtype != torch.int8
        or not prepacked_b.is_cuda
        or prepacked_b.device != qactivation.device
    ):
        raise XQTBackendError(
            "prepacked_b must be CUDA int8 [N,K]="
            f"[{n},{k}], got {prepacked_b.dtype} {tuple(prepacked_b.shape)}"
        )

    if scale_bias is None:
        activation = _device_activation_scale(activation_scale, qactivation.device)
        weight = weight_scale.detach().to(
            device=qactivation.device,
            dtype=torch.float32,
        ).reshape(-1)
        if weight.numel() != n:
            raise XQTBackendError("weight_scale must have N elements")
        bias_value = (
            torch.zeros_like(weight)
            if bias is None
            else bias.detach()
            .to(device=qactivation.device, dtype=torch.float32)
            .reshape(-1)
        )
        if bias_value.numel() != n:
            raise XQTBackendError("bias must have N elements")
        scale_bias = torch.stack((activation * weight, bias_value), dim=1).contiguous()
    elif (
        not scale_bias.is_cuda
        or scale_bias.device != qactivation.device
        or scale_bias.dtype != torch.float32
        or scale_bias.shape != (n, 2)
        or not scale_bias.is_contiguous()
    ):
        raise XQTBackendError(
            "scale_bias must be a contiguous CUDA float32 [N,2] tensor on the input device"
        )

    lib = _load_lib()
    if not hasattr(lib, "int8mma_run_cutlass_64x128_prepacked_b"):
        raise XQTBackendError(
            "CUTLASS W8A8 symbol is unavailable; rebuild int8mma_sm89.so"
        )
    output = torch.empty((m, n), device=qactivation.device, dtype=torch.float16)
    err = lib.int8mma_run_cutlass_64x128_prepacked_b(
        qactivation.contiguous().data_ptr(),
        prepacked_b.data_ptr(),
        output.data_ptr(),
        scale_bias.data_ptr(),
        m,
        n,
        k,
    )
    if err != 0:
        raise XQTBackendError(f"CUTLASS cuda_sm89 W8A8 GEMM failed with cuda error {err}")
    return output


def convrot_cutlass_w8a8_sm89(
    inputs: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    rotated_input_features: int,
    rot_size: int,
    output_dtype: torch.dtype,
    prepacked_b: torch.Tensor | None = None,
    quantized_activation: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    accumulator: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    weight_scale_buffer: torch.Tensor | None = None,
    bias_buffer: torch.Tensor | None = None,
    scaled_output: torch.Tensor | None = None,
    scale_bias_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CUDA ConvRot quantization plus a CUTLASS INT8 GEMM on sm_89."""

    if not inputs.is_cuda or inputs.ndim != 2:
        raise XQTBackendError("CUTLASS ConvRot expects a two-dimensional CUDA input")
    if inputs.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("CUTLASS ConvRot expects FP16 or BF16 inputs")
    if qweight_t.dtype != torch.int8 or not qweight_t.is_cuda:
        raise XQTBackendError("CUTLASS ConvRot expects CUDA int8 qweight_t")
    if qweight_t.ndim != 2 or qweight_t.shape[0] != int(rotated_input_features):
        raise XQTBackendError("CUTLASS ConvRot qweight shape does not match rotated K")
    major, minor = torch.cuda.get_device_capability(inputs.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"CUTLASS ConvRot is tuned for sm_89, got sm_{major}{minor}"
        )
    if output_dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("CUTLASS ConvRot output must be FP16 or BF16")

    m = int(inputs.shape[0])
    logical_k = int(inputs.shape[1])
    rotated_k = int(rotated_input_features)
    n = int(qweight_t.shape[1])
    padded_m = ((m + 255) // 256) * 256
    padded_k = int(qweight_t.shape[0])
    if rotated_k % 256 != 0 or padded_k % 256 != 0:
        raise XQTBackendError("CUTLASS ConvRot requires K aligned to 256")
    if logical_k > rotated_k or n % 8 != 0:
        raise XQTBackendError("CUTLASS ConvRot shape is not Tensor Core aligned")

    if quantized_activation is None:
        quantized_activation = torch.empty(
            (padded_m, padded_k),
            device=inputs.device,
            dtype=torch.int8,
        )
    if activation_scales is None:
        activation_scales = torch.empty(
            padded_m,
            device=inputs.device,
            dtype=torch.float32,
        )
    if (
        quantized_activation.shape != (padded_m, padded_k)
        or quantized_activation.dtype != torch.int8
        or quantized_activation.device != inputs.device
    ):
        raise XQTBackendError("invalid CUTLASS ConvRot quantized activation workspace")
    if (
        activation_scales.shape != (padded_m,)
        or activation_scales.dtype != torch.float32
        or activation_scales.device != inputs.device
    ):
        raise XQTBackendError("invalid CUTLASS ConvRot activation scale workspace")

    lib = _load_lib()
    required = (
        "int8mma_convrot_quantize_rows",
        "int8mma_run_cutlass_visitor_convrot_bf16_prepacked_b_stream"
        if output_dtype == torch.bfloat16
        else "int8mma_run_cutlass_visitor_convrot_half_prepacked_b_stream",
    )
    if not all(hasattr(lib, name) for name in required):
        raise XQTBackendError(
            "CUTLASS ConvRot symbols are unavailable; rebuild int8mma_sm89.so"
        )
    if prepacked_b is None:
        prepacked_b = prepack_qweight_t_for_ptx_sm89(qweight_t)
    if (
        prepacked_b.shape != (n, padded_k)
        or prepacked_b.dtype != torch.int8
        or prepacked_b.device != inputs.device
    ):
        raise XQTBackendError("invalid CUTLASS ConvRot prepacked B workspace")

    stream = torch.cuda.current_stream(inputs.device).cuda_stream
    if output is None:
        output = torch.empty(
            (padded_m, n),
            device=inputs.device,
            dtype=output_dtype,
        )
    if (
        output.shape != (padded_m, n)
        or output.dtype != output_dtype
        or output.device != inputs.device
        or not output.is_contiguous()
    ):
        raise XQTBackendError("invalid CUTLASS ConvRot output workspace")
    visitor_weight_scale = (
        weight_scale
        if weight_scale_buffer is None
        else weight_scale_buffer
    ).detach().to(
        device=inputs.device,
        dtype=torch.float32,
    ).reshape(-1).contiguous()
    if visitor_weight_scale.numel() != n:
        raise XQTBackendError(
            "weight_scale must contain one value per output channel"
        )
    visitor_bias = bias_buffer
    if visitor_bias is None:
        visitor_bias = (
            torch.zeros(
                n,
                device=inputs.device,
                dtype=torch.float32,
            )
            if bias is None
            else bias.detach()
        )
    visitor_bias = visitor_bias.to(
        device=inputs.device,
        dtype=torch.float32,
    ).reshape(-1).contiguous()
    if visitor_bias.numel() != n:
        raise XQTBackendError(
            "bias must contain one value per output channel"
        )
    input_tensor = inputs if inputs.is_contiguous() else inputs.contiguous()
    visitor_symbol = (
        "int8mma_run_cutlass_visitor_convrot_bf16_prepacked_b_stream"
        if output_dtype == torch.bfloat16
        else "int8mma_run_cutlass_visitor_convrot_half_prepacked_b_stream"
    )
    err = getattr(lib, visitor_symbol)(
        input_tensor.data_ptr(),
        quantized_activation.data_ptr(),
        activation_scales.data_ptr(),
        prepacked_b.data_ptr(),
        output.data_ptr(),
        visitor_weight_scale.data_ptr(),
        visitor_bias.data_ptr(),
        m,
        logical_k,
        rotated_k,
        padded_m,
        padded_k,
        n,
        stream,
    )
    if err != 0:
        raise XQTBackendError(
            f"CUTLASS ConvRot visitor GEMM failed with CUDA error {err}"
        )
    return output[:m], quantized_activation, activation_scales

    if scaled_output is None:
        scaled_output = torch.empty(
            (padded_m, n),
            device=inputs.device,
            dtype=output_dtype,
        )
    if (
        scaled_output.shape != (padded_m, n)
        or scaled_output.dtype != output_dtype
        or scaled_output.device != inputs.device
        or not scaled_output.is_contiguous()
    ):
        raise XQTBackendError("invalid CUTLASS ConvRot scaled output workspace")

    wscale = (
        weight_scale
        if weight_scale_buffer is None
        else weight_scale_buffer
    )
    wscale = wscale.detach().to(
        device=inputs.device,
        dtype=torch.float32,
    ).reshape(-1).contiguous()
    if wscale.numel() != n:
        raise XQTBackendError("weight_scale must contain one value per output channel")
    if scale_bias_buffer is None:
        scale_bias_buffer = torch.stack(
            (wscale, torch.zeros_like(wscale)),
            dim=1,
        ).contiguous()
    if (
        scale_bias_buffer.shape != (n, 2)
        or scale_bias_buffer.dtype != torch.float32
        or scale_bias_buffer.device != inputs.device
        or not scale_bias_buffer.is_contiguous()
    ):
        raise XQTBackendError("invalid CUTLASS ConvRot scale/bias workspace")

    # Klein's transformer is dominated by wide projections.  On Ada, the
    # 128x256 tile wins once N has enough columns to fill the larger epilogue;
    # keep 64x128 for narrow heads and the final 128-channel projection.
    gemm_symbol = (
        "int8mma_run_cutlass_scale_bf16_128x256_prepacked_b_stream"
        if output_dtype == torch.bfloat16 and n >= 512
        else "int8mma_run_cutlass_scale_bf16_64x128_prepacked_b_stream"
        if output_dtype == torch.bfloat16
        else "int8mma_run_cutlass_scale_128x256_prepacked_b_stream"
        if n >= 512
        else "int8mma_run_cutlass_scale_64x128_prepacked_b_stream"
    )
    err = getattr(lib, gemm_symbol)(
        quantized_activation.data_ptr(),
        prepacked_b.data_ptr(),
        scaled_output.data_ptr(),
        scale_bias_buffer.data_ptr(),
        padded_m,
        n,
        padded_k,
        stream,
    )
    if err != 0:
        raise XQTBackendError(
            f"CUTLASS ConvRot GEMM failed with CUDA error {err}"
        )

    bias_value = bias_buffer
    if bias_value is None:
        bias_value = (
            torch.zeros(n, device=inputs.device, dtype=torch.float32)
            if bias is None
            else bias.detach()
        )
    bias_value = bias_value.to(
        device=inputs.device,
        dtype=torch.float32,
    ).reshape(-1).contiguous()
    if bias_value.numel() != n:
        raise XQTBackendError("bias must contain one value per output channel")

    if output is None:
        output = torch.empty(
            (padded_m, n),
            device=inputs.device,
            dtype=output_dtype,
        )
    if (
        output.shape != (padded_m, n)
        or output.dtype != output_dtype
        or output.device != inputs.device
        or not output.is_contiguous()
    ):
        raise XQTBackendError("invalid CUTLASS ConvRot output workspace")
    output_kind = 0 if output_dtype == torch.bfloat16 else 1
    apply_symbol = (
        "int8mma_apply_bf16_rowwise_scale_bias"
        if output_dtype == torch.bfloat16
        else "int8mma_apply_half_rowwise_scale_bias"
    )
    err = getattr(lib, apply_symbol)(
        scaled_output.data_ptr(),
        output.data_ptr(),
        activation_scales.data_ptr(),
        bias_value.data_ptr(),
        padded_m,
        n,
        output_kind,
        stream,
    )
    if err != 0:
        raise XQTBackendError(
            f"CUTLASS ConvRot epilogue failed with CUDA error {err}"
        )
    return output[:m], quantized_activation, activation_scales


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


def _device_activation_scale(
    activation_scale: torch.Tensor | float,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(activation_scale, torch.Tensor):
        scale = activation_scale.detach().to(device=device, dtype=torch.float32).reshape(1)
    else:
        value = float(activation_scale)
        if value <= 0.0:
            raise XQTBackendError("activation_scale must be positive")
        scale = torch.tensor([value], device=device, dtype=torch.float32)
    return scale.contiguous()


def _cublaslt_workspace(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    workspace = _CUBLASLT_WORKSPACES.get(key)
    if workspace is None or workspace.device != device:
        workspace = torch.empty(
            _CUBLASLT_WORKSPACE_BYTES,
            device=device,
            dtype=torch.uint8,
        )
        _CUBLASLT_WORKSPACES[key] = workspace
    return workspace


def int8_linear_cublaslt_sm89(
    qactivation: torch.Tensor,
    qweight_t: torch.Tensor,
    activation_scale: torch.Tensor | float,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run cuBLASLt INT8 GEMM plus a CUDA fp16 scale/bias epilogue on sm_89."""

    if not qactivation.is_cuda or not qweight_t.is_cuda:
        raise XQTBackendError("cuBLASLt W8A8 path requires CUDA tensors")
    if qactivation.dtype != torch.int8 or qweight_t.dtype != torch.int8:
        raise XQTBackendError("cuBLASLt W8A8 path expects int8 inputs")
    if qactivation.ndim != 2 or qweight_t.ndim != 2:
        raise XQTBackendError("cuBLASLt W8A8 path expects 2D tensors")
    if qactivation.shape[1] != qweight_t.shape[0]:
        raise XQTBackendError("cuBLASLt W8A8 path requires A.shape[1] == B.shape[0]")
    major, minor = torch.cuda.get_device_capability(qactivation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"cuBLASLt W8A8 fast path is tuned for sm_89, got sm_{major}{minor}"
        )

    m, k = int(qactivation.shape[0]), int(qactivation.shape[1])
    n = int(qweight_t.shape[1])
    scale = _device_activation_scale(activation_scale, qactivation.device)
    wscale = weight_scale.detach().to(
        device=qactivation.device,
        dtype=torch.float32,
    ).contiguous()
    if wscale.numel() != n:
        raise XQTBackendError("weight_scale must have N elements")
    bias_half = (
        None
        if bias is None
        else bias.detach().to(device=qactivation.device, dtype=torch.float16).contiguous()
    )
    accum = torch.empty((m, n), device=qactivation.device, dtype=torch.int32)
    output = torch.empty((m, n), device=qactivation.device, dtype=torch.float16)
    workspace = _cublaslt_workspace(qactivation.device)
    stream = torch.cuda.current_stream(qactivation.device).cuda_stream
    lib = _load_lib()
    if not (
        hasattr(lib, "int8mma_run_cublaslt_i32")
        and hasattr(lib, "int8mma_dequant_i32")
    ):
        raise XQTBackendError(
            "cuBLASLt W8A8 symbols are unavailable; rebuild int8mma_sm89.so"
        )
    err = lib.int8mma_run_cublaslt_i32(
        qactivation.contiguous().data_ptr(),
        qweight_t.contiguous().data_ptr(),
        accum.data_ptr(),
        m,
        n,
        k,
        workspace.data_ptr(),
        workspace.numel(),
        stream,
    )
    if err != 0:
        raise XQTBackendError(f"cuBLASLt W8A8 GEMM failed with status {err}")
    err = lib.int8mma_dequant_i32(
        accum.data_ptr(),
        output.data_ptr(),
        scale.data_ptr(),
        wscale.data_ptr(),
        0 if bias_half is None else bias_half.data_ptr(),
        m,
        n,
    )
    if err != 0:
        raise XQTBackendError(f"cuBLASLt W8A8 epilogue failed with cuda error {err}")
    return output if output_dtype == torch.float16 else output.to(output_dtype)


def int8_gemv_m1_ptx_sm89(
    qactivation: torch.Tensor,
    qweight_t: torch.Tensor,
    activation_scale: torch.Tensor | float,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    output_dtype: torch.dtype = torch.float16,
    prepacked_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """True M=1 INT8 GEMV via DP4A (no MMA tile padding)."""

    if not qactivation.is_cuda or not qweight_t.is_cuda:
        raise XQTBackendError("ptx_sm89 INT8 GEMV requires CUDA tensors")
    if qactivation.dtype != torch.int8 or qweight_t.dtype != torch.int8:
        raise XQTBackendError("ptx_sm89 INT8 GEMV expects int8 inputs")
    if qactivation.ndim != 2 or qweight_t.ndim != 2:
        raise XQTBackendError("ptx_sm89 INT8 GEMV expects 2D tensors")
    if int(qactivation.shape[0]) != 1:
        raise XQTBackendError("ptx_sm89 INT8 GEMV requires M=1")
    if qactivation.shape[1] != qweight_t.shape[0]:
        raise XQTBackendError("ptx_sm89 INT8 GEMV requires A.shape[1] == B.shape[0]")

    k = int(qactivation.shape[1])
    n = int(qweight_t.shape[1])
    sa = _device_activation_scale(activation_scale, qactivation.device)
    sw = weight_scale.detach().to(device=qactivation.device, dtype=torch.float32).contiguous()
    if sw.numel() != n:
        raise XQTBackendError("weight_scale must have N elements")

    a = qactivation.contiguous()
    c_half = torch.empty(1, n, device=qactivation.device, dtype=torch.float16)
    lib = _load_lib()
    if prepacked_b is not None:
        if prepacked_b.shape != (n, k) or prepacked_b.dtype != torch.int8:
            raise XQTBackendError(
                f"prepacked_b must be int8 [N,K]=[{n},{k}], got "
                f"{prepacked_b.dtype} {tuple(prepacked_b.shape)}"
            )
        if not hasattr(lib, "int8_gemv_m1_run_prepacked_b"):
            raise XQTBackendError("int8_gemv_m1_run_prepacked_b missing; rebuild SO")
        err = lib.int8_gemv_m1_run_prepacked_b(
            a.data_ptr(),
            prepacked_b.contiguous().data_ptr(),
            c_half.data_ptr(),
            sa.data_ptr(),
            sw.data_ptr(),
            n,
            k,
        )
    else:
        if not hasattr(lib, "int8_gemv_m1_run"):
            raise XQTBackendError("int8_gemv_m1_run missing; rebuild SO")
        err = lib.int8_gemv_m1_run(
            a.data_ptr(),
            qweight_t.contiguous().data_ptr(),
            c_half.data_ptr(),
            sa.data_ptr(),
            sw.data_ptr(),
            n,
            k,
        )
    if err != 0:
        raise XQTBackendError(f"int8_gemv_m1_run failed with cuda error code {err}")
    out = c_half if output_dtype == torch.float16 else c_half.to(output_dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


def int8_gemv_m1_fused_static_ptx_sm89(
    activations: torch.Tensor,
    qweight_t: torch.Tensor,
    activation_scale: torch.Tensor | float,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    output_dtype: torch.dtype = torch.float16,
    prepacked_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused static quant + M=1 INT8 GEMV. activations: half [1,K]."""

    if not activations.is_cuda or not qweight_t.is_cuda:
        raise XQTBackendError("ptx_sm89 fused GEMV requires CUDA tensors")
    if activations.dtype != torch.float16:
        raise XQTBackendError("ptx_sm89 fused GEMV expects float16 activations")
    if qweight_t.dtype != torch.int8 or activations.ndim != 2 or qweight_t.ndim != 2:
        raise XQTBackendError("ptx_sm89 fused GEMV expects half[1,K] x int8[K,N]")
    if int(activations.shape[0]) != 1:
        raise XQTBackendError("ptx_sm89 fused GEMV requires M=1")
    if activations.shape[1] != qweight_t.shape[0]:
        raise XQTBackendError("ptx_sm89 fused GEMV requires A.shape[1] == B.shape[0]")

    k = int(activations.shape[1])
    n = int(qweight_t.shape[1])
    sa = _device_activation_scale(activation_scale, activations.device)
    sw = weight_scale.detach().to(device=activations.device, dtype=torch.float32).contiguous()
    if sw.numel() != n:
        raise XQTBackendError("weight_scale must have N elements")

    x = activations.contiguous()
    c_half = torch.empty(1, n, device=activations.device, dtype=torch.float16)
    lib = _load_lib()
    if prepacked_b is None:
        prepacked_b = prepack_qweight_t_for_ptx_sm89(qweight_t)
    if prepacked_b.shape != (n, k) or prepacked_b.dtype != torch.int8:
        raise XQTBackendError(
            f"prepacked_b must be int8 [N,K]=[{n},{k}], got "
            f"{prepacked_b.dtype} {tuple(prepacked_b.shape)}"
        )
    if not hasattr(lib, "int8_gemv_m1_run_fused_static_prepacked_b"):
        raise XQTBackendError(
            "int8_gemv_m1_run_fused_static_prepacked_b missing; rebuild SO"
        )
    err = lib.int8_gemv_m1_run_fused_static_prepacked_b(
        x.data_ptr(),
        prepacked_b.contiguous().data_ptr(),
        c_half.data_ptr(),
        sa.data_ptr(),
        sw.data_ptr(),
        n,
        k,
    )
    if err != 0:
        raise XQTBackendError(
            f"int8_gemv_m1_run_fused_static_prepacked_b failed with cuda error code {err}"
        )
    out = c_half if output_dtype == torch.float16 else c_half.to(output_dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


__all__ = [
    "int8mma_available",
    "int8mma_version",
    "int8_linear_cutlass_sm89",
    "int8_linear_cublaslt_sm89",
    "int8_linear_ptx_sm89",
    "int8_linear_fused_static_ptx_sm89",
    "int8_gemv_m1_ptx_sm89",
    "int8_gemv_m1_fused_static_ptx_sm89",
    "prepack_qweight_t_for_ptx_sm89",
]
