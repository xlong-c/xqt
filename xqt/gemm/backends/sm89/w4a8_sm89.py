"""SM89 native W4A8 adapter.

The artifact decodes signed INT4 nibbles into INT8/FP8 MMA operands inside the
mainloop and applies ``[N,G]`` weight scales plus per-tensor/per-token/blockwise
activation scales per K group.  This is deliberately a group-scale mainloop
kernel, not a dequantize-then-dense fallback.  The registry entry stays
``metadata_only`` until the artifact manifest has passed its correctness gate.
"""

from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.gemm.common.contracts import GemmSpec, PackedWeight
from xqt.gemm.common.preflight import artifact_manifest_path, artifact_ready_for_execution
from xqt.gemm.common.quantize import quantize_int8_activation
from xqt.gemm.common.fp8 import quantize_fp8
from xqt.gemm.common.registry import GemmKernelRegistry


_DEFAULT_ARTIFACT = Path.home() / ".cache/xqt/gemm/sm89/w4a8_sm89.so"

_OUTPUT_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
_INT8_SYMBOL = {
    "fp16": "xqt_w4a8_int8_sm89_fp16_run",
    "bf16": "xqt_w4a8_int8_sm89_bf16_run",
    "fp32": "xqt_w4a8_int8_sm89_fp32_run",
}
_FP8_SYMBOL = {
    ("fp8_e4m3", "fp16"): "xqt_w4a8_fp8_e4m3_sm89_fp16_run",
    ("fp8_e4m3", "bf16"): "xqt_w4a8_fp8_e4m3_sm89_bf16_run",
    ("fp8_e4m3", "fp32"): "xqt_w4a8_fp8_e4m3_sm89_fp32_run",
    ("fp8_e5m2", "fp16"): "xqt_w4a8_fp8_e5m2_sm89_fp16_run",
    ("fp8_e5m2", "bf16"): "xqt_w4a8_fp8_e5m2_sm89_bf16_run",
    ("fp8_e5m2", "fp32"): "xqt_w4a8_fp8_e5m2_sm89_fp32_run",
}


def sm89_w4a8_artifact_available(artifact: str | Path | None = None) -> bool:
    """Return whether the W4A8 shared object exports the native ABI symbols."""

    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    if not path.is_file():
        return False
    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        return False
    for symbol in ("xqt_w4a8_int8_sm89_fp16_run", "xqt_w4a8_fp8_e4m3_sm89_fp16_run"):
        if not hasattr(library, symbol):
            return False
    return True


def _canonical_w4a8_weight(weight: PackedWeight, *, spec: GemmSpec) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Validate and return ``(qweight, weight_scales, padded_k)``."""

    if not isinstance(weight, PackedWeight) or weight.metadata.weight_dtype != "int4":
        raise XQTBackendError("SM89 W4A8 requires an INT4 PackedWeight")
    if weight.metadata.storage_layout != "xqt_int4_nk_v1":
        raise XQTBackendError(
            "SM89 W4A8 requires storage_layout='xqt_int4_nk_v1', "
            f"got {weight.metadata.storage_layout!r}"
        )
    if tuple(weight.metadata.logical_shape) != (spec.problem.n, spec.problem.k):
        raise XQTBackendError("SM89 W4A8 weight logical shape disagrees with GemmProblem")
    if weight.zero_points is not None:
        raise XQTBackendError("SM89 W4A8 native path is symmetric and rejects zero points")
    if weight.scales is None:
        raise XQTBackendError("SM89 W4A8 requires explicit groupwise weight scales")
    qweight = weight.qweight
    if not isinstance(qweight, torch.Tensor) or not qweight.is_cuda:
        raise XQTBackendError("SM89 W4A8 qweight must be a CUDA tensor")
    if qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise XQTBackendError("SM89 W4A8 qweight must be uint8 [N, padded_K/2]")
    padded_k = int(weight.metadata.padded_k)
    if int(qweight.shape[1]) != (padded_k + 1) // 2:
        raise XQTBackendError("SM89 W4A8 qweight columns disagree with padded_k")
    scales = weight.scales
    if not isinstance(scales, torch.Tensor) or not scales.is_cuda:
        raise XQTBackendError("SM89 W4A8 weight scales must be CUDA tensors")
    if scales.dtype != torch.float32 or scales.ndim != 2:
        raise XQTBackendError("SM89 W4A8 weight scales must be float32 [N,G]")
    if tuple(scales.shape) != (spec.problem.n, padded_k // int(weight.metadata.group_size or 1)):
        raise XQTBackendError("SM89 W4A8 weight scales shape disagrees with [N,G]")
    return qweight.contiguous(), scales.contiguous(), padded_k


def _validate_problem_shape(spec: GemmSpec, *, group_size: int, padded_k: int) -> None:
    m, n = (int(spec.problem.m), int(spec.problem.n))
    if m == 0:
        return
    if m % 16 != 0:
        raise XQTBackendError("SM89 W4A8 native path requires M % 16 == 0")
    if n % 8 != 0:
        raise XQTBackendError("SM89 W4A8 native path requires N % 8 == 0")
    if padded_k % 32 != 0 or padded_k % group_size != 0:
        raise XQTBackendError("SM89 W4A8 requires padded_k % 32 == 0 and padded_k % group_size == 0")
    if group_size % 32 != 0:
        raise XQTBackendError("SM89 W4A8 requires group_size % 32 == 0")


def _run_symbol(
    library: ctypes.CDLL,
    symbol: str,
    *,
    activation: torch.Tensor,
    qweight: torch.Tensor,
    weight_scales: torch.Tensor,
    activation_scales: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    m: int,
    n: int,
    padded_k: int,
    group_size: int,
    activation_mode: int,
    has_bias: int,
) -> None:
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    weight_scale_ptr = ctypes.cast(weight_scales.data_ptr(), ctypes.POINTER(ctypes.c_float))
    activation_scale_ptr = ctypes.cast(
        activation_scales.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_float))
    status = int(
        function(
            activation.data_ptr(),
            qweight.data_ptr(),
            weight_scale_ptr,
            activation_scale_ptr,
            bias_ptr,
            output.data_ptr(),
            m,
            n,
            padded_k,
            group_size,
            activation_mode,
            has_bias,
            stream,
        )
    )
    if status != 0:
        raise XQTBackendError(f"SM89 W4A8 native launch failed with CUDA status {status}")


def sm89_w4a8_executor(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    weight_zero_points: torch.Tensor | None = None,
    activation_zero_points: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    artifact: str | Path | None = None,
) -> torch.Tensor:
    """Run the gated SM89 W4A8 INT8/FP8 native kernel for aligned shapes."""

    quant = spec.quant
    if quant.weight_dtype != "int4":
        raise XQTBackendError("SM89 W4A8 requires weight_dtype='int4'")
    if quant.activation_dtype not in {"int8", "fp8_e4m3", "fp8_e5m2"}:
        raise XQTBackendError("SM89 W4A8 requires int8 or fp8_e4m3/e5m2 activation")
    if quant.weight_granularity != "groupwise":
        raise XQTBackendError("SM89 W4A8 native path supports groupwise weight scales only")
    if weight_zero_points is not None or activation_zero_points is not None:
        raise XQTBackendError("SM89 W4A8 native path is symmetric with no zero points")
    if residual is not None:
        raise XQTBackendError("SM89 W4A8 has no residual epilogue")
    if spec.epilogue.activation != "none":
        raise XQTBackendError("SM89 W4A8 supports bias-only epilogue")
    if quant.output_dtype not in _OUTPUT_DTYPES or spec.epilogue.output_dtype != quant.output_dtype:
        raise XQTBackendError("SM89 W4A8 output dtype must be fp16, bf16, or fp32")
    if not isinstance(weight, PackedWeight):
        raise XQTBackendError("SM89 W4A8 requires a canonical PackedWeight")
    if weight_scales is not None and not torch.equal(weight_scales, weight.scales):
        raise XQTBackendError("external weight_scales disagree with PackedWeight.scales")
    if not isinstance(activation, torch.Tensor) or not activation.is_cuda:
        raise XQTBackendError("SM89 W4A8 activation must be CUDA")
    if activation.ndim != 2 or tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("SM89 W4A8 activation shape disagrees with GemmProblem")
    major, minor = torch.cuda.get_device_capability(activation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 W4A8 executor received sm_{major}{minor}")
    if bias is not None and int(bias.numel()) != spec.problem.n:
        raise XQTBackendError("SM89 W4A8 bias must have N elements")
    if spec.epilogue.has_bias and bias is None:
        raise XQTBackendError("SM89 W4A8 epilogue declares bias but no bias was supplied")

    qweight, w4a8_weight_scales, padded_k = _canonical_w4a8_weight(weight, spec=spec)
    group_size = int(weight.metadata.group_size or 1)
    _validate_problem_shape(spec, group_size=group_size, padded_k=padded_k)
    output_dtype = _OUTPUT_DTYPES[quant.output_dtype]
    if spec.problem.m == 0:
        return torch.empty((0, spec.problem.n), device=activation.device, dtype=output_dtype)

    artifact_path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    try:
        library = ctypes.CDLL(str(artifact_path))
    except OSError as exc:
        raise XQTBackendError(f"SM89 W4A8 artifact is unavailable: {exc}") from exc

    groups = padded_k // group_size
    if quant.activation_dtype == "int8":
        if quant.activation_granularity not in {"per_tensor", "per_token"}:
            raise XQTBackendError("SM89 W4A8 INT8 activation supports per_tensor/per_token only")
        encoded = quantize_int8_activation(
            activation,
            granularity=quant.activation_granularity,
            source="activation_dynamic",
        )
        padded_values = F.pad(encoded.values, (0, padded_k - spec.problem.k))
        activation_mode = 0 if quant.activation_granularity == "per_tensor" else 1
        runtime_scales = encoded.scales.reshape(-1)
        symbol = _INT8_SYMBOL[quant.output_dtype]
    else:
        if quant.activation_granularity == "blockwise" and group_size != 32:
            raise XQTBackendError("SM89 W4A8 FP8 blockwise scale uses block_k == weight group_size")
        block_k = group_size if quant.activation_granularity == "blockwise" else None
        encoded_fp8 = quantize_fp8(
            activation,
            format_name=quant.activation_dtype,
            granularity=quant.activation_granularity,
            role="activation",
            source=quant.activation_scale_source,
            scale=activation_scales,
            block_k=block_k,
        )
        padded_values = F.pad(encoded_fp8.storage, (0, padded_k - spec.problem.k))
        activation_mode = {"per_tensor": 0, "per_token": 1, "blockwise": 2}[
            quant.activation_granularity
        ]
        runtime_scales = encoded_fp8.scale.reshape(-1)
        if quant.activation_granularity == "blockwise":
            if int(runtime_scales.numel()) != spec.problem.m * groups:
                raise XQTBackendError("SM89 W4A8 FP8 blockwise scale shape disagrees with [M,G]")
        symbol = _FP8_SYMBOL[(quant.activation_dtype, quant.output_dtype)]
    if not hasattr(library, symbol):
        raise XQTBackendError(f"SM89 W4A8 artifact lacks symbol {symbol}")

    runtime_scales = runtime_scales.detach().to(
        device=activation.device, dtype=torch.float32
    ).contiguous()
    bias_value = (
        torch.zeros(spec.problem.n, device=activation.device, dtype=torch.float32)
        if bias is None
        else bias.detach().to(device=activation.device, dtype=torch.float32).contiguous()
    )
    output = torch.empty((spec.problem.m, spec.problem.n), device=activation.device, dtype=output_dtype)
    _run_symbol(
        library,
        symbol,
        activation=padded_values.contiguous(),
        qweight=qweight,
        weight_scales=w4a8_weight_scales,
        activation_scales=runtime_scales,
        bias=bias_value,
        output=output,
        m=spec.problem.m,
        n=spec.problem.n,
        padded_k=padded_k,
        group_size=group_size,
        activation_mode=activation_mode,
        has_bias=1 if bias is not None else 0,
    )
    return output


def install_sm89_w4a8_executors(
    registry: GemmKernelRegistry,
    *,
    artifact: str | Path | None = None,
    manifest: str | Path | None = None,
) -> bool:
    """Promote both W4A8 registry entries after artifact and correctness gates."""

    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    if not sm89_w4a8_artifact_available(path):
        return False
    manifest_path = Path(manifest) if manifest is not None else artifact_manifest_path(path)
    if not artifact_ready_for_execution(
        manifest_path, kernel_name="sm89_w4a8_cutlass", target_arch="sm_89"
    ):
        return False
    for name in ("sm89_w4a8_int8_cutlass", "sm89_w4a8_fp8_cutlass"):
        entry = registry.get(name)
        registry.replace(
            replace(
                entry,
                maturity="executable",
                implementation="cutlass_sm89_w4a8_native_mma",
                executor=lambda *args, **kwargs: sm89_w4a8_executor(
                    *args, artifact=path, **kwargs
                ),
            )
        )
    return True


__all__ = [
    "install_sm89_w4a8_executors",
    "sm89_w4a8_artifact_available",
    "sm89_w4a8_executor",
]
