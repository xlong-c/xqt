"""Manifest-gated SM89 FP8 CUTLASS GEMM adapter."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ..contracts import GemmSpec, PackedWeight
from ..preflight import artifact_manifest_path, artifact_ready_for_execution
from ..registry import GemmCapability, GemmKernelRegistration, GemmKernelRegistry
from ..fp8 import fp8_block_count, fp8_format_spec, validate_fp8_block_k


_SYMBOLS = {
    ("fp8_e4m3", "fp16"): "fp8_sm89_e4m3_fp16_run",
    ("fp8_e4m3", "bf16"): "fp8_sm89_e4m3_bf16_run",
    ("fp8_e5m2", "fp16"): "fp8_sm89_e5m2_fp16_run",
    ("fp8_e5m2", "bf16"): "fp8_sm89_e5m2_bf16_run",
}
_BLOCKWISE_SYMBOLS = {
    ("fp8_e4m3", "fp16"): "fp8_sm89_e4m3_fp16_run_blockwise",
    ("fp8_e4m3", "bf16"): "fp8_sm89_e4m3_bf16_run_blockwise",
    ("fp8_e5m2", "fp16"): "fp8_sm89_e5m2_fp16_run_blockwise",
    ("fp8_e5m2", "bf16"): "fp8_sm89_e5m2_bf16_run_blockwise",
}


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 FP8 artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 FP8 artifact: {path}") from exc
    for symbol in _SYMBOLS.values():
        if not hasattr(library, symbol):
            raise XQTBackendError(f"SM89 FP8 artifact lacks {symbol}")
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        ]
        function.restype = ctypes.c_int
    for symbol in _BLOCKWISE_SYMBOLS.values():
        if not hasattr(library, symbol):
            raise XQTBackendError(f"SM89 FP8 artifact lacks {symbol}")
        function = getattr(library, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        function.restype = ctypes.c_int
    return library


def sm89_fp8_artifact_available(artifact: str | Path) -> bool:
    """Return whether all four format/output symbols are loadable."""

    try:
        _load_library(artifact)
    except XQTBackendError:
        return False
    return True


def _scale_scalar(value: torch.Tensor | None, *, name: str) -> float:
    if value is None:
        raise XQTBackendError(f"{name} is required for native tensorwise FP8")
    if not isinstance(value, torch.Tensor):
        raise XQTBackendError(f"{name} must be a tensor")
    if value.ndim == 0:
        scalar = value
    elif tuple(value.shape) == (1, 1):
        scalar = value.reshape(())
    else:
        raise XQTBackendError(
            f"native tensorwise FP8 requires {name} scalar or [1,1], got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(scalar).all()) or bool((scalar <= 0).any()):
        raise XQTBackendError(f"{name} must be finite and positive")
    return float(scalar.item())


def _scale_blockwise(
    value: torch.Tensor | None,
    *,
    rows: int,
    cols: int,
    block_k: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        raise XQTBackendError(f"{name} is required for native blockwise FP8")
    if not isinstance(value, torch.Tensor):
        raise XQTBackendError(f"{name} must be a tensor")
    block_k = validate_fp8_block_k(block_k)
    blocks = fp8_block_count(cols, block_k)
    if tuple(value.shape) != (rows, blocks):
        raise XQTBackendError(
            f"native blockwise FP8 requires {name} [{rows},{blocks}] "
            f"(ceil(cols={cols}/block_k={block_k})), got {tuple(value.shape)}"
        )
    if not value.is_cuda:
        raise XQTBackendError(f"native blockwise FP8 requires CUDA {name}")
    if value.device != device:
        raise XQTBackendError(f"{name} must be on the activation device")
    if value.dtype != torch.float32:
        raise XQTBackendError(f"native blockwise FP8 requires FP32 {name}")
    return value.contiguous()


def _encoded_fp8(value: torch.Tensor, *, format_name: str, name: str) -> torch.Tensor:
    format_spec = fp8_format_spec(format_name)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise XQTBackendError(f"native FP8 {name} must be rank-2 encoded storage")
    if value.dtype == torch.uint8:
        return value
    if value.dtype == format_spec.torch_dtype:
        return value.view(torch.uint8)
    raise XQTBackendError(
        f"native FP8 {name} must be uint8 or {format_spec.torch_dtype}, got {value.dtype}"
    )


def _weight_payload(weight: torch.Tensor | PackedWeight) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(weight, PackedWeight):
        if not isinstance(weight.qweight, torch.Tensor):
            raise XQTBackendError("native FP8 PackedWeight qweight must be a tensor")
        return weight.qweight, weight.scales
    if not isinstance(weight, torch.Tensor):
        raise XQTBackendError("native FP8 weight must be a tensor or PackedWeight")
    return weight, None


def fp8_sm89_executor(
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
    artifact: str | Path,
) -> torch.Tensor:
    """Run SM89 FP8 native paths; unsupported scale modes raise for fallback."""

    quant = spec.quant
    if quant.weight_dtype not in {"fp8_e4m3", "fp8_e5m2"}:
        raise XQTBackendError("native SM89 FP8 requires an FP8 weight dtype")
    if quant.activation_dtype != quant.weight_dtype:
        raise XQTBackendError("native SM89 FP8 currently requires matching A/W FP8 formats")
    if quant.output_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("native SM89 FP8 supports fp16 or bf16 output")
    tensorwise = quant.weight_granularity == "per_tensor" and quant.activation_granularity == "per_tensor"
    blockwise = quant.weight_granularity == "blockwise" and quant.activation_granularity == "blockwise"
    if not tensorwise and not blockwise:
        raise XQTBackendError(
            "native SM89 FP8 supports tensorwise or matched K-blockwise scales only"
        )
    if quant.activation_scale_source != "activation_static":
        raise XQTBackendError(
            "dynamic FP8 activation quantization is a separate path and is not fused here"
        )
    if spec.epilogue.activation != "none":
        raise XQTBackendError("native SM89 FP8 has no fused activation epilogue")
    if weight_zero_points is not None or activation_zero_points is not None:
        raise XQTBackendError("FP8 native GEMM does not accept zero points")
    qweight, packed_scales = _weight_payload(weight)
    if weight_scales is None:
        weight_scales = packed_scales
    if weight_scales is None:
        raise XQTBackendError("native FP8 weight scale is missing")
    if not activation.is_cuda or not qweight.is_cuda:
        raise XQTBackendError("native SM89 FP8 requires CUDA tensors")
    if tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("activation shape does not match GemmProblem")
    if tuple(qweight.shape) != (spec.problem.n, spec.problem.k):
        raise XQTBackendError("FP8 weight shape does not match GemmProblem")
    a_bytes = _encoded_fp8(activation, format_name=quant.activation_dtype, name="activation")
    w_bytes = _encoded_fp8(qweight, format_name=quant.weight_dtype, name="weight")
    if not a_bytes.is_cuda or not w_bytes.is_cuda:
        raise XQTBackendError("native SM89 FP8 storage must be CUDA")
    if bias is not None and tuple(bias.shape) not in {(spec.problem.n,), (1, spec.problem.n)}:
        raise XQTBackendError("FP8 bias must have shape [N] or [1,N]")
    if residual is not None and tuple(residual.shape) != (spec.problem.m, spec.problem.n):
        raise XQTBackendError("FP8 residual must have output shape")
    if spec.epilogue.has_bias and bias is None:
        raise XQTBackendError("FP8 epilogue declares bias but no bias was supplied")
    if spec.epilogue.has_residual and residual is None:
        raise XQTBackendError("FP8 epilogue declares residual but no residual was supplied")
    major, minor = torch.cuda.get_device_capability(activation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"native SM89 FP8 received sm_{major}{minor}")
    library = _load_library(artifact)
    output_dtype = torch.float16 if quant.output_dtype == "fp16" else torch.bfloat16
    has_c = bias is not None or residual is not None

    if blockwise:
        if quant.group_axis != "k":
            raise XQTBackendError("native blockwise FP8 requires group_axis='k'")
        if quant.group_size is None:
            raise XQTBackendError("native blockwise FP8 requires QuantSpec.group_size")
        block_k = validate_fp8_block_k(int(quant.group_size))
        a_scale = _scale_blockwise(
            activation_scales,
            rows=spec.problem.m,
            cols=spec.problem.k,
            block_k=block_k,
            device=activation.device,
            name="activation_scales",
        )
        w_scale = _scale_blockwise(
            weight_scales,
            rows=spec.problem.n,
            cols=spec.problem.k,
            block_k=block_k,
            device=activation.device,
            name="weight_scales",
        )
        c_source = None
        if has_c:
            c_source = torch.zeros(
                (spec.problem.m, spec.problem.n),
                device=activation.device,
                dtype=output_dtype,
            )
            if bias is not None:
                c_source.add_(bias.to(device=activation.device, dtype=output_dtype).reshape(1, -1))
            if residual is not None:
                c_source.add_(residual.to(device=activation.device, dtype=output_dtype))
        output = torch.empty((spec.problem.m, spec.problem.n), device=activation.device, dtype=output_dtype)
        symbol = _BLOCKWISE_SYMBOLS[(quant.weight_dtype, quant.output_dtype)]
        a_runtime = a_bytes.contiguous()
        w_runtime = w_bytes.contiguous()
        error = getattr(library, symbol)(
            a_runtime.data_ptr(),
            w_runtime.data_ptr(),
            a_scale.data_ptr(),
            w_scale.data_ptr(),
            None if c_source is None else c_source.data_ptr(),
            output.data_ptr(),
            spec.problem.m,
            spec.problem.n,
            spec.problem.k,
            block_k,
            ctypes.c_float(1.0 if has_c else 0.0),
        )
        if error != 0:
            raise XQTBackendError(f"SM89 FP8 blockwise GEMM failed with CUDA error {error}")
        return output

    alpha = _scale_scalar(weight_scales, name="weight_scales") * _scale_scalar(
        activation_scales, name="activation_scales"
    )
    alignment_m, alignment_n, alignment_k = (8, 8, 32)
    padded_m = (spec.problem.m + alignment_m - 1) // alignment_m * alignment_m
    padded_n = (spec.problem.n + alignment_n - 1) // alignment_n * alignment_n
    padded_k = (spec.problem.k + alignment_k - 1) // alignment_k * alignment_k
    a_padded = a_bytes
    if tuple(a_bytes.shape) != (padded_m, padded_k):
        a_padded = F.pad(a_bytes, (0, padded_k - spec.problem.k, 0, padded_m - spec.problem.m))
    w_padded = w_bytes
    if tuple(w_bytes.shape) != (padded_n, padded_k):
        w_padded = F.pad(w_bytes, (0, padded_k - spec.problem.k, 0, padded_n - spec.problem.n))
    c_source = torch.zeros((padded_m, padded_n), device=activation.device, dtype=output_dtype)
    if bias is not None:
        c_source[:, : spec.problem.n].add_(
            bias.to(device=activation.device, dtype=output_dtype).reshape(1, -1)
        )
    if residual is not None:
        c_source[: spec.problem.m, : spec.problem.n].add_(
            residual.to(device=activation.device, dtype=output_dtype)
        )
    output = torch.empty_like(c_source)
    symbol = _SYMBOLS[(quant.weight_dtype, quant.output_dtype)]
    a_runtime = a_padded.contiguous()
    w_runtime = w_padded.contiguous()
    error = getattr(library, symbol)(
        a_runtime.data_ptr(),
        w_runtime.data_ptr(),
        c_source.data_ptr(),
        output.data_ptr(),
        padded_m,
        padded_n,
        padded_k,
        ctypes.c_float(alpha),
        ctypes.c_float(1.0 if has_c else 0.0),
    )
    if error != 0:
        raise XQTBackendError(f"SM89 FP8 CUTLASS GEMM failed with CUDA error {error}")
    return output[: spec.problem.m, : spec.problem.n]


def install_sm89_fp8_executors(
    registry: GemmKernelRegistry,
    *,
    artifact: str | Path,
    manifest: str | Path | None = None,
) -> bool:
    """Promote both FP8 format entries only after an executable manifest gate."""

    if not sm89_fp8_artifact_available(artifact):
        return False
    manifest_path = Path(manifest) if manifest is not None else artifact_manifest_path(artifact)
    if not artifact_ready_for_execution(
        manifest_path,
        kernel_name="sm89_fp8_cutlass",
        target_arch="sm_89",
    ):
        return False
    for name in ("sm89_fp8_e4m3_cutlass", "sm89_fp8_e5m2_cutlass"):
        entry = registry.get(name)
        registry.replace(
            GemmKernelRegistration(
                name=entry.name,
                backend=entry.backend,
                maturity="executable",
                capability=GemmCapability(
                    architectures=("sm_89",),
                    weight_dtypes=entry.capability.weight_dtypes,
                    activation_dtypes=entry.capability.activation_dtypes,
                    scale_modes=("w:per_tensor/a:per_tensor", "w:blockwise/a:blockwise"),
                    phases=("generic", "prefill"),
                    epilogues=("none",),
                    min_sm=89,
                ),
                kernel_family=entry.kernel_family,
                layout=entry.layout,
                tile_shape=entry.tile_shape,
                warp_count=entry.warp_count,
                stage_count=entry.stage_count,
                alignment=entry.alignment,
                priority=entry.priority,
                implementation="cutlass_sm89_fp8_artifact",
                executor=lambda *args, **kwargs: fp8_sm89_executor(
                    *args, artifact=artifact, **kwargs
                ),
            )
        )
    return True


__all__ = [
    "fp8_sm89_executor",
    "install_sm89_fp8_executors",
    "sm89_fp8_artifact_available",
]
