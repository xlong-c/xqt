"""SM89 W4A16 CUDA dequant fallback adapter.

The adapter is artifact- and manifest-gated. Its family name intentionally
states that it is a fallback: it is a correctness baseline for canonical W4
storage, not a claim that SM89 has a fused CUTLASS INT4 weight-only mainloop.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from xqt.core.errors import XQTBackendError

from xqt.gemm.common.contracts import GemmSpec, PackedWeight
from xqt.gemm.common.layout import validate_w4a16_packed_weight
from xqt.gemm.common.preflight import artifact_manifest_path, artifact_ready_for_execution
from xqt.gemm.common.registry import GemmKernelRegistration, GemmKernelRegistry


_RESOURCE_VARIANTS: dict[str, tuple[int, int, int]] = {
    "m1_gemv": (0, 256, 0),
    "small_m_2_8": (1, 128, 0),
    "tile_m_8x16x32": (2, 128, 2560),
}


@dataclass(frozen=True, slots=True)
class Sm89W4A16ResourceReport:
    """Runtime CUDA resource and occupancy evidence for one W4A16 variant."""

    artifact: str
    variant: str
    block_threads: int
    dynamic_shared_bytes: int
    registers_per_thread: int
    static_shared_bytes: int
    max_active_blocks_per_sm: int
    multiprocessor_count: int
    max_threads_per_sm: int
    occupancy: float | None
    device: str
    device_name: str
    capability: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "variant": self.variant,
            "block_threads": self.block_threads,
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
            "registers_per_thread": self.registers_per_thread,
            "static_shared_bytes": self.static_shared_bytes,
            "max_active_blocks_per_sm": self.max_active_blocks_per_sm,
            "multiprocessor_count": self.multiprocessor_count,
            "max_threads_per_sm": self.max_threads_per_sm,
            "occupancy": self.occupancy,
            "device": self.device,
            "device_name": self.device_name,
            "capability": self.capability,
        }


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 W4A16 artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 W4A16 artifact: {path}") from exc
    argument_types = [
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
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    for name in (
        "xqt_w4a16_sm89_fp16_run",
        "xqt_w4a16_sm89_bf16_run",
        "xqt_w4a16_sm89_fp16_run_persistent",
        "xqt_w4a16_sm89_bf16_run_persistent",
    ):
        if hasattr(library, name):
            function = getattr(library, name)
            function.argtypes = argument_types
            function.restype = ctypes.c_int
    if hasattr(library, "xqt_w4a16_sm89_resource_query"):
        function = library.xqt_w4a16_sm89_resource_query
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        function.restype = ctypes.c_int
    return library


def sm89_w4a16_dequant_artifact_available(artifact: str | Path) -> bool:
    """Return whether both FP16 and BF16 fallback entry points are exported."""

    try:
        library = _load_library(artifact)
    except XQTBackendError:
        return False
    return hasattr(library, "xqt_w4a16_sm89_fp16_run") and hasattr(
        library, "xqt_w4a16_sm89_bf16_run"
    )


def query_sm89_w4a16_resources(
    artifact: str | Path,
    *,
    variant: str,
    device: torch.device | str | None = None,
) -> Sm89W4A16ResourceReport:
    """Query CUDA function attributes and occupancy for a loaded artifact.

    The query calls CUDA runtime introspection only; it does not launch a
    kernel.  ``variant`` must match the shape branch used by the launch ABI.
    """

    if variant not in _RESOURCE_VARIANTS:
        raise ValueError(f"unsupported W4A16 resource variant: {variant!r}")
    if not torch.cuda.is_available():
        raise XQTBackendError("W4A16 resource query requires CUDA")
    cuda_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if cuda_device.type != "cuda":
        raise ValueError(f"W4A16 resource query requires a CUDA device, got {cuda_device}")
    major, minor = torch.cuda.get_device_capability(cuda_device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 W4A16 resource query received sm_{major}{minor}")
    library = _load_library(artifact)
    if not hasattr(library, "xqt_w4a16_sm89_resource_query"):
        raise XQTBackendError("SM89 W4A16 artifact lacks resource query symbol")
    variant_id, block_threads, dynamic_shared = _RESOURCE_VARIANTS[variant]
    registers = ctypes.c_int()
    static_shared = ctypes.c_int()
    max_blocks = ctypes.c_int()
    error = library.xqt_w4a16_sm89_resource_query(
        variant_id,
        block_threads,
        dynamic_shared,
        ctypes.byref(registers),
        ctypes.byref(static_shared),
        ctypes.byref(max_blocks),
    )
    if error != 0:
        raise XQTBackendError(f"SM89 W4A16 resource query failed with CUDA error {error}")
    properties = torch.cuda.get_device_properties(cuda_device)
    multiprocessors = int(properties.multi_processor_count)
    max_threads_per_sm = int(properties.max_threads_per_multi_processor)
    active_threads = max_blocks.value * block_threads
    occupancy = (
        float(active_threads) / float(max_threads_per_sm)
        if max_threads_per_sm > 0
        else None
    )
    return Sm89W4A16ResourceReport(
        artifact=str(Path(artifact).expanduser()),
        variant=variant,
        block_threads=block_threads,
        dynamic_shared_bytes=dynamic_shared,
        registers_per_thread=int(registers.value),
        static_shared_bytes=int(static_shared.value),
        max_active_blocks_per_sm=int(max_blocks.value),
        multiprocessor_count=multiprocessors,
        max_threads_per_sm=max_threads_per_sm,
        occupancy=occupancy,
        device=str(cuda_device),
        device_name=str(torch.cuda.get_device_name(cuda_device)),
        capability=f"sm_{major}{minor}",
    )


def sm89_w4a16_dequant_executor(
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
    persistent: bool = False,
) -> torch.Tensor:
    """Run the explicit CUDA dequant fallback for canonical W4A16 weights."""

    if not isinstance(weight, PackedWeight):
        raise XQTBackendError("SM89 W4A16 fallback requires a canonical PackedWeight")
    if not isinstance(persistent, bool):
        raise TypeError("SM89 W4A16 persistent must be bool")
    if spec.quant.activation_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("SM89 W4A16 fallback supports fp16 or bf16 activations")
    if spec.quant.output_dtype != spec.quant.activation_dtype:
        raise XQTBackendError("SM89 W4A16 fallback requires output dtype equal to activation dtype")
    if spec.epilogue.activation != "none" or residual is not None:
        raise XQTBackendError("SM89 W4A16 fallback supports bias only, not residual/activation")
    if activation_scales is not None or activation_zero_points is not None:
        raise XQTBackendError("W4A16 does not accept activation quantization scales")
    validate_w4a16_packed_weight(
        weight,
        spec=spec.quant,
        logical_shape=(spec.problem.n, spec.problem.k),
    )
    if weight_scales is not None and not torch.equal(weight_scales, weight.scales):
        raise XQTBackendError("external weight_scales disagree with PackedWeight.scales")
    if weight_zero_points is not None and not torch.equal(
        weight_zero_points, weight.zero_points
    ):
        raise XQTBackendError("external weight_zero_points disagree with PackedWeight.zero_points")
    if not activation.is_cuda or activation.ndim != 2:
        raise XQTBackendError("SM89 W4A16 activation must be CUDA rank-2")
    expected_dtype = torch.float16 if spec.quant.activation_dtype == "fp16" else torch.bfloat16
    if activation.dtype != expected_dtype:
        raise XQTBackendError("SM89 W4A16 activation dtype disagrees with QuantSpec")
    if tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("SM89 W4A16 activation shape does not match GemmProblem")
    qweight = weight.qweight
    scales = weight.scales
    zero_points = weight.zero_points
    if not isinstance(qweight, torch.Tensor) or not isinstance(scales, torch.Tensor):
        raise XQTBackendError("SM89 W4A16 PackedWeight tensors are missing")
    if not qweight.is_cuda or not scales.is_cuda:
        raise XQTBackendError("SM89 W4A16 qweight and scales must be CUDA tensors")
    if scales.dtype != torch.float32 or (zero_points is not None and zero_points.dtype != torch.float32):
        raise XQTBackendError("SM89 W4A16 scales and zero_points must be float32")
    if zero_points is not None and not zero_points.is_cuda:
        raise XQTBackendError("SM89 W4A16 zero_points must be on CUDA")
    if bias is not None:
        if tuple(bias.shape) not in {(spec.problem.n,), (1, spec.problem.n)}:
            raise XQTBackendError("SM89 W4A16 bias must have shape [N] or [1,N]")
        bias_device = bias.to(device=activation.device, dtype=torch.float32).reshape(-1).contiguous()
        if bias_device.numel() != spec.problem.n:
            raise XQTBackendError("SM89 W4A16 bias must have N elements")
    else:
        bias_device = None
    if spec.epilogue.has_bias and bias_device is None:
        raise XQTBackendError("W4A16 epilogue declares bias but no bias was supplied")
    major, minor = torch.cuda.get_device_capability(activation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 W4A16 fallback received sm_{major}{minor}")
    library = _load_library(artifact)
    if expected_dtype == torch.float16:
        symbol = "xqt_w4a16_sm89_fp16_run_persistent" if persistent else "xqt_w4a16_sm89_fp16_run"
    else:
        symbol = "xqt_w4a16_sm89_bf16_run_persistent" if persistent else "xqt_w4a16_sm89_bf16_run"
    if not hasattr(library, symbol):
        raise XQTBackendError(f"SM89 W4A16 artifact lacks {symbol}")
    output = torch.empty(
        (spec.problem.m, spec.problem.n), device=activation.device, dtype=expected_dtype
    )
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    zero_device = None if zero_points is None else zero_points.contiguous()
    function = getattr(library, symbol)
    error = function(
        activation.contiguous().data_ptr(),
        qweight.contiguous().data_ptr(),
        scales.contiguous().data_ptr(),
        0 if zero_device is None else zero_device.data_ptr(),
        0 if bias_device is None else bias_device.data_ptr(),
        output.data_ptr(),
        spec.problem.m,
        spec.problem.n,
        spec.problem.k,
        weight.metadata.padded_k,
        int(weight.metadata.group_size or 0),
        int(weight.metadata.nibble_signed),
        int(zero_device is not None),
        int(bias_device is not None),
        stream,
    )
    if error != 0:
        raise XQTBackendError(f"SM89 W4A16 CUDA fallback failed with error {error}")
    return output


def install_sm89_w4a16_dequant_executor(
    registry: GemmKernelRegistry,
    *,
    artifact: str | Path,
    manifest: str | Path | None = None,
) -> bool:
    """Promote only the explicitly named fallback entry after numeric gates."""

    if not sm89_w4a16_dequant_artifact_available(artifact):
        return False
    manifest_path = Path(manifest) if manifest is not None else artifact_manifest_path(artifact)
    if not artifact_ready_for_execution(
        manifest_path,
        kernel_name="sm89_w4a16_dequant_fallback",
        target_arch="sm_89",
    ):
        return False
    entry = registry.get("sm89_w4a16_dequant_fallback")
    registry.replace(
        GemmKernelRegistration(
            name=entry.name,
            backend=entry.backend,
            maturity="executable",
            capability=entry.capability,
            kernel_family=entry.kernel_family,
            layout=entry.layout,
            tile_shape=entry.tile_shape,
            warp_count=entry.warp_count,
            stage_count=entry.stage_count,
            alignment=entry.alignment,
            priority=entry.priority,
            implementation="cuda_sm89_w4a16_dequant_artifact",
            executor=lambda *args, **kwargs: sm89_w4a16_dequant_executor(
                *args, artifact=artifact, **kwargs
            ),
        )
    )
    return True


__all__ = [
    "Sm89W4A16ResourceReport",
    "install_sm89_w4a16_dequant_executor",
    "query_sm89_w4a16_resources",
    "sm89_w4a16_dequant_artifact_available",
    "sm89_w4a16_dequant_executor",
]
