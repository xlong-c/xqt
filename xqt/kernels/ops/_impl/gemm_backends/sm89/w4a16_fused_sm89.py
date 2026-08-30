"""Manifest-gated SM89 fused W4A16 CUTLASS warp-MMA adapter."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops.gemm.contracts import GemmSpec, PackedWeight
from xqt.kernels.ops.gemm.layout import validate_w4a16_packed_weight
from xqt.kernels.ops.gemm.preflight import artifact_manifest_path, artifact_ready_for_execution
from xqt.kernels.ops.gemm.registry import GemmKernelRegistration, GemmKernelRegistry


_SYMBOL = "xqt_w4a16_cutlass_fused_sm89_fp16_run"
_SPLITK_SYMBOL = "xqt_w4a16_cutlass_fused_sm89_fp16_run_splitk"
_DECODE_SYMBOL = "xqt_w4a16_cutlass_fused_sm89_fp16_run_decode"


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 fused W4A16 artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 fused W4A16 artifact: {path}") from exc
    if not hasattr(library, _SYMBOL):
        raise XQTBackendError(f"SM89 fused W4A16 artifact lacks {_SYMBOL}")
    function = getattr(library, _SYMBOL)
    function.argtypes = [
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
    function.restype = ctypes.c_int
    return library


def _load_splitk_function(library: ctypes.CDLL) -> Any:
    if not hasattr(library, _SPLITK_SYMBOL):
        raise XQTBackendError(f"SM89 fused W4A16 artifact lacks {_SPLITK_SYMBOL}")
    function = getattr(library, _SPLITK_SYMBOL)
    function.argtypes = [
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
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function


def sm89_w4a16_fused_splitk_supported(artifact: str | Path) -> bool:
    """Return whether the fused artifact exports the split-K workspace ABI."""

    try:
        _load_splitk_function(_load_library(artifact))
    except XQTBackendError:
        return False
    return True


def _load_decode_function(library: ctypes.CDLL) -> Any:
    if not hasattr(library, _DECODE_SYMBOL):
        raise XQTBackendError(f"SM89 fused W4A16 artifact lacks {_DECODE_SYMBOL}")
    function = getattr(library, _DECODE_SYMBOL)
    function.argtypes = [
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
    function.restype = ctypes.c_int
    return function


def sm89_w4a16_fused_decode_supported(artifact: str | Path) -> bool:
    """Return whether the fused artifact exports the M=1..8 decode ABI."""

    try:
        _load_decode_function(_load_library(artifact))
    except XQTBackendError:
        return False
    return True


def split_k_partition(k: int, split_k: int) -> tuple[int, int]:
    """Return (k_per_split, split_count) for one split-K launch over k."""

    k_tiles = (k + 15) // 16
    tiles_per_split = -(-k_tiles // split_k)
    k_per_split = tiles_per_split * 16
    split_count = -(-k // k_per_split)
    return k_per_split, split_count


def select_fused_split_k(
    *,
    m: int,
    n: int,
    k: int,
    sm_count: int,
    max_split: int = 16,
    k_tiles_per_split: int = 16,
) -> int:
    """Shape heuristic for the optional fused split-K path.

    The full-K CTA is one warp per 16x8 output tile, so K runs serially inside
    a single block.  Split-K is useful when the K loop is long relative to the
    available (M, N) parallelism.  The heuristic asks for at least
    ``k_tiles_per_split`` K tiles per split and never exceeds ``max_split``;
    it returns 1 (full-K) when the K loop is already short or the (M, N) grid
    alone fills the device.  ``split_k=1`` must route to the full-K ABI.
    """

    if m <= 0 or n <= 0 or k <= 0 or sm_count <= 0:
        raise XQTBackendError("select_fused_split_k requires positive m/n/k/sm_count")
    k_tiles = (k + 15) // 16
    if k_tiles <= k_tiles_per_split:
        return 1
    base_blocks = ((m + 15) // 16) * ((n + 7) // 8)
    if base_blocks >= 2 * sm_count and k_tiles <= 2 * k_tiles_per_split:
        return 1
    split_k = -(-k_tiles // k_tiles_per_split)
    return max(1, min(split_k, max_split, k_tiles))


def sm89_w4a16_fused_artifact_available(artifact: str | Path) -> bool:
    """Return whether the fused CUTLASS warp-MMA symbol is loadable."""

    try:
        _load_library(artifact)
    except XQTBackendError:
        return False
    return True


def sm89_w4a16_fused_executor(
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
    split_k: int | None = None,
) -> torch.Tensor:
    """Run the experimental K-tile decode/scale + CUTLASS warp MMA path.

    ``M=1..8`` routes to the fused decode kernel (same nibble/scale semantics,
    no MMA), which replaces the old ``m1_gemv``/``small_m_2_8`` decode branch;
    those variants stay in the fallback artifact as the evidence chain.
    ``split_k=None`` or ``split_k=1`` keeps the full-K one-warp CTA ABI.
    ``split_k>=2`` routes to the split-K workspace + reduction ABI, which is
    an explicit opt-in on the M%16 MMA path; callers record the chosen value
    in their own report.
    """

    if not isinstance(weight, PackedWeight):
        raise XQTBackendError("fused W4A16 requires a canonical PackedWeight")
    if spec.quant.activation_dtype != "fp16" or spec.quant.output_dtype != "fp16":
        raise XQTBackendError("fused SM89 W4A16 currently supports fp16 input/output only")
    if spec.epilogue.activation != "none" or residual is not None:
        raise XQTBackendError("fused SM89 W4A16 supports bias only")
    if activation_scales is not None or activation_zero_points is not None:
        raise XQTBackendError("fused W4A16 does not accept activation quantization scales")
    is_decode = 1 <= spec.problem.m <= 8
    if not is_decode and (spec.problem.m % 16 or spec.problem.n % 8):
        raise XQTBackendError(
            "fused SM89 W4A16 requires M in 1..8 (decode) or M%16=0 and N%8=0 (MMA)"
        )
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
    if not activation.is_cuda or activation.dtype != torch.float16 or activation.ndim != 2:
        raise XQTBackendError("fused W4A16 activation must be CUDA fp16 rank-2")
    if tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("fused W4A16 activation shape does not match GemmProblem")
    execution_k = ((spec.problem.k + 15) // 16) * 16
    if execution_k > weight.metadata.padded_k:
        raise XQTBackendError(
            "fused W4A16 padded_k is too small for the K-tile execution padding"
        )
    qweight = weight.qweight
    scales = weight.scales
    zero_points = weight.zero_points
    if not isinstance(qweight, torch.Tensor) or not isinstance(scales, torch.Tensor):
        raise XQTBackendError("fused W4A16 PackedWeight tensors are missing")
    if not qweight.is_cuda or not scales.is_cuda or scales.dtype != torch.float32:
        raise XQTBackendError("fused W4A16 qweight/scales must be CUDA (scales float32)")
    if zero_points is not None and (
        not zero_points.is_cuda or zero_points.dtype != torch.float32
    ):
        raise XQTBackendError("fused W4A16 zero_points must be CUDA float32")
    if bias is not None and tuple(bias.shape) not in {(spec.problem.n,), (1, spec.problem.n)}:
        raise XQTBackendError("fused W4A16 bias must have shape [N] or [1,N]")
    if spec.epilogue.has_bias and bias is None:
        raise XQTBackendError("fused W4A16 epilogue declares bias but no bias was supplied")
    major, minor = torch.cuda.get_device_capability(activation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"fused W4A16 received sm_{major}{minor}")
    if split_k is not None and split_k < 1:
        raise XQTBackendError("fused W4A16 split_k must be >= 1 (1 keeps the full-K path)")
    library = _load_library(artifact)
    bias_device = (
        None
        if bias is None
        else bias.to(device=activation.device, dtype=torch.float32).reshape(-1).contiguous()
    )
    output = torch.empty(
        (spec.problem.m, spec.problem.n), device=activation.device, dtype=torch.float16
    )
    activation_exec = activation
    if execution_k != spec.problem.k:
        activation_exec = F.pad(activation, (0, execution_k - spec.problem.k))
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    if is_decode:
        if split_k is not None and split_k >= 2:
            raise XQTBackendError(
                "fused W4A16 split_k applies to the M%16 MMA path, not the decode path"
            )
        error = _load_decode_function(library)(
            activation_exec.contiguous().data_ptr(),
            qweight.contiguous().data_ptr(),
            scales.contiguous().data_ptr(),
            0 if zero_points is None else zero_points.contiguous().data_ptr(),
            0 if bias_device is None else bias_device.data_ptr(),
            output.data_ptr(),
            spec.problem.m,
            spec.problem.n,
            execution_k,
            weight.metadata.padded_k,
            int(weight.metadata.group_size or 0),
            int(weight.metadata.nibble_signed),
            int(zero_points is not None),
            int(bias_device is not None),
            stream,
        )
    elif split_k is not None and split_k >= 2:
        _, split_count = split_k_partition(execution_k, split_k)
        workspace = torch.empty(
            (split_count, spec.problem.m, spec.problem.n),
            device=activation.device,
            dtype=torch.float32,
        )
        error = _load_splitk_function(library)(
            activation_exec.contiguous().data_ptr(),
            qweight.contiguous().data_ptr(),
            scales.contiguous().data_ptr(),
            0 if zero_points is None else zero_points.contiguous().data_ptr(),
            0 if bias_device is None else bias_device.data_ptr(),
            output.data_ptr(),
            workspace.data_ptr(),
            spec.problem.m,
            spec.problem.n,
            execution_k,
            weight.metadata.padded_k,
            int(weight.metadata.group_size or 0),
            int(weight.metadata.nibble_signed),
            int(zero_points is not None),
            int(bias_device is not None),
            int(split_k),
            stream,
        )
    else:
        error = getattr(library, _SYMBOL)(
            activation_exec.contiguous().data_ptr(),
            qweight.contiguous().data_ptr(),
            scales.contiguous().data_ptr(),
            0 if zero_points is None else zero_points.contiguous().data_ptr(),
            0 if bias_device is None else bias_device.data_ptr(),
            output.data_ptr(),
            spec.problem.m,
            spec.problem.n,
            execution_k,
            weight.metadata.padded_k,
            int(weight.metadata.group_size or 0),
            int(weight.metadata.nibble_signed),
            int(zero_points is not None),
            int(bias_device is not None),
            stream,
        )
    if error != 0:
        raise XQTBackendError(f"fused SM89 W4A16 failed with CUDA error {error}")
    return output


def install_sm89_w4a16_fused_executor(
    registry: GemmKernelRegistry,
    *,
    artifact: str | Path,
    manifest: str | Path | None = None,
) -> bool:
    """Promote only after the fused artifact's correctness manifest gate."""

    if not sm89_w4a16_fused_artifact_available(artifact):
        return False
    manifest_path = Path(manifest) if manifest is not None else artifact_manifest_path(artifact)
    if not artifact_ready_for_execution(
        manifest_path,
        kernel_name="sm89_w4a16_cutlass_fused_mma",
        target_arch="sm_89",
    ):
        return False
    entry = registry.get("sm89_w4a16_cutlass_fused")
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
            implementation="custom_cuda_cutlass_mma",
            executor=lambda *args, **kwargs: sm89_w4a16_fused_executor(
                *args, artifact=artifact, **kwargs
            ),
        )
    )
    return True


__all__ = [
    "install_sm89_w4a16_fused_executor",
    "select_fused_split_k",
    "sm89_w4a16_fused_artifact_available",
    "sm89_w4a16_fused_decode_supported",
    "sm89_w4a16_fused_executor",
    "sm89_w4a16_fused_splitk_supported",
    "split_k_partition",
]
