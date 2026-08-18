"""Manifest-gated SM89 FP8 CUTLASS GEMM adapter."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import weakref

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.gemm.common.contracts import GemmSpec, PackedWeight
from xqt.gemm.common.preflight import artifact_manifest_path, artifact_ready_for_execution
from xqt.gemm.common.registry import GemmCapability, GemmKernelRegistration, GemmKernelRegistry
from xqt.gemm.common.fp8 import fp8_block_count, fp8_format_spec, validate_fp8_block_k


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
_BLOCKWISE_SPLITK_SYMBOLS = {
    ("fp8_e4m3", "fp16"): "fp8_sm89_e4m3_fp16_run_blockwise_splitk",
    ("fp8_e4m3", "bf16"): "fp8_sm89_e4m3_bf16_run_blockwise_splitk",
    ("fp8_e5m2", "fp16"): "fp8_sm89_e5m2_fp16_run_blockwise_splitk",
    ("fp8_e5m2", "bf16"): "fp8_sm89_e5m2_bf16_run_blockwise_splitk",
}
_RESOURCE_QUERY_SYMBOL = "fp8_sm89_blockwise_resource_query"
_BLOCKWISE_RESOURCE_FORMATS = {"fp8_e4m3": 0, "fp8_e5m2": 1}
_BLOCKWISE_RESOURCE_OUTPUTS = {"fp16": 0, "bf16": 1}
_BLOCKWISE_ALIGNMENT = (16, 8, 32)
_TENSORWISE_SCALE_CACHE: dict[int, "_TensorwiseScaleCacheEntry"] = {}


@dataclass(frozen=True, slots=True)
class Sm89Fp8BlockwiseResourceReport:
    """Runtime CUDA resource and occupancy evidence for one blockwise variant."""

    artifact: str
    format_name: str
    output_dtype: str
    block_k: int
    block_threads: int
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
            "format_name": self.format_name,
            "output_dtype": self.output_dtype,
            "block_k": self.block_k,
            "block_threads": self.block_threads,
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


@dataclass(frozen=True, slots=True)
class _TensorwiseScaleCacheEntry:
    """A validated host scalar keyed by one static scale tensor."""

    tensor_ref: weakref.ReferenceType[torch.Tensor]
    version: int
    data_ptr: int
    scalar: float


def _tensor_version(value: torch.Tensor) -> int | None:
    """Return a tensor mutation version when PyTorch tracks one."""

    try:
        return int(value._version)
    except RuntimeError:
        # Inference tensors intentionally do not expose a version counter.  Do
        # not cache them because an in-place mutation cannot be invalidated.
        return None


def _cached_tensorwise_scale(value: torch.Tensor, *, version: int) -> float | None:
    """Return a valid cached scalar when tensor identity and storage still match."""

    key = id(value)
    entry = _TENSORWISE_SCALE_CACHE.get(key)
    if entry is None:
        return None
    if (
        entry.tensor_ref() is value
        and entry.version == version
        and entry.data_ptr == value.data_ptr()
    ):
        return entry.scalar
    _TENSORWISE_SCALE_CACHE.pop(key, None)
    return None


def _cache_tensorwise_scale(value: torch.Tensor, *, version: int, scalar: float) -> None:
    """Cache one validated static scale without retaining its tensor lifetime."""

    key = id(value)

    def _remove(dead_ref: weakref.ReferenceType[torch.Tensor]) -> None:
        entry = _TENSORWISE_SCALE_CACHE.get(key)
        if entry is not None and entry.tensor_ref is dead_ref:
            _TENSORWISE_SCALE_CACHE.pop(key, None)

    _TENSORWISE_SCALE_CACHE[key] = _TensorwiseScaleCacheEntry(
        tensor_ref=weakref.ref(value, _remove),
        version=version,
        data_ptr=value.data_ptr(),
        scalar=scalar,
    )


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
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        function.restype = ctypes.c_int
    for symbol in _BLOCKWISE_SPLITK_SYMBOLS.values():
        if not hasattr(library, symbol):
            raise XQTBackendError(f"SM89 FP8 artifact lacks {symbol}")
        function = getattr(library, symbol)
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
            ctypes.c_float,
        ]
        function.restype = ctypes.c_int
    if hasattr(library, _RESOURCE_QUERY_SYMBOL):
        function = getattr(library, _RESOURCE_QUERY_SYMBOL)
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


def sm89_fp8_artifact_available(artifact: str | Path) -> bool:
    """Return whether all four format/output symbols are loadable."""

    try:
        _load_library(artifact)
    except XQTBackendError:
        return False
    return True


def fp8_blockwise_split_k_partition(
    padded_k: int, block_k: int, split_k: int
) -> tuple[int, int]:
    """Return (k_per_split, split_count) for one blockwise split-K launch.

    Splits are block_k aligned so a scale block never straddles a split
    boundary; the reduction pass is then a plain deterministic sum.
    """

    block_k = validate_fp8_block_k(block_k)
    if int(padded_k) <= 0 or int(padded_k) % 32 != 0:
        raise ValueError(f"blockwise split-K requires padded_k % 32 == 0, got {padded_k!r}")
    if int(split_k) < 2:
        raise ValueError(f"blockwise split-K requires split_k >= 2, got {split_k!r}")
    execution_blocks = (int(padded_k) + block_k - 1) // block_k
    blocks_per_split = -(-execution_blocks // int(split_k))
    k_per_split = blocks_per_split * block_k
    split_count = -(-int(padded_k) // k_per_split)
    return k_per_split, split_count


def select_fp8_blockwise_split_k(
    *,
    m: int,
    n: int,
    k: int,
    block_k: int,
    sm_count: int,
    max_split: int = 16,
    k_tiles_per_split: int = 16,
) -> int:
    """Shape heuristic for the opt-in blockwise split-K path.

    The full-K warp-MMA CTA is one warp per 16x8 output tile, so K runs
    serially inside a single warp.  Split-K is useful when the K loop is long
    relative to the available (M, N) parallelism.  The heuristic asks for at
    least ``k_tiles_per_split`` 32-wide K tiles per split and never exceeds
    ``max_split``; it returns 1 (full-K) when the K loop is already short or
    the (M, N) grid alone fills the device.  ``split_k=1`` must route to the
    full-K ABI.  This mirrors ``select_fused_split_k`` for the fused W4A16
    path; the blockwise partition itself is aligned to ``block_k`` by
    ``fp8_blockwise_split_k_partition``.
    """

    validate_fp8_block_k(block_k)
    if m <= 0 or n <= 0 or k <= 0 or sm_count <= 0:
        raise XQTBackendError(
            "select_fp8_blockwise_split_k requires positive m/n/k/sm_count"
        )
    k_tiles = (k + 31) // 32
    if k_tiles <= k_tiles_per_split:
        return 1
    base_blocks = ((m + 15) // 16) * ((n + 7) // 8)
    if base_blocks >= 2 * sm_count and k_tiles <= 2 * k_tiles_per_split:
        return 1
    split_k = -(-k_tiles // k_tiles_per_split)
    return max(1, min(split_k, max_split, k_tiles))


def query_sm89_fp8_blockwise_resources(
    artifact: str | Path,
    *,
    format_name: str,
    output_dtype: str,
    block_k: int,
    device: torch.device | str | None = None,
) -> Sm89Fp8BlockwiseResourceReport:
    """Query CUDA function attributes and occupancy for one blockwise variant.

    The query calls CUDA runtime introspection only; it does not launch a
    kernel.  ``block_k`` selects the templated mainloop unroll actually used by
    the launch ABI.
    """

    block_k = validate_fp8_block_k(block_k)
    try:
        format_id = _BLOCKWISE_RESOURCE_FORMATS[str(format_name)]
        output_id = _BLOCKWISE_RESOURCE_OUTPUTS[str(output_dtype)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported FP8 blockwise resource variant: {format_name!r}/{output_dtype!r}"
        ) from exc
    if not torch.cuda.is_available():
        raise XQTBackendError("FP8 blockwise resource query requires CUDA")
    cuda_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if cuda_device.type != "cuda":
        raise ValueError(f"FP8 blockwise resource query requires a CUDA device, got {cuda_device}")
    major, minor = torch.cuda.get_device_capability(cuda_device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 FP8 resource query received sm_{major}{minor}")
    library = _load_library(artifact)
    if not hasattr(library, _RESOURCE_QUERY_SYMBOL):
        raise XQTBackendError("SM89 FP8 artifact lacks resource query symbol")
    registers = ctypes.c_int()
    static_shared = ctypes.c_int()
    max_blocks = ctypes.c_int()
    error = getattr(library, _RESOURCE_QUERY_SYMBOL)(
        format_id,
        output_id,
        block_k,
        ctypes.byref(registers),
        ctypes.byref(static_shared),
        ctypes.byref(max_blocks),
    )
    if error != 0:
        raise XQTBackendError(f"SM89 FP8 resource query failed with CUDA error {error}")
    properties = torch.cuda.get_device_properties(cuda_device)
    multiprocessors = int(properties.multi_processor_count)
    max_threads_per_sm = int(properties.max_threads_per_multi_processor)
    block_threads = 32
    active_threads = max_blocks.value * block_threads
    occupancy = (
        float(active_threads) / float(max_threads_per_sm)
        if max_threads_per_sm > 0
        else None
    )
    return Sm89Fp8BlockwiseResourceReport(
        artifact=str(Path(artifact).expanduser()),
        format_name=str(format_name),
        output_dtype=str(output_dtype),
        block_k=block_k,
        block_threads=block_threads,
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
    version = _tensor_version(value)
    if version is not None:
        cached = _cached_tensorwise_scale(value, version=version)
        if cached is not None:
            return cached
    if not bool(torch.isfinite(scalar).all()) or bool((scalar <= 0).any()):
        raise XQTBackendError(f"{name} must be finite and positive")
    result = float(scalar.item())
    if version is not None:
        _cache_tensorwise_scale(value, version=version, scalar=result)
    return result


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
    split_k: int | None = None,
) -> torch.Tensor:
    """Run SM89 FP8 native paths; unsupported scale modes raise for fallback.

    The tensorwise path pads bytes to (8, 8, 32) and runs the CUTLASS device
    GEMM.  The blockwise path pads bytes to (16, 8, 32), pads scale rows to the
    padded M/N, and runs the K-block warp MMA mainloop; ``split_k>=2`` opts
    into the block_k-aligned split-K workspace/reduction ABI, while
    ``split_k=None`` or ``1`` keeps the full-K ABI.  Zero padding contributes
    exactly zero to every MMA tile and padded scale rows are multiplied into
    discarded output rows only.
    """

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
    if split_k is not None:
        if not isinstance(split_k, bool) and int(split_k) >= 1:
            split_k = int(split_k)
        else:
            raise XQTBackendError("FP8 split_k must be an int >= 1")
    if tensorwise and split_k not in {None, 1}:
        raise XQTBackendError("FP8 split_k applies to the blockwise path, not tensorwise")
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
        alignment_m, alignment_n, alignment_k = _BLOCKWISE_ALIGNMENT
        padded_m = (spec.problem.m + alignment_m - 1) // alignment_m * alignment_m
        padded_n = (spec.problem.n + alignment_n - 1) // alignment_n * alignment_n
        padded_k = (spec.problem.k + alignment_k - 1) // alignment_k * alignment_k
        a_padded = a_bytes
        if tuple(a_bytes.shape) != (padded_m, padded_k):
            a_padded = F.pad(a_bytes, (0, padded_k - spec.problem.k, 0, padded_m - spec.problem.m))
        w_padded = w_bytes
        if tuple(w_bytes.shape) != (padded_n, padded_k):
            w_padded = F.pad(w_bytes, (0, padded_k - spec.problem.k, 0, padded_n - spec.problem.n))
        a_scale_padded = a_scale
        if padded_m != spec.problem.m:
            a_scale_padded = F.pad(a_scale, (0, 0, 0, padded_m - spec.problem.m))
        w_scale_padded = w_scale
        if padded_n != spec.problem.n:
            w_scale_padded = F.pad(w_scale, (0, 0, 0, padded_n - spec.problem.n))
        c_source = None
        if has_c:
            c_source = torch.zeros(
                (padded_m, padded_n),
                device=activation.device,
                dtype=output_dtype,
            )
            if bias is not None:
                c_source[:, : spec.problem.n].add_(
                    bias.to(device=activation.device, dtype=output_dtype).reshape(1, -1)
                )
            if residual is not None:
                c_source[: spec.problem.m, : spec.problem.n].add_(
                    residual.to(device=activation.device, dtype=output_dtype)
                )
        output = torch.empty(
            (padded_m, padded_n), device=activation.device, dtype=output_dtype
        )
        a_runtime = a_padded.contiguous()
        w_runtime = w_padded.contiguous()
        a_scale_runtime = a_scale_padded.contiguous()
        w_scale_runtime = w_scale_padded.contiguous()
        if split_k is not None and split_k >= 2:
            _, split_count = fp8_blockwise_split_k_partition(padded_k, block_k, split_k)
            workspace = torch.empty(
                (split_count, padded_m, padded_n),
                device=activation.device,
                dtype=torch.float32,
            )
            symbol = _BLOCKWISE_SPLITK_SYMBOLS[(quant.weight_dtype, quant.output_dtype)]
            error = getattr(library, symbol)(
                a_runtime.data_ptr(),
                w_runtime.data_ptr(),
                a_scale_runtime.data_ptr(),
                w_scale_runtime.data_ptr(),
                None if c_source is None else c_source.data_ptr(),
                output.data_ptr(),
                workspace.data_ptr(),
                padded_m,
                padded_n,
                padded_k,
                block_k,
                split_k,
                ctypes.c_float(1.0 if has_c else 0.0),
            )
        else:
            symbol = _BLOCKWISE_SYMBOLS[(quant.weight_dtype, quant.output_dtype)]
            error = getattr(library, symbol)(
                a_runtime.data_ptr(),
                w_runtime.data_ptr(),
                a_scale_runtime.data_ptr(),
                w_scale_runtime.data_ptr(),
                None if c_source is None else c_source.data_ptr(),
                output.data_ptr(),
                padded_m,
                padded_n,
                padded_k,
                block_k,
                ctypes.c_float(1.0 if has_c else 0.0),
            )
        if error != 0:
            raise XQTBackendError(f"SM89 FP8 blockwise GEMM failed with CUDA error {error}")
        return output[: spec.problem.m, : spec.problem.n]

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
    "Sm89Fp8BlockwiseResourceReport",
    "fp8_blockwise_split_k_partition",
    "fp8_sm89_executor",
    "install_sm89_fp8_executors",
    "query_sm89_fp8_blockwise_resources",
    "select_fp8_blockwise_split_k",
    "sm89_fp8_artifact_available",
]
