"""Manifest-gated SM89 grouped W4A16 CUDA adapter.

This module owns a decode-oriented grouped launch for routing-heavy W4A16
workloads. It deliberately takes prepacked expert tensors and a prebuilt device
schedule so Python expert iteration, tensor stacking, and route-table transfer
are not part of the steady-state forward path.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops.gemm.contracts import (
    GroupedGemmProblem,
    PackedWeight,
    PackedWeightMetadata,
    QuantSpec,
)
from xqt.kernels.ops.gemm.grouped_dispatch import (
    GroupedGemmCandidateOutput,
    GroupedGemmDispatchResult,
    GroupedGemmNativeCandidate,
    dispatch_grouped_gemm,
)
from xqt.kernels.ops.gemm.layout import validate_w4a16_packed_weight
from xqt.kernels.ops.gemm.preflight import artifact_ready_for_execution
from xqt.kernels.ops.gemm.reference import reference_packed_grouped_gemm
from xqt.kernels.ops.gemm.tuning_cache import (
    GemmTuningCache,
    GemmTuningLookup,
    build_grouped_tuning_key,
    resolve_tuning_record,
)


_SYMBOL = "xqt_w4a16_grouped_sm89_fp16_run"
_PERSISTENT_SYMBOL = "xqt_w4a16_grouped_sm89_fp16_run_persistent"
_RESOURCE_QUERY_SYMBOL = "xqt_w4a16_grouped_sm89_resource_query"
_PERSISTENT_RESOURCE_QUERY_SYMBOL = (
    "xqt_w4a16_grouped_sm89_persistent_resource_query"
)
_KERNEL_NAME = "sm89_w4a16_grouped_decode"
_ROW_BOUNDS = (1, 2, 4, 8)
_DEFAULT_PERSISTENT_BLOCKS_PER_SM = 4
_MAX_PERSISTENT_BLOCKS_PER_SM = 8
_MAX_ACTIVE_PERSISTENT_BLOCKS = 0  # sentinel: query device max_active_blocks


@dataclass(frozen=True, slots=True)
class Sm89GroupedW4A16PackedWeights:
    """Static contiguous W4A16 payload shared by one set of MoE experts."""

    qweight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    bias: torch.Tensor | None
    expert_count: int
    n: int
    k: int
    padded_k: int
    group_size: int
    nibble_signed: bool

    def __post_init__(self) -> None:
        if self.expert_count <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("grouped W4A16 expert_count/N/K must be positive")
        if self.padded_k < self.k or self.padded_k % 2:
            raise ValueError("grouped W4A16 padded_k must be even and at least logical K")
        if self.group_size not in {32, 64, 128}:
            raise ValueError("grouped W4A16 group_size must be one of 32, 64, or 128")
        packed_columns = self.padded_k // 2
        group_count = (self.padded_k + self.group_size - 1) // self.group_size
        if (
            self.qweight.dtype != torch.uint8
            or self.qweight.ndim != 3
            or tuple(self.qweight.shape) != (self.expert_count, self.n, packed_columns)
        ):
            raise ValueError(
                "grouped W4A16 qweight must be CUDA uint8 [expert_count,N,padded_K/2]"
            )
        if (
            self.scales.dtype != torch.float32
            or self.scales.ndim != 3
            or tuple(self.scales.shape) != (self.expert_count, self.n, group_count)
        ):
            raise ValueError(
                "grouped W4A16 scales must be CUDA float32 [expert_count,N,group_count]"
            )
        values = [self.qweight, self.scales]
        if self.zero_points is not None:
            if (
                self.zero_points.dtype != torch.float32
                or tuple(self.zero_points.shape)
                != (self.expert_count, self.n, group_count)
            ):
                raise ValueError("grouped W4A16 zero_points must match stacked scales")
            values.append(self.zero_points)
        if self.bias is not None:
            if self.bias.dtype != torch.float32 or tuple(self.bias.shape) != (
                self.expert_count,
                self.n,
            ):
                raise ValueError("grouped W4A16 bias must be float32 [expert_count,N]")
            values.append(self.bias)
        device = self.qweight.device
        if any(not value.is_cuda or value.device != device for value in values):
            raise ValueError("grouped W4A16 packed tensors must be co-located on CUDA")
        if any(not value.is_contiguous() for value in values):
            raise ValueError("grouped W4A16 packed tensors must be contiguous")

    @property
    def device(self) -> torch.device:
        """Return the CUDA device carrying all static payload tensors."""

        return self.qweight.device


@dataclass(frozen=True, slots=True)
class Sm89GroupedW4A16Schedule:
    """Device task tables derived once from one grouped routing contract."""

    grouped_problem: GroupedGemmProblem
    m_offsets: torch.Tensor
    task_table: torch.Tensor
    bucket_task_tables: tuple[tuple[int, torch.Tensor], ...]
    output_rows: torch.Tensor | None
    multiprocessor_count: int
    row_tile: int = 8

    def __post_init__(self) -> None:
        if self.row_tile != 8:
            raise ValueError("SM89 grouped W4A16 currently uses row_tile=8")
        if self.multiprocessor_count <= 0:
            raise ValueError(
                "SM89 grouped W4A16 multiprocessor_count must be positive"
            )
        expected_offsets = self.grouped_problem.group_count + 1
        if (
            self.m_offsets.dtype != torch.int32
            or self.m_offsets.ndim != 1
            or int(self.m_offsets.numel()) != expected_offsets
            or not self.m_offsets.is_cuda
            or not self.m_offsets.is_contiguous()
        ):
            raise ValueError("grouped W4A16 m_offsets must be contiguous CUDA int32 [E+1]")
        expected_tasks = self.task_count
        if (
            self.task_table.dtype != torch.int32
            or self.task_table.ndim != 2
            or tuple(self.task_table.shape) != (expected_tasks, 3)
            or not self.task_table.is_cuda
            or not self.task_table.is_contiguous()
        ):
            raise ValueError("grouped W4A16 task_table must be contiguous CUDA int32 [task_count,3]")
        if self.output_rows is not None:
            if (
                self.output_rows.dtype != torch.int32
                or self.output_rows.ndim != 1
                or int(self.output_rows.numel()) != self.total_m
                or not self.output_rows.is_cuda
                or not self.output_rows.is_contiguous()
            ):
                raise ValueError("grouped W4A16 output_rows must be contiguous CUDA int32 [total_M]")
        device = self.m_offsets.device
        if self.task_table.device != device or (
            self.output_rows is not None and self.output_rows.device != device
        ):
            raise ValueError("grouped W4A16 schedule tensors must share one CUDA device")
        observed_bounds: list[int] = []
        observed_tasks = 0
        for row_bound, table in self.bucket_task_tables:
            if row_bound not in _ROW_BOUNDS:
                raise ValueError(f"unsupported grouped W4A16 row_bound: {row_bound}")
            if (
                table.dtype != torch.int32
                or table.ndim != 2
                or int(table.shape[1]) != 3
                or not table.is_cuda
                or not table.is_contiguous()
                or table.device != device
            ):
                raise ValueError("grouped W4A16 bucket task tables must be CUDA int32 [count,3]")
            observed_bounds.append(row_bound)
            observed_tasks += int(table.shape[0])
        if tuple(observed_bounds) != tuple(sorted(set(observed_bounds))):
            raise ValueError("grouped W4A16 bucket task tables must use sorted unique row bounds")
        if observed_tasks != expected_tasks:
            raise ValueError("grouped W4A16 bucket task count must equal direct task count")

    @property
    def device(self) -> torch.device:
        """Return the CUDA device holding the static routing schedule."""

        return self.m_offsets.device

    @property
    def total_m(self) -> int:
        """Return the number of packed activation rows for this routing state."""

        return self.grouped_problem.total_m

    @property
    def task_count(self) -> int:
        """Return the number of eight-row-or-smaller direct-grid tasks."""

        return sum((problem.m + self.row_tile - 1) // self.row_tile for problem in self.grouped_problem.problems)

    @property
    def expert_rows(self) -> tuple[int, ...]:
        """Return per-expert routed token counts, retaining zero-row experts."""

        return tuple(problem.m for problem in self.grouped_problem.problems)

    @property
    def empty_expert_count(self) -> int:
        """Return how many expert identities have no task in this schedule."""

        return sum(rows == 0 for rows in self.expert_rows)


@dataclass(frozen=True, slots=True)
class Sm89GroupedW4A16DispatchReport:
    """Selected grouped scheduler and observable routing/launch facts."""

    artifact: str
    scheduler: str
    launch_count: int
    expert_count: int
    expert_rows: tuple[int, ...]
    m_offsets: tuple[int, ...]
    task_count: int
    row_bounds: tuple[int, ...]
    persistent_blocks_per_sm: int | None
    persistent_grid_blocks: int | None
    n: int
    k: int
    padded_k: int
    group_size: int
    output_scatter: bool
    scatter_mode: str
    scatter_launch_count: int
    workspace_bytes: int
    shape_variant: str
    fallback_chain: tuple[str, ...]
    fallback_reason: str | None
    native: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report without hiding scheduler or scatter details."""

        return {
            "artifact": self.artifact,
            "scheduler": self.scheduler,
            "launch_count": self.launch_count,
            "expert_count": self.expert_count,
            "expert_rows": list(self.expert_rows),
            "m_offsets": list(self.m_offsets),
            "task_count": self.task_count,
            "row_bounds": list(self.row_bounds),
            "persistent_blocks_per_sm": self.persistent_blocks_per_sm,
            "persistent_grid_blocks": self.persistent_grid_blocks,
            "n": self.n,
            "k": self.k,
            "padded_k": self.padded_k,
            "group_size": self.group_size,
            "output_scatter": self.output_scatter,
            "scatter_mode": self.scatter_mode,
            "scatter_launch_count": self.scatter_launch_count,
            "workspace_bytes": self.workspace_bytes,
            "shape_variant": self.shape_variant,
            "fallback_chain": list(self.fallback_chain),
            "fallback_reason": self.fallback_reason,
            "native": self.native,
        }


@dataclass(frozen=True, slots=True)
class Sm89GroupedW4A16DispatchResult:
    """Native output plus the selected grouped CUDA scheduling report."""

    output: torch.Tensor
    report: Sm89GroupedW4A16DispatchReport


@dataclass(frozen=True, slots=True)
class Sm89GroupedW4A16ResourceReport:
    """CUDA function-attribute evidence for one grouped row-bound kernel."""

    artifact: str
    row_bound: int
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
        """Return a JSON-ready resource query record."""

        return {
            "artifact": self.artifact,
            "row_bound": self.row_bound,
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
class Sm89GroupedW4A16PersistentResourceReport:
    """CUDA resource and residency evidence for the persistent kernel."""

    artifact: str
    requested_blocks_per_sm: int
    resident_blocks_per_sm: int
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
        """Return a JSON-ready persistent resource query record."""

        return {
            "artifact": self.artifact,
            "requested_blocks_per_sm": self.requested_blocks_per_sm,
            "resident_blocks_per_sm": self.resident_blocks_per_sm,
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


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    """Load the grouped artifact and bind its fixed C ABI."""

    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 grouped W4A16 artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 grouped W4A16 artifact: {path}") from exc
    if not hasattr(library, _SYMBOL):
        raise XQTBackendError(f"SM89 grouped W4A16 artifact lacks {_SYMBOL}")
    function = getattr(library, _SYMBOL)
    function.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    if hasattr(library, _PERSISTENT_SYMBOL):
        persistent_function = getattr(library, _PERSISTENT_SYMBOL)
        persistent_function.argtypes = function.argtypes
        persistent_function.restype = ctypes.c_int
    if hasattr(library, _RESOURCE_QUERY_SYMBOL):
        resource_query = getattr(library, _RESOURCE_QUERY_SYMBOL)
        resource_query.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        resource_query.restype = ctypes.c_int
    if hasattr(library, _PERSISTENT_RESOURCE_QUERY_SYMBOL):
        persistent_resource_query = getattr(
            library,
            _PERSISTENT_RESOURCE_QUERY_SYMBOL,
        )
        persistent_resource_query.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        persistent_resource_query.restype = ctypes.c_int
    return library


def sm89_grouped_w4a16_artifact_available(artifact: str | Path) -> bool:
    """Return whether the grouped CUDA symbol can be loaded from an artifact."""

    try:
        _load_library(artifact)
    except XQTBackendError:
        return False
    return True


def _offsets_for_problem(grouped_problem: GroupedGemmProblem) -> tuple[int, ...]:
    """Return explicit packed-row offsets even when the logical contract omitted them."""

    if grouped_problem.m_offsets is not None:
        return grouped_problem.m_offsets
    offsets = [0]
    for problem in grouped_problem.problems:
        offsets.append(offsets[-1] + problem.m)
    return tuple(offsets)


def _row_bound(row_count: int) -> int:
    """Map one task's actual row count to a compiled accumulator bound."""

    if row_count <= 0 or row_count > 8:
        raise ValueError("grouped W4A16 task rows must be in [1,8]")
    for bound in _ROW_BOUNDS:
        if row_count <= bound:
            return bound
    raise AssertionError("row bound selection is exhaustive")


def pack_sm89_grouped_w4a16_weights(
    weights: Sequence[PackedWeight],
    *,
    quant: QuantSpec,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> Sm89GroupedW4A16PackedWeights:
    """Stack compatible canonical W4A16 expert weights for one SM89 artifact.

    Packing is intentionally an explicit static step. The forward ABI never
    iterates the input expert list or converts a bias tensor, so callers can
    keep this object across routed-token batches.
    """

    if not weights:
        raise ValueError("grouped W4A16 prepack requires at least one expert weight")
    if quant.weight_dtype != "int4" or quant.activation_dtype != "fp16":
        raise XQTBackendError("SM89 grouped W4A16 requires int4 weights and fp16 activations")
    if quant.output_dtype != "fp16" or quant.weight_granularity not in {"groupwise", "blockwise"}:
        raise XQTBackendError("SM89 grouped W4A16 requires fp16 output and grouped weight scales")
    first = weights[0]
    n, k = (int(first.metadata.logical_shape[0]), int(first.metadata.logical_shape[1]))
    padded_k = int(first.metadata.padded_k)
    group_size = int(first.metadata.group_size or 0)
    nibble_signed = bool(first.metadata.nibble_signed)
    if bias is not None and len(bias) != len(weights):
        raise ValueError("grouped W4A16 bias count must match expert weight count")
    normalized_bias = tuple(None for _ in weights) if bias is None else tuple(bias)
    device: torch.device | None = None
    qweights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    zero_points: list[torch.Tensor] = []
    bias_rows: list[torch.Tensor] = []
    has_bias = any(value is not None for value in normalized_bias)
    for index, weight in enumerate(weights):
        validate_w4a16_packed_weight(weight, spec=quant, logical_shape=(n, k))
        metadata = weight.metadata
        if (
            metadata.padded_k != padded_k
            or metadata.group_size != group_size
            or metadata.nibble_signed != nibble_signed
        ):
            raise ValueError(
                "grouped W4A16 experts must share padded_K, group_size, and nibble signedness"
            )
        if not isinstance(weight.qweight, torch.Tensor) or not isinstance(weight.scales, torch.Tensor):
            raise TypeError("grouped W4A16 experts require tensor qweight and scales")
        if not weight.qweight.is_cuda or not weight.scales.is_cuda:
            raise XQTBackendError("grouped W4A16 prepack requires CUDA expert payloads")
        if weight.qweight.dtype != torch.uint8 or weight.scales.dtype != torch.float32:
            raise XQTBackendError("grouped W4A16 requires uint8 qweight and float32 scales")
        if device is None:
            device = weight.qweight.device
        if weight.qweight.device != device or weight.scales.device != device:
            raise XQTBackendError("grouped W4A16 expert payloads must share one CUDA device")
        qweights.append(weight.qweight.contiguous())
        scales.append(weight.scales.contiguous())
        if quant.weight_zero_point:
            if not isinstance(weight.zero_points, torch.Tensor):
                raise XQTBackendError("asymmetric grouped W4A16 requires zero_points for every expert")
            if weight.zero_points.device != device or weight.zero_points.dtype != torch.float32:
                raise XQTBackendError("grouped W4A16 zero_points must be float32 on the expert device")
            zero_points.append(weight.zero_points.contiguous())
        elif weight.zero_points is not None:
            raise XQTBackendError("symmetric grouped W4A16 experts cannot carry zero_points")
        if has_bias:
            current_bias = normalized_bias[index]
            if current_bias is None:
                bias_rows.append(torch.zeros((n,), device=device, dtype=torch.float32))
            else:
                if current_bias.device != device or tuple(current_bias.shape) not in {(n,), (1, n)}:
                    raise XQTBackendError("grouped W4A16 bias must be on the expert device with shape [N] or [1,N]")
                bias_rows.append(current_bias.to(dtype=torch.float32).reshape(n).contiguous())
    if device is None:
        raise AssertionError("non-empty expert weights must resolve a CUDA device")
    return Sm89GroupedW4A16PackedWeights(
        qweight=torch.stack(qweights, dim=0).contiguous(),
        scales=torch.stack(scales, dim=0).contiguous(),
        zero_points=None if not zero_points else torch.stack(zero_points, dim=0).contiguous(),
        bias=None if not has_bias else torch.stack(bias_rows, dim=0).contiguous(),
        expert_count=len(weights),
        n=n,
        k=k,
        padded_k=padded_k,
        group_size=group_size,
        nibble_signed=nibble_signed,
    )


def pack_sm89_grouped_w4a16_weights_multi_stream(
    weights: Sequence[PackedWeight],
    *,
    quant: QuantSpec,
    bias: Sequence[torch.Tensor | None] | None = None,
    stream_count: int = 4,
) -> Sm89GroupedW4A16PackedWeights:
    """Prepack expert payloads across round-robin CUDA streams.

    The canonical single-stream pack still owns every contract and shape gate;
    the multi-stream path only parallelizes the per-expert normalization copies
    and is rejected if its payload does not match the canonical result exactly.
    """

    if isinstance(stream_count, bool) or int(stream_count) != stream_count or int(stream_count) < 1:
        raise ValueError("stream_count must be a positive int")
    canonical = pack_sm89_grouped_w4a16_weights(weights, quant=quant, bias=bias)
    if int(stream_count) == 1 or len(weights) == 1:
        return canonical
    device = canonical.qweight.device
    normalized_bias = tuple(None for _ in weights) if bias is None else tuple(bias)
    has_bias = any(value is not None for value in normalized_bias)
    stream_count = min(int(stream_count), len(weights))
    streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
    qweight_copies: list[torch.Tensor | None] = [None] * len(weights)
    scale_copies: list[torch.Tensor | None] = [None] * len(weights)
    zero_point_copies: list[torch.Tensor | None] = [None] * len(weights)
    bias_copies: list[torch.Tensor | None] = [None] * len(weights)
    events: list[torch.cuda.Event] = []
    for index, weight in enumerate(weights):
        with torch.cuda.stream(streams[index % stream_count]):
            qweight_copies[index] = weight.qweight.contiguous()
            scale_copies[index] = weight.scales.contiguous()
            if quant.weight_zero_point:
                zero_point_copies[index] = weight.zero_points.contiguous()
            if has_bias:
                current_bias = normalized_bias[index]
                if current_bias is None:
                    bias_copies[index] = torch.zeros(
                        (canonical.n,), device=device, dtype=torch.float32
                    )
                else:
                    bias_copies[index] = current_bias.to(dtype=torch.float32).reshape(
                        canonical.n
                    ).contiguous()
            events.append(torch.cuda.Event())
            events[-1].record(streams[index % stream_count])
    current = torch.cuda.current_stream(device)
    for event in events:
        current.wait_event(event)
    payload = Sm89GroupedW4A16PackedWeights(
        qweight=torch.stack([value for value in qweight_copies if value is not None], dim=0).contiguous(),
        scales=torch.stack([value for value in scale_copies if value is not None], dim=0).contiguous(),
        zero_points=(
            None
            if not quant.weight_zero_point
            else torch.stack(
                [value for value in zero_point_copies if value is not None], dim=0
            ).contiguous()
        ),
        bias=(
            None
            if not has_bias
            else torch.stack([value for value in bias_copies if value is not None], dim=0).contiguous()
        ),
        expert_count=canonical.expert_count,
        n=canonical.n,
        k=canonical.k,
        padded_k=canonical.padded_k,
        group_size=canonical.group_size,
        nibble_signed=canonical.nibble_signed,
    )
    if not torch.equal(payload.qweight, canonical.qweight):
        raise RuntimeError("multi-stream grouped W4A16 qweight disagrees with canonical pack")
    if not torch.equal(payload.scales, canonical.scales):
        raise RuntimeError("multi-stream grouped W4A16 scales disagree with canonical pack")
    for name in ("zero_points", "bias"):
        multi = getattr(payload, name)
        single = getattr(canonical, name)
        if (multi is None) != (single is None):
            raise RuntimeError(f"multi-stream grouped W4A16 {name} presence disagrees")
        if multi is not None and not torch.equal(multi, single):
            raise RuntimeError(f"multi-stream grouped W4A16 {name} disagrees with canonical pack")
    return payload


def warmup_sm89_grouped_w4a16(
    activation: torch.Tensor,
    weights: Sm89GroupedW4A16PackedWeights,
    schedule: Sm89GroupedW4A16Schedule,
    *,
    quant: QuantSpec,
    artifact: str | Path,
    runs: int = 5,
    scheduler: str = "auto",
    persistent_blocks_per_sm: int = _DEFAULT_PERSISTENT_BLOCKS_PER_SM,
    allow_unverified_artifact: bool = False,
) -> dict[str, Any]:
    """Warm instruction/data caches with discardable grouped W4A16 launches."""

    if isinstance(runs, bool) or int(runs) != runs or int(runs) < 1:
        raise ValueError("warmup runs must be a positive int")
    for _ in range(int(runs)):
        sm89_grouped_w4a16_executor(
            activation,
            weights,
            schedule,
            artifact=artifact,
            scheduler=scheduler,
            persistent_blocks_per_sm=persistent_blocks_per_sm,
            allow_unverified_artifact=allow_unverified_artifact,
        )
    torch.cuda.synchronize(activation.device)
    return {
        "runs": int(runs),
        "artifact": str(artifact),
        "scheduler": scheduler,
        "persistent_blocks_per_sm": persistent_blocks_per_sm,
        "allow_unverified_artifact": allow_unverified_artifact,
        "expert_count": weights.expert_count,
    }


def build_sm89_grouped_w4a16_schedule(
    grouped_problem: GroupedGemmProblem,
    *,
    device: torch.device | str,
) -> Sm89GroupedW4A16Schedule:
    """Materialize direct and row-bound bucket task tables on a CUDA device.

    Empty experts are preserved in ``m_offsets`` but produce no task. The
    optional logical output permutation is copied once to CUDA and consumed by
    the native kernel, avoiding a standalone output-scatter launch.
    """

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError(f"grouped W4A16 schedule requires CUDA, got {cuda_device}")
    offsets = _offsets_for_problem(grouped_problem)
    direct_tasks: list[tuple[int, int, int]] = []
    bucketed_tasks: dict[int, list[tuple[int, int, int]]] = {
        bound: [] for bound in _ROW_BOUNDS
    }
    for expert, problem in enumerate(grouped_problem.problems):
        row = offsets[expert]
        remaining = problem.m
        while remaining:
            row_count = min(8, remaining)
            task = (expert, row, row_count)
            direct_tasks.append(task)
            bucketed_tasks[_row_bound(row_count)].append(task)
            row += row_count
            remaining -= row_count
    direct_table = (
        torch.empty((0, 3), dtype=torch.int32, device=cuda_device)
        if not direct_tasks
        else torch.tensor(direct_tasks, dtype=torch.int32, device=cuda_device).contiguous()
    )
    bucket_tables = tuple(
        (
            bound,
            torch.tensor(tasks, dtype=torch.int32, device=cuda_device).contiguous(),
        )
        for bound, tasks in bucketed_tasks.items()
        if tasks
    )
    output_rows = (
        None
        if grouped_problem.output_rows is None
        else torch.tensor(grouped_problem.output_rows, dtype=torch.int32, device=cuda_device).contiguous()
    )
    return Sm89GroupedW4A16Schedule(
        grouped_problem=grouped_problem,
        m_offsets=torch.tensor(offsets, dtype=torch.int32, device=cuda_device).contiguous(),
        task_table=direct_table,
        bucket_task_tables=bucket_tables,
        output_rows=output_rows,
        multiprocessor_count=int(
            torch.cuda.get_device_properties(cuda_device).multi_processor_count
        ),
    )


def query_sm89_grouped_w4a16_resources(
    artifact: str | Path,
    *,
    row_bound: int,
    device: torch.device | str | None = None,
) -> Sm89GroupedW4A16ResourceReport:
    """Query runtime CUDA resource limits for one compiled row-bound variant."""

    if row_bound not in _ROW_BOUNDS:
        raise ValueError(f"row_bound must be one of {_ROW_BOUNDS}, got {row_bound}")
    if not torch.cuda.is_available():
        raise XQTBackendError("grouped W4A16 resource query requires CUDA")
    cuda_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if cuda_device.type != "cuda":
        raise ValueError(f"grouped W4A16 resource query requires CUDA, got {cuda_device}")
    major, minor = torch.cuda.get_device_capability(cuda_device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 grouped W4A16 resource query received sm_{major}{minor}")
    library = _load_library(artifact)
    if not hasattr(library, _RESOURCE_QUERY_SYMBOL):
        raise XQTBackendError("SM89 grouped W4A16 artifact lacks resource query symbol")
    registers = ctypes.c_int()
    static_shared = ctypes.c_int()
    max_blocks = ctypes.c_int()
    error = getattr(library, _RESOURCE_QUERY_SYMBOL)(
        row_bound,
        ctypes.byref(registers),
        ctypes.byref(static_shared),
        ctypes.byref(max_blocks),
    )
    if error != 0:
        raise XQTBackendError(f"SM89 grouped W4A16 resource query failed with CUDA error {error}")
    properties = torch.cuda.get_device_properties(cuda_device)
    max_threads_per_sm = int(properties.max_threads_per_multi_processor)
    active_threads = int(max_blocks.value) * 256
    occupancy = float(active_threads) / float(max_threads_per_sm) if max_threads_per_sm else None
    return Sm89GroupedW4A16ResourceReport(
        artifact=str(Path(artifact).expanduser()),
        row_bound=row_bound,
        block_threads=256,
        registers_per_thread=int(registers.value),
        static_shared_bytes=int(static_shared.value),
        max_active_blocks_per_sm=int(max_blocks.value),
        multiprocessor_count=int(properties.multi_processor_count),
        max_threads_per_sm=max_threads_per_sm,
        occupancy=occupancy,
        device=str(cuda_device),
        device_name=str(torch.cuda.get_device_name(cuda_device)),
        capability=f"sm_{major}{minor}",
    )


def query_sm89_grouped_w4a16_persistent_resources(
    artifact: str | Path,
    *,
    blocks_per_sm: int,
    device: torch.device | str | None = None,
) -> Sm89GroupedW4A16PersistentResourceReport:
    """Query resource limits and requested residency for the persistent kernel."""

    _validate_persistent_blocks_per_sm(blocks_per_sm)
    if not torch.cuda.is_available():
        raise XQTBackendError("grouped W4A16 persistent resource query requires CUDA")
    cuda_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if cuda_device.type != "cuda":
        raise ValueError(
            f"grouped W4A16 persistent resource query requires CUDA, got {cuda_device}"
        )
    major, minor = torch.cuda.get_device_capability(cuda_device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"SM89 grouped W4A16 persistent resource query received sm_{major}{minor}"
        )
    library = _load_library(artifact)
    if not hasattr(library, _PERSISTENT_RESOURCE_QUERY_SYMBOL):
        raise XQTBackendError(
            "SM89 grouped W4A16 artifact lacks persistent resource query symbol"
        )
    registers = ctypes.c_int()
    static_shared = ctypes.c_int()
    max_blocks = ctypes.c_int()
    error = getattr(library, _PERSISTENT_RESOURCE_QUERY_SYMBOL)(
        ctypes.byref(registers),
        ctypes.byref(static_shared),
        ctypes.byref(max_blocks),
    )
    if error != 0:
        raise XQTBackendError(
            "SM89 grouped W4A16 persistent resource query failed with CUDA "
            f"error {error}"
        )
    properties = torch.cuda.get_device_properties(cuda_device)
    max_threads_per_sm = int(properties.max_threads_per_multi_processor)
    if blocks_per_sm == _MAX_ACTIVE_PERSISTENT_BLOCKS:
        resident_blocks = int(max_blocks.value)
    else:
        resident_blocks = min(blocks_per_sm, int(max_blocks.value))
    active_threads = resident_blocks * 256
    occupancy = (
        float(active_threads) / float(max_threads_per_sm)
        if max_threads_per_sm
        else None
    )
    return Sm89GroupedW4A16PersistentResourceReport(
        artifact=str(Path(artifact).expanduser()),
        requested_blocks_per_sm=blocks_per_sm,
        resident_blocks_per_sm=resident_blocks,
        block_threads=256,
        registers_per_thread=int(registers.value),
        static_shared_bytes=int(static_shared.value),
        max_active_blocks_per_sm=int(max_blocks.value),
        multiprocessor_count=int(properties.multi_processor_count),
        max_threads_per_sm=max_threads_per_sm,
        occupancy=occupancy,
        device=str(cuda_device),
        device_name=str(torch.cuda.get_device_name(cuda_device)),
        capability=f"sm_{major}{minor}",
    )


def _validate_persistent_blocks_per_sm(blocks_per_sm: int) -> None:
    if isinstance(blocks_per_sm, bool) or not isinstance(blocks_per_sm, int):
        raise TypeError("persistent_blocks_per_sm must be int")
    if not blocks_per_sm == _MAX_ACTIVE_PERSISTENT_BLOCKS and not (
        1 <= blocks_per_sm <= _MAX_PERSISTENT_BLOCKS_PER_SM
    ):
        raise ValueError(
            "persistent_blocks_per_sm must be in "
            f"[1,{_MAX_PERSISTENT_BLOCKS_PER_SM}] "
            f"or {_MAX_ACTIVE_PERSISTENT_BLOCKS} (max_active)"
        )


def _selected_tables(
    schedule: Sm89GroupedW4A16Schedule, *, scheduler: str
) -> tuple[str, tuple[tuple[int, torch.Tensor], ...]]:
    """Choose a direct one-launch grid or explicit row-bound bucket launches."""

    if scheduler == "direct":
        return "direct_task_grid", ((8, schedule.task_table),)
    if scheduler == "bucketed":
        return "bucketed_direct_task_grid", schedule.bucket_task_tables
    if scheduler == "persistent":
        return "persistent_grid_stride", ()
    if scheduler == "auto":
        if len(schedule.bucket_task_tables) == 1:
            return "bucketed_direct_task_grid", schedule.bucket_task_tables
        # The profiled SM89 decode shape (N=1024) pays less kernel work when
        # each row bound is compiled separately. Keep a single launch for
        # small-N shapes where the extra bucket launches dominate.
        if schedule.grouped_problem.n >= 256:
            return "bucketed_direct_task_grid", schedule.bucket_task_tables
        return "direct_task_grid", ((8, schedule.task_table),)
    raise ValueError(
        "grouped W4A16 scheduler must be 'auto', 'direct', 'bucketed', "
        "or 'persistent'"
    )


def _scheduler_candidate_name(selected_scheduler: str) -> str:
    names = {
        "direct_task_grid": "sm89_grouped_w4a16_direct_task_grid",
        "bucketed_direct_task_grid": "sm89_grouped_w4a16_bucketed_task_grid",
        "persistent_grid_stride": "sm89_grouped_w4a16_persistent_grid_stride",
    }
    try:
        return names[selected_scheduler]
    except KeyError as exc:
        raise ValueError(
            f"unsupported grouped W4A16 selected scheduler: {selected_scheduler}"
        ) from exc


def _resolve_cached_configuration(
    lookup: GemmTuningLookup,
    *,
    fallback_scheduler: str,
    fallback_persistent_blocks_per_sm: int,
) -> tuple[str, int, GemmTuningLookup]:
    if lookup.status != "hit":
        return fallback_scheduler, fallback_persistent_blocks_per_sm, lookup
    assert lookup.record is not None
    cached_scheduler = lookup.selection.get("scheduler")
    cached_blocks = lookup.selection.get("persistent_blocks_per_sm")
    if lookup.record.selected_kernel == _KERNEL_NAME:
        if cached_scheduler in {"direct", "bucketed"}:
            return (
                str(cached_scheduler),
                fallback_persistent_blocks_per_sm,
                lookup,
            )
        if (
            cached_scheduler == "persistent"
            and isinstance(cached_blocks, int)
            and not isinstance(cached_blocks, bool)
            and (
                cached_blocks == _MAX_ACTIVE_PERSISTENT_BLOCKS
                or 1 <= cached_blocks <= _MAX_PERSISTENT_BLOCKS_PER_SM
            )
        ):
            return "persistent", cached_blocks, lookup
    return (
        fallback_scheduler,
        fallback_persistent_blocks_per_sm,
        GemmTuningLookup(
            status="invalid",
            key=lookup.key,
            reason=(
                "cached W4A16 selection must target sm89_w4a16_grouped_decode "
                "with scheduler direct/bucketed, or persistent plus a valid "
                "persistent_blocks_per_sm"
            ),
            source="deterministic_default",
        ),
    )


def sm89_grouped_w4a16_executor(
    activation: torch.Tensor,
    weights: Sm89GroupedW4A16PackedWeights,
    schedule: Sm89GroupedW4A16Schedule,
    *,
    artifact: str | Path,
    scheduler: str = "auto",
    persistent_blocks_per_sm: int = _DEFAULT_PERSISTENT_BLOCKS_PER_SM,
    allow_unverified_artifact: bool = False,
) -> Sm89GroupedW4A16DispatchResult:
    """Run grouped W4A16 with no Python expert loop in the launch path.

    ``allow_unverified_artifact`` exists solely for the dedicated correctness
    promotion workflow. Normal inference must use the manifest-gated default.
    """

    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise XQTBackendError("grouped W4A16 activation must be a rank-2 CUDA fp16 tensor")
    if not activation.is_cuda or activation.dtype != torch.float16:
        raise XQTBackendError("grouped W4A16 activation must be CUDA fp16")
    if activation.device != weights.device or activation.device != schedule.device:
        raise XQTBackendError("grouped W4A16 activation, weights, and schedule must share one device")
    if tuple(activation.shape) != (schedule.total_m, weights.k):
        raise XQTBackendError(
            "grouped W4A16 activation must have packed shape "
            f"[{schedule.total_m},{weights.k}], got {tuple(activation.shape)}"
        )
    if schedule.grouped_problem.group_count != weights.expert_count:
        raise XQTBackendError("grouped W4A16 schedule expert count does not match packed weights")
    if schedule.grouped_problem.n != weights.n or schedule.grouped_problem.k != weights.k:
        raise XQTBackendError("grouped W4A16 schedule N/K does not match packed weights")
    _validate_persistent_blocks_per_sm(persistent_blocks_per_sm)
    effective_blocks_per_sm = persistent_blocks_per_sm
    if persistent_blocks_per_sm == _MAX_ACTIVE_PERSISTENT_BLOCKS:
        library_check = _load_library(artifact)
        if hasattr(library_check, _PERSISTENT_RESOURCE_QUERY_SYMBOL):
            reg = ctypes.c_int()
            shm = ctypes.c_int()
            max_blocks = ctypes.c_int()
            err = getattr(library_check, _PERSISTENT_RESOURCE_QUERY_SYMBOL)(
                ctypes.byref(reg), ctypes.byref(shm), ctypes.byref(max_blocks),
            )
            if err == 0:
                effective_blocks_per_sm = int(max_blocks.value)
                if effective_blocks_per_sm < 1:
                    effective_blocks_per_sm = _DEFAULT_PERSISTENT_BLOCKS_PER_SM
            else:
                effective_blocks_per_sm = _DEFAULT_PERSISTENT_BLOCKS_PER_SM
        else:
            effective_blocks_per_sm = _DEFAULT_PERSISTENT_BLOCKS_PER_SM
    major, minor = torch.cuda.get_device_capability(activation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 grouped W4A16 received sm_{major}{minor}")
    if not allow_unverified_artifact and not artifact_ready_for_execution(
        artifact,
        kernel_name=_KERNEL_NAME,
        target_arch="sm_89",
    ):
        raise XQTBackendError(
            "SM89 grouped W4A16 artifact is not correctness-promoted; run the grouped "
            "correctness gate before inference dispatch"
        )
    library = _load_library(artifact)
    selected_scheduler, selected_tables = _selected_tables(schedule, scheduler=scheduler)
    native_candidate_name = _scheduler_candidate_name(selected_scheduler)
    output = torch.empty((schedule.total_m, weights.n), device=activation.device, dtype=torch.float16)
    if schedule.total_m == 0:
        return Sm89GroupedW4A16DispatchResult(
            output=output,
            report=Sm89GroupedW4A16DispatchReport(
                artifact=str(Path(artifact).expanduser()),
                scheduler="empty",
                launch_count=0,
                expert_count=weights.expert_count,
                expert_rows=schedule.expert_rows,
                m_offsets=_offsets_for_problem(schedule.grouped_problem),
                task_count=0,
                row_bounds=(),
                persistent_blocks_per_sm=None,
                persistent_grid_blocks=None,
                n=weights.n,
                k=weights.k,
                padded_k=weights.padded_k,
                group_size=weights.group_size,
                output_scatter=schedule.output_rows is not None,
                scatter_mode="in_kernel_permutation" if schedule.output_rows is not None else "identity",
                scatter_launch_count=0,
                workspace_bytes=0,
                shape_variant="empty",
                fallback_chain=(native_candidate_name, "grouped_reference"),
                fallback_reason=None,
                native=False,
            ),
        )
    execution_k = ((weights.k + 15) // 16) * 16
    if execution_k > weights.padded_k:
        raise XQTBackendError("grouped W4A16 padded_K is too small for 16-wide execution padding")
    activation_exec = activation if execution_k == weights.k else F.pad(activation, (0, execution_k - weights.k))
    if not activation_exec.is_contiguous():
        activation_exec = activation_exec.contiguous()
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    launches = 0
    selected_bounds: list[int] = []
    persistent_grid_blocks: int | None = None
    if selected_scheduler == "persistent_grid_stride":
        if not hasattr(library, _PERSISTENT_SYMBOL):
            raise XQTBackendError(
                "SM89 grouped W4A16 artifact lacks persistent scheduler symbol"
            )
        work_items = schedule.task_count * weights.n
        persistent_grid_blocks = min(
            work_items,
            schedule.multiprocessor_count * effective_blocks_per_sm,
        )
        error = getattr(library, _PERSISTENT_SYMBOL)(
            activation_exec.data_ptr(),
            weights.qweight.data_ptr(),
            weights.scales.data_ptr(),
            0 if weights.zero_points is None else weights.zero_points.data_ptr(),
            0 if weights.bias is None else weights.bias.data_ptr(),
            schedule.task_table.data_ptr(),
            0 if schedule.output_rows is None else schedule.output_rows.data_ptr(),
            output.data_ptr(),
            schedule.task_count,
            weights.n,
            execution_k,
            weights.padded_k,
            weights.group_size,
            weights.expert_count,
            int(weights.nibble_signed),
            int(weights.zero_points is not None),
            int(weights.bias is not None),
            persistent_grid_blocks,
            stream,
        )
        if error != 0:
            raise XQTBackendError(f"SM89 grouped W4A16 failed with CUDA error {error}")
        launches = 1
        selected_bounds.extend(bound for bound, _ in schedule.bucket_task_tables)
    else:
        for row_bound, task_table in selected_tables:
            if int(task_table.shape[0]) == 0:
                continue
            error = getattr(library, _SYMBOL)(
                activation_exec.data_ptr(),
                weights.qweight.data_ptr(),
                weights.scales.data_ptr(),
                0 if weights.zero_points is None else weights.zero_points.data_ptr(),
                0 if weights.bias is None else weights.bias.data_ptr(),
                task_table.data_ptr(),
                0 if schedule.output_rows is None else schedule.output_rows.data_ptr(),
                output.data_ptr(),
                int(task_table.shape[0]),
                weights.n,
                execution_k,
                weights.padded_k,
                weights.group_size,
                weights.expert_count,
                int(weights.nibble_signed),
                int(weights.zero_points is not None),
                int(weights.bias is not None),
                row_bound,
                stream,
            )
            if error != 0:
                raise XQTBackendError(
                    f"SM89 grouped W4A16 failed with CUDA error {error}"
                )
            launches += 1
            selected_bounds.append(row_bound)
    shape_variant: str
    if selected_scheduler == "persistent_grid_stride":
        shape_variant = "persistent_grid_stride_max_rows_8"
    elif selected_scheduler == "direct_task_grid":
        shape_variant = "direct_max_rows_8"
    elif selected_scheduler == "bucketed_direct_task_grid":
        max_bound = max(selected_bounds) if selected_bounds else 8
        shape_variant = f"bucketed_max_rows_{max_bound}"
    else:
        shape_variant = "empty"
    return Sm89GroupedW4A16DispatchResult(
        output=output,
        report=Sm89GroupedW4A16DispatchReport(
            artifact=str(Path(artifact).expanduser()),
            scheduler=selected_scheduler,
            launch_count=launches,
            expert_count=weights.expert_count,
            expert_rows=schedule.expert_rows,
            m_offsets=_offsets_for_problem(schedule.grouped_problem),
            task_count=schedule.task_count,
            row_bounds=tuple(selected_bounds),
            persistent_blocks_per_sm=(
                effective_blocks_per_sm
                if selected_scheduler == "persistent_grid_stride"
                else None
            ),
            persistent_grid_blocks=persistent_grid_blocks,
            n=weights.n,
            k=weights.k,
            padded_k=weights.padded_k,
            group_size=weights.group_size,
            output_scatter=schedule.output_rows is not None,
            scatter_mode="in_kernel_permutation" if schedule.output_rows is not None else "identity",
            scatter_launch_count=0,
            workspace_bytes=0,
            shape_variant=shape_variant,
            fallback_chain=(native_candidate_name, "grouped_reference"),
            fallback_reason=None,
            native=True,
        ),
    )


def dispatch_sm89_grouped_w4a16(
    activation: torch.Tensor,
    weights: Sm89GroupedW4A16PackedWeights,
    schedule: Sm89GroupedW4A16Schedule,
    *,
    quant: QuantSpec,
    artifact: str | Path,
    scheduler: str = "auto",
    persistent_blocks_per_sm: int = _DEFAULT_PERSISTENT_BLOCKS_PER_SM,
    tuning_cache: GemmTuningCache | None = None,
    allow_reference: bool = True,
    allow_unverified_artifact: bool = False,
) -> GroupedGemmDispatchResult:
    """Dispatch SM89 grouped W4A16 with an explicit grouped reference fallback."""

    if (
        quant.weight_dtype != "int4"
        or quant.activation_dtype != "fp16"
        or quant.output_dtype != "fp16"
        or quant.weight_granularity not in {"groupwise", "blockwise"}
        or quant.group_size != weights.group_size
        or quant.storage_layout != "xqt_int4_nk_v1"
        or quant.weight_zero_point != (weights.zero_points is not None)
        or quant.symmetric != weights.nibble_signed
    ):
        raise ValueError(
            "grouped W4A16 quant contract does not match the packed payload"
        )
    tuning_key = build_grouped_tuning_key(
        kernel_family=_KERNEL_NAME,
        backend="custom_cuda",
        target_arch="sm_89",
        grouped_problem=schedule.grouped_problem,
        quant=quant,
        has_bias=weights.bias is not None,
        persistent=scheduler == "persistent",
    )
    tuning = resolve_tuning_record(
        tuning_cache,
        key=tuning_key,
        artifact=artifact,
        explicit=scheduler != "auto",
    )
    selected_scheduler, selected_persistent_blocks_per_sm, tuning = (
        _resolve_cached_configuration(
            tuning,
            fallback_scheduler=scheduler,
            fallback_persistent_blocks_per_sm=persistent_blocks_per_sm,
        )
    )

    def execute_native() -> GroupedGemmCandidateOutput:
        result = sm89_grouped_w4a16_executor(
            activation,
            weights,
            schedule,
            artifact=artifact,
            scheduler=selected_scheduler,
            persistent_blocks_per_sm=selected_persistent_blocks_per_sm,
            allow_unverified_artifact=allow_unverified_artifact,
        )
        return GroupedGemmCandidateOutput(
            output=result.output,
            details=result.report.to_dict(),
            native=result.report.native,
        )

    def execute_reference() -> torch.Tensor:
        reference_weights = tuple(
            PackedWeight(
                qweight=weights.qweight[index],
                scales=weights.scales[index],
                zero_points=(
                    None
                    if weights.zero_points is None
                    else weights.zero_points[index]
                ),
                metadata=PackedWeightMetadata(
                    logical_shape=(weights.n, weights.k),
                    storage_layout=quant.storage_layout,
                    pack_version=quant.pack_version,
                    weight_dtype=quant.weight_dtype,
                    padded_k=weights.padded_k,
                    group_size=weights.group_size,
                    packed_bits=4,
                    nibble_order="low_high",
                    nibble_signed=weights.nibble_signed,
                ),
            )
            for index in range(weights.expert_count)
        )
        reference_bias = (
            tuple(None for _ in range(weights.expert_count))
            if weights.bias is None
            else tuple(
                weights.bias[index]
                for index in range(weights.expert_count)
            )
        )
        return reference_packed_grouped_gemm(
            schedule.grouped_problem,
            activation,
            reference_weights,
            quant_specs=quant,
            bias=reference_bias,
        )

    candidate = GroupedGemmNativeCandidate(
        name=_KERNEL_NAME,
        backend="custom_cuda",
        executor=execute_native,
    )
    return dispatch_grouped_gemm(
        activation,
        None,
        grouped_problem=schedule.grouped_problem,
        quant_specs=quant,
        candidates=(candidate,),
        requested_kernel=_KERNEL_NAME,
        allow_reference=allow_reference,
        reference_executor=execute_reference,
        tuning=tuning,
    )


__all__ = [
    "Sm89GroupedW4A16DispatchReport",
    "Sm89GroupedW4A16DispatchResult",
    "Sm89GroupedW4A16PackedWeights",
    "Sm89GroupedW4A16PersistentResourceReport",
    "Sm89GroupedW4A16ResourceReport",
    "Sm89GroupedW4A16Schedule",
    "_MAX_ACTIVE_PERSISTENT_BLOCKS",
    "build_sm89_grouped_w4a16_schedule",
    "dispatch_sm89_grouped_w4a16",
    "pack_sm89_grouped_w4a16_weights",
    "pack_sm89_grouped_w4a16_weights_multi_stream",
    "query_sm89_grouped_w4a16_resources",
    "query_sm89_grouped_w4a16_persistent_resources",
    "sm89_grouped_w4a16_artifact_available",
    "sm89_grouped_w4a16_executor",
    "warmup_sm89_grouped_w4a16",
]
