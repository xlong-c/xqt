"""Manifest-gated SM89 grouped W8A8 adapter.

The native ABI consumes already quantized activations and static expert
payloads.  Routing tables are materialized once, so the steady-state call is
one CUDA launch for all non-empty experts and optional in-kernel scattering.
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

from xqt.kernels.ops.gemm.contracts import GroupedGemmProblem, PackedWeight, QuantSpec
from xqt.kernels.ops.gemm.grouped_dispatch import (
    GroupedGemmCandidateOutput,
    GroupedGemmDispatchResult,
    GroupedGemmNativeCandidate,
    dispatch_grouped_gemm,
)
from xqt.kernels.ops.gemm.preflight import artifact_ready_for_execution
from xqt.kernels.ops.gemm.reference import reference_packed_grouped_gemm
from xqt.kernels.ops.gemm.tuning_cache import (
    GemmTuningCache,
    GemmTuningLookup,
    build_grouped_tuning_key,
    resolve_tuning_record,
)


_SYMBOL = "xqt_w8a8_grouped_sm89_fp16_run"
_RESOURCE_QUERY_SYMBOL = "xqt_w8a8_grouped_sm89_resource_query"
_KERNEL_NAME = "sm89_w8a8_grouped_mma"
_ROW_TILE = 8
_K_TILE = 32


@dataclass(frozen=True, slots=True)
class Sm89GroupedW8A8PackedWeights:
    """Static contiguous INT8 expert payload for grouped W8A8."""

    weights_nk: torch.Tensor
    weight_scales: torch.Tensor
    bias: torch.Tensor | None
    expert_count: int
    n: int
    k: int
    padded_k: int

    def __post_init__(self) -> None:
        if self.expert_count <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("grouped W8A8 expert_count/N/K must be positive")
        if self.padded_k < self.k or self.padded_k % _K_TILE:
            raise ValueError("grouped W8A8 padded_k must be a positive multiple of 32")
        if (
            self.weights_nk.dtype != torch.int8
            or self.weights_nk.ndim != 3
            or tuple(self.weights_nk.shape) != (self.expert_count, self.n, self.padded_k)
        ):
            raise ValueError("grouped W8A8 weights_nk must be CUDA int8 [E,N,padded_K]")
        if (
            self.weight_scales.dtype != torch.float32
            or self.weight_scales.ndim != 2
            or tuple(self.weight_scales.shape) != (self.expert_count, self.n)
        ):
            raise ValueError("grouped W8A8 weight_scales must be CUDA float32 [E,N]")
        values = [self.weights_nk, self.weight_scales]
        if self.bias is not None:
            if self.bias.dtype != torch.float32 or tuple(self.bias.shape) != (
                self.expert_count,
                self.n,
            ):
                raise ValueError("grouped W8A8 bias must be CUDA float32 [E,N]")
            values.append(self.bias)
        device = self.weights_nk.device
        if any(not value.is_cuda or value.device != device for value in values):
            raise ValueError("grouped W8A8 payload tensors must share one CUDA device")
        if any(not value.is_contiguous() for value in values):
            raise ValueError("grouped W8A8 payload tensors must be contiguous")

    @property
    def device(self) -> torch.device:
        """Return the device carrying the static expert payload."""

        return self.weights_nk.device


@dataclass(frozen=True, slots=True)
class Sm89GroupedW8A8Schedule:
    """Device task table for one grouped routing state."""

    grouped_problem: GroupedGemmProblem
    m_offsets: torch.Tensor
    task_table: torch.Tensor
    output_rows: torch.Tensor | None
    row_tile: int = _ROW_TILE

    def __post_init__(self) -> None:
        if self.row_tile != _ROW_TILE:
            raise ValueError("SM89 grouped W8A8 currently uses row_tile=8")
        expected_offsets = self.grouped_problem.group_count + 1
        if (
            self.m_offsets.dtype != torch.int32
            or self.m_offsets.ndim != 1
            or int(self.m_offsets.numel()) != expected_offsets
            or not self.m_offsets.is_cuda
            or not self.m_offsets.is_contiguous()
        ):
            raise ValueError("grouped W8A8 m_offsets must be CUDA int32 [E+1]")
        if (
            self.task_table.dtype != torch.int32
            or self.task_table.ndim != 2
            or tuple(self.task_table.shape) != (self.task_count, 3)
            or not self.task_table.is_cuda
            or not self.task_table.is_contiguous()
        ):
            raise ValueError("grouped W8A8 task_table must be CUDA int32 [task_count,3]")
        if self.output_rows is not None:
            if (
                self.output_rows.dtype != torch.int32
                or self.output_rows.ndim != 1
                or int(self.output_rows.numel()) != self.total_m
                or not self.output_rows.is_cuda
                or not self.output_rows.is_contiguous()
            ):
                raise ValueError("grouped W8A8 output_rows must be CUDA int32 [total_M]")
        device = self.m_offsets.device
        if self.task_table.device != device or (
            self.output_rows is not None and self.output_rows.device != device
        ):
            raise ValueError("grouped W8A8 schedule tensors must share one CUDA device")

    @property
    def device(self) -> torch.device:
        """Return the device holding the routing schedule."""

        return self.m_offsets.device

    @property
    def total_m(self) -> int:
        """Return the packed activation row count."""

        return self.grouped_problem.total_m

    @property
    def task_count(self) -> int:
        """Return the number of row tiles in the task table."""

        return sum((problem.m + _ROW_TILE - 1) // _ROW_TILE for problem in self.grouped_problem.problems)

    @property
    def expert_rows(self) -> tuple[int, ...]:
        """Return routed token counts, retaining empty experts."""

        return tuple(problem.m for problem in self.grouped_problem.problems)


@dataclass(frozen=True, slots=True)
class Sm89GroupedW8A8DispatchReport:
    """Observable grouped scheduler and scale contract."""

    artifact: str
    scheduler: str
    launch_count: int
    expert_count: int
    expert_rows: tuple[int, ...]
    m_offsets: tuple[int, ...]
    task_count: int
    n: int
    k: int
    padded_k: int
    warps_per_block: int
    weight_scale_mode: str
    activation_scale_mode: str
    output_scatter: bool
    scatter_mode: str
    scatter_launch_count: int
    workspace_bytes: int
    fallback_chain: tuple[str, ...]
    fallback_reason: str | None
    native: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""

        return {
            "artifact": self.artifact,
            "scheduler": self.scheduler,
            "launch_count": self.launch_count,
            "expert_count": self.expert_count,
            "expert_rows": list(self.expert_rows),
            "m_offsets": list(self.m_offsets),
            "task_count": self.task_count,
            "n": self.n,
            "k": self.k,
            "padded_k": self.padded_k,
            "warps_per_block": self.warps_per_block,
            "weight_scale_mode": self.weight_scale_mode,
            "activation_scale_mode": self.activation_scale_mode,
            "output_scatter": self.output_scatter,
            "scatter_mode": self.scatter_mode,
            "scatter_launch_count": self.scatter_launch_count,
            "workspace_bytes": self.workspace_bytes,
            "fallback_chain": list(self.fallback_chain),
            "fallback_reason": self.fallback_reason,
            "native": self.native,
        }


@dataclass(frozen=True, slots=True)
class Sm89GroupedW8A8DispatchResult:
    """Native grouped output together with its dispatch report."""

    output: torch.Tensor
    report: Sm89GroupedW8A8DispatchReport


@dataclass(frozen=True, slots=True)
class Sm89GroupedW8A8ResourceReport:
    """CUDA function-attribute evidence for the grouped MMA kernel."""

    artifact: str
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
        """Return a JSON-ready resource record."""

        return {
            "artifact": self.artifact,
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
    """Load the artifact and bind its fixed C ABI."""

    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 grouped W8A8 artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 grouped W8A8 artifact: {path}") from exc
    if not hasattr(library, _SYMBOL):
        raise XQTBackendError(f"SM89 grouped W8A8 artifact lacks {_SYMBOL}")
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
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    if hasattr(library, _RESOURCE_QUERY_SYMBOL):
        query = getattr(library, _RESOURCE_QUERY_SYMBOL)
        query.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        query.restype = ctypes.c_int
    return library


def sm89_grouped_w8a8_artifact_available(artifact: str | Path) -> bool:
    """Return whether the grouped W8A8 symbol can be loaded."""

    try:
        _load_library(artifact)
    except XQTBackendError:
        return False
    return True


def _offsets_for_problem(grouped_problem: GroupedGemmProblem) -> tuple[int, ...]:
    """Return packed-row offsets for the grouped contract."""

    if grouped_problem.m_offsets is not None:
        return grouped_problem.m_offsets
    offsets = [0]
    for problem in grouped_problem.problems:
        offsets.append(offsets[-1] + problem.m)
    return tuple(offsets)


def _as_canonical_int8(weight: PackedWeight | torch.Tensor) -> torch.Tensor:
    """Extract a canonical [N,K] INT8 tensor from a packed weight."""

    if isinstance(weight, PackedWeight):
        candidate = weight.canonical_qweight
        if not isinstance(candidate, torch.Tensor):
            candidate = weight.qweight
    else:
        candidate = weight
    if not isinstance(candidate, torch.Tensor) or candidate.ndim != 2 or candidate.dtype != torch.int8:
        raise XQTBackendError("grouped W8A8 weights require rank-2 CUDA int8 tensors")
    return candidate


def pack_sm89_grouped_w8a8_weights(
    weights: Sequence[PackedWeight | torch.Tensor],
    *,
    quant: QuantSpec,
    weight_scales: Sequence[torch.Tensor] | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> Sm89GroupedW8A8PackedWeights:
    """Stack compatible canonical INT8 expert weights and per-channel scales."""

    if not weights:
        raise ValueError("grouped W8A8 prepack requires at least one expert weight")
    if quant.weight_dtype != "int8" or quant.activation_dtype != "int8":
        raise XQTBackendError("SM89 grouped W8A8 requires int8 weights and int8 activations")
    if quant.output_dtype != "fp16":
        raise XQTBackendError("SM89 grouped W8A8 currently outputs fp16")
    if quant.weight_granularity != "per_channel":
        raise XQTBackendError("SM89 grouped W8A8 requires per-channel weight scales")
    if quant.activation_granularity not in {"per_token", "per_tensor"}:
        raise XQTBackendError("SM89 grouped W8A8 supports per-token or per-tensor activation scales")
    if not quant.symmetric or quant.weight_zero_point or quant.activation_zero_point:
        raise XQTBackendError("SM89 grouped W8A8 requires symmetric zero-point-free INT8")
    if bias is not None and len(bias) != len(weights):
        raise ValueError("grouped W8A8 bias count must match expert count")
    if weight_scales is not None and len(weight_scales) != len(weights):
        raise ValueError("grouped W8A8 scale count must match expert count")

    first = _as_canonical_int8(weights[0])
    n, k = (int(first.shape[0]), int(first.shape[1]))
    padded_k = ((k + _K_TILE - 1) // _K_TILE) * _K_TILE
    device: torch.device | None = None
    stacked_weights: list[torch.Tensor] = []
    stacked_scales: list[torch.Tensor] = []
    stacked_bias: list[torch.Tensor] = []
    normalized_bias = tuple(None for _ in weights) if bias is None else tuple(bias)
    has_bias = any(value is not None for value in normalized_bias)
    for index, weight in enumerate(weights):
        current = _as_canonical_int8(weight)
        if tuple(current.shape) != (n, k):
            raise ValueError("grouped W8A8 experts must share logical [N,K]")
        if not current.is_cuda:
            raise XQTBackendError("grouped W8A8 prepack requires CUDA expert payloads")
        if device is None:
            device = current.device
        if current.device != device:
            raise XQTBackendError("grouped W8A8 experts must share one CUDA device")
        current = current.contiguous()
        if padded_k != k:
            current = F.pad(current, (0, padded_k - k))
        stacked_weights.append(current.contiguous())
        scale = weight_scales[index] if weight_scales is not None else (
            weights[index].scales if isinstance(weights[index], PackedWeight) else None
        )
        if not isinstance(scale, torch.Tensor) or int(scale.numel()) != n:
            raise XQTBackendError("grouped W8A8 requires one float scale per output channel")
        if scale.device != device:
            raise XQTBackendError("grouped W8A8 scales must share the expert device")
        stacked_scales.append(scale.detach().to(dtype=torch.float32).reshape(n).contiguous())
        if has_bias:
            current_bias = normalized_bias[index]
            if current_bias is None:
                stacked_bias.append(torch.zeros((n,), device=device, dtype=torch.float32))
            else:
                if current_bias.device != device or int(current_bias.numel()) != n:
                    raise XQTBackendError("grouped W8A8 bias must have N elements on the expert device")
                stacked_bias.append(current_bias.detach().to(dtype=torch.float32).reshape(n).contiguous())
    if device is None:
        raise AssertionError("non-empty expert weights must resolve a CUDA device")
    return Sm89GroupedW8A8PackedWeights(
        weights_nk=torch.stack(stacked_weights, dim=0).contiguous(),
        weight_scales=torch.stack(stacked_scales, dim=0).contiguous(),
        bias=None if not has_bias else torch.stack(stacked_bias, dim=0).contiguous(),
        expert_count=len(weights),
        n=n,
        k=k,
        padded_k=padded_k,
    )


def build_sm89_grouped_w8a8_schedule(
    grouped_problem: GroupedGemmProblem,
    *,
    device: torch.device | str,
) -> Sm89GroupedW8A8Schedule:
    """Materialize one direct task grid and optional output permutation."""

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError(f"grouped W8A8 schedule requires CUDA, got {cuda_device}")
    offsets = _offsets_for_problem(grouped_problem)
    tasks: list[tuple[int, int, int]] = []
    for expert, problem in enumerate(grouped_problem.problems):
        row = offsets[expert]
        remaining = problem.m
        while remaining:
            count = min(_ROW_TILE, remaining)
            tasks.append((expert, row, count))
            row += count
            remaining -= count
    task_table = (
        torch.empty((0, 3), dtype=torch.int32, device=cuda_device)
        if not tasks
        else torch.tensor(tasks, dtype=torch.int32, device=cuda_device).contiguous()
    )
    output_rows = (
        None
        if grouped_problem.output_rows is None
        else torch.tensor(grouped_problem.output_rows, dtype=torch.int32, device=cuda_device).contiguous()
    )
    return Sm89GroupedW8A8Schedule(
        grouped_problem=grouped_problem,
        m_offsets=torch.tensor(offsets, dtype=torch.int32, device=cuda_device).contiguous(),
        task_table=task_table,
        output_rows=output_rows,
    )


def query_sm89_grouped_w8a8_resources(
    artifact: str | Path,
    *,
    warps_per_block: int = 4,
    device: torch.device | str | None = None,
) -> Sm89GroupedW8A8ResourceReport:
    """Query CUDA function attributes for the grouped MMA kernel."""

    if warps_per_block not in {1, 2, 4, 8}:
        raise ValueError("grouped W8A8 warps_per_block must be 1, 2, 4, or 8")
    if not torch.cuda.is_available():
        raise XQTBackendError("grouped W8A8 resource query requires CUDA")
    cuda_device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(cuda_device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 grouped W8A8 resource query received sm_{major}{minor}")
    library = _load_library(artifact)
    if not hasattr(library, _RESOURCE_QUERY_SYMBOL):
        raise XQTBackendError("SM89 grouped W8A8 artifact lacks resource query symbol")
    registers = ctypes.c_int()
    shared = ctypes.c_int()
    blocks = ctypes.c_int()
    error = getattr(library, _RESOURCE_QUERY_SYMBOL)(
        warps_per_block,
        ctypes.byref(registers),
        ctypes.byref(shared),
        ctypes.byref(blocks),
    )
    if error != 0:
        raise XQTBackendError(f"SM89 grouped W8A8 resource query failed with CUDA error {error}")
    properties = torch.cuda.get_device_properties(cuda_device)
    max_threads = int(properties.max_threads_per_multi_processor)
    block_threads = warps_per_block * 32
    occupancy = float(blocks.value * block_threads) / float(max_threads) if max_threads else None
    return Sm89GroupedW8A8ResourceReport(
        artifact=str(Path(artifact).expanduser()),
        block_threads=block_threads,
        registers_per_thread=int(registers.value),
        static_shared_bytes=int(shared.value),
        max_active_blocks_per_sm=int(blocks.value),
        multiprocessor_count=int(properties.multi_processor_count),
        max_threads_per_sm=max_threads,
        occupancy=occupancy,
        device=str(cuda_device),
        device_name=str(torch.cuda.get_device_name(cuda_device)),
        capability=f"sm_{major}{minor}",
    )


def _resolve_cached_configuration(
    lookup: GemmTuningLookup,
    *,
    fallback_scheduler: str,
    fallback_warps: int | str,
) -> tuple[str, int | str, GemmTuningLookup]:
    if lookup.status != "hit":
        return fallback_scheduler, fallback_warps, lookup
    assert lookup.record is not None
    cached_scheduler = lookup.selection.get("scheduler")
    cached_warps = lookup.selection.get("warps_per_block")
    if (
        lookup.record.selected_kernel == _KERNEL_NAME
        and cached_scheduler == "direct"
        and isinstance(cached_warps, int)
        and not isinstance(cached_warps, bool)
        and cached_warps in {1, 2, 4, 8}
    ):
        return "direct", cached_warps, lookup
    return fallback_scheduler, fallback_warps, GemmTuningLookup(
        status="invalid",
        key=lookup.key,
        reason=(
            "cached W8A8 selection must target sm89_w8a8_grouped_mma "
            "with scheduler direct and 1, 2, 4, or 8 warps"
        ),
        source="deterministic_default",
    )


def sm89_grouped_w8a8_executor(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weights: Sm89GroupedW8A8PackedWeights,
    schedule: Sm89GroupedW8A8Schedule,
    *,
    artifact: str | Path,
    scheduler: str = "auto",
    warps_per_block: int | str = "auto",
    allow_unverified_artifact: bool = False,
) -> Sm89GroupedW8A8DispatchResult:
    """Execute grouped INT8 MMA with per-channel W and per-token A scales."""

    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise XQTBackendError("grouped W8A8 activation must be rank-2 CUDA int8")
    if not activation.is_cuda or activation.dtype != torch.int8:
        raise XQTBackendError("grouped W8A8 activation must be CUDA int8")
    if activation.device != weights.device or activation.device != schedule.device:
        raise XQTBackendError("grouped W8A8 activation, weights, and schedule must share one device")
    if tuple(activation.shape) != (schedule.total_m, weights.k):
        raise XQTBackendError(
            f"grouped W8A8 activation must have packed shape [{schedule.total_m},{weights.k}]"
        )
    if schedule.grouped_problem.group_count != weights.expert_count:
        raise XQTBackendError("grouped W8A8 schedule expert count does not match packed weights")
    if schedule.grouped_problem.n != weights.n or schedule.grouped_problem.k != weights.k:
        raise XQTBackendError("grouped W8A8 schedule N/K does not match packed weights")
    if scheduler not in {"auto", "direct"}:
        raise ValueError("grouped W8A8 scheduler must be 'auto' or 'direct'")
    if warps_per_block == "auto":
        n_tiles = (weights.n + 7) // 8
        selected_warps = next(
            candidate for candidate in (4, 2, 1) if candidate <= n_tiles
        )
    elif isinstance(warps_per_block, int) and not isinstance(warps_per_block, bool):
        selected_warps = warps_per_block
    else:
        raise TypeError("grouped W8A8 warps_per_block must be an int or 'auto'")
    if selected_warps not in {1, 2, 4, 8}:
        raise ValueError("grouped W8A8 warps_per_block must resolve to 1, 2, 4, or 8")
    if not isinstance(activation_scales, torch.Tensor) or not activation_scales.is_cuda:
        raise XQTBackendError("grouped W8A8 activation scales must be a CUDA tensor")
    if activation_scales.device != activation.device or activation_scales.dtype not in {
        torch.float16,
        torch.float32,
        torch.bfloat16,
    }:
        raise XQTBackendError("grouped W8A8 activation scales must be fp16/bf16/fp32 on the input device")
    per_token = int(activation_scales.numel()) == schedule.total_m
    if not per_token and int(activation_scales.numel()) != 1:
        raise XQTBackendError("grouped W8A8 activation scales must have one or total_M elements")
    major, minor = torch.cuda.get_device_capability(activation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 grouped W8A8 received sm_{major}{minor}")
    if not allow_unverified_artifact and not artifact_ready_for_execution(
        artifact, kernel_name=_KERNEL_NAME, target_arch="sm_89"
    ):
        raise XQTBackendError(
            "SM89 grouped W8A8 artifact is not correctness-promoted; run the grouped correctness gate"
        )
    library = _load_library(artifact)
    output = torch.empty((schedule.total_m, weights.n), device=activation.device, dtype=torch.float16)
    artifact_path = str(Path(artifact).expanduser())
    base_report = dict(
        artifact=artifact_path,
        expert_count=weights.expert_count,
        expert_rows=schedule.expert_rows,
        m_offsets=_offsets_for_problem(schedule.grouped_problem),
        task_count=schedule.task_count,
        n=weights.n,
        k=weights.k,
        padded_k=weights.padded_k,
        warps_per_block=selected_warps,
        weight_scale_mode="per_channel",
        activation_scale_mode="per_token" if per_token else "per_tensor",
        output_scatter=schedule.output_rows is not None,
        scatter_mode="in_kernel_permutation" if schedule.output_rows is not None else "identity",
        scatter_launch_count=0,
        workspace_bytes=0,
        fallback_chain=("sm89_grouped_w8a8_mma", "grouped_reference"),
        fallback_reason=None,
    )
    if schedule.total_m == 0:
        return Sm89GroupedW8A8DispatchResult(
            output=output,
            report=Sm89GroupedW8A8DispatchReport(
                scheduler="empty", launch_count=0, native=False, **base_report
            ),
        )
    scale_input = activation_scales.detach().to(dtype=torch.float32).reshape(-1).contiguous()
    activation_exec = activation.contiguous()
    if weights.padded_k != weights.k:
        activation_exec = F.pad(activation_exec, (0, weights.padded_k - weights.k)).contiguous()
    task_table = schedule.task_table
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    error = getattr(library, _SYMBOL)(
        activation_exec.data_ptr(),
        weights.weights_nk.data_ptr(),
        weights.weight_scales.data_ptr(),
        scale_input.data_ptr(),
        0 if weights.bias is None else weights.bias.data_ptr(),
        task_table.data_ptr(),
        0 if schedule.output_rows is None else schedule.output_rows.data_ptr(),
        output.data_ptr(),
        int(schedule.task_count),
        weights.n,
        weights.padded_k,
        weights.expert_count,
        int(weights.bias is not None),
        int(schedule.output_rows is not None),
        int(per_token),
        selected_warps,
        stream,
    )
    if error != 0:
        raise XQTBackendError(f"SM89 grouped W8A8 MMA failed with CUDA error {error}")
    return Sm89GroupedW8A8DispatchResult(
        output=output,
        report=Sm89GroupedW8A8DispatchReport(
            scheduler="direct_task_grid", launch_count=1, native=True, **base_report
        ),
    )


def dispatch_sm89_grouped_w8a8(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weights: Sm89GroupedW8A8PackedWeights,
    schedule: Sm89GroupedW8A8Schedule,
    *,
    quant: QuantSpec,
    artifact: str | Path,
    scheduler: str = "auto",
    warps_per_block: int | str = "auto",
    tuning_cache: GemmTuningCache | None = None,
    allow_reference: bool = True,
    allow_unverified_artifact: bool = False,
) -> GroupedGemmDispatchResult:
    """Dispatch SM89 grouped W8A8 with an explicit grouped reference fallback."""

    if (
        quant.weight_dtype != "int8"
        or quant.activation_dtype != "int8"
        or quant.output_dtype != "fp16"
        or quant.weight_granularity != "per_channel"
        or quant.activation_granularity not in {"per_token", "per_tensor"}
        or not quant.symmetric
        or quant.weight_zero_point
        or quant.activation_zero_point
    ):
        raise ValueError(
            "grouped W8A8 quant contract does not match the packed payload"
        )
    if (
        quant.activation_granularity == "per_token"
        and int(activation_scales.numel()) != schedule.total_m
    ):
        raise ValueError(
            "grouped W8A8 per-token scales must have total_M elements"
        )
    if (
        quant.activation_granularity == "per_tensor"
        and int(activation_scales.numel()) != 1
    ):
        raise ValueError("grouped W8A8 per-tensor scales must be scalar")
    tuning_key = build_grouped_tuning_key(
        kernel_family=_KERNEL_NAME,
        backend="custom_cuda",
        target_arch="sm_89",
        grouped_problem=schedule.grouped_problem,
        quant=quant,
        has_bias=weights.bias is not None,
    )
    tuning = resolve_tuning_record(
        tuning_cache,
        key=tuning_key,
        artifact=artifact,
        explicit=scheduler != "auto" or warps_per_block != "auto",
    )
    selected_scheduler, selected_warps, tuning = _resolve_cached_configuration(
        tuning,
        fallback_scheduler=scheduler,
        fallback_warps=warps_per_block,
    )

    def execute_native() -> GroupedGemmCandidateOutput:
        result = sm89_grouped_w8a8_executor(
            activation,
            activation_scales,
            weights,
            schedule,
            artifact=artifact,
            scheduler=selected_scheduler,
            warps_per_block=selected_warps,
            allow_unverified_artifact=allow_unverified_artifact,
        )
        return GroupedGemmCandidateOutput(
            output=result.output,
            details=result.report.to_dict(),
            native=result.report.native,
        )

    def execute_reference() -> torch.Tensor:
        reference_weights = tuple(
            weights.weights_nk[index, :, : weights.k]
            for index in range(weights.expert_count)
        )
        reference_weight_scales = tuple(
            weights.weight_scales[index]
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
            weight_scales=reference_weight_scales,
            activation_scales=activation_scales,
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
    "Sm89GroupedW8A8DispatchReport",
    "Sm89GroupedW8A8DispatchResult",
    "Sm89GroupedW8A8PackedWeights",
    "Sm89GroupedW8A8ResourceReport",
    "Sm89GroupedW8A8Schedule",
    "build_sm89_grouped_w8a8_schedule",
    "dispatch_sm89_grouped_w8a8",
    "pack_sm89_grouped_w8a8_weights",
    "query_sm89_grouped_w8a8_resources",
    "sm89_grouped_w8a8_artifact_available",
    "sm89_grouped_w8a8_executor",
]
