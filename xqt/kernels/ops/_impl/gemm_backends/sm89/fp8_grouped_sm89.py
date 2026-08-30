"""Manifest-gated SM89 grouped FP8 MMA adapter."""

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
from xqt.kernels.ops.gemm.fp8 import fp8_block_count, fp8_format_spec, validate_fp8_block_k
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


_SYMBOL = "xqt_fp8_grouped_sm89_fp16_run"
_RESOURCE_QUERY_SYMBOL = "xqt_fp8_grouped_sm89_resource_query"
_KERNEL_NAME = "sm89_fp8_grouped_mma"
_FORMAT_IDS = {"fp8_e4m3": 0, "fp8_e5m2": 1}
_ROW_TILE = 8
_K_TILE = 32


@dataclass(frozen=True, slots=True)
class Sm89GroupedFp8PackedWeights:
    """Static grouped FP8 bytes and scale payload."""

    qweight: torch.Tensor
    weight_scales: torch.Tensor
    bias: torch.Tensor | None
    expert_count: int
    n: int
    k: int
    padded_k: int
    format_name: str
    scale_mode: str
    block_k: int | None

    def __post_init__(self) -> None:
        if self.format_name not in _FORMAT_IDS:
            raise ValueError(f"unsupported grouped FP8 format: {self.format_name}")
        if self.scale_mode not in {"tensorwise", "blockwise"}:
            raise ValueError("grouped FP8 scale_mode must be tensorwise or blockwise")
        if self.expert_count <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("grouped FP8 expert_count/N/K must be positive")
        if self.padded_k < self.k or self.padded_k % _K_TILE:
            raise ValueError("grouped FP8 padded_k must be a multiple of 32")
        if (
            self.qweight.dtype != torch.uint8
            or self.qweight.ndim != 3
            or tuple(self.qweight.shape)
            != (self.expert_count, self.n, self.padded_k)
        ):
            raise ValueError("grouped FP8 qweight must be CUDA uint8 [E,N,padded_K]")
        if self.scale_mode == "tensorwise":
            if self.block_k is not None:
                raise ValueError("tensorwise grouped FP8 cannot declare block_k")
            expected_scale_shape = (self.expert_count,)
        else:
            if self.block_k is None:
                raise ValueError("blockwise grouped FP8 requires block_k")
            validate_fp8_block_k(self.block_k)
            expected_scale_shape = (
                self.expert_count,
                self.n,
                fp8_block_count(self.k, self.block_k),
            )
        if (
            self.weight_scales.dtype != torch.float32
            or tuple(self.weight_scales.shape) != expected_scale_shape
        ):
            raise ValueError(
                f"grouped FP8 weight_scales must be float32 {expected_scale_shape}"
            )
        values = [self.qweight, self.weight_scales]
        if self.bias is not None:
            if self.bias.dtype != torch.float32 or tuple(self.bias.shape) != (
                self.expert_count,
                self.n,
            ):
                raise ValueError("grouped FP8 bias must be float32 [E,N]")
            values.append(self.bias)
        device = self.qweight.device
        if any(not value.is_cuda or value.device != device for value in values):
            raise ValueError("grouped FP8 payload tensors must share one CUDA device")
        if any(not value.is_contiguous() for value in values):
            raise ValueError("grouped FP8 payload tensors must be contiguous")

    @property
    def device(self) -> torch.device:
        """Return the device carrying the static grouped payload."""

        return self.qweight.device


@dataclass(frozen=True, slots=True)
class Sm89GroupedFp8Schedule:
    """Device task table for one grouped FP8 routing state."""

    grouped_problem: GroupedGemmProblem
    m_offsets: torch.Tensor
    task_table: torch.Tensor
    output_rows: torch.Tensor | None
    row_tile: int = _ROW_TILE

    def __post_init__(self) -> None:
        if self.row_tile != _ROW_TILE:
            raise ValueError("SM89 grouped FP8 currently uses row_tile=8")
        if (
            self.m_offsets.dtype != torch.int32
            or self.m_offsets.ndim != 1
            or int(self.m_offsets.numel()) != self.grouped_problem.group_count + 1
            or not self.m_offsets.is_cuda
            or not self.m_offsets.is_contiguous()
        ):
            raise ValueError("grouped FP8 m_offsets must be CUDA int32 [E+1]")
        if (
            self.task_table.dtype != torch.int32
            or self.task_table.ndim != 2
            or tuple(self.task_table.shape) != (self.task_count, 3)
            or not self.task_table.is_cuda
            or not self.task_table.is_contiguous()
        ):
            raise ValueError("grouped FP8 task_table must be CUDA int32 [task_count,3]")
        if self.output_rows is not None:
            if (
                self.output_rows.dtype != torch.int32
                or self.output_rows.ndim != 1
                or int(self.output_rows.numel()) != self.total_m
                or not self.output_rows.is_cuda
                or not self.output_rows.is_contiguous()
            ):
                raise ValueError("grouped FP8 output_rows must be CUDA int32 [total_M]")
        device = self.m_offsets.device
        if self.task_table.device != device or (
            self.output_rows is not None and self.output_rows.device != device
        ):
            raise ValueError("grouped FP8 schedule tensors must share one CUDA device")

    @property
    def device(self) -> torch.device:
        """Return the device holding the grouped routing state."""

        return self.m_offsets.device

    @property
    def total_m(self) -> int:
        """Return the packed activation row count."""

        return self.grouped_problem.total_m

    @property
    def task_count(self) -> int:
        """Return the number of eight-row-or-smaller tasks."""

        return sum(
            (problem.m + _ROW_TILE - 1) // _ROW_TILE
            for problem in self.grouped_problem.problems
        )

    @property
    def expert_rows(self) -> tuple[int, ...]:
        """Return routed row counts, retaining empty experts."""

        return tuple(problem.m for problem in self.grouped_problem.problems)


@dataclass(frozen=True, slots=True)
class Sm89GroupedFp8DispatchReport:
    """Grouped FP8 dispatch and scale-layout evidence."""

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
    format_name: str
    scale_mode: str
    block_k: int | None
    warps_per_block: int
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
            "format_name": self.format_name,
            "scale_mode": self.scale_mode,
            "block_k": self.block_k,
            "warps_per_block": self.warps_per_block,
            "output_scatter": self.output_scatter,
            "scatter_mode": self.scatter_mode,
            "scatter_launch_count": self.scatter_launch_count,
            "workspace_bytes": self.workspace_bytes,
            "fallback_chain": list(self.fallback_chain),
            "fallback_reason": self.fallback_reason,
            "native": self.native,
        }


@dataclass(frozen=True, slots=True)
class Sm89GroupedFp8DispatchResult:
    """Native grouped FP8 output and dispatch report."""

    output: torch.Tensor
    report: Sm89GroupedFp8DispatchReport


@dataclass(frozen=True, slots=True)
class Sm89GroupedFp8ResourceReport:
    """CUDA function-attribute evidence for one grouped FP8 variant."""

    artifact: str
    format_name: str
    scale_mode: str
    block_k: int | None
    warps_per_block: int
    block_threads: int
    registers_per_thread: int
    static_shared_bytes: int
    max_active_blocks_per_sm: int
    max_threads_per_sm: int
    occupancy: float | None
    device: str
    device_name: str
    capability: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready resource record."""

        return {
            "artifact": self.artifact,
            "format_name": self.format_name,
            "scale_mode": self.scale_mode,
            "block_k": self.block_k,
            "warps_per_block": self.warps_per_block,
            "block_threads": self.block_threads,
            "registers_per_thread": self.registers_per_thread,
            "static_shared_bytes": self.static_shared_bytes,
            "max_active_blocks_per_sm": self.max_active_blocks_per_sm,
            "max_threads_per_sm": self.max_threads_per_sm,
            "occupancy": self.occupancy,
            "device": self.device,
            "device_name": self.device_name,
            "capability": self.capability,
        }


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    """Load the grouped FP8 artifact and bind its fixed ABI."""

    path = Path(artifact).expanduser()
    if not path.is_file():
        raise XQTBackendError(f"SM89 grouped FP8 artifact not found: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise XQTBackendError(f"unable to load SM89 grouped FP8 artifact: {path}") from exc
    if not hasattr(library, _SYMBOL):
        raise XQTBackendError(f"SM89 grouped FP8 artifact lacks {_SYMBOL}")
    function = getattr(library, _SYMBOL)
    function.argtypes = [
        *([ctypes.c_void_p] * 8),
        *([ctypes.c_int] * 9),
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    if hasattr(library, _RESOURCE_QUERY_SYMBOL):
        query = getattr(library, _RESOURCE_QUERY_SYMBOL)
        query.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        query.restype = ctypes.c_int
    return library


def sm89_grouped_fp8_artifact_available(artifact: str | Path) -> bool:
    """Return whether the grouped FP8 symbol can be loaded."""

    try:
        _load_library(artifact)
    except XQTBackendError:
        return False
    return True


def _encoded_bytes(
    value: torch.Tensor,
    *,
    format_name: str,
    name: str,
) -> torch.Tensor:
    """Return canonical uint8 storage without changing FP8 bits."""

    format_spec = fp8_format_spec(format_name)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise XQTBackendError(f"grouped FP8 {name} must be rank-2")
    if value.dtype == torch.uint8:
        return value
    if value.dtype == format_spec.torch_dtype:
        return value.view(torch.uint8)
    raise XQTBackendError(
        f"grouped FP8 {name} must be uint8 or {format_spec.torch_dtype}"
    )


def _weight_payload(weight: PackedWeight | torch.Tensor) -> tuple[torch.Tensor, Any]:
    """Extract one expert's encoded weight and optional scale artifact."""

    if isinstance(weight, PackedWeight):
        if not isinstance(weight.qweight, torch.Tensor):
            raise XQTBackendError("grouped FP8 PackedWeight qweight must be a tensor")
        return weight.qweight, weight.scales
    if not isinstance(weight, torch.Tensor):
        raise XQTBackendError("grouped FP8 weight must be a tensor or PackedWeight")
    return weight, None


def _offsets_for_problem(grouped_problem: GroupedGemmProblem) -> tuple[int, ...]:
    """Return explicit packed row offsets."""

    if grouped_problem.m_offsets is not None:
        return grouped_problem.m_offsets
    offsets = [0]
    for problem in grouped_problem.problems:
        offsets.append(offsets[-1] + problem.m)
    return tuple(offsets)


def pack_sm89_grouped_fp8_weights(
    weights: Sequence[PackedWeight | torch.Tensor],
    *,
    quant: QuantSpec,
    weight_scales: Sequence[torch.Tensor] | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> Sm89GroupedFp8PackedWeights:
    """Stack compatible expert FP8 bytes and explicit scale artifacts."""

    if not weights:
        raise ValueError("grouped FP8 prepack requires at least one expert weight")
    if quant.weight_dtype not in _FORMAT_IDS or quant.activation_dtype != quant.weight_dtype:
        raise XQTBackendError("SM89 grouped FP8 requires matching E4M3 or E5M2 A/W")
    if quant.output_dtype != "fp16":
        raise XQTBackendError("SM89 grouped FP8 currently outputs fp16")
    tensorwise = (
        quant.weight_granularity == "per_tensor"
        and quant.activation_granularity == "per_tensor"
    )
    blockwise = (
        quant.weight_granularity == "blockwise"
        and quant.activation_granularity == "blockwise"
    )
    if not tensorwise and not blockwise:
        raise XQTBackendError("grouped FP8 requires matched tensorwise or blockwise scales")
    block_k = None
    if blockwise:
        if quant.group_axis != "k" or quant.group_size is None:
            raise XQTBackendError("grouped blockwise FP8 requires group_axis='k' and group_size")
        block_k = validate_fp8_block_k(quant.group_size)
    if weight_scales is not None and len(weight_scales) != len(weights):
        raise ValueError("grouped FP8 scale count must match expert count")
    if bias is not None and len(bias) != len(weights):
        raise ValueError("grouped FP8 bias count must match expert count")
    first_payload, _ = _weight_payload(weights[0])
    first = _encoded_bytes(first_payload, format_name=quant.weight_dtype, name="weight")
    n, k = (int(first.shape[0]), int(first.shape[1]))
    padded_k = ((k + _K_TILE - 1) // _K_TILE) * _K_TILE
    scale_blocks = None if block_k is None else fp8_block_count(k, block_k)
    normalized_bias = tuple(None for _ in weights) if bias is None else tuple(bias)
    has_bias = any(value is not None for value in normalized_bias)
    device: torch.device | None = None
    qweights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for index, weight in enumerate(weights):
        payload, packed_scale = _weight_payload(weight)
        encoded = _encoded_bytes(payload, format_name=quant.weight_dtype, name="weight")
        if tuple(encoded.shape) != (n, k):
            raise ValueError("grouped FP8 experts must share logical [N,K]")
        if not encoded.is_cuda:
            raise XQTBackendError("grouped FP8 prepack requires CUDA expert payloads")
        if device is None:
            device = encoded.device
        if encoded.device != device:
            raise XQTBackendError("grouped FP8 experts must share one CUDA device")
        encoded = encoded.contiguous()
        if padded_k != k:
            encoded = F.pad(encoded, (0, padded_k - k))
        qweights.append(encoded.contiguous())
        scale = weight_scales[index] if weight_scales is not None else packed_scale
        if not isinstance(scale, torch.Tensor) or scale.device != device:
            raise XQTBackendError("grouped FP8 weight scales must be CUDA tensors on the expert device")
        if tensorwise:
            if int(scale.numel()) != 1:
                raise XQTBackendError("grouped tensorwise FP8 requires one scale per expert")
            scales.append(scale.detach().to(dtype=torch.float32).reshape(()))
        else:
            expected = (n, scale_blocks)
            if tuple(scale.shape) != expected:
                raise XQTBackendError(
                    f"grouped blockwise FP8 weight scale must have shape {expected}"
                )
            scales.append(scale.detach().to(dtype=torch.float32).contiguous())
        if has_bias:
            current_bias = normalized_bias[index]
            if current_bias is None:
                biases.append(torch.zeros((n,), device=device, dtype=torch.float32))
            else:
                if current_bias.device != device or int(current_bias.numel()) != n:
                    raise XQTBackendError("grouped FP8 bias must have N elements on the expert device")
                biases.append(current_bias.detach().to(dtype=torch.float32).reshape(n).contiguous())
    if device is None:
        raise AssertionError("non-empty FP8 experts must resolve a CUDA device")
    return Sm89GroupedFp8PackedWeights(
        qweight=torch.stack(qweights, dim=0).contiguous(),
        weight_scales=torch.stack(scales, dim=0).contiguous(),
        bias=None if not has_bias else torch.stack(biases, dim=0).contiguous(),
        expert_count=len(weights),
        n=n,
        k=k,
        padded_k=padded_k,
        format_name=quant.weight_dtype,
        scale_mode="tensorwise" if tensorwise else "blockwise",
        block_k=block_k,
    )


def build_sm89_grouped_fp8_schedule(
    grouped_problem: GroupedGemmProblem,
    *,
    device: torch.device | str,
) -> Sm89GroupedFp8Schedule:
    """Materialize the direct grouped task table and optional permutation."""

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError(f"grouped FP8 schedule requires CUDA, got {cuda_device}")
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
        else torch.tensor(
            grouped_problem.output_rows, dtype=torch.int32, device=cuda_device
        ).contiguous()
    )
    return Sm89GroupedFp8Schedule(
        grouped_problem=grouped_problem,
        m_offsets=torch.tensor(
            offsets, dtype=torch.int32, device=cuda_device
        ).contiguous(),
        task_table=task_table,
        output_rows=output_rows,
    )


def query_sm89_grouped_fp8_resources(
    artifact: str | Path,
    *,
    format_name: str,
    block_k: int | None,
    warps_per_block: int = 4,
    device: torch.device | str | None = None,
) -> Sm89GroupedFp8ResourceReport:
    """Query CUDA resource attributes for one grouped FP8 template."""

    if format_name not in _FORMAT_IDS:
        raise ValueError(f"unsupported grouped FP8 format: {format_name}")
    if block_k is not None:
        block_k = validate_fp8_block_k(block_k)
    if warps_per_block not in {1, 2, 4, 8}:
        raise ValueError("grouped FP8 warps_per_block must be 1, 2, 4, or 8")
    if not torch.cuda.is_available():
        raise XQTBackendError("grouped FP8 resource query requires CUDA")
    cuda_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if cuda_device.type != "cuda":
        raise ValueError(f"grouped FP8 resource query requires CUDA, got {cuda_device}")
    major, minor = torch.cuda.get_device_capability(cuda_device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 grouped FP8 resource query received sm_{major}{minor}")
    library = _load_library(artifact)
    registers = ctypes.c_int()
    shared = ctypes.c_int()
    blocks = ctypes.c_int()
    error = getattr(library, _RESOURCE_QUERY_SYMBOL)(
        _FORMAT_IDS[format_name],
        0 if block_k is None else block_k,
        warps_per_block,
        ctypes.byref(registers),
        ctypes.byref(shared),
        ctypes.byref(blocks),
    )
    if error != 0:
        raise XQTBackendError(f"SM89 grouped FP8 resource query failed with CUDA error {error}")
    properties = torch.cuda.get_device_properties(cuda_device)
    block_threads = warps_per_block * 32
    max_threads = int(properties.max_threads_per_multi_processor)
    occupancy = float(blocks.value * block_threads) / float(max_threads) if max_threads else None
    return Sm89GroupedFp8ResourceReport(
        artifact=str(Path(artifact).expanduser()),
        format_name=format_name,
        scale_mode="tensorwise" if block_k is None else "blockwise",
        block_k=block_k,
        warps_per_block=warps_per_block,
        block_threads=block_threads,
        registers_per_thread=int(registers.value),
        static_shared_bytes=int(shared.value),
        max_active_blocks_per_sm=int(blocks.value),
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
            "cached FP8 selection must target sm89_fp8_grouped_mma "
            "with scheduler direct and 1, 2, 4, or 8 warps"
        ),
        source="deterministic_default",
    )


def sm89_grouped_fp8_executor(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weights: Sm89GroupedFp8PackedWeights,
    schedule: Sm89GroupedFp8Schedule,
    *,
    artifact: str | Path,
    scheduler: str = "auto",
    warps_per_block: int | str = "auto",
    allow_unverified_artifact: bool = False,
) -> Sm89GroupedFp8DispatchResult:
    """Execute grouped E4M3/E5M2 tensorwise or blockwise FP8 MMA."""

    encoded = _encoded_bytes(
        activation, format_name=weights.format_name, name="activation"
    )
    if not encoded.is_cuda:
        raise XQTBackendError("grouped FP8 activation must be CUDA")
    if encoded.device != weights.device or encoded.device != schedule.device:
        raise XQTBackendError("grouped FP8 activation, weights, and schedule must share one device")
    if tuple(encoded.shape) != (schedule.total_m, weights.k):
        raise XQTBackendError(
            f"grouped FP8 activation must have packed shape [{schedule.total_m},{weights.k}]"
        )
    if schedule.grouped_problem.group_count != weights.expert_count:
        raise XQTBackendError("grouped FP8 schedule expert count does not match packed weights")
    if schedule.grouped_problem.n != weights.n or schedule.grouped_problem.k != weights.k:
        raise XQTBackendError("grouped FP8 schedule N/K does not match packed weights")
    if scheduler not in {"auto", "direct"}:
        raise ValueError("grouped FP8 scheduler must be 'auto' or 'direct'")
    if warps_per_block == "auto":
        n_tiles = (weights.n + 7) // 8
        selected_warps = next(
            candidate for candidate in (4, 2, 1) if candidate <= n_tiles
        )
    elif isinstance(warps_per_block, int) and not isinstance(warps_per_block, bool):
        selected_warps = warps_per_block
    else:
        raise TypeError("grouped FP8 warps_per_block must be an int or 'auto'")
    if selected_warps not in {1, 2, 4, 8}:
        raise ValueError("grouped FP8 warps_per_block must resolve to 1, 2, 4, or 8")
    if not isinstance(activation_scales, torch.Tensor) or not activation_scales.is_cuda:
        raise XQTBackendError("grouped FP8 activation scales must be CUDA tensors")
    if activation_scales.device != encoded.device or activation_scales.dtype != torch.float32:
        raise XQTBackendError("grouped FP8 activation scales must be FP32 on the input device")
    if weights.scale_mode == "tensorwise":
        if int(activation_scales.numel()) == 1:
            scale_input = activation_scales.reshape(1).expand(weights.expert_count).contiguous()
        elif tuple(activation_scales.shape) == (weights.expert_count,):
            scale_input = activation_scales.contiguous()
        else:
            raise XQTBackendError("grouped tensorwise FP8 activation scales must be scalar or [E]")
    else:
        assert weights.block_k is not None
        expected = (
            schedule.total_m,
            fp8_block_count(weights.k, weights.block_k),
        )
        if tuple(activation_scales.shape) != expected:
            raise XQTBackendError(
                f"grouped blockwise FP8 activation scales must have shape {expected}"
            )
        scale_input = activation_scales.contiguous()
    major, minor = torch.cuda.get_device_capability(encoded.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 grouped FP8 received sm_{major}{minor}")
    if not allow_unverified_artifact and not artifact_ready_for_execution(
        artifact, kernel_name=_KERNEL_NAME, target_arch="sm_89"
    ):
        raise XQTBackendError(
            "SM89 grouped FP8 artifact is not correctness-promoted; run the grouped correctness gate"
        )
    library = _load_library(artifact)
    output = torch.empty(
        (schedule.total_m, weights.n), device=encoded.device, dtype=torch.float16
    )
    base_report = dict(
        artifact=str(Path(artifact).expanduser()),
        expert_count=weights.expert_count,
        expert_rows=schedule.expert_rows,
        m_offsets=_offsets_for_problem(schedule.grouped_problem),
        task_count=schedule.task_count,
        n=weights.n,
        k=weights.k,
        padded_k=weights.padded_k,
        format_name=weights.format_name,
        scale_mode=weights.scale_mode,
        block_k=weights.block_k,
        warps_per_block=selected_warps,
        output_scatter=schedule.output_rows is not None,
        scatter_mode="in_kernel_permutation" if schedule.output_rows is not None else "identity",
        scatter_launch_count=0,
        workspace_bytes=0,
        fallback_chain=("sm89_grouped_fp8_mma", "grouped_reference"),
        fallback_reason=None,
    )
    if schedule.total_m == 0:
        return Sm89GroupedFp8DispatchResult(
            output=output,
            report=Sm89GroupedFp8DispatchReport(
                scheduler="empty", launch_count=0, native=False, **base_report
            ),
        )
    activation_runtime = encoded.contiguous()
    if weights.padded_k != weights.k:
        activation_runtime = F.pad(
            activation_runtime, (0, weights.padded_k - weights.k)
        ).contiguous()
    stream = torch.cuda.current_stream(encoded.device).cuda_stream
    error = getattr(library, _SYMBOL)(
        activation_runtime.data_ptr(),
        weights.qweight.data_ptr(),
        scale_input.data_ptr(),
        weights.weight_scales.data_ptr(),
        0 if weights.bias is None else weights.bias.data_ptr(),
        schedule.task_table.data_ptr(),
        0 if schedule.output_rows is None else schedule.output_rows.data_ptr(),
        output.data_ptr(),
        schedule.task_count,
        weights.n,
        weights.padded_k,
        weights.expert_count,
        _FORMAT_IDS[weights.format_name],
        0 if weights.block_k is None else weights.block_k,
        int(weights.bias is not None),
        int(schedule.output_rows is not None),
        selected_warps,
        stream,
    )
    if error != 0:
        raise XQTBackendError(f"SM89 grouped FP8 MMA failed with CUDA error {error}")
    return Sm89GroupedFp8DispatchResult(
        output=output,
        report=Sm89GroupedFp8DispatchReport(
            scheduler="direct_task_grid", launch_count=1, native=True, **base_report
        ),
    )


def dispatch_sm89_grouped_fp8(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    weights: Sm89GroupedFp8PackedWeights,
    schedule: Sm89GroupedFp8Schedule,
    *,
    quant: QuantSpec,
    artifact: str | Path,
    scheduler: str = "auto",
    warps_per_block: int | str = "auto",
    tuning_cache: GemmTuningCache | None = None,
    allow_reference: bool = True,
    allow_unverified_artifact: bool = False,
) -> GroupedGemmDispatchResult:
    """Dispatch SM89 grouped FP8 with an explicit grouped reference fallback."""

    expected_granularity = (
        "per_tensor" if weights.scale_mode == "tensorwise" else "blockwise"
    )
    if (
        quant.weight_dtype != weights.format_name
        or quant.activation_dtype != weights.format_name
        or quant.output_dtype != "fp16"
        or quant.weight_granularity != expected_granularity
        or quant.activation_granularity != expected_granularity
        or quant.group_size != weights.block_k
        or (weights.block_k is not None and quant.group_axis != "k")
        or quant.weight_zero_point
        or quant.activation_zero_point
    ):
        raise ValueError(
            "grouped FP8 quant contract does not match the packed payload"
        )
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
        result = sm89_grouped_fp8_executor(
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
            weights.qweight[index, :, : weights.k]
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
    "Sm89GroupedFp8DispatchReport",
    "Sm89GroupedFp8DispatchResult",
    "Sm89GroupedFp8PackedWeights",
    "Sm89GroupedFp8ResourceReport",
    "Sm89GroupedFp8Schedule",
    "build_sm89_grouped_fp8_schedule",
    "dispatch_sm89_grouped_fp8",
    "pack_sm89_grouped_fp8_weights",
    "query_sm89_grouped_fp8_resources",
    "sm89_grouped_fp8_artifact_available",
    "sm89_grouped_fp8_executor",
]
