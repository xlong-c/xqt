"""SM120 low-precision GEMM capability boundary.

The SM120 CUTLASS path is deliberately separate from SM90.  The current
CollectiveBuilder exposes FP8 block-scaled and NVFP4 type families for SM120;
it does not provide a portable FP16 dense builder for this target.  The build
API therefore compiles only the supported low-precision probes and keeps them
metadata-only until RTX 50 correctness and SASS gates are collected.
"""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from xqt.core.errors import XQTBackendError

from ..contracts import GemmSpec, PackedWeight, QuantSpec
from ..fp8 import fp8_format_spec
from ..preflight import (
    GemmArtifactManifest,
    build_compile_flags,
    default_cache_dir,
    probe_cuda_cutlass,
)
from ..reference import reference_gemm
from ._sm1xx_runtime import (
    current_cuda_stream,
    load_runtime_function,
    pad_matrix,
    prepare_cutlass_blockscales,
    prepare_cutlass_nvfp4_scales,
    raise_runtime_error,
    require_same_cuda_device,
    require_target_cuda,
)


_SOURCE = Path(__file__).with_name("sm120_gemm.cu")
_DEFAULT_ARTIFACT = default_cache_dir() / "sm120" / "low_precision_tcgen05_sm120.so"
_SM120_CAPABILITY = (12, 0)
_SM120_FP8_SYMBOLS = {
    ("fp8_e4m3", "blockwise", "cooperative"): "xqt_sm120_fp8_e4m3_blockwise_bf16_run",
    ("fp8_e5m2", "blockwise", "cooperative"): "xqt_sm120_fp8_e5m2_blockwise_bf16_run",
    ("fp8_e4m3", "blockwise", "pingpong"):
        "xqt_sm120_fp8_e4m3_blockwise_pingpong_bf16_run",
    ("fp8_e5m2", "blockwise", "pingpong"):
        "xqt_sm120_fp8_e5m2_blockwise_pingpong_bf16_run",
    ("fp8_e4m3", "groupwise", "cooperative"): "xqt_sm120_fp8_e4m3_groupwise_bf16_run",
    ("fp8_e5m2", "groupwise", "cooperative"): "xqt_sm120_fp8_e5m2_groupwise_bf16_run",
    ("fp8_e4m3", "groupwise", "pingpong"): "xqt_sm120_fp8_e4m3_groupwise_pingpong_bf16_run",
    ("fp8_e5m2", "groupwise", "pingpong"): "xqt_sm120_fp8_e5m2_groupwise_pingpong_bf16_run",
}
_SM120_NVFP4_SYMBOLS = {
    (128, "cooperative"): "xqt_sm120_nvfp4_bf16_run",
    (256, "cooperative"): "xqt_sm120_nvfp4_k256_bf16_run",
    (128, "pingpong"): "xqt_sm120_nvfp4_pingpong_bf16_run",
    (256, "pingpong"): "xqt_sm120_nvfp4_k256_pingpong_bf16_run",
}


@dataclass(frozen=True, slots=True)
class Sm120Fp8Contract:
    """SM120 FP8 blockwise/groupwise scale contract."""

    format_name: str = "fp8_e4m3"
    scale_granularity: str = "blockwise"
    block_m: int = 1
    block_n: int = 128
    block_k: int = 128
    output_dtype: str = "bf16"
    use_tcgen05: bool = True
    use_tma: bool = True
    pack_version: str = "xqt-sm120-fp8-blockwise-v1"

    def __post_init__(self) -> None:
        if self.format_name not in {"fp8_e4m3", "fp8_e5m2"}:
            raise ValueError("SM120 FP8 format must be fp8_e4m3 or fp8_e5m2")
        if self.scale_granularity not in {"blockwise", "groupwise"}:
            raise ValueError("SM120 FP8 scale_granularity must be blockwise or groupwise")
        if (self.block_m, self.block_n, self.block_k) != (1, 128, 128):
            raise ValueError("SM120 FP8 uses the fixed 1x128x128 scale contract")
        if self.output_dtype != "bf16":
            raise ValueError("SM120 FP8 output_dtype must be bf16")
        if not isinstance(self.use_tcgen05, bool) or not isinstance(self.use_tma, bool):
            raise TypeError("SM120 FP8 execution flags must be bool")

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype=self.format_name,
            activation_dtype=self.format_name,
            output_dtype=self.output_dtype,
            weight_granularity="blockwise",
            activation_granularity="blockwise",
            group_size=self.block_k,
            weight_scale_source="weight_offline",
            activation_scale_source="activation_dynamic",
            storage_layout="xqt_fp8_rowmajor_v1",
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "sm_120",
            "format": self.format_name,
            "scale_granularity": self.scale_granularity,
            "block_shape": [self.block_m, self.block_n, self.block_k],
            "xqt_scale_mode": "w:blockwise/a:blockwise",
            "output_dtype": self.output_dtype,
            "use_tcgen05": self.use_tcgen05,
            "use_tma": self.use_tma,
            "pack_version": self.pack_version,
            "maturity": "metadata_only",
        }


@dataclass(frozen=True, slots=True)
class Sm120Nvfp4Contract:
    """SM120 NVFP4 block-scaled MMA contract."""

    format_name: str = "nvfp4"
    group_size: int = 16
    output_dtype: str = "bf16"
    use_tcgen05: bool = True
    use_tma: bool = True
    pack_version: str = "xqt-sm120-nvfp4-v1"

    def __post_init__(self) -> None:
        if self.group_size != 16:
            raise ValueError("SM120 NVFP4 uses group_size=16")
        if self.output_dtype != "bf16":
            raise ValueError("SM120 NVFP4 output_dtype must be bf16")
        if not isinstance(self.use_tcgen05, bool) or not isinstance(self.use_tma, bool):
            raise TypeError("SM120 NVFP4 execution flags must be bool")

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype="nvfp4",
            activation_dtype="fp16",
            output_dtype=self.output_dtype,
            weight_granularity="groupwise",
            activation_granularity="per_tensor",
            group_size=self.group_size,
            weight_scale_source="weight_offline",
            activation_scale_source="none",
            storage_layout="xqt_fp4_nk_v1",
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "sm_120",
            "format": self.format_name,
            "group_size": self.group_size,
            "activation_storage": "packed_e2m1",
            "scale_dtype": "float_ue4m3",
            "scale_layout": "sm1xx_blockscaled_sfvec16",
            "tile_shapes": [[128, 128, 128], [128, 128, 256]],
            "output_dtype": self.output_dtype,
            "use_tcgen05": self.use_tcgen05,
            "use_tma": self.use_tma,
            "pack_version": self.pack_version,
            "maturity": "metadata_only",
        }


@dataclass(frozen=True, slots=True)
class Sm120BuildConfig:
    """Inputs for the SM120 low-precision CUTLASS build."""

    source: Path = _SOURCE
    output: Path = field(default_factory=lambda: _DEFAULT_ARTIFACT)
    target_arch: str = "sm_120"
    extra_flags: tuple[str, ...] = ("-lineinfo", "-lcudart")

    def __post_init__(self) -> None:
        if self.target_arch != "sm_120":
            raise ValueError("SM120 build target_arch must be sm_120")


def build_sm120_artifact(
    config: Sm120BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile SM120 FP8/NVFP4 probes and retain metadata-only maturity."""

    resolved = config or Sm120BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError("SM120 build preflight failed: " + "; ".join(report.reasons))
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM120 CUDA source not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = build_compile_flags(
        report,
        source=source,
        output=output,
        extra_flags=resolved.extra_flags,
    )
    try:
        subprocess.run(list(flags), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XQTBackendError(f"SM120 CUTLASS build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm120_low_precision_tcgen05",
        target_arch="sm_120",
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(128, 128, 128),
        warp_count=4,
        stage_count=4,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_rtx50_correctness_gate",
            "correctness_verified": False,
            "native_tcgen05_verified": False,
            "tma_verified": False,
            "cluster_shape": [1, 1, 1],
            "fp8_blockwise_probe_present": True,
            "fp8_groupwise_probe_present": True,
            "runtime_probe_present": True,
            "runtime_abi": "cutlass_internal_tiled_scale_layout_float32_and_ue4m3",
            "runtime_stream_abi": "torch_current_stream_void_p",
            "nvfp4_probe_present": True,
            "nvfp4_k256_probe_present": True,
            "nvfp4_tile_shapes": [[128, 128, 128], [128, 128, 256]],
            "nvfp4_schedule_variants": ["cooperative", "pingpong"],
            "dense_fp16_supported_by_builder": False,
            "nvfp4_runtime_routing": "explicit_packed_activation_and_ue4m3_scale_probe",
            "nvfp4_scale_layout": "sm1xx_blockscaled_sfvec16",
            "fp8_groupwise_pingpong_probe_present": True,
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def sm120_artifact_available(artifact: str | Path | None = None) -> bool:
    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    return path.expanduser().is_file()


def _sm120_fp8_function(
    artifact: str | Path,
    *,
    format_name: str,
    scale_granularity: str,
    schedule: str,
) -> Any:
    try:
        symbol = _SM120_FP8_SYMBOLS[(format_name, scale_granularity, schedule)]
    except KeyError as exc:
        raise XQTBackendError(
            "unsupported SM120 FP8 format/scale granularity: "
            f"{format_name}/{scale_granularity}/{schedule}"
        ) from exc
    return load_runtime_function(
        artifact,
        symbol=symbol,
        argtypes=[
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ],
    )


def _sm120_encoded_fp8(value: torch.Tensor, *, format_name: str, name: str) -> torch.Tensor:
    format_spec = fp8_format_spec(format_name)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise XQTBackendError(f"{name} must be a rank-2 FP8 tensor")
    if value.dtype == torch.uint8:
        return value.contiguous()
    if value.dtype == format_spec.torch_dtype:
        return value.contiguous().view(torch.uint8)
    raise XQTBackendError(
        f"{name} must be uint8 or {format_spec.torch_dtype}, got {value.dtype}"
    )


def _sm120_weight_payload(
    weight: torch.Tensor | PackedWeight,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(weight, PackedWeight):
        if not isinstance(weight.qweight, torch.Tensor):
            raise XQTBackendError("SM120 FP8 qweight must be a tensor")
        return weight.qweight, weight.scales
    if not isinstance(weight, torch.Tensor):
        raise XQTBackendError("SM120 FP8 weight must be a tensor or PackedWeight")
    return weight, None


def run_sm120_fp8_tcgen05_probe(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    format_name: str,
    scale_granularity: str,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor | None = None,
    schedule: str = "cooperative",
    artifact: str | Path,
) -> torch.Tensor:
    """Run the explicit SM120 FP8 tcgen05 probe on an RTX 5090.

    For ``blockwise``, cooperative scale shapes are
    ``[ceil(M/128), ceil(K/128)]`` and ``[ceil(N/128), ceil(K/128)]``.
    The blockwise pingpong tile uses ``SFVecM=64`` for A, so its A scale
    shape is ``[ceil(M/64), ceil(K/128)]``.
    For ``groupwise``, the actual CUTLASS probe uses
    ``Sm120BlockwiseScaleConfig<1,128,128>``: A scales are
    ``[M, ceil(K/128)]`` and B scales are
    ``[ceil(N/128), ceil(K/128)]``.  Neither mode accepts XQT scale tensors
    with a different K block size.
    """

    if format_name not in {"fp8_e4m3", "fp8_e5m2"}:
        raise XQTBackendError(f"unsupported SM120 FP8 format: {format_name!r}")
    if scale_granularity not in {"blockwise", "groupwise"}:
        raise XQTBackendError(
            "SM120 FP8 probe scale_granularity must be blockwise or groupwise"
        )
    if schedule not in {"cooperative", "pingpong"}:
        raise XQTBackendError("SM120 FP8 probe schedule must be cooperative or pingpong")
    qweight, packed_scales = _sm120_weight_payload(weight)
    if scale_b is None:
        scale_b = packed_scales
    if scale_b is None:
        raise XQTBackendError("SM120 FP8 probe requires scale_b")
    device = require_target_cuda(
        activation,
        capability=_SM120_CAPABILITY,
        name="SM120 FP8 probe",
    )
    require_same_cuda_device(
        activation,
        (qweight, scale_a, scale_b),
        names=("weight", "scale_a", "scale_b"),
    )
    if activation.ndim != 2 or qweight.ndim != 2:
        raise XQTBackendError("SM120 FP8 probe expects activation [M,K] and weight [N,K]")
    m, k = (int(item) for item in activation.shape)
    n, weight_k = (int(item) for item in qweight.shape)
    if weight_k != k:
        raise XQTBackendError(f"weight K={weight_k} does not match activation K={k}")
    a_bytes = _sm120_encoded_fp8(activation, format_name=format_name, name="activation")
    w_bytes = _sm120_encoded_fp8(qweight, format_name=format_name, name="weight")
    activation_runtime = pad_matrix(
        a_bytes,
        rows=m,
        cols=k,
        row_multiple=128,
        col_multiple=128,
    )
    weight_runtime = pad_matrix(
        w_bytes,
        rows=n,
        cols=k,
        row_multiple=128,
        col_multiple=128,
    )
    padded_m, padded_k = (int(item) for item in activation_runtime.shape)
    padded_n = int(weight_runtime.shape[0])
    row_block_a = (
        1
        if scale_granularity == "groupwise"
        else 64
        if schedule == "pingpong"
        else 128
    )
    scale_a_runtime = prepare_cutlass_blockscales(
        scale_a,
        rows=m,
        cols=k,
        padded_rows=padded_m,
        padded_cols=padded_k,
        row_block=row_block_a,
        block_k=128,
        name="scale_a",
    )
    scale_b_runtime = prepare_cutlass_blockscales(
        scale_b,
        rows=n,
        cols=k,
        padded_rows=padded_n,
        padded_cols=padded_k,
        row_block=128,
        block_k=128,
        name="scale_b",
    )
    c_source = torch.zeros(
        (padded_m, padded_n),
        device=device,
        dtype=torch.bfloat16,
    )
    output = torch.empty_like(c_source)
    function = _sm120_fp8_function(
        artifact,
        format_name=format_name,
        scale_granularity=scale_granularity,
        schedule=schedule,
    )
    symbol = _SM120_FP8_SYMBOLS[(format_name, scale_granularity, schedule)]
    error = function(
        padded_m,
        padded_n,
        padded_k,
        activation_runtime.data_ptr(),
        weight_runtime.data_ptr(),
        scale_a_runtime.data_ptr(),
        scale_b_runtime.data_ptr(),
        c_source.data_ptr(),
        output.data_ptr(),
        current_cuda_stream(device),
    )
    raise_runtime_error(symbol, error)
    return output[:m, :n]


def _pad_nvfp4_matrix(
    value: torch.Tensor,
    *,
    rows: int,
    logical_cols: int,
    padded_rows: int,
    padded_cols: int,
    name: str,
) -> torch.Tensor:
    expected_cols = (int(logical_cols) + 1) // 2
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise XQTBackendError(f"{name} must be a rank-2 packed NVFP4 tensor")
    if value.dtype != torch.uint8:
        raise XQTBackendError(f"{name} must use uint8 packed storage")
    actual_cols = int(value.shape[1])
    if int(value.shape[0]) != rows or actual_cols < expected_cols:
        raise XQTBackendError(
            f"{name} must have at least packed shape {(rows, expected_cols)}, "
            f"got {tuple(value.shape)}"
        )
    if actual_cols > int(padded_cols) // 2:
        raise XQTBackendError(
            f"{name} packed columns {actual_cols} exceed padded K {padded_cols}"
        )
    target = torch.zeros(
        (int(padded_rows), int(padded_cols) // 2),
        device=value.device,
        dtype=torch.uint8,
    )
    target[:rows, :actual_cols].copy_(value)
    return target


def run_sm120_nvfp4_probe(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    logical_k: int,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor | None = None,
    activation_global_scale: torch.Tensor | None = None,
    tile_k: int = 128,
    schedule: str = "cooperative",
    artifact: str | Path,
) -> torch.Tensor:
    """Run an explicit SM120 NVFP4 cooperative or pingpong probe.

    Both operands use packed uint8 E2M1 storage.  ``scale_a`` and ``scale_b``
    are conceptual ``[rows, ceil(K / 16)]`` grids; the helper converts them to
    CUTLASS's interleaved ``float_ue4m3_t`` storage.  A PackedWeight global
    scale is folded into the B scale before the call.
    """

    if int(logical_k) <= 0:
        raise ValueError("logical_k must be positive")
    if int(tile_k) not in {128, 256}:
        raise XQTBackendError("SM120 NVFP4 tile_k must be 128 or 256")
    if schedule not in {"cooperative", "pingpong"}:
        raise XQTBackendError("SM120 NVFP4 schedule must be cooperative or pingpong")
    qweight, packed_scales = _sm120_weight_payload(weight)
    if scale_b is None:
        scale_b = packed_scales
    if scale_b is None:
        raise XQTBackendError("SM120 NVFP4 probe requires scale_b")
    device = require_target_cuda(
        activation,
        capability=_SM120_CAPABILITY,
        name="SM120 NVFP4 probe",
    )
    require_same_cuda_device(
        activation,
        (qweight, scale_a, scale_b),
        names=("weight", "scale_a", "scale_b"),
    )
    if activation.ndim != 2 or qweight.ndim != 2:
        raise XQTBackendError(
            "SM120 NVFP4 probe expects packed activation [M,ceil(K/2)] "
            "and weight [N,ceil(K/2)]"
        )
    m = int(activation.shape[0])
    n = int(qweight.shape[0])
    weight_padded_k = (
        int(weight.metadata.padded_k)
        if isinstance(weight, PackedWeight)
        else int(qweight.shape[1]) * 2
    )
    if weight_padded_k < int(logical_k):
        raise XQTBackendError("PackedWeight padded_k cannot be smaller than logical_k")
    if int(qweight.shape[1]) != (weight_padded_k + 1) // 2:
        raise XQTBackendError(
            "SM120 NVFP4 weight storage does not match PackedWeight padded_k"
        )
    padded_k = (
        (max(int(logical_k), weight_padded_k) + int(tile_k) - 1)
        // int(tile_k)
    ) * int(tile_k)
    padded_m = ((m + 127) // 128) * 128
    padded_n = ((n + 127) // 128) * 128
    activation_runtime = _pad_nvfp4_matrix(
        activation,
        rows=m,
        logical_cols=int(logical_k),
        padded_rows=padded_m,
        padded_cols=padded_k,
        name="activation",
    )
    weight_runtime = _pad_nvfp4_matrix(
        qweight,
        rows=n,
        logical_cols=weight_padded_k,
        padded_rows=padded_n,
        padded_cols=padded_k,
        name="weight",
    )
    weight_global_scale = (
        weight.global_scale if isinstance(weight, PackedWeight) else None
    )
    scale_a_runtime = prepare_cutlass_nvfp4_scales(
        scale_a,
        rows=m,
        cols=int(logical_k),
        padded_rows=padded_m,
        padded_cols=padded_k,
        global_scale=activation_global_scale,
        name="scale_a",
    )
    scale_b_runtime = prepare_cutlass_nvfp4_scales(
        scale_b,
        rows=n,
        cols=weight_padded_k,
        padded_rows=padded_n,
        padded_cols=padded_k,
        global_scale=weight_global_scale,
        name="scale_b",
    )
    c_source = torch.zeros(
        (padded_m, padded_n),
        device=device,
        dtype=torch.bfloat16,
    )
    output = torch.empty_like(c_source)
    symbol = _SM120_NVFP4_SYMBOLS[(int(tile_k), schedule)]
    function = load_runtime_function(
        artifact,
        symbol=symbol,
        argtypes=[
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ],
    )
    error = function(
        padded_m,
        padded_n,
        padded_k,
        activation_runtime.data_ptr(),
        weight_runtime.data_ptr(),
        scale_a_runtime.data_ptr(),
        scale_b_runtime.data_ptr(),
        c_source.data_ptr(),
        output.data_ptr(),
        current_cuda_stream(device),
    )
    raise_runtime_error(symbol, error)
    return output[:m, :n]


def sm120_gemm_executor(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    artifact: str | Path | None = None,
) -> torch.Tensor:
    """Reject native execution until RTX 50 correctness evidence exists."""

    del activation, weight, spec, weight_scales, activation_scales, artifact
    raise XQTBackendError(
        "SM120 FP8/NVFP4 CUTLASS executor is metadata_only; RTX 50 "
        "correctness and SASS evidence are required"
    )


def sm120_gemm_reference(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference result for SM120 shape and scale validation."""

    return reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=weight_scales if weight_scales is not None else weight.scales,
        activation_scales=activation_scales,
    )


def install_sm120_executor(*_: Any, **__: Any) -> bool:
    """Keep SM120 metadata-only until target hardware gates pass."""

    return False


__all__ = [
    "Sm120BuildConfig",
    "Sm120Fp8Contract",
    "Sm120Nvfp4Contract",
    "build_sm120_artifact",
    "install_sm120_executor",
    "run_sm120_fp8_tcgen05_probe",
    "run_sm120_nvfp4_probe",
    "sm120_artifact_available",
    "sm120_gemm_executor",
    "sm120_gemm_reference",
]
