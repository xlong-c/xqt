"""SM90 FP8 WGMMA/TMA capability boundary.

The repository does not currently have an SM90 validation device.  This
module therefore owns the independent contract and build/manifest path while
keeping execution reference-guarded.  It must not reuse the SM89 executor or
artifact.
"""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from xqt.core.errors import XQTBackendError

from xqt.gemm.common.contracts import GemmSpec, GroupedGemmProblem, PackedWeight, QuantSpec
from xqt.gemm.common.fp8 import fp8_format_spec
from xqt.gemm.common.preflight import (
    GemmArtifactManifest,
    build_compile_flags,
    default_cache_dir,
    probe_cuda_cutlass,
)
from xqt.gemm.common.reference import reference_gemm, reference_packed_grouped_gemm
from xqt.gemm.backends._sm1xx_runtime import (
    current_cuda_stream,
    load_runtime_function,
    pad_matrix,
    prepare_cutlass_blockscales,
    raise_runtime_error,
    require_same_cuda_device,
    require_target_cuda,
)


_SOURCE = Path(__file__).with_name("sm90_fp8_wgmma.cu")
_DEFAULT_ARTIFACT = default_cache_dir() / "sm90" / "fp8_wgmma_sm90.so"
_DEFAULT_GROUPED_ARTIFACT = default_cache_dir() / "sm90" / "fp8_grouped_wgmma_sm90.so"
_SM90_CAPABILITY = (9, 0)
_SM90_DENSE_SYMBOLS = {
    ("fp16", "auto", 128, "2x1"): "xqt_sm90_dense_fp16_wgmma_run",
    ("bf16", "auto", 128, "2x1"): "xqt_sm90_dense_bf16_wgmma_run",
    ("fp16", "cooperative", 128, "1x1"):
        "xqt_sm90_dense_cooperative_128_fp16_wgmma_run",
    ("bf16", "cooperative", 128, "1x1"):
        "xqt_sm90_dense_cooperative_128_bf16_wgmma_run",
    ("fp16", "pingpong", 128, "1x1"):
        "xqt_sm90_dense_pingpong_128_fp16_wgmma_run",
    ("bf16", "pingpong", 128, "1x1"):
        "xqt_sm90_dense_pingpong_128_bf16_wgmma_run",
    ("fp16", "cooperative", 256, "1x2"):
        "xqt_sm90_dense_cooperative_256_fp16_wgmma_run",
    ("bf16", "cooperative", 256, "1x2"):
        "xqt_sm90_dense_cooperative_256_bf16_wgmma_run",
    ("fp16", "cooperative", 128, "2x2"):
        "xqt_sm90_dense_cooperative_128_cluster2x2_fp16_wgmma_run",
    ("bf16", "cooperative", 128, "2x2"):
        "xqt_sm90_dense_cooperative_128_cluster2x2_bf16_wgmma_run",
    ("fp16", "pingpong", 128, "2x2"):
        "xqt_sm90_dense_pingpong_128_cluster2x2_fp16_wgmma_run",
    ("bf16", "pingpong", 128, "2x2"):
        "xqt_sm90_dense_pingpong_128_cluster2x2_bf16_wgmma_run",
    ("fp16", "pingpong", 64, "2x2"):
        "xqt_sm90_dense_pingpong_64_cluster2x2_fp16_wgmma_run",
    ("bf16", "pingpong", 64, "2x2"):
        "xqt_sm90_dense_pingpong_64_cluster2x2_bf16_wgmma_run",
}
_SM90_FP8_SYMBOLS = {
    ("fp8_e4m3", "blockwise", "cooperative", "fp16"):
        "xqt_sm90_fp8_e4m3_fp16_wgmma_run",
    ("fp8_e4m3", "blockwise", "cooperative", "bf16"):
        "xqt_sm90_fp8_e4m3_bf16_wgmma_run",
    ("fp8_e5m2", "blockwise", "cooperative", "fp16"):
        "xqt_sm90_fp8_e5m2_fp16_wgmma_run",
    ("fp8_e5m2", "blockwise", "cooperative", "bf16"):
        "xqt_sm90_fp8_e5m2_bf16_wgmma_run",
    ("fp8_e4m3", "groupwise", "pingpong", "fp16"):
        "xqt_sm90_fp8_e4m3_groupwise_pingpong_fp16_run",
    ("fp8_e4m3", "groupwise", "pingpong", "bf16"):
        "xqt_sm90_fp8_e4m3_groupwise_pingpong_bf16_run",
    ("fp8_e5m2", "groupwise", "pingpong", "fp16"):
        "xqt_sm90_fp8_e5m2_groupwise_pingpong_fp16_run",
    ("fp8_e5m2", "groupwise", "pingpong", "bf16"):
        "xqt_sm90_fp8_e5m2_groupwise_pingpong_bf16_run",
    ("fp8_e4m3", "groupwise", "cooperative", "fp16"):
        "xqt_sm90_fp8_e4m3_groupwise_cooperative_256_fp16_run",
    ("fp8_e4m3", "groupwise", "cooperative", "bf16"):
        "xqt_sm90_fp8_e4m3_groupwise_cooperative_256_bf16_run",
    ("fp8_e5m2", "groupwise", "cooperative", "fp16"):
        "xqt_sm90_fp8_e5m2_groupwise_cooperative_256_fp16_run",
    ("fp8_e5m2", "groupwise", "cooperative", "bf16"):
        "xqt_sm90_fp8_e5m2_groupwise_cooperative_256_bf16_run",
}


@dataclass(frozen=True, slots=True)
class Sm90Fp8WgmmaContract:
    """Fine-grained FP8 scale contract for Hopper WGMMA/TMA."""

    format_name: str = "fp8_e4m3"
    weight_granularity: str = "blockwise"
    activation_granularity: str = "blockwise"
    block_k: int = 64
    output_dtype: str = "fp16"
    use_wgmma: bool = True
    use_tma: bool = True
    grouped: bool = False
    scale_mainloop: bool = True
    pack_version: str = "xqt-sm90-fp8-wgmma-v1"

    def __post_init__(self) -> None:
        if self.format_name not in {"fp8_e4m3", "fp8_e5m2"}:
            raise ValueError("SM90 WGMMA format must be fp8_e4m3 or fp8_e5m2")
        if self.weight_granularity not in {"per_tensor", "per_channel", "blockwise"}:
            raise ValueError("unsupported SM90 weight scale granularity")
        if self.activation_granularity not in {"per_tensor", "per_token", "blockwise"}:
            raise ValueError("unsupported SM90 activation scale granularity")
        if self.weight_granularity == "blockwise" or self.activation_granularity == "blockwise":
            if int(self.block_k) not in {32, 64, 128}:
                raise ValueError("SM90 blockwise FP8 block_k must be 32, 64, or 128")
        if (
            self.activation_granularity == "blockwise"
            and self.weight_granularity != "blockwise"
        ):
            raise ValueError(
                "SM90 blockwise activation requires blockwise weight under the shared group_size ABI"
            )
        if self.output_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("SM90 WGMMA output_dtype must be fp16, bf16, or fp32")
        if not all(
            isinstance(value, bool)
            for value in (self.use_wgmma, self.use_tma, self.grouped, self.scale_mainloop)
        ):
            raise TypeError("SM90 WGMMA/TMA flags must be bool")

    @property
    def capability(self) -> str:
        return "sm90_wgmma_tma"

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype=self.format_name,
            activation_dtype=self.format_name,
            output_dtype=self.output_dtype,
            weight_granularity=self.weight_granularity,
            activation_granularity=self.activation_granularity,
            group_size=int(self.block_k) if (
                self.weight_granularity == "blockwise"
                or self.activation_granularity == "blockwise"
            ) else None,
            weight_scale_source="weight_offline",
            activation_scale_source="activation_static",
            storage_layout="xqt_fp8_rowmajor_v1",
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "sm_90",
            "format": self.format_name,
            "weight_granularity": self.weight_granularity,
            "activation_granularity": self.activation_granularity,
            "block_k": int(self.block_k),
            "output_dtype": self.output_dtype,
            "use_wgmma": self.use_wgmma,
            "use_tma": self.use_tma,
            "grouped": self.grouped,
            "scale_mainloop": self.scale_mainloop,
            "pack_version": self.pack_version,
            "maturity": "metadata_only",
        }


Sm90GroupedFp8WgmmaContract = Sm90Fp8WgmmaContract


@dataclass(frozen=True, slots=True)
class Sm90Fp8WgmmaBuildConfig:
    """Independent SM90 CollectiveBuilder compile inputs."""

    source: Path = _SOURCE
    output: Path = field(default_factory=lambda: _DEFAULT_ARTIFACT)
    target_arch: str = "sm_90"
    grouped: bool = False
    extra_flags: tuple[str, ...] = ("-lineinfo", "-lcudart")

    def __post_init__(self) -> None:
        if self.target_arch != "sm_90":
            raise ValueError("SM90 WGMMA build target_arch must be sm_90")


Sm90GroupedFp8WgmmaBuildConfig = Sm90Fp8WgmmaBuildConfig


def _build(config: Sm90Fp8WgmmaBuildConfig, *, kernel_name: str) -> GemmArtifactManifest:
    report = probe_cuda_cutlass(config.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError("SM90 WGMMA build preflight failed: " + "; ".join(report.reasons))
    source = config.source.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM90 WGMMA source not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = build_compile_flags(report, source=source, output=output, extra_flags=config.extra_flags)
    try:
        subprocess.run(list(flags), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XQTBackendError(f"SM90 WGMMA CollectiveBuilder build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing artifact: {output}")
    is_fp8 = "fp8" in kernel_name
    manifest = GemmArtifactManifest(
        kernel_name=kernel_name,
        target_arch="sm_90",
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(128, 128, 128) if is_fp8 else (128, 128, 64),
        warp_count=4,
        stage_count=4,
        preflight=report,
        metadata={
            "build_status": "collective_builder_compiled_pending_wgmma_validation",
            "correctness_verified": False,
            "native_wgmma_verified": False,
            "tma_verified": False,
            "dense_probe_present": True,
            "dense_schedule_variants": [
                {
                    "schedule": "auto",
                    "tile_shape": [128, 128, 64],
                    "cluster_shape": [2, 1, 1],
                },
                {
                    "schedule": "cooperative",
                    "tile_shape": [128, 128, 64],
                    "cluster_shape": [1, 1, 1],
                },
                {
                    "schedule": "pingpong",
                    "tile_shape": [128, 128, 64],
                    "cluster_shape": [1, 1, 1],
                },
                {
                    "schedule": "cooperative",
                    "tile_shape": [256, 128, 64],
                    "cluster_shape": [1, 2, 1],
                },
                {
                    "schedule": "cooperative",
                    "tile_shape": [128, 128, 64],
                    "cluster_shape": [2, 2, 1],
                },
                {
                    "schedule": "pingpong",
                    "tile_shape": [128, 128, 64],
                    "cluster_shape": [2, 2, 1],
                },
                {
                    "schedule": "pingpong",
                    "tile_shape": [64, 128, 64],
                    "cluster_shape": [2, 2, 1],
                },
            ],
            "fp8_probe_present": True,
            "fp8_blockwise_schedule": "cooperative",
            "fp8_groupwise_pingpong_probe_present": True,
            "fp8_groupwise_schedule_variants": [
                {
                    "schedule": "pingpong",
                    "tile_shape": [128, 128, 128],
                },
                {
                    "schedule": "cooperative",
                    "tile_shape": [256, 128, 128],
                },
            ],
            "fp8_groupwise_scale_layout": "sm90_blockwise_1x128x128_sfa_mn_sfb_k",
            "runtime_probe_present": True,
            "runtime_abi": "cutlass_internal_tiled_scale_layout_float32",
            "runtime_stream_abi": "torch_current_stream_void_p",
            "scale_application": "fine_grained_mainloop_contract_only",
            "alignment": [16, 16, 32] if is_fp8 else [8, 8, 8],
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm90_fp8_wgmma_artifact(
    config: Sm90Fp8WgmmaBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Build the SM90 FP8 CollectiveBuilder probe and retain metadata-only maturity."""

    resolved = config or Sm90Fp8WgmmaBuildConfig()
    return _build(resolved, kernel_name="sm90_fp8_wgmma")


def build_sm90_dense_artifact(
    config: Sm90Fp8WgmmaBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Build the SM90 dense WGMMA CollectiveBuilder probe."""

    resolved = config or Sm90Fp8WgmmaBuildConfig(
        output=default_cache_dir() / "sm90" / "dense_wgmma_sm90.so",
    )
    return _build(resolved, kernel_name="sm90_dense_wgmma")


def build_sm90_grouped_fp8_wgmma_artifact(
    config: Sm90GroupedFp8WgmmaBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Build the independent grouped SM90 scaffold."""

    resolved = config or Sm90Fp8WgmmaBuildConfig(
        output=_DEFAULT_GROUPED_ARTIFACT,
        grouped=True,
    )
    if not resolved.grouped:
        resolved = Sm90Fp8WgmmaBuildConfig(
            source=resolved.source,
            output=resolved.output,
            target_arch=resolved.target_arch,
            grouped=True,
            extra_flags=resolved.extra_flags,
        )
    return _build(resolved, kernel_name="sm90_grouped_fp8_wgmma")


def sm90_fp8_wgmma_artifact_available(artifact: str | Path | None = None) -> bool:
    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    return path.expanduser().is_file()


def sm90_grouped_fp8_wgmma_artifact_available(artifact: str | Path | None = None) -> bool:
    path = Path(artifact) if artifact is not None else _DEFAULT_GROUPED_ARTIFACT
    return path.expanduser().is_file()


def _sm90_dense_function(
    artifact: str | Path,
    *,
    dtype_name: str,
    schedule: str,
    tile_m: int,
    cluster_shape: str,
) -> Any:
    try:
        symbol = _SM90_DENSE_SYMBOLS[
            (dtype_name, schedule, int(tile_m), cluster_shape)
        ]
    except KeyError as exc:
        raise XQTBackendError(
            "unsupported SM90 dense dtype/schedule/tile/cluster: "
            f"{dtype_name}/{schedule}/{tile_m}/{cluster_shape}"
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
        ],
    )


def _sm90_fp8_function(
    artifact: str | Path,
    *,
    format_name: str,
    scale_granularity: str,
    schedule: str,
    output_dtype: str,
) -> Any:
    try:
        symbol = _SM90_FP8_SYMBOLS[
            (format_name, scale_granularity, schedule, output_dtype)
        ]
    except KeyError as exc:
        raise XQTBackendError(
            "unsupported SM90 FP8 format/scale/schedule/output: "
            f"{format_name}/{scale_granularity}/{schedule}/{output_dtype}"
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


def _sm90_encoded_fp8(value: torch.Tensor, *, format_name: str, name: str) -> torch.Tensor:
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


def _sm90_weight_payload(
    weight: torch.Tensor | PackedWeight,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(weight, PackedWeight):
        if not isinstance(weight.qweight, torch.Tensor):
            raise XQTBackendError("SM90 FP8 qweight must be a tensor")
        return weight.qweight, weight.scales
    if not isinstance(weight, torch.Tensor):
        raise XQTBackendError("SM90 FP8 weight must be a tensor or PackedWeight")
    return weight, None


def run_sm90_dense_wgmma_probe(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    schedule: str = "auto",
    tile_m: int = 128,
    cluster_shape: str | None = None,
    artifact: str | Path,
) -> torch.Tensor:
    """Run the explicit SM90 dense WGMMA probe on an H100.

    The logical input layout is ``activation[M,K]`` and ``weight[N,K]``.
    Weight remains contiguous ``[N,K]`` storage because CUTLASS interprets
    that buffer as column-major ``KxN``.  This function is intentionally
    outside the automatic registry dispatch.
    """

    if schedule not in {"auto", "cooperative", "pingpong"}:
        raise XQTBackendError(
            "SM90 dense probe schedule must be auto, cooperative, or pingpong"
        )
    if int(tile_m) not in {64, 128, 256}:
        raise XQTBackendError("SM90 dense probe tile_m must be 64, 128, or 256")
    if int(tile_m) == 256 and schedule != "cooperative":
        raise XQTBackendError("SM90 dense tile_m=256 only exposes cooperative schedule")
    resolved_cluster = (
        "2x1"
        if cluster_shape is None and schedule == "auto"
        else "1x1"
        if cluster_shape is None
        else cluster_shape
    )
    if resolved_cluster not in {"1x1", "1x2", "2x1", "2x2"}:
        raise XQTBackendError(
            "SM90 dense probe cluster_shape must be 1x1, 1x2, 2x1, or 2x2"
        )
    if schedule == "auto" and resolved_cluster != "2x1":
        raise XQTBackendError("SM90 dense auto schedule requires cluster_shape=2x1")
    if int(tile_m) == 256 and resolved_cluster != "1x2":
        raise XQTBackendError("SM90 dense tile_m=256 requires cluster_shape=1x2")
    if int(tile_m) == 128 and schedule != "auto" and resolved_cluster not in {
        "1x1",
        "2x2",
    }:
        raise XQTBackendError(
            "SM90 dense tile_m=128 explicit schedules require cluster_shape=1x1 or 2x2"
        )
    if int(tile_m) == 64 and (
        schedule != "pingpong" or resolved_cluster != "2x2"
    ):
        raise XQTBackendError(
            "SM90 dense tile_m=64 requires pingpong schedule and cluster_shape=2x2"
        )
    if not isinstance(activation, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise XQTBackendError("SM90 dense probe requires tensor inputs")
    device = require_target_cuda(
        activation,
        capability=_SM90_CAPABILITY,
        name="SM90 dense probe",
    )
    require_same_cuda_device(activation, (weight,), names=("weight",))
    if activation.ndim != 2 or weight.ndim != 2:
        raise XQTBackendError("SM90 dense probe expects activation [M,K] and weight [N,K]")
    m, k = (int(item) for item in activation.shape)
    n, weight_k = (int(item) for item in weight.shape)
    if weight_k != k:
        raise XQTBackendError(f"weight K={weight_k} does not match activation K={k}")
    if activation.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("SM90 dense probe supports fp16 or bf16 inputs")
    if weight.dtype != activation.dtype:
        raise XQTBackendError("SM90 dense probe requires matching A/W dtypes")
    dtype_name = "fp16" if activation.dtype == torch.float16 else "bf16"
    activation_runtime = pad_matrix(
        activation,
        rows=m,
        cols=k,
        row_multiple=int(tile_m),
        col_multiple=64,
    )
    weight_runtime = pad_matrix(
        weight,
        rows=n,
        cols=k,
        row_multiple=128,
        col_multiple=64,
    )
    padded_m, padded_k = (int(item) for item in activation_runtime.shape)
    padded_n = int(weight_runtime.shape[0])
    c_source = torch.zeros(
        (padded_m, padded_n),
        device=device,
        dtype=activation.dtype,
    )
    output = torch.empty_like(c_source)
    function = _sm90_dense_function(
        artifact,
        dtype_name=dtype_name,
        schedule=schedule,
        tile_m=int(tile_m),
        cluster_shape=resolved_cluster,
    )
    error = function(
        padded_m,
        padded_n,
        padded_k,
        activation_runtime.data_ptr(),
        weight_runtime.data_ptr(),
        c_source.data_ptr(),
        output.data_ptr(),
        current_cuda_stream(device),
    )
    symbol = _SM90_DENSE_SYMBOLS[
        (dtype_name, schedule, int(tile_m), resolved_cluster)
    ]
    raise_runtime_error(symbol, error)
    return output[:m, :n]


def run_sm90_fp8_wgmma_probe(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    format_name: str,
    output_dtype: str,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor | None = None,
    scale_granularity: str = "blockwise",
    schedule: str = "cooperative",
    artifact: str | Path,
) -> torch.Tensor:
    """Run SM90 FP8 WGMMA with the explicit CUTLASS tiled scale ABI.

    For blockwise mode, ``scale_a`` has shape
    ``[ceil(M/128), ceil(K/128)]``.  For groupwise mode it has shape
    ``[M, ceil(K/128)]``.  ``scale_b`` has shape
    ``[ceil(N/128), ceil(K/128)]`` in both modes.  These are CUTLASS tiled
    scales, not XQT's canonical per-row ``[rows, ceil(K/G)]`` tensors.
    """

    if format_name not in {"fp8_e4m3", "fp8_e5m2"}:
        raise XQTBackendError(f"unsupported SM90 FP8 format: {format_name!r}")
    if output_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("SM90 FP8 probe output_dtype must be fp16 or bf16")
    if scale_granularity not in {"blockwise", "groupwise"}:
        raise XQTBackendError(
            "SM90 FP8 probe scale_granularity must be blockwise or groupwise"
        )
    if schedule not in {"cooperative", "pingpong"}:
        raise XQTBackendError("SM90 FP8 probe schedule must be cooperative or pingpong")
    if scale_granularity == "blockwise" and schedule != "cooperative":
        raise XQTBackendError("SM90 FP8 blockwise probe only exposes cooperative schedule")
    qweight, packed_scales = _sm90_weight_payload(weight)
    if scale_b is None and packed_scales is not None:
        scale_b = packed_scales
    if scale_b is None:
        raise XQTBackendError("SM90 FP8 probe requires scale_b")
    device = require_target_cuda(
        activation,
        capability=_SM90_CAPABILITY,
        name="SM90 FP8 probe",
    )
    require_same_cuda_device(
        activation,
        (qweight, scale_a, scale_b),
        names=("weight", "scale_a", "scale_b"),
    )
    if activation.ndim != 2 or qweight.ndim != 2:
        raise XQTBackendError("SM90 FP8 probe expects activation [M,K] and weight [N,K]")
    m, k = (int(item) for item in activation.shape)
    n, weight_k = (int(item) for item in qweight.shape)
    if weight_k != k:
        raise XQTBackendError(f"weight K={weight_k} does not match activation K={k}")
    a_bytes = _sm90_encoded_fp8(activation, format_name=format_name, name="activation")
    w_bytes = _sm90_encoded_fp8(qweight, format_name=format_name, name="weight")
    activation_runtime = pad_matrix(
        a_bytes,
        rows=m,
        cols=k,
        row_multiple=(
            256
            if scale_granularity == "groupwise" and schedule == "cooperative"
            else 128
        ),
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
    if scale_granularity == "groupwise":
        scale_a_runtime = prepare_cutlass_blockscales(
            scale_a,
            rows=m,
            cols=k,
            padded_rows=padded_m,
            padded_cols=padded_k,
            row_block=1,
            block_k=128,
            major="mn",
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
            major="k",
            name="scale_b",
        )
    else:
        scale_a_runtime = prepare_cutlass_blockscales(
            scale_a,
            rows=m,
            cols=k,
            padded_rows=padded_m,
            padded_cols=padded_k,
            name="scale_a",
        )
        scale_b_runtime = prepare_cutlass_blockscales(
            scale_b,
            rows=n,
            cols=k,
            padded_rows=padded_n,
            padded_cols=padded_k,
            name="scale_b",
        )
    output_torch_dtype = torch.float16 if output_dtype == "fp16" else torch.bfloat16
    c_source = torch.zeros(
        (padded_m, padded_n),
        device=device,
        dtype=output_torch_dtype,
    )
    output = torch.empty_like(c_source)
    function = _sm90_fp8_function(
        artifact,
        format_name=format_name,
        scale_granularity=scale_granularity,
        schedule=schedule,
        output_dtype=output_dtype,
    )
    symbol = _SM90_FP8_SYMBOLS[
        (format_name, scale_granularity, schedule, output_dtype)
    ]
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


def sm90_fp8_wgmma_executor(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    artifact: str | Path | None = None,
) -> torch.Tensor:
    """Reject native execution until an SM90 WGMMA correctness gate exists."""

    del activation, weight, spec, weight_scales, activation_scales, artifact
    raise XQTBackendError(
        "SM90 FP8 WGMMA/TMA executor is metadata_only; target SM90 hardware "
        "and an independently validated WGMMA artifact are required"
    )


def sm90_fp8_wgmma_reference(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference result for the SM90 contract, for shape and scale gates."""

    return reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=weight_scales if weight_scales is not None else weight.scales,
        activation_scales=activation_scales,
    )


def sm90_grouped_fp8_wgmma_reference(
    grouped_problem: GroupedGemmProblem,
    activation: torch.Tensor,
    weights: Sequence[PackedWeight],
    *,
    quant: QuantSpec,
    weight_scales: Sequence[torch.Tensor | None] | None = None,
    activation_scales: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Reference grouped SM90 result with empty-expert and scatter coverage."""

    return reference_packed_grouped_gemm(
        grouped_problem,
        activation,
        weights,
        quant_specs=quant,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
        bias=bias,
    )


def install_sm90_fp8_wgmma_executor(*_: Any, **__: Any) -> bool:
    """Never promote the scaffold without target-specific native evidence."""

    return False


install_sm90_grouped_fp8_wgmma_executor = install_sm90_fp8_wgmma_executor


__all__ = [
    "Sm90Fp8WgmmaBuildConfig",
    "Sm90Fp8WgmmaContract",
    "Sm90GroupedFp8WgmmaBuildConfig",
    "Sm90GroupedFp8WgmmaContract",
    "build_sm90_dense_artifact",
    "build_sm90_fp8_wgmma_artifact",
    "build_sm90_grouped_fp8_wgmma_artifact",
    "install_sm90_fp8_wgmma_executor",
    "install_sm90_grouped_fp8_wgmma_executor",
    "run_sm90_dense_wgmma_probe",
    "run_sm90_fp8_wgmma_probe",
    "sm90_fp8_wgmma_artifact_available",
    "sm90_fp8_wgmma_executor",
    "sm90_fp8_wgmma_reference",
    "sm90_grouped_fp8_wgmma_artifact_available",
    "sm90_grouped_fp8_wgmma_reference",
]
