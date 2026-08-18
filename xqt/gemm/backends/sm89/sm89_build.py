"""Explicit SM89 CUDA/CUTLASS build entry.

The build entry performs preflight first, records the exact command, and leaves
the resulting artifact at ``metadata_only`` until a correctness gate promotes
the registry entry.  It is intentionally a small Python API, not a serving or
training CLI.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from xqt.core.errors import XQTBackendError

from xqt.gemm.common.preflight import (
    GemmArtifactManifest,
    build_compile_flags,
    default_cache_dir,
    probe_cuda_cutlass,
)


_SEED_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "operator_opt"
    / "kernels"
    / "cute"
    / "int8mma_kernel.cu"
)
_DENSE_SOURCE = Path(__file__).resolve().with_name("dense_sm89.cu")
_W4A16_SOURCE = Path(__file__).resolve().with_name("w4a16_sm89.cu")
_W4A16_FUSED_SOURCE = Path(__file__).resolve().with_name("w4a16_cutlass_fused_sm89.cu")
_W4A16_GROUPED_SOURCE = Path(__file__).resolve().with_name("w4a16_grouped_sm89.cu")
_W8A8_GROUPED_SOURCE = Path(__file__).resolve().with_name("w8a8_grouped_sm89.cu")
_W8A16_SOURCE = _SEED_SOURCE
_W4A8_SOURCE = Path(__file__).resolve().with_name("w4a8_sm89.cu")
_FP8_PROBE_SOURCE = Path(__file__).resolve().with_name("fp8_cutlass_probe_sm89.cu")
_FP8_SOURCE = Path(__file__).resolve().with_name("fp8_cutlass_sm89.cu")
_FP8_GROUPED_SOURCE = Path(__file__).resolve().with_name("fp8_grouped_sm89.cu")
_MIXED_INPUT_PROBE_SOURCE = Path(__file__).resolve().with_name("mixed_input_probe_sm89.cu")


@dataclass(frozen=True, slots=True)
class Sm89BuildConfig:
    """Inputs for one explicit SM89 artifact build."""

    source: Path = _SEED_SOURCE
    output: Path = field(default_factory=lambda: default_cache_dir() / "sm89" / "int8mma_sm89.so")
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = (
        "-use_fast_math",
        "-lineinfo",
        "-lcublasLt",
        "-lcublas",
        "-lcudart",
    )

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("SM89 build source must be non-empty")
        if not self.output:
            raise ValueError("SM89 build output must be non-empty")


@dataclass(frozen=True, slots=True)
class Sm89DenseBuildConfig:
    """Inputs for the FP16/BF16 dense SM89 artifact."""

    source: Path = _DENSE_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "dense_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89W4A16BuildConfig:
    """Inputs for the explicitly named W4A16 dequant fallback artifact."""

    source: Path = _W4A16_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "w4a16_dequant_fallback_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89W4A16FusedBuildConfig:
    """Inputs for the experimental CUTLASS warp-MMA W4A16 artifact."""

    source: Path = _W4A16_FUSED_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "w4a16_cutlass_fused_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89GroupedW4A16BuildConfig:
    """Inputs for the manifest-gated grouped W4A16 decode artifact."""

    source: Path = _W4A16_GROUPED_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "w4a16_grouped_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89GroupedW8A8BuildConfig:
    """Inputs for the manifest-gated grouped W8A8 native MMA artifact."""

    source: Path = _W8A8_GROUPED_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "w8a8_grouped_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89W8A16BuildConfig:
    """Inputs for the SM89 W8A16 artifact build using INT8 MMA kernel as main path."""

    source: Path = _W8A16_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "w8a16_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = (
        "-use_fast_math",
        "-lineinfo",
        "-lcublasLt",
        "-lcublas",
        "-lcudart",
    )


@dataclass(frozen=True, slots=True)
class Sm89W4A8BuildConfig:
    """Inputs for the SM89 native W4A8 INT8/FP8 group-scale mainloop artifact."""

    source: Path = _W4A8_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "w4a8_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89Fp8ProbeBuildConfig:
    """Inputs for the non-production SM89 E4M3/E5M2 MMA capability probe."""

    source: Path = _FP8_PROBE_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "fp8_cutlass_probe_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89Fp8BuildConfig:
    """Inputs for the SM89 FP8 CUTLASS GEMM artifact."""

    source: Path = _FP8_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "fp8_cutlass_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89GroupedFp8BuildConfig:
    """Inputs for the manifest-gated grouped FP8 native MMA artifact."""

    source: Path = _FP8_GROUPED_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "fp8_grouped_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-use_fast_math", "-lineinfo", "-lcudart")


@dataclass(frozen=True, slots=True)
class Sm89MixedInputProbeBuildConfig:
    """Inputs for the non-production CUTLASS mixed-input capability probe."""

    source: Path = _MIXED_INPUT_PROBE_SOURCE
    output: Path = field(
        default_factory=lambda: default_cache_dir() / "sm89" / "mixed_input_probe_sm89.so"
    )
    target_arch: str = "sm_89"
    extra_flags: tuple[str, ...] = ("-lineinfo", "-lcudart")


def build_sm89_artifact(config: Sm89BuildConfig | None = None) -> GemmArtifactManifest:
    """Compile one SM89 artifact after preflight and write its manifest."""

    resolved = config or Sm89BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_int8_mma_cutlass",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(64, 128, 64),
        warp_count=8,
        stage_count=3,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_dense_artifact(
    config: Sm89DenseBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the dense CUTLASS seed and keep it metadata-only pending tests."""

    resolved = config or Sm89DenseBuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 dense build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 dense CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 dense nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing dense artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_dense_cutlass",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(128, 128, 32),
        warp_count=4,
        stage_count=3,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_w4a16_dequant_artifact(
    config: Sm89W4A16BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the W4A16 fallback and keep it metadata-only pending tests."""

    resolved = config or Sm89W4A16BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 W4A16 fallback build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 W4A16 fallback CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 W4A16 fallback nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing W4A16 artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_w4a16_dequant_fallback",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(8, 16, 32),
        warp_count=4,
        stage_count=1,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "kernel_role": "dequant_fallback",
            "cutlass_mainloop": False,
            "shape_variants": {
                "m1_gemv": {"block_threads": 256, "persistent": False},
                "small_m_2_8": {"block_threads": 128, "persistent": False},
                "tile_m_8x16x32": {
                    "block_threads": 128,
                    "persistent_supported": True,
                    "default_persistent": False,
                },
            },
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_w4a16_fused_artifact(
    config: Sm89W4A16FusedBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the CUTLASS warp-MMA W4A16 experiment as metadata-only."""

    resolved = config or Sm89W4A16FusedBuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 fused W4A16 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 fused W4A16 CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 fused W4A16 nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing fused W4A16 artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_w4a16_cutlass_fused_mma",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(16, 8, 16),
        warp_count=1,
        stage_count=1,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "kernel_role": "fused_groupwise_mainloop",
            "cutlass_mainloop": True,
            "implementation": "custom_cuda_cutlass_mma",
            "scale_application": "k_tile_before_mma",
            "supported_alignment": {"m": 16, "n": 8, "k": 16},
            "logical_k_padding": "adapter_zero_pad_to_16_within_padded_k",
            "split_k": {
                "abi": "workspace_reduction",
                "workspace": "float32 [split_count, M, N]",
                "reduction": "second_kernel_deterministic_sum",
                "opt_in": "executor split_k>=2, split_k=1 keeps full-K ABI",
            },
            "decode_m_1_8": {
                "abi": "k_parallel_simt_block_per_column row_bound_template_1_2_4_8",
                "replaces": "old m1_gemv/small_m_2_8 decode branch (kept in fallback artifact)",
                "nibble_semantics": "shared with prefill mainloop, signed GPTQ / unsigned AWQ",
            },
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_grouped_w4a16_artifact(
    config: Sm89GroupedW4A16BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the grouped W4A16 decode kernel pending numeric promotion."""

    resolved = config or Sm89GroupedW4A16BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 grouped W4A16 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 grouped W4A16 CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 grouped W4A16 nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(
            f"nvcc completed without producing grouped W4A16 artifact: {output}"
        )
    manifest = GemmArtifactManifest(
        kernel_name="sm89_w4a16_grouped_decode",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(8, 1, 16),
        warp_count=8,
        stage_count=1,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "kernel_role": "grouped_w4a16_decode",
            "implementation": "custom_cuda_simt_task_grid",
            "weight_layout": {
                "qweight": "[expert,N,padded_K/2] uint8",
                "scales": "[expert,N,G] float32",
                "zero_points": "optional [expert,N,G] float32",
                "bias": "optional [expert,N] float32",
            },
            "task_table": "int32 [task_count,3] = [expert,packed_row_base,row_count]",
            "row_bounds": [1, 2, 4, 8],
            "scheduler_candidates": [
                "direct_task_grid",
                "bucketed_direct_task_grid",
                "persistent_grid_stride",
            ],
            "persistent_blocks_per_sm_candidates": [
                1,
                2,
                4,
                "max_active",
            ],
            "shape_variants": {
                "direct_max_rows_8": {"row_bound": 8, "persistent": False},
                "bucketed_max_rows_1": {"row_bound": 1, "persistent": False},
                "bucketed_max_rows_2": {"row_bound": 2, "persistent": False},
                "bucketed_max_rows_4": {"row_bound": 4, "persistent": False},
                "bucketed_max_rows_8": {"row_bound": 8, "persistent": False},
                "persistent_grid_stride_max_rows_8": {
                    "row_bound": 8,
                    "persistent": True,
                    "default_blocks_per_sm": 4,
                },
            },
            "output_scatter": "in_kernel_permutation",
            "scatter_launch_count": 0,
            "tensor_core_mma": False,
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_grouped_w8a8_artifact(
    config: Sm89GroupedW8A8BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile grouped W8A8 native MMA and keep it gated pending correctness."""

    resolved = config or Sm89GroupedW8A8BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 grouped W8A8 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 grouped W8A8 CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 grouped W8A8 nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing grouped W8A8 artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_w8a8_grouped_mma",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(16, 32, 32),
        warp_count=4,
        stage_count=1,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "kernel_role": "grouped_w8a8_decode",
            "implementation": "custom_cuda_mma_task_grid",
            "weight_layout": "[expert,N,padded_K] int8",
            "weight_scale_layout": "[expert,N] float32 per_channel",
            "activation_scale_layout": "[total_M] float32 per_token or [1] per_tensor",
            "task_table": "int32 [task_count,3] = [expert,packed_row_base,row_count]",
            "row_tile": 8,
            "scheduler": "single_direct_task_grid",
            "warp_candidates": [1, 2, 4, 8],
            "default_warps_per_block": 4,
            "shared_a_staging": True,
            "output_scatter": "in_kernel_permutation",
            "scatter_launch_count": 0,
            "tensor_core_mma": True,
            "instruction_shape": [16, 8, 32],
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_fp8_probe_artifact(
    config: Sm89Fp8ProbeBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the SM89 FP8 MMA capability probe as metadata-only evidence."""

    resolved = config or Sm89Fp8ProbeBuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 FP8 probe build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 FP8 probe source not found: {source}")
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
        raise XQTBackendError(f"SM89 FP8 probe nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing FP8 probe artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_fp8_cutlass_probe",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(16, 8, 32),
        warp_count=1,
        stage_count=1,
        preflight=report,
        metadata={
            "build_status": "compiled_probe_pending_runtime_sass_gate",
            "correctness_verified": False,
            "kernel_role": "fp8_capability_probe",
            "cutlass_mainloop": True,
            "implementation": "cutlass_arch_mma_probe",
            "formats": ["fp8_e4m3", "fp8_e5m2"],
            "instruction_shape": [16, 8, 32],
            "native_gemm": False,
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_fp8_artifact(
    config: Sm89Fp8BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the real SM89 FP8 CUTLASS GEMM and keep it gated."""

    resolved = config or Sm89Fp8BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 FP8 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 FP8 CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 FP8 nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing FP8 artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_fp8_cutlass",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(128, 256, 64),
        warp_count=8,
        stage_count=3,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "kernel_role": "fp8_native_gemm",
            "cutlass_mainloop": True,
            "implementation": "cutlass_device_gemm_sm89",
            "formats": ["fp8_e4m3", "fp8_e5m2"],
            "instruction_shape": [16, 8, 32],
            "supported_scale_modes": ["w:per_tensor/a:per_tensor", "w:blockwise/a:blockwise"],
            "unsupported_scale_modes_reason": (
                "per-channel/per-token scales require a separate row/column epilogue"
            ),
            "native_gemm": True,
            "blockwise": {
                "implementation": "custom_cuda_cutlass_warp_mma",
                "tile_shape": [16, 8, 32],
                "warp_count": 1,
                "scale_application": "per_k_block_promotion_inside_mainloop",
                "scale_layout": "a [M, ceil(K/block_k)] fp32, w [N, ceil(K/block_k)] fp32",
                "block_k_values": [32, 64, 128],
                "adapter_padding": "bytes zero-padded to M%16=0, N%8=0, K%32=0; scale rows padded with zeros",
                "partial_block": "trailing partial K block keeps its own scale slot; zero padding contributes zero",
                "split_k": {
                    "abi": "workspace_reduction",
                    "workspace": "float32 [split_count, padded_M, padded_N]",
                    "reduction": "second_kernel_deterministic_sum_beta_c_once",
                    "alignment": "splits are block_k aligned; a scale block never straddles a split",
                    "opt_in": "executor split_k>=2, split_k=1 keeps the full-K ABI",
                },
                "sm90_note": (
                    "SM90 WGMMA/TMA blockwise is a separate registry entry "
                    "(sm90_fp8_*_wgmma, metadata_only); the SM89 kernel has no "
                    "architecture if-else and no SM90 evidence exists on this host"
                ),
            },
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_grouped_fp8_artifact(
    config: Sm89GroupedFp8BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile grouped FP8 MMA and keep it gated pending correctness."""

    resolved = config or Sm89GroupedFp8BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 grouped FP8 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 grouped FP8 CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 grouped FP8 nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing grouped FP8 artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_fp8_grouped_mma",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(16, 32, 32),
        warp_count=4,
        stage_count=1,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "kernel_role": "grouped_fp8_decode",
            "implementation": "custom_cuda_cutlass_warp_mma_task_grid",
            "formats": ["fp8_e4m3", "fp8_e5m2"],
            "output_dtype": "fp16",
            "scale_modes": ["tensorwise", "blockwise"],
            "block_k_values": [32, 64, 128],
            "warp_candidates": [1, 2, 4, 8],
            "default_warps_per_block": 4,
            "shared_a_staging": True,
            "weight_layout": "[expert,N,padded_K] canonical FP8 bytes",
            "tensorwise_scales": "activation [expert], weight [expert]",
            "blockwise_scales": (
                "activation [total_M,Kb], weight [expert,N,Kb]"
            ),
            "task_table": "int32 [task_count,3] = [expert,packed_row_base,row_count]",
            "output_scatter": "in_kernel_permutation",
            "tensor_core_mma": True,
            "instruction_shape": [16, 8, 32],
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_mixed_input_probe_artifact(
    config: Sm89MixedInputProbeBuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the CUTLASS int8 x int4 capability/nibble-order probe.

    The result is intentionally metadata-only.  The probe is evidence for the
    available SM89 CUTLASS primitive, not an executable W4A16 registry entry.
    """

    resolved = config or Sm89MixedInputProbeBuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 mixed-input probe build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 mixed-input probe source not found: {source}")
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
        raise XQTBackendError(f"SM89 mixed-input probe build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing probe artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_cutlass_mixed_input_probe",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(128, 128, 128),
        warp_count=4,
        stage_count=2,
        preflight=report,
        metadata={
            "build_status": "compiled_probe_only",
            "correctness_verified": False,
            "probe_role": "cutlass_mixed_input_capability",
            "mixed_input_pair": "int8_x_int4_to_int32",
            "canonical_nibble_order": "low_high",
            "w4a16_native": False,
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_w8a16_artifact(
    config: Sm89W8A16BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Build SM89 W8A16 using the existing INT8 MMA kernel as the main path.

    This provides the primary executable entry for weight-only INT8 with per-channel
    scale and fp16/bf16 activation, as required by T040.
    """
    resolved = config or Sm89W8A16BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 W8A16 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 W8A16 source not found: {source}")
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
        raise XQTBackendError(f"SM89 W8A16 build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing W8A16 artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_w8a16_cutlass",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(64, 128, 64),
        warp_count=8,
        stage_count=3,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "execution_mode": "dynamic_int8_activation_then_mma",
            "weight_granularity": "per_channel",
            "activation_dtype": "fp16|bf16",
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def build_sm89_w4a8_artifact(
    config: Sm89W4A8BuildConfig | None = None,
) -> GemmArtifactManifest:
    """Compile the SM89 W4A8 INT8/FP8 group-scale MMA artifact."""

    resolved = config or Sm89W4A8BuildConfig()
    report = probe_cuda_cutlass(resolved.target_arch, require_device=False)
    if not report.ready_for_compile:
        raise XQTBackendError(
            "SM89 W4A8 build preflight failed: " + "; ".join(report.reasons)
        )
    source = resolved.source.expanduser().resolve()
    output = resolved.output.expanduser().resolve()
    if not source.is_file():
        raise XQTBackendError(f"SM89 W4A8 CUDA source not found: {source}")
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
        raise XQTBackendError(f"SM89 W4A8 nvcc build failed: {exc}") from exc
    if not output.is_file():
        raise XQTBackendError(f"nvcc completed without producing W4A8 artifact: {output}")
    manifest = GemmArtifactManifest(
        kernel_name="sm89_w4a8_cutlass",
        target_arch=report.target_arch,
        maturity="metadata_only",
        source=str(source),
        artifact=str(output),
        compile_flags=flags,
        tile_shape=(16, 8, 32),
        warp_count=1,
        stage_count=1,
        preflight=report,
        metadata={
            "build_status": "compiled_pending_correctness_gate",
            "correctness_verified": False,
            "kernel_role": "w4a8_native_group_scale_mainloop",
            "implementation": "custom_cuda_cutlass_mma",
            "weight_layout": {
                "qweight": "[N, padded_K/2] uint8 signed-int4 low-nibble-first",
                "weight_scales": "[N, padded_K/group_size] float32",
            },
            "activation_paths": {
                "int8": "mma.m16n8k32 s8s8s32, per_tensor/per_token dynamic scale",
                "fp8": "mma.m16n8k32 e4m3/e5m2, per_tensor/per_token/blockwise scale",
            },
            "shape_gates": {
                "m": "multiple of 16",
                "n": "multiple of 8",
                "padded_k": "multiple of 32 and group_size",
                "group_size": "multiple of 32",
            },
            "instruction_shape": [16, 8, 32],
            "tensor_core_mma": True,
        },
    )
    manifest.write_json(output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


__all__ = [
    "Sm89BuildConfig",
    "Sm89DenseBuildConfig",
    "Sm89W4A16BuildConfig",
    "Sm89W4A16FusedBuildConfig",
    "Sm89GroupedW4A16BuildConfig",
    "Sm89GroupedW8A8BuildConfig",
    "Sm89W4A8BuildConfig",
    "Sm89Fp8ProbeBuildConfig",
    "Sm89Fp8BuildConfig",
    "Sm89GroupedFp8BuildConfig",
    "Sm89MixedInputProbeBuildConfig",
    "build_sm89_artifact",
    "build_sm89_dense_artifact",
    "build_sm89_w4a16_dequant_artifact",
    "build_sm89_w4a16_fused_artifact",
    "build_sm89_grouped_w4a16_artifact",
    "build_sm89_grouped_w8a8_artifact",
    "build_sm89_w4a8_artifact",
    "build_sm89_fp8_probe_artifact",
    "build_sm89_fp8_artifact",
    "build_sm89_grouped_fp8_artifact",
    "build_sm89_mixed_input_probe_artifact",
    "build_sm89_w8a16_artifact",
]
