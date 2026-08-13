from __future__ import annotations

from pathlib import Path

import pytest

from xqt.gemm import (
    GemmArtifactManifest,
    GemmPreflightReport,
    Sm89BuildConfig,
    artifact_ready_for_execution,
    artifact_manifest_path,
    build_compile_flags,
    promote_artifact_manifest,
    probe_cuda_cutlass,
)


def test_preflight_rejects_invalid_arch() -> None:
    with pytest.raises(ValueError, match="sm_89"):
        probe_cuda_cutlass("ada")


def test_preflight_report_is_serializable() -> None:
    report = probe_cuda_cutlass("sm_89", require_device=False)
    payload = report.to_dict()
    assert payload["target_arch"] == "sm_89"
    assert payload["status"] in {"ready", "partial", "unavailable"}
    assert isinstance(payload["reasons"], list)


def test_manifest_rejects_executable_without_artifact() -> None:
    report = probe_cuda_cutlass("sm_89", require_device=False)
    with pytest.raises(ValueError, match="artifact path"):
        GemmArtifactManifest(
            kernel_name="test",
            target_arch="sm_89",
            maturity="executable",
            source="kernel.cu",
            artifact=None,
            compile_flags=(),
            tile_shape=None,
            warp_count=None,
            stage_count=None,
            preflight=report,
        )


def test_compile_flags_use_reported_include_and_arch(tmp_path: Path) -> None:
    report = probe_cuda_cutlass("sm_89", require_device=False)
    if not report.ready_for_compile:
        pytest.skip("CUDA/CUTLASS headers are not installed in this environment")
    flags = build_compile_flags(
        report,
        source=tmp_path / "kernel.cu",
        output=tmp_path / "kernel.so",
    )
    assert flags[0].endswith("nvcc")
    assert "-gencode=arch=compute_89,code=sm_89" in flags
    assert report.cutlass_include in flags


def test_cross_arch_report_allows_compile_but_not_executable_promotion(
    tmp_path: Path,
) -> None:
    report = GemmPreflightReport(
        target_arch="sm_90",
        status="partial",
        nvcc_path="/usr/local/cuda/bin/nvcc",
        nvcc_version="CUDA",
        cuda_runtime_version="13.1",
        compiler_version="g++",
        cutlass_version=None,
        cutlass_python_path=None,
        cutlass_include="/tmp/cutlass/include",
        device_name="RTX 4070 Ti SUPER",
        device_arch="sm_89",
        reasons=("device arch sm_89 does not match target sm_90",),
    )
    assert report.ready_for_compile is True

    artifact = tmp_path / "kernel.so"
    artifact.write_bytes(b"placeholder")
    manifest = GemmArtifactManifest(
        kernel_name="cross_arch_probe",
        target_arch="sm_90",
        maturity="executable",
        source="kernel.cu",
        artifact=str(artifact),
        compile_flags=("nvcc",),
        tile_shape=(128, 128, 128),
        warp_count=4,
        stage_count=4,
        preflight=report,
        metadata={
            "correctness_verified": True,
            "correctness": {"max_abs_error": 0.0},
        },
    )
    assert manifest.executable_ready is False


def test_sm89_build_config_defaults_to_metadata_only_seed() -> None:
    config = Sm89BuildConfig()
    assert config.target_arch == "sm_89"
    assert config.source.name == "int8mma_kernel.cu"
    assert "-lcublasLt" in config.extra_flags


def test_artifact_requires_correctness_promotion(tmp_path: Path) -> None:
    artifact = tmp_path / "kernel.so"
    artifact.write_bytes(b"placeholder")
    preflight = GemmPreflightReport(
        target_arch="sm_89",
        status="ready",
        nvcc_path="/usr/local/cuda/bin/nvcc",
        nvcc_version="CUDA",
        cuda_runtime_version="13.0",
        compiler_version="g++",
        cutlass_version="4.6.1",
        cutlass_python_path="cutlass",
        cutlass_include="/tmp/cutlass/include",
        device_name="GPU",
        device_arch="sm_89",
    )
    manifest = GemmArtifactManifest(
        kernel_name="test_kernel",
        target_arch="sm_89",
        maturity="metadata_only",
        source="kernel.cu",
        artifact=str(artifact),
        compile_flags=("nvcc",),
        tile_shape=(64, 64, 32),
        warp_count=4,
        stage_count=3,
        preflight=preflight,
        metadata={"correctness_verified": False},
    )
    manifest_path = artifact_manifest_path(artifact)
    manifest.write_json(manifest_path)
    assert artifact_ready_for_execution(manifest_path, kernel_name="test_kernel") is False
    promoted = promote_artifact_manifest(
        artifact,
        evidence={"cases": 2, "max_abs_error": 0.0},
    )
    assert promoted.executable_ready is True
    assert artifact_ready_for_execution(manifest_path, kernel_name="test_kernel") is True
