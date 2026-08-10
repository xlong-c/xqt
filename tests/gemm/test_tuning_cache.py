from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

import xqt.gemm.tuning_cache as tuning_cache_module
from xqt.core.errors import XQTArtifactError
from xqt.gemm import (
    GemmArtifactManifest,
    GemmPreflightReport,
    GemmProblem,
    GemmTuningArtifactIdentity,
    GemmTuningCache,
    GemmTuningKey,
    GemmTuningLookup,
    GemmTuningRecord,
    GroupedGemmCandidateOutput,
    GroupedGemmNativeCandidate,
    GroupedGemmProblem,
    QuantSpec,
    artifact_manifest_path,
    dispatch_grouped_gemm,
    resolve_tuning_record,
)
from xqt.gemm.backends.fp8_grouped_sm89 import (
    _resolve_cached_configuration as _resolve_fp8_configuration,
)
from xqt.gemm.backends.w4a16_grouped_sm89 import (
    _resolve_cached_configuration as _resolve_w4a16_configuration,
)
from xqt.gemm.backends.w8a8_grouped_sm89 import (
    _resolve_cached_configuration as _resolve_w8a8_configuration,
)


_CREATED_AT = "2026-08-09T00:00:00+00:00"
_ARTIFACT_DIGEST = "a" * 64
_MANIFEST_DIGEST = "b" * 64


def _key(
    kernel_family: str = "sm89_w4a16_grouped_decode",
    *,
    rows: tuple[int, ...] = (0, 1, 2, 4, 8),
    weight_dtype: str = "int4",
    activation_dtype: str = "fp16",
    compute_dtype: str = "fp32",
    accum_dtype: str = "fp32",
    group_size: int | None = 128,
    scale_mode: str = "w:groupwise/a:per_tensor",
    symmetric: bool = True,
    weight_zero_point: bool = False,
    activation_zero_point: bool = False,
    weight_scale_source: str = "weight_offline",
    activation_scale_source: str = "none",
    storage_layout: str = "xqt_int4_nk_v1",
    pack_version: str = "xqt-int4-v1",
    output_scatter: bool = False,
    has_bias: bool = True,
) -> GemmTuningKey:
    return GemmTuningKey(
        kernel_family=kernel_family,
        backend="custom_cuda",
        target_arch="sm_89",
        m=sum(rows),
        n=1024,
        k=4096,
        expert_rows=rows,
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        output_dtype="fp16",
        group_axis="k",
        group_size=group_size,
        scale_mode=scale_mode,
        symmetric=symmetric,
        weight_zero_point=weight_zero_point,
        activation_zero_point=activation_zero_point,
        weight_scale_source=weight_scale_source,
        activation_scale_source=activation_scale_source,
        storage_layout=storage_layout,
        pack_version=pack_version,
        persistent=False,
        output_scatter=output_scatter,
        has_bias=has_bias,
    )


def _identity(
    kernel_name: str = "sm89_w4a16_grouped_decode",
) -> GemmTuningArtifactIdentity:
    return GemmTuningArtifactIdentity(
        artifact_path=f"/cache/{kernel_name}.so",
        kernel_name=kernel_name,
        target_arch="sm_89",
        artifact_sha256=_ARTIFACT_DIGEST,
        manifest_sha256=_MANIFEST_DIGEST,
    )


def _record(
    key: GemmTuningKey,
    *,
    artifact: GemmTuningArtifactIdentity | None = None,
    selection: dict[str, object] | None = None,
    correctness_verified: bool = True,
    expires_at: str | None = None,
    selected_kernel: str | None = None,
) -> GemmTuningRecord:
    return GemmTuningRecord(
        key=key,
        selected_kernel=selected_kernel or key.kernel_family,
        selection=selection or {"scheduler": "bucketed"},
        artifact=artifact or _identity(key.kernel_family),
        correctness_verified=correctness_verified,
        benchmark={
            "method": "cuda_event",
            "latency_ms": 0.241664,
            "warmup": 20,
            "repeats": 15,
        },
        resources={"registers_per_thread": 40, "static_shared_bytes": 0},
        cache_sensitivity={
            "method": "cuda_event_after_256MiB_l2_thrash",
            "cold_over_hot": 1.02,
        },
        evidence_paths=("research/xqt-gemm/artifacts/example/baseline.json",),
        created_at=_CREATED_AT,
        expires_at=expires_at,
    )


def _promoted_artifact(
    tmp_path: Path,
    *,
    kernel_name: str = "sm89_w4a16_grouped_decode",
) -> tuple[Path, GemmTuningArtifactIdentity]:
    artifact = tmp_path / f"{kernel_name}.so"
    artifact.write_bytes(b"promoted native artifact")
    preflight = GemmPreflightReport(
        target_arch="sm_89",
        status="ready",
        nvcc_path="/usr/bin/nvcc",
        nvcc_version="test",
        cuda_runtime_version="test",
        compiler_version="test",
        cutlass_version="test",
        cutlass_python_path=None,
        cutlass_include="/tmp/cutlass/include",
        device_name="test GPU",
        device_arch="sm_89",
    )
    manifest = GemmArtifactManifest(
        kernel_name=kernel_name,
        target_arch="sm_89",
        maturity="executable",
        source="kernel.cu",
        artifact=str(artifact),
        compile_flags=(),
        tile_shape=None,
        warp_count=None,
        stage_count=None,
        preflight=preflight,
        metadata={
            "correctness_verified": True,
            "correctness": {"status": "passed", "max_abs_error": 0.0},
        },
    )
    manifest.write_json(artifact_manifest_path(artifact))
    return artifact, GemmTuningArtifactIdentity.from_artifact(
        artifact,
        kernel_name=kernel_name,
        target_arch="sm_89",
    )


def test_tuning_key_is_stable_and_round_trips_exact_fields() -> None:
    key = _key(output_scatter=True)

    loaded = GemmTuningKey.from_dict(key.to_dict())

    assert loaded == key
    assert loaded.cache_key == key.cache_key
    assert _key(symmetric=False, weight_zero_point=True).cache_key != key.cache_key
    assert _key(pack_version="xqt-int4-v2").cache_key != key.cache_key
    assert loaded.to_dict()["persistent"] is False
    assert loaded.to_dict()["shape"] == {"m": 15, "n": 1024, "k": 4096}
    with pytest.raises(TypeError, match="persistent must be bool"):
        GemmTuningKey.from_dict({**key.to_dict(), "persistent": "false"})


def test_tuning_cache_round_trip_checksum_and_atomic_replace(tmp_path: Path) -> None:
    cache_path = tmp_path / "tuning-cache.json"
    cache = GemmTuningCache(records=(_record(_key()),), created_at=_CREATED_AT)

    written = cache.write_json(cache_path)
    loaded = GemmTuningCache.load_json(written)
    cache.write_json(cache_path)

    assert loaded.to_dict() == cache.to_dict()
    assert loaded.payload_sha256 == cache.payload_sha256
    assert list(tmp_path.glob(".tuning-cache.json.*.tmp")) == []
    json.loads(cache_path.read_text(encoding="utf-8"))


def test_tuning_cache_rejects_duplicate_corrupt_and_invalid_json(
    tmp_path: Path,
) -> None:
    record = _record(_key())
    with pytest.raises(ValueError, match="duplicate tuning cache key"):
        GemmTuningCache(records=(record, record), created_at=_CREATED_AT)

    cache_path = tmp_path / "tuning-cache.json"
    payload = GemmTuningCache(
        records=(record,),
        created_at=_CREATED_AT,
    ).to_dict()
    payload["records"][0]["selection"]["scheduler"] = "direct"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(XQTArtifactError, match="checksum mismatch"):
        GemmTuningCache.load_json(cache_path)

    cache_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(XQTArtifactError, match="failed to load tuning cache"):
        GemmTuningCache.load_json(cache_path)


def test_tuning_lookup_reports_hit_miss_expired_and_unverified(
    tmp_path: Path,
) -> None:
    artifact, identity = _promoted_artifact(tmp_path)
    key = _key()
    hit_record = _record(key, artifact=identity)
    hit_cache = GemmTuningCache(records=(hit_record,), created_at=_CREATED_AT)

    hit = hit_cache.lookup(
        key,
        artifact=artifact,
        now=datetime(2026, 8, 9, 1, tzinfo=timezone.utc),
    )
    miss = hit_cache.lookup(
        _key(rows=(0, 1, 2, 4, 9)),
        artifact=artifact,
    )
    expired_cache = GemmTuningCache(
        records=(
            _record(
                key,
                artifact=identity,
                expires_at="2026-08-10T00:00:00+00:00",
            ),
        ),
        created_at=_CREATED_AT,
    )
    expired = expired_cache.lookup(
        key,
        artifact=artifact,
        now=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )
    unverified_cache = GemmTuningCache(
        records=(
            _record(
                key,
                artifact=identity,
                correctness_verified=False,
            ),
        ),
        created_at=_CREATED_AT,
    )
    unverified = unverified_cache.lookup(key, artifact=artifact)

    assert hit.status == "hit"
    assert hit.selection == {"scheduler": "bucketed"}
    assert miss.status == "miss"
    assert expired.status == "expired"
    assert unverified.status == "invalid"
    assert unverified.source == "deterministic_default"


def test_tuning_lookup_rejects_changed_artifact_and_manifest(tmp_path: Path) -> None:
    artifact, identity = _promoted_artifact(tmp_path)
    key = _key()
    cache = GemmTuningCache(
        records=(_record(key, artifact=identity),),
        created_at=_CREATED_AT,
    )
    assert cache.lookup(key, artifact=artifact).status == "hit"

    artifact.write_bytes(b"changed artifact")
    cache.invalidate_artifact(artifact)
    changed_artifact = cache.lookup(key, artifact=artifact)
    assert changed_artifact.status == "invalid"
    assert "artifact checksum" in changed_artifact.reason

    artifact.write_bytes(b"promoted native artifact")
    manifest_path = artifact_manifest_path(artifact)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["metadata"]["evidence_revision"] = 2
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cache.invalidate_artifact(artifact)
    changed_manifest = cache.lookup(key, artifact=artifact)
    assert changed_manifest.status == "invalid"
    assert "manifest checksum" in changed_manifest.reason


def test_tuning_lookup_hashes_identity_once_per_unchanged_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, identity = _promoted_artifact(tmp_path)
    key = _key()
    cache = GemmTuningCache(
        records=(_record(key, artifact=identity),),
        created_at=_CREATED_AT,
    )
    original = tuning_cache_module.file_sha256
    calls: list[Path] = []

    def counted_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
        calls.append(Path(path))
        return original(path, chunk_size=chunk_size)

    monkeypatch.setattr(tuning_cache_module, "file_sha256", counted_sha256)
    monkeypatch.chdir(tmp_path)
    relative_artifact = Path(artifact.name)

    cache.prime_artifact(relative_artifact)
    assert cache.lookup(key, artifact=artifact).status == "hit"
    assert len(calls) == 2


def test_prime_artifact_rejects_missing_runtime_identity(tmp_path: Path) -> None:
    cache = GemmTuningCache(records=(), created_at=_CREATED_AT)

    with pytest.raises(XQTArtifactError, match="artifact or manifest file is missing"):
        cache.prime_artifact(tmp_path / "missing.so")


def test_explicit_configuration_bypasses_cache_and_missing_cache_is_a_miss() -> None:
    key = _key()

    explicit = resolve_tuning_record(
        None,
        key=key,
        artifact="unused.so",
        explicit=True,
    )
    missing = resolve_tuning_record(
        None,
        key=key,
        artifact="unused.so",
        explicit=False,
    )

    assert explicit.status == "bypassed_explicit"
    assert explicit.source == "explicit_request"
    assert missing.status == "miss"
    assert missing.source == "deterministic_default"


def test_grouped_dispatch_serializes_tuning_state_without_online_autotune() -> None:
    problem = GroupedGemmProblem(
        problems=(GemmProblem(m=2, n=3, k=4),),
    )
    activation = torch.zeros((2, 4), dtype=torch.float32)
    output = torch.ones((2, 3), dtype=torch.float32)
    candidate_calls = 0

    def execute_native() -> GroupedGemmCandidateOutput:
        nonlocal candidate_calls
        candidate_calls += 1
        return GroupedGemmCandidateOutput(output, details={"launch_count": 1})

    tuning = resolve_tuning_record(
        None,
        key=GemmTuningKey(
            kernel_family="native_grouped",
            backend="custom_cuda",
            target_arch="sm_89",
            m=2,
            n=3,
            k=4,
            expert_rows=(2,),
            weight_dtype="fp32",
            activation_dtype="fp32",
            compute_dtype="fp32",
            accum_dtype="fp32",
            output_dtype="fp32",
            group_axis="k",
            group_size=None,
            scale_mode="w:per_tensor/a:per_tensor",
            symmetric=True,
            weight_zero_point=False,
            activation_zero_point=False,
            weight_scale_source="none",
            activation_scale_source="none",
            storage_layout="canonical",
            pack_version="canonical-v1",
            persistent=False,
            output_scatter=False,
            has_bias=False,
        ),
        artifact="unused.so",
        explicit=False,
    )
    result = dispatch_grouped_gemm(
        activation,
        None,
        grouped_problem=problem,
        quant_specs=QuantSpec(
            weight_dtype="fp32",
            activation_dtype="fp32",
            output_dtype="fp32",
        ),
        candidates=(
            GroupedGemmNativeCandidate(
                name="native_grouped",
                backend="custom_cuda",
                executor=execute_native,
            ),
        ),
        tuning=tuning,
    )

    report = json.loads(json.dumps(result.report.to_dict()))
    assert candidate_calls == 1
    assert result.output is output
    assert report["tuning_cache_status"] == "miss"
    assert report["tuning_source"] == "deterministic_default"
    assert report["tuning_cache_key_id"] == tuning.key.cache_key


def test_backend_cached_knobs_are_validated_before_use() -> None:
    w4_key = _key()
    w4_lookup = GemmTuningLookup(
        status="hit",
        key=w4_key,
        reason="test hit",
        source="offline_cuda_event",
        record=_record(w4_key, selection={"scheduler": "bucketed"}),
    )
    scheduler, blocks, resolved_w4 = _resolve_w4a16_configuration(
        w4_lookup,
        fallback_scheduler="auto",
        fallback_persistent_blocks_per_sm=4,
    )
    assert (scheduler, blocks) == ("bucketed", 4)
    assert resolved_w4.status == "hit"

    persistent_lookup = GemmTuningLookup(
        status="hit",
        key=w4_key,
        reason="test hit",
        source="offline_cuda_event",
        record=_record(
            w4_key,
            selection={
                "scheduler": "persistent",
                "persistent_blocks_per_sm": 4,
            },
        ),
    )
    scheduler, blocks, resolved_w4 = _resolve_w4a16_configuration(
        persistent_lookup,
        fallback_scheduler="auto",
        fallback_persistent_blocks_per_sm=2,
    )
    assert (scheduler, blocks, resolved_w4.status) == ("persistent", 4, "hit")

    invalid_persistent_lookup = GemmTuningLookup(
        status="hit",
        key=w4_key,
        reason="test hit",
        source="offline_cuda_event",
        record=_record(
            w4_key,
            selection={
                "scheduler": "persistent",
                "persistent_blocks_per_sm": 9,
            },
        ),
    )
    scheduler, blocks, resolved_w4 = _resolve_w4a16_configuration(
        invalid_persistent_lookup,
        fallback_scheduler="auto",
        fallback_persistent_blocks_per_sm=4,
    )
    assert (scheduler, blocks, resolved_w4.status) == ("auto", 4, "invalid")

    w8_key = _key(
        "sm89_w8a8_grouped_mma",
        weight_dtype="int8",
        activation_dtype="int8",
        group_size=None,
        scale_mode="w:per_channel/a:per_token",
        storage_layout="canonical",
    )
    w8_lookup = GemmTuningLookup(
        status="hit",
        key=w8_key,
        reason="test hit",
        source="offline_cuda_event",
        record=_record(
            w8_key,
            selection={"scheduler": "direct", "warps_per_block": 4},
        ),
    )
    scheduler, warps, resolved_w8 = _resolve_w8a8_configuration(
        w8_lookup,
        fallback_scheduler="auto",
        fallback_warps="auto",
    )
    assert (scheduler, warps, resolved_w8.status) == ("direct", 4, "hit")

    fp8_key = _key(
        "sm89_fp8_grouped_mma",
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        group_size=64,
        scale_mode="w:blockwise/a:blockwise",
        storage_layout="xqt_fp8_rowmajor_v1",
    )
    invalid_fp8_lookup = GemmTuningLookup(
        status="hit",
        key=fp8_key,
        reason="test hit",
        source="offline_cuda_event",
        record=_record(
            fp8_key,
            selection={"scheduler": "direct", "warps_per_block": 16},
        ),
    )
    scheduler, warps, resolved_fp8 = _resolve_fp8_configuration(
        invalid_fp8_lookup,
        fallback_scheduler="auto",
        fallback_warps="auto",
    )
    assert (scheduler, warps) == ("auto", "auto")
    assert resolved_fp8.status == "invalid"
    assert resolved_fp8.source == "deterministic_default"
