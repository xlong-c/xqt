"""Versioned offline tuning records consumed by XQT GEMM dispatchers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xqt.core.artifact import file_sha256
from xqt.core.errors import XQTArtifactError

from .contracts import GroupedGemmProblem, QuantSpec
from .preflight import (
    artifact_manifest_path,
    artifact_ready_for_execution,
    load_artifact_manifest,
)


GEMM_TUNING_CACHE_VERSION = "xqt-gemm-tuning-v1"
_LOOKUP_STATUSES = frozenset(
    {"hit", "miss", "expired", "invalid", "bypassed_explicit"}
)
_SHA256_HEX_LENGTH = 64


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("tuning cache payload must be JSON serializable") from exc


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_timestamp(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA256 hex digest")
    return normalized


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


@dataclass(frozen=True, slots=True)
class GemmTuningKey:
    """Exact runtime contract used to identify one offline tuning record."""

    kernel_family: str
    backend: str
    target_arch: str
    m: int
    n: int
    k: int
    expert_rows: tuple[int, ...]
    weight_dtype: str
    activation_dtype: str
    compute_dtype: str
    accum_dtype: str
    output_dtype: str
    group_axis: str
    group_size: int | None
    scale_mode: str
    symmetric: bool
    weight_zero_point: bool
    activation_zero_point: bool
    weight_scale_source: str
    activation_scale_source: str
    storage_layout: str
    pack_version: str
    persistent: bool
    output_scatter: bool
    has_bias: bool
    cuda_graph: bool = False
    workspace_bytes: int = 0
    _cache_key: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in (
            "kernel_family",
            "backend",
            "target_arch",
            "weight_dtype",
            "activation_dtype",
            "compute_dtype",
            "accum_dtype",
            "output_dtype",
            "group_axis",
            "scale_mode",
            "weight_scale_source",
            "activation_scale_source",
            "storage_layout",
            "pack_version",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"GemmTuningKey.{field_name} cannot be empty")
            object.__setattr__(self, field_name, value)
        if not self.target_arch.startswith("sm_"):
            raise ValueError("GemmTuningKey.target_arch must use sm_* form")
        for field_name in ("m", "n", "k"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"GemmTuningKey.{field_name} must be int")
        if self.m < 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("GemmTuningKey requires M >= 0 and N/K > 0")
        if any(isinstance(row, bool) or not isinstance(row, int) for row in self.expert_rows):
            raise TypeError("GemmTuningKey.expert_rows must contain ints")
        rows = tuple(self.expert_rows)
        if not rows or any(row < 0 for row in rows):
            raise ValueError("GemmTuningKey.expert_rows must be non-empty and non-negative")
        if sum(rows) != self.m:
            raise ValueError("GemmTuningKey.expert_rows must sum to M")
        object.__setattr__(self, "expert_rows", rows)
        if isinstance(self.group_size, bool):
            raise TypeError("GemmTuningKey.group_size must be int or None")
        if self.group_size is not None and not isinstance(self.group_size, int):
            raise TypeError("GemmTuningKey.group_size must be int or None")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("GemmTuningKey.group_size must be positive when present")
        for field_name in (
            "symmetric",
            "weight_zero_point",
            "activation_zero_point",
            "persistent",
            "output_scatter",
            "has_bias",
            "cuda_graph",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"GemmTuningKey.{field_name} must be bool")
        if isinstance(self.workspace_bytes, bool) or not isinstance(
            self.workspace_bytes,
            int,
        ):
            raise TypeError("GemmTuningKey.workspace_bytes must be int")
        if self.workspace_bytes < 0:
            raise ValueError("GemmTuningKey.workspace_bytes cannot be negative")
        object.__setattr__(self, "_cache_key", _payload_sha256(self.to_dict()))

    @property
    def cache_key(self) -> str:
        """Return a stable SHA256 identifier for this exact contract."""

        return self._cache_key

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-ready lookup key."""

        return {
            "kernel_family": self.kernel_family,
            "backend": self.backend,
            "target_arch": self.target_arch,
            "shape": {"m": self.m, "n": self.n, "k": self.k},
            "expert_rows": list(self.expert_rows),
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "compute_dtype": self.compute_dtype,
            "accum_dtype": self.accum_dtype,
            "output_dtype": self.output_dtype,
            "group_axis": self.group_axis,
            "group_size": self.group_size,
            "scale_mode": self.scale_mode,
            "symmetric": self.symmetric,
            "weight_zero_point": self.weight_zero_point,
            "activation_zero_point": self.activation_zero_point,
            "weight_scale_source": self.weight_scale_source,
            "activation_scale_source": self.activation_scale_source,
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
            "persistent": self.persistent,
            "output_scatter": self.output_scatter,
            "has_bias": self.has_bias,
            "cuda_graph": self.cuda_graph,
            "workspace_bytes": self.workspace_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GemmTuningKey":
        """Load an exact key without compatibility rewriting."""

        if not isinstance(payload, Mapping):
            raise TypeError("GemmTuningKey.from_dict expects a mapping")
        shape = payload.get("shape")
        if not isinstance(shape, Mapping):
            raise ValueError("tuning key requires a shape mapping")
        for field_name in ("m", "n", "k"):
            if isinstance(shape.get(field_name), bool) or not isinstance(
                shape.get(field_name),
                int,
            ):
                raise TypeError(f"tuning key shape.{field_name} must be int")
        group_size = payload.get("group_size")
        if group_size is not None and (
            isinstance(group_size, bool) or not isinstance(group_size, int)
        ):
            raise TypeError("tuning key group_size must be int or null")
        boolean_fields = {
            "symmetric": payload.get("symmetric"),
            "weight_zero_point": payload.get("weight_zero_point"),
            "activation_zero_point": payload.get("activation_zero_point"),
            "persistent": payload.get("persistent"),
            "output_scatter": payload.get("output_scatter"),
            "has_bias": payload.get("has_bias"),
            "cuda_graph": payload.get("cuda_graph", False),
        }
        for field_name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(f"tuning key {field_name} must be bool")
        workspace_bytes = payload.get("workspace_bytes", 0)
        if isinstance(workspace_bytes, bool) or not isinstance(
            workspace_bytes,
            int,
        ):
            raise TypeError("tuning key workspace_bytes must be int")
        return cls(
            kernel_family=str(payload["kernel_family"]),
            backend=str(payload["backend"]),
            target_arch=str(payload["target_arch"]),
            m=shape["m"],
            n=shape["n"],
            k=shape["k"],
            expert_rows=tuple(int(row) for row in payload["expert_rows"]),
            weight_dtype=str(payload["weight_dtype"]),
            activation_dtype=str(payload["activation_dtype"]),
            compute_dtype=str(payload["compute_dtype"]),
            accum_dtype=str(payload["accum_dtype"]),
            output_dtype=str(payload["output_dtype"]),
            group_axis=str(payload["group_axis"]),
            group_size=group_size,
            scale_mode=str(payload["scale_mode"]),
            symmetric=boolean_fields["symmetric"],
            weight_zero_point=boolean_fields["weight_zero_point"],
            activation_zero_point=boolean_fields["activation_zero_point"],
            weight_scale_source=str(payload["weight_scale_source"]),
            activation_scale_source=str(payload["activation_scale_source"]),
            storage_layout=str(payload["storage_layout"]),
            pack_version=str(payload["pack_version"]),
            persistent=boolean_fields["persistent"],
            output_scatter=boolean_fields["output_scatter"],
            has_bias=boolean_fields["has_bias"],
            cuda_graph=boolean_fields["cuda_graph"],
            workspace_bytes=workspace_bytes,
        )


@dataclass(frozen=True, slots=True)
class GemmTuningArtifactIdentity:
    """Content identity binding a tuning winner to one promoted artifact."""

    artifact_path: str
    kernel_name: str
    target_arch: str
    artifact_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("artifact_path", "kernel_name", "target_arch"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"GemmTuningArtifactIdentity.{field_name} cannot be empty")
            object.__setattr__(self, field_name, value)
        if not self.target_arch.startswith("sm_"):
            raise ValueError("tuning artifact target_arch must use sm_* form")
        object.__setattr__(
            self,
            "artifact_sha256",
            _validate_sha256(
                self.artifact_sha256,
                field_name="GemmTuningArtifactIdentity.artifact_sha256",
            ),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _validate_sha256(
                self.manifest_sha256,
                field_name="GemmTuningArtifactIdentity.manifest_sha256",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready artifact identity."""

        return {
            "artifact_path": self.artifact_path,
            "kernel_name": self.kernel_name,
            "target_arch": self.target_arch,
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "GemmTuningArtifactIdentity":
        """Load a persisted artifact identity."""

        if not isinstance(payload, Mapping):
            raise TypeError("GemmTuningArtifactIdentity.from_dict expects a mapping")
        return cls(
            artifact_path=str(payload["artifact_path"]),
            kernel_name=str(payload["kernel_name"]),
            target_arch=str(payload["target_arch"]),
            artifact_sha256=str(payload["artifact_sha256"]),
            manifest_sha256=str(payload["manifest_sha256"]),
        )

    @classmethod
    def from_artifact(
        cls,
        artifact: str | Path,
        *,
        kernel_name: str,
        target_arch: str,
    ) -> "GemmTuningArtifactIdentity":
        """Build identity only for a correctness-promoted native artifact."""

        artifact_path = Path(artifact).expanduser().resolve()
        manifest_path = artifact_manifest_path(artifact_path)
        if not artifact_ready_for_execution(
            artifact_path,
            kernel_name=kernel_name,
            target_arch=target_arch,
        ):
            raise XQTArtifactError(
                "tuning identity requires a correctness-promoted artifact and manifest"
            )
        return cls(
            artifact_path=str(artifact_path),
            kernel_name=kernel_name,
            target_arch=target_arch,
            artifact_sha256=file_sha256(artifact_path),
            manifest_sha256=file_sha256(manifest_path),
        )


@dataclass(frozen=True, slots=True)
class GemmTuningRecord:
    """Offline benchmark winner and the evidence required to trust it."""

    key: GemmTuningKey
    selected_kernel: str
    selection: Mapping[str, Any]
    artifact: GemmTuningArtifactIdentity
    correctness_verified: bool
    benchmark: Mapping[str, Any]
    resources: Mapping[str, Any] = field(default_factory=dict)
    cache_sensitivity: Mapping[str, Any] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()
    source: str = "offline_cuda_event"
    created_at: str = field(default_factory=_utc_timestamp)
    expires_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, GemmTuningKey):
            raise TypeError("GemmTuningRecord.key must be GemmTuningKey")
        if not isinstance(self.artifact, GemmTuningArtifactIdentity):
            raise TypeError(
                "GemmTuningRecord.artifact must be GemmTuningArtifactIdentity"
            )
        for field_name in ("selected_kernel", "source"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"GemmTuningRecord.{field_name} cannot be empty")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.correctness_verified, bool):
            raise TypeError("GemmTuningRecord.correctness_verified must be bool")
        for field_name in (
            "selection",
            "benchmark",
            "resources",
            "cache_sensitivity",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"GemmTuningRecord.{field_name} must be a mapping")
            normalized = dict(value)
            _canonical_json(normalized)
            object.__setattr__(self, field_name, normalized)
        paths = tuple(str(path).strip() for path in self.evidence_paths)
        if any(not path for path in paths):
            raise ValueError("GemmTuningRecord.evidence_paths cannot contain empty paths")
        object.__setattr__(self, "evidence_paths", paths)
        object.__setattr__(
            self,
            "created_at",
            _normalize_timestamp(self.created_at, field_name="created_at"),
        )
        if self.expires_at is not None:
            normalized_expiry = _normalize_timestamp(
                self.expires_at,
                field_name="expires_at",
            )
            if _parse_timestamp(normalized_expiry) <= _parse_timestamp(self.created_at):
                raise ValueError("GemmTuningRecord.expires_at must be after created_at")
            object.__setattr__(self, "expires_at", normalized_expiry)

    @property
    def record_id(self) -> str:
        """Return the stable key identifier used for duplicate detection."""

        return self.key.cache_key

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Return whether this record has passed its explicit expiry."""

        if self.expires_at is None:
            return False
        current = datetime.now(timezone.utc) if now is None else now
        if current.tzinfo is None:
            raise ValueError("tuning lookup time must include a timezone")
        return current.astimezone(timezone.utc) >= _parse_timestamp(self.expires_at)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready tuning record."""

        return {
            "record_id": self.record_id,
            "key": self.key.to_dict(),
            "selected_kernel": self.selected_kernel,
            "selection": dict(self.selection),
            "artifact": self.artifact.to_dict(),
            "correctness_verified": self.correctness_verified,
            "benchmark": dict(self.benchmark),
            "resources": dict(self.resources),
            "cache_sensitivity": dict(self.cache_sensitivity),
            "evidence_paths": list(self.evidence_paths),
            "source": self.source,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GemmTuningRecord":
        """Load a versioned tuning record without filling missing evidence."""

        if not isinstance(payload, Mapping):
            raise TypeError("GemmTuningRecord.from_dict expects a mapping")
        record = cls(
            key=GemmTuningKey.from_dict(payload["key"]),
            selected_kernel=str(payload["selected_kernel"]),
            selection=dict(payload["selection"]),
            artifact=GemmTuningArtifactIdentity.from_dict(payload["artifact"]),
            correctness_verified=payload["correctness_verified"],
            benchmark=dict(payload["benchmark"]),
            resources=dict(payload.get("resources", {})),
            cache_sensitivity=dict(payload.get("cache_sensitivity", {})),
            evidence_paths=tuple(str(path) for path in payload.get("evidence_paths", [])),
            source=str(payload["source"]),
            created_at=str(payload["created_at"]),
            expires_at=(
                None
                if payload.get("expires_at") is None
                else str(payload["expires_at"])
            ),
        )
        if str(payload.get("record_id", record.record_id)) != record.record_id:
            raise ValueError("tuning record_id does not match its canonical key")
        return record


@dataclass(frozen=True, slots=True)
class GemmTuningLookup:
    """Runtime cache decision propagated into a dispatch report."""

    status: str
    key: GemmTuningKey
    reason: str
    source: str
    record: GemmTuningRecord | None = None

    def __post_init__(self) -> None:
        if self.status not in _LOOKUP_STATUSES:
            raise ValueError(f"unsupported tuning cache status: {self.status!r}")
        if not isinstance(self.key, GemmTuningKey):
            raise TypeError("GemmTuningLookup.key must be GemmTuningKey")
        if not self.reason or not self.source:
            raise ValueError("tuning lookup reason and source cannot be empty")
        if self.status == "hit" and self.record is None:
            raise ValueError("tuning cache hit requires a record")
        if self.record is not None and self.record.key != self.key:
            raise ValueError("tuning lookup record key must match lookup key")

    @property
    def selection(self) -> Mapping[str, Any]:
        """Return cached knobs only for a verified hit."""

        return {} if self.record is None else self.record.selection

    def to_report_dict(self) -> dict[str, Any]:
        """Return fields embedded into grouped dispatch reports."""

        return {
            "tuning_cache_status": self.status,
            "tuning_cache_key": self.key.to_dict(),
            "tuning_cache_key_id": self.key.cache_key,
            "tuning_cache_reason": self.reason,
            "tuning_source": self.source,
            "tuning_record_id": (
                None if self.record is None else self.record.record_id
            ),
        }


@dataclass(frozen=True, slots=True)
class _ArtifactSnapshot:
    ready: bool
    kernel_name: str | None
    target_arch: str | None
    artifact_sha256: str | None
    manifest_sha256: str | None
    reason: str


@dataclass(slots=True)
class GemmTuningCache:
    """In-memory tuning cache loaded once before steady-state dispatch."""

    records: tuple[GemmTuningRecord, ...]
    created_at: str = field(default_factory=_utc_timestamp)
    schema_version: str = GEMM_TUNING_CACHE_VERSION
    _records_by_key: dict[str, GemmTuningRecord] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _artifact_snapshots: dict[str, _ArtifactSnapshot] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if self.schema_version != GEMM_TUNING_CACHE_VERSION:
            raise ValueError(
                "unsupported tuning cache schema_version: "
                f"{self.schema_version!r}"
            )
        self.created_at = _normalize_timestamp(
            self.created_at,
            field_name="created_at",
        )
        records = tuple(self.records)
        if any(not isinstance(record, GemmTuningRecord) for record in records):
            raise TypeError("GemmTuningCache.records must contain GemmTuningRecord")
        normalized_records = tuple(sorted(records, key=lambda item: item.record_id))
        records_by_key: dict[str, GemmTuningRecord] = {}
        for record in normalized_records:
            if record.record_id in records_by_key:
                raise ValueError(
                    f"duplicate tuning cache key: {record.record_id}"
                )
            records_by_key[record.record_id] = record
        self.records = normalized_records
        self._records_by_key = records_by_key

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def payload_sha256(self) -> str:
        """Return the checksum covering schema metadata and all records."""

        return _payload_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        """Return the checksummed cache envelope."""

        payload = self._payload()
        payload["payload_sha256"] = _payload_sha256(payload)
        return payload

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist the complete checksummed tuning cache."""

        output_path = Path(path).expanduser()
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        return _atomic_write_text(output_path, text)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GemmTuningCache":
        """Validate checksum and load one exact cache schema version."""

        if not isinstance(payload, Mapping):
            raise TypeError("GemmTuningCache.from_dict expects a mapping")
        expected_checksum = payload.get("payload_sha256")
        if not isinstance(expected_checksum, str):
            raise XQTArtifactError("tuning cache requires payload_sha256")
        unsigned_payload = dict(payload)
        unsigned_payload.pop("payload_sha256", None)
        actual_checksum = _payload_sha256(unsigned_payload)
        if expected_checksum != actual_checksum:
            raise XQTArtifactError(
                "tuning cache payload checksum mismatch: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
        try:
            records_payload = unsigned_payload["records"]
            if not isinstance(records_payload, Sequence) or isinstance(
                records_payload,
                (str, bytes),
            ):
                raise TypeError("tuning cache records must be a sequence")
            return cls(
                schema_version=str(unsigned_payload["schema_version"]),
                created_at=str(unsigned_payload["created_at"]),
                records=tuple(
                    GemmTuningRecord.from_dict(record)
                    for record in records_payload
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise XQTArtifactError(f"invalid tuning cache payload: {exc}") from exc

    @classmethod
    def load_json(cls, path: str | Path) -> "GemmTuningCache":
        """Load a cache file once for reuse across forward calls."""

        source = Path(path).expanduser()
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise XQTArtifactError(
                f"failed to load tuning cache {source}: {exc}"
            ) from exc
        return cls.from_dict(payload)

    def _artifact_snapshot(self, artifact: str | Path) -> _ArtifactSnapshot:
        artifact_path = Path(artifact).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = artifact_path.absolute()
        cache_key = str(artifact_path)
        cached = self._artifact_snapshots.get(cache_key)
        if cached is not None:
            return cached
        manifest_path = artifact_manifest_path(artifact_path)
        artifact_stat = artifact_path.stat() if artifact_path.is_file() else None
        manifest_stat = manifest_path.stat() if manifest_path.is_file() else None
        if artifact_stat is None or manifest_stat is None:
            snapshot = _ArtifactSnapshot(
                ready=False,
                kernel_name=None,
                target_arch=None,
                artifact_sha256=None,
                manifest_sha256=None,
                reason="artifact or manifest file is missing",
            )
        else:
            try:
                manifest = load_artifact_manifest(artifact_path)
                ready = artifact_ready_for_execution(manifest)
                snapshot = _ArtifactSnapshot(
                    ready=ready,
                    kernel_name=manifest.kernel_name,
                    target_arch=manifest.target_arch,
                    artifact_sha256=(
                        file_sha256(artifact_path) if ready else None
                    ),
                    manifest_sha256=(
                        file_sha256(manifest_path) if ready else None
                    ),
                    reason=(
                        "artifact and manifest are correctness-promoted"
                        if ready
                        else "artifact manifest is not correctness-promoted"
                    ),
                )
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                snapshot = _ArtifactSnapshot(
                    ready=False,
                    kernel_name=None,
                    target_arch=None,
                    artifact_sha256=None,
                    manifest_sha256=None,
                    reason=f"artifact identity validation failed: {exc}",
                )
        self._artifact_snapshots[cache_key] = snapshot
        return snapshot

    def prime_artifact(self, artifact: str | Path) -> None:
        """Validate and memoize one immutable runtime artifact before forward."""

        snapshot = self._artifact_snapshot(artifact)
        if not snapshot.ready:
            raise XQTArtifactError(snapshot.reason)

    def invalidate_artifact(self, artifact: str | Path) -> None:
        """Drop a memoized identity after an artifact is rebuilt or replaced."""

        artifact_path = Path(artifact).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = artifact_path.absolute()
        self._artifact_snapshots.pop(str(artifact_path), None)

    def lookup(
        self,
        key: GemmTuningKey,
        *,
        artifact: str | Path,
        now: datetime | None = None,
    ) -> GemmTuningLookup:
        """Resolve one verified record without benchmarking or compiling."""

        if not isinstance(key, GemmTuningKey):
            raise TypeError("GemmTuningCache.lookup requires GemmTuningKey")
        record = self._records_by_key.get(key.cache_key)
        if record is None:
            return GemmTuningLookup(
                status="miss",
                key=key,
                reason="no exact offline tuning record matched the runtime contract",
                source="deterministic_default",
            )
        if record.is_expired(now=now):
            return GemmTuningLookup(
                status="expired",
                key=key,
                reason=f"tuning record expired at {record.expires_at}",
                source="deterministic_default",
            )
        if not record.correctness_verified:
            return GemmTuningLookup(
                status="invalid",
                key=key,
                reason="tuning record lacks correctness verification",
                source="deterministic_default",
            )
        if not record.selection or not record.benchmark:
            return GemmTuningLookup(
                status="invalid",
                key=key,
                reason="tuning record lacks selection or benchmark evidence",
                source="deterministic_default",
            )
        snapshot = self._artifact_snapshot(artifact)
        if not snapshot.ready:
            return GemmTuningLookup(
                status="invalid",
                key=key,
                reason=snapshot.reason,
                source="deterministic_default",
            )
        expected = record.artifact
        if snapshot.kernel_name != expected.kernel_name:
            reason = "runtime artifact kernel_name does not match tuning record"
        elif snapshot.target_arch != expected.target_arch:
            reason = "runtime artifact target_arch does not match tuning record"
        elif snapshot.artifact_sha256 != expected.artifact_sha256:
            reason = "runtime artifact checksum does not match tuning record"
        elif snapshot.manifest_sha256 != expected.manifest_sha256:
            reason = "runtime manifest checksum does not match tuning record"
        else:
            return GemmTuningLookup(
                status="hit",
                key=key,
                reason="verified offline record matched artifact and manifest identity",
                source=record.source,
                record=record,
            )
        return GemmTuningLookup(
            status="invalid",
            key=key,
            reason=reason,
            source="deterministic_default",
        )


def build_grouped_tuning_key(
    *,
    kernel_family: str,
    backend: str,
    target_arch: str,
    grouped_problem: GroupedGemmProblem,
    quant: QuantSpec,
    has_bias: bool,
    persistent: bool = False,
    cuda_graph: bool = False,
    workspace_bytes: int = 0,
) -> GemmTuningKey:
    """Build the canonical key used by all grouped GEMM bridges."""

    if not isinstance(grouped_problem, GroupedGemmProblem):
        raise TypeError("build_grouped_tuning_key requires GroupedGemmProblem")
    if not isinstance(quant, QuantSpec):
        raise TypeError("build_grouped_tuning_key requires QuantSpec")
    return GemmTuningKey(
        kernel_family=kernel_family,
        backend=backend,
        target_arch=target_arch,
        m=grouped_problem.total_m,
        n=grouped_problem.n,
        k=grouped_problem.k,
        expert_rows=tuple(problem.m for problem in grouped_problem.problems),
        weight_dtype=quant.weight_dtype,
        activation_dtype=quant.activation_dtype,
        compute_dtype=quant.compute_dtype,
        accum_dtype=quant.accum_dtype,
        output_dtype=quant.output_dtype,
        group_axis=quant.group_axis,
        group_size=quant.group_size,
        scale_mode=quant.scale_mode,
        symmetric=quant.symmetric,
        weight_zero_point=quant.weight_zero_point,
        activation_zero_point=quant.activation_zero_point,
        weight_scale_source=quant.weight_scale_source,
        activation_scale_source=quant.activation_scale_source,
        storage_layout=quant.storage_layout,
        pack_version=quant.pack_version,
        persistent=persistent,
        output_scatter=grouped_problem.output_rows is not None,
        has_bias=has_bias,
        cuda_graph=cuda_graph,
        workspace_bytes=workspace_bytes,
    )


def resolve_tuning_record(
    tuning_cache: GemmTuningCache | None,
    *,
    key: GemmTuningKey,
    artifact: str | Path,
    explicit: bool,
) -> GemmTuningLookup:
    """Resolve a cache hit or a deterministic non-autotuning fallback state."""

    if explicit:
        return GemmTuningLookup(
            status="bypassed_explicit",
            key=key,
            reason="explicit scheduler or launch configuration bypassed tuning cache",
            source="explicit_request",
        )
    if tuning_cache is None:
        return GemmTuningLookup(
            status="miss",
            key=key,
            reason="no preloaded tuning cache was provided",
            source="deterministic_default",
        )
    if not isinstance(tuning_cache, GemmTuningCache):
        raise TypeError("tuning_cache must be GemmTuningCache or None")
    return tuning_cache.lookup(key, artifact=artifact)


__all__ = [
    "GEMM_TUNING_CACHE_VERSION",
    "GemmTuningArtifactIdentity",
    "GemmTuningCache",
    "GemmTuningKey",
    "GemmTuningLookup",
    "GemmTuningRecord",
    "build_grouped_tuning_key",
    "resolve_tuning_record",
]
