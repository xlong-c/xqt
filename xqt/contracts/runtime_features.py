"""Self-owned runtime feature metadata (not serving configuration).

This module is the fact source for the runtime features XQT may describe on the
model side: ``prefix_cache``, ``paged_kv``, ``kv_cache_quant``,
``chunked_prefill``, ``speculative_decode`` and ``continuous_batching``.

Every feature explicitly declares:

- scope: whether it is model-side metadata or a real runtime capability;
- xqt_status: what XQT itself implements (metadata only, optional adapter,
  reference entity, or not implemented);
- owner: ``xqt`` or an external runtime.

XQT never implements cache management, batching schedulers, or online decoding.
The report must be able to explain supported / unsupported / unverified reasons
for every feature without promising serving capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from xqt.core.errors import XQTConfigError

RUNTIME_FEATURES_KEY = "runtime_features"
RUNTIME_FEATURES_SCHEMA_VERSION = 1

FEATURE_SCOPES = ("model_side_metadata", "runtime_capability")
FEATURE_STATUSES = (
    "supported",
    "unsupported",
    "unverified",
    "metadata_only",
    "not_implemented",
)
FEATURE_OWNERS = ("xqt", "external_runtime")


@dataclass(frozen=True, slots=True)
class RuntimeFeatureSpec:
    """One canonical runtime feature declaration."""

    name: str
    scope: str
    xqt_status: str
    owner: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope,
            "xqt_status": self.xqt_status,
            "owner": self.owner,
            "notes": list(self.notes),
        }


_RUNTIME_FEATURE_SPECS: tuple[RuntimeFeatureSpec, ...] = (
    RuntimeFeatureSpec(
        "prefix_cache",
        "model_side_metadata",
        "metadata_only",
        "external_runtime",
        (
            "XQT records the switch, cache block size and hit rate as runtime "
            "metrics; prefix cache management belongs to the serving engine.",
        ),
    ),
    RuntimeFeatureSpec(
        "paged_kv",
        "runtime_capability",
        "metadata_only",
        "external_runtime",
        (
            "XQT records switch, cache block size and hit rate as runtime "
            "metrics; page tables, block pools and eviction are external.",
        ),
    ),
    RuntimeFeatureSpec(
        "kv_cache_quant",
        "model_side_metadata",
        "reference_entity",
        "xqt",
        (
            "XQT produces per-tensor K/V scale artifacts and a reference "
            "attention entity; real KV storage dtype and kernels belong to the "
            "backend, CUDA kernel verification is pending.",
        ),
    ),
    RuntimeFeatureSpec(
        "chunked_prefill",
        "runtime_capability",
        "not_implemented",
        "external_runtime",
        (
            "XQT records the switch and chunk size only; chunk scheduling is "
            "not implemented in XQT.",
        ),
    ),
    RuntimeFeatureSpec(
        "speculative_decode",
        "model_side_metadata",
        "adapter_only",
        "external_runtime",
        (
            "XQT records draft/target model relation, acceptance rate and "
            "backend support; no decode engine is implemented in XQT.",
        ),
    ),
    RuntimeFeatureSpec(
        "continuous_batching",
        "runtime_capability",
        "not_implemented",
        "external_runtime",
        (
            "XQT records the switch and maximum batch size as metadata; "
            "request queue and scheduling are external.",
        ),
    ),
)

_RUNTIME_FEATURE_SPECS_BY_NAME: dict[str, RuntimeFeatureSpec] = {
    spec.name: spec for spec in _RUNTIME_FEATURE_SPECS
}

_DEFAULT_STATUS_BY_XQT_STATUS: dict[str, str] = {
    "metadata_only": "metadata_only",
    "adapter_only": "unverified",
    "reference_entity": "unverified",
    "not_implemented": "not_implemented",
}


def runtime_feature_specs() -> list[dict[str, Any]]:
    """Return all canonical runtime feature specs."""

    return [spec.to_dict() for spec in _RUNTIME_FEATURE_SPECS]


def describe_runtime_feature(name: str) -> dict[str, Any]:
    """Describe one canonical runtime feature, raising on unknown names."""

    spec = _RUNTIME_FEATURE_SPECS_BY_NAME.get(str(name).strip().lower())
    if spec is None:
        allowed = ", ".join(_RUNTIME_FEATURE_SPECS_BY_NAME)
        raise ValueError(f"Unknown runtime feature {name!r}. Allowed: {allowed}")
    return spec.to_dict()


@dataclass(frozen=True, slots=True)
class RuntimeFeatureEntry:
    """One feature's recorded state in a runtime feature metadata payload."""

    name: str
    enabled: bool
    status: str
    provider: str | None
    notes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": bool(self.enabled),
            "status": self.status,
            "provider": self.provider,
            "notes": list(self.notes),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class RuntimeFeatureMetadata:
    """Aggregated model-side runtime feature metadata payload."""

    schema_version: int = RUNTIME_FEATURES_SCHEMA_VERSION
    entries: tuple[RuntimeFeatureEntry, ...] = ()

    def by_name(self, name: str) -> RuntimeFeatureEntry | None:
        for entry in self.entries:
            if entry.name == str(name).strip().lower():
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeFeatureMetadata:
        if not isinstance(payload, Mapping):
            raise XQTConfigError(
                "RuntimeFeatureMetadata.from_dict expects a mapping; "
                f"got {type(payload).__name__}"
            )
        raw_entries = payload.get("entries", ())
        if not isinstance(raw_entries, (list, tuple)):
            raise XQTConfigError(
                "RuntimeFeatureMetadata.entries must be a sequence"
            )
        entries: list[RuntimeFeatureEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise XQTConfigError(
                    "RuntimeFeatureMetadata.entries items must be mappings"
                )
            name = str(raw.get("name", "")).strip().lower()
            spec = _RUNTIME_FEATURE_SPECS_BY_NAME.get(name)
            if spec is None:
                raise XQTConfigError(
                    f"Unknown runtime feature {name!r} in RuntimeFeatureMetadata"
                )
            enabled = raw.get("enabled", False)
            if not isinstance(enabled, bool):
                raise XQTConfigError(
                    f"RuntimeFeatureMetadata.{name}.enabled must be bool"
                )
            status = str(raw.get("status", "")).strip().lower()
            if status not in FEATURE_STATUSES:
                raise XQTConfigError(
                    f"RuntimeFeatureMetadata.{name}.status must be one of "
                    f"{', '.join(FEATURE_STATUSES)}; got {status!r}"
                )
            provider = raw.get("provider")
            raw_notes = raw.get("notes", ())
            if not isinstance(raw_notes, (list, tuple)):
                raise XQTConfigError(
                    f"RuntimeFeatureMetadata.{name}.notes must be a sequence"
                )
            raw_metrics = raw.get("metrics")
            metrics: dict[str, Any] = {}
            if raw_metrics is not None:
                if not isinstance(raw_metrics, Mapping):
                    raise XQTConfigError(
                        f"RuntimeFeatureMetadata.{name}.metrics must be a mapping"
                    )
                metrics = {str(key): value for key, value in raw_metrics.items()}
            entries.append(
                RuntimeFeatureEntry(
                    name=name,
                    enabled=enabled,
                    status=status,
                    provider=None if provider is None else str(provider),
                    notes=tuple(str(item) for item in raw_notes),
                    metrics=metrics,
                )
            )
        version = payload.get("schema_version", RUNTIME_FEATURES_SCHEMA_VERSION)
        try:
            version_i = int(version)
        except (TypeError, ValueError) as exc:
            raise XQTConfigError(
                f"RuntimeFeatureMetadata.schema_version must be int; got {version!r}"
            ) from exc
        return cls(schema_version=version_i, entries=tuple(entries))

    def report(self) -> dict[str, Any]:
        """Explain every feature's status and reason (support contract)."""

        return {
            "schema_version": int(self.schema_version),
            "features": {
                entry.name: {
                    "enabled": entry.enabled,
                    "status": entry.status,
                    "scope": _RUNTIME_FEATURE_SPECS_BY_NAME[entry.name].scope,
                    "provider": entry.provider,
                    "reason": (
                        "verified in XQT" if entry.status == "supported" else
                        "not supported by XQT" if entry.status == "unsupported" else
                        "implementation exists but verification pending"
                        if entry.status == "unverified" else
                        "model-side metadata only; runtime behavior belongs to "
                        "the external runtime"
                        if entry.status == "metadata_only" else
                        "not implemented in XQT"
                    ),
                    "notes": self._report_notes(entry),
                    "metrics": dict(entry.metrics),
                }
                for entry in self.entries
            },
        }

    def _report_notes(self, entry: RuntimeFeatureEntry) -> list[str]:
        spec = _RUNTIME_FEATURE_SPECS_BY_NAME[entry.name]
        return list(entry.notes) or list(spec.notes)


def _default_status(spec: RuntimeFeatureSpec) -> str:
    return _DEFAULT_STATUS_BY_XQT_STATUS[spec.xqt_status]


def build_runtime_feature_metadata(
    *,
    enabled: Mapping[str, bool] | None = None,
    statuses: Mapping[str, str] | None = None,
    providers: Mapping[str, str] | None = None,
    metrics: Mapping[str, Mapping[str, Any]] | None = None,
    notes: Mapping[str, Sequence[str]] | None = None,
) -> RuntimeFeatureMetadata:
    """Build canonical metadata for every declared feature.

    Unknown feature names raise ``XQTConfigError``; explicit statuses are
    validated against ``FEATURE_STATUSES``.
    """

    enabled_map = dict(enabled or {})
    status_map = dict(statuses or {})
    provider_map = dict(providers or {})
    metrics_map = dict(metrics or {})
    notes_map = dict(notes or {})
    entries: list[RuntimeFeatureEntry] = []
    for spec in _RUNTIME_FEATURE_SPECS:
        if spec.name in enabled_map:
            enabled_value = enabled_map[spec.name]
            if not isinstance(enabled_value, bool):
                raise XQTConfigError(
                    f"runtime feature {spec.name}.enabled must be bool"
                )
        else:
            enabled_value = False
        raw_status = status_map.get(spec.name)
        if raw_status is not None:
            status = str(raw_status).strip().lower()
            if status not in FEATURE_STATUSES:
                raise XQTConfigError(
                    f"runtime feature {spec.name}.status must be one of "
                    f"{', '.join(FEATURE_STATUSES)}; got {status!r}"
                )
        else:
            status = _default_status(spec)
        provider = provider_map.get(spec.name)
        raw_metrics = metrics_map.get(spec.name)
        if raw_metrics is not None and not isinstance(raw_metrics, Mapping):
            raise XQTConfigError(
                f"runtime feature {spec.name}.metrics must be a mapping"
            )
        raw_notes = notes_map.get(spec.name)
        if raw_notes is not None and not isinstance(raw_notes, (list, tuple)):
            raise XQTConfigError(
                f"runtime feature {spec.name}.notes must be a sequence"
            )
        entries.append(
            RuntimeFeatureEntry(
                name=spec.name,
                enabled=enabled_value,
                status=status,
                provider=None if provider is None else str(provider),
                notes=tuple(str(item) for item in (raw_notes or ())),
                metrics=dict(raw_metrics) if raw_metrics is not None else {},
            )
        )
    return RuntimeFeatureMetadata(entries=tuple(entries))


def runtime_feature_report(
    metadata: RuntimeFeatureMetadata | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the support/unsupported/unverified report for feature metadata."""

    if metadata is None:
        return {
            "schema_version": RUNTIME_FEATURES_SCHEMA_VERSION,
            "features": {},
            "notes": ["runtime_features_absent"],
        }
    if isinstance(metadata, RuntimeFeatureMetadata):
        return metadata.report()
    if isinstance(metadata, Mapping):
        return RuntimeFeatureMetadata.from_dict(metadata).report()
    raise XQTConfigError(
        "runtime_feature_report expects RuntimeFeatureMetadata, mapping, or None; "
        f"got {type(metadata).__name__}"
    )


def speculative_decode_metadata(
    *,
    draft_model: str,
    target_model: str,
    backend: str,
    acceptance_rate: float | None = None,
    enabled: bool = True,
) -> RuntimeFeatureMetadata:
    """Record speculative decode relations only (no decode engine)."""

    if not str(draft_model).strip() or not str(target_model).strip():
        raise XQTConfigError(
            "speculative_decode_metadata requires draft_model and target_model"
        )
    if acceptance_rate is not None and not 0.0 <= float(acceptance_rate) <= 1.0:
        raise XQTConfigError("acceptance_rate must be in [0, 1]")
    metrics: dict[str, Any] = {
        "draft_model": str(draft_model),
        "target_model": str(target_model),
    }
    if acceptance_rate is not None:
        metrics["acceptance_rate"] = float(acceptance_rate)
    return build_runtime_feature_metadata(
        enabled={"speculative_decode": enabled},
        statuses={"speculative_decode": "unverified"},
        providers={"speculative_decode": str(backend)},
        metrics={"speculative_decode": metrics},
        notes={
            "speculative_decode": (
                "draft/target relation and acceptance rate recorded as "
                "model-side metadata; no decode engine in XQT.",
            )
        },
    )


def prefix_paged_kv_metadata(
    *,
    prefix_enabled: bool = False,
    paged_enabled: bool = False,
    cache_block_size: int | None = None,
    hit_rate: float | None = None,
    backend: str | None = None,
) -> RuntimeFeatureMetadata:
    """Record prefix cache / paged KV switches and runtime metrics only."""

    if cache_block_size is not None and int(cache_block_size) <= 0:
        raise XQTConfigError("cache_block_size must be positive")
    if hit_rate is not None and not 0.0 <= float(hit_rate) <= 1.0:
        raise XQTConfigError("hit_rate must be in [0, 1]")
    metrics: dict[str, Any] = {}
    if cache_block_size is not None:
        metrics["cache_block_size"] = int(cache_block_size)
    if hit_rate is not None:
        metrics["hit_rate"] = float(hit_rate)
    return build_runtime_feature_metadata(
        enabled={
            "prefix_cache": bool(prefix_enabled),
            "paged_kv": bool(paged_enabled),
        },
        statuses={
            "prefix_cache": "metadata_only",
            "paged_kv": "metadata_only",
        },
        providers={
            "prefix_cache": backend,
            "paged_kv": backend,
        },
        metrics={
            "prefix_cache": dict(metrics),
            "paged_kv": dict(metrics),
        },
        notes={
            "prefix_cache": (
                "switch/block size/hit rate recorded as runtime metrics; "
                "cache management stays in the serving engine.",
            ),
            "paged_kv": (
                "switch/block size/hit rate recorded as runtime metrics; "
                "page tables and block pools stay in the serving engine.",
            ),
        },
    )


__all__ = [
    "FEATURE_OWNERS",
    "FEATURE_SCOPES",
    "FEATURE_STATUSES",
    "RUNTIME_FEATURES_KEY",
    "RUNTIME_FEATURES_SCHEMA_VERSION",
    "RuntimeFeatureEntry",
    "RuntimeFeatureMetadata",
    "RuntimeFeatureSpec",
    "build_runtime_feature_metadata",
    "describe_runtime_feature",
    "prefix_paged_kv_metadata",
    "runtime_feature_report",
    "runtime_feature_specs",
    "speculative_decode_metadata",
]
