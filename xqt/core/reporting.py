"""Shared reporting schemas for XQT optimization runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .artifact import ArtifactManifest, MetricRecord
from .serialization import json_safe_value


CAPABILITY_STATUSES = (
    "available",
    "adapter",
    "planned",
    "unavailable",
    "unsupported",
    "skipped",
    "not_verified",
)

CAPABILITY_MATURITIES = (
    "executable",
    "reference_guarded",
    "metadata_only",
    "planned",
)


def _tuple_of_str(value: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(str(item) for item in value or ())


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe report value without persisting live handles."""

    return json_safe_value(value)


def _find_first_numeric(value: Any, key: str) -> float | int | None:
    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, (float, int)):
            return raw
        for item in value.values():
            found = _find_first_numeric(item, key)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_numeric(item, key)
            if found is not None:
                return found
    return None


def _find_first_text(value: Any, key: str) -> str | None:
    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str):
                    return item
                if item is not None and not isinstance(item, (Mapping, list, tuple)):
                    return str(item)
        if raw is not None and not isinstance(raw, (Mapping, list, tuple)):
            return str(raw)
        for item in value.values():
            found = _find_first_text(item, key)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_text(item, key)
            if found is not None:
                return found
    return None


def _find_first_bool(value: Any, key: str) -> bool | None:
    if isinstance(value, Mapping):
        raw = value.get(key)
        if isinstance(raw, bool):
            return raw
        for item in value.values():
            found = _find_first_bool(item, key)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_bool(item, key)
            if found is not None:
                return found
    return None


def _find_first_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_first_value(item, key)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_value(item, key)
            if found is not None:
                return found
    return None


def _shape_signature(value: Any) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(dimension) for dimension in shape]
        except (TypeError, ValueError):
            return str(shape)
    if isinstance(value, Mapping):
        return {str(key): _shape_signature(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_signature(item) for item in value]
    return None


def _collect_text_values(value: Any, key: str) -> list[str]:
    if isinstance(value, Mapping):
        values: list[str] = []
        raw = value.get(key)
        if isinstance(raw, str):
            values.append(raw)
        for item in value.values():
            values.extend(_collect_text_values(item, key))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_collect_text_values(item, key))
        return values
    return []


def _artifact_kinds(
    metrics: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> list[str]:
    values: list[str] = []
    for key in ("artifact_kind", "format"):
        for value in _collect_text_values(metrics, key):
            if value not in values:
                values.append(value)
    if values:
        return values
    return [str(key) for key in artifacts]


@dataclass(frozen=True)
class OptimizationCapability:
    """Unified capability description for quant, prune, operator, and export paths."""

    kind: str
    name: str
    engine: str
    status: str
    maturity: str
    runtime: str
    artifact_kind: str
    requires_cuda: bool = False
    requires_calibration: bool = False
    requires_exportable_graph: bool = False
    available: bool | None = None
    supported: bool | None = None
    methods: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()
    target_module_types: tuple[str, ...] = ()
    precisions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable capability payload."""

        return {
            "kind": self.kind,
            "name": self.name,
            "engine": self.engine,
            "status": self.status,
            "maturity": self.maturity,
            "runtime": self.runtime,
            "artifact_kind": self.artifact_kind,
            "requires_cuda": self.requires_cuda,
            "requires_calibration": self.requires_calibration,
            "requires_exportable_graph": self.requires_exportable_graph,
            "available": self.available,
            "supported": self.supported,
            "methods": list(self.methods),
            "model_families": list(self.model_families),
            "target_module_types": list(self.target_module_types),
            "precisions": list(self.precisions),
            "notes": list(self.notes),
            "limitations": list(self.limitations),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        kind: str,
        name: str,
        engine: str | None = None,
    ) -> "OptimizationCapability":
        """Create a unified capability from a capability dictionary."""

        raw_status = str(payload.get("status", "not_verified"))
        raw_maturity = str(payload.get("maturity", "planned"))
        runtime = str(payload.get("runtime", "unknown"))
        artifact_kind = str(payload.get("artifact_kind", "unknown"))
        return cls(
            kind=kind,
            name=name,
            engine=str(engine or payload.get("engine") or name),
            status=normalize_capability_status(raw_status),
            maturity=normalize_capability_maturity(raw_maturity),
            runtime=runtime,
            artifact_kind=artifact_kind,
            requires_cuda=bool(payload.get("requires_cuda", False)),
            requires_calibration=bool(payload.get("requires_calibration", False)),
            requires_exportable_graph=bool(
                payload.get("requires_exportable_graph", False)
            ),
            available=(bool(payload["available"]) if "available" in payload else None),
            supported=(bool(payload["supported"]) if "supported" in payload else None),
            methods=_tuple_of_str(payload.get("methods")),
            model_families=_tuple_of_str(payload.get("model_families")),
            target_module_types=_tuple_of_str(
                payload.get("primary_module_types")
                or payload.get("target_module_types")
                or payload.get("candidate_module_types")
            ),
            precisions=_tuple_of_str(payload.get("precisions")),
            notes=_tuple_of_str(payload.get("notes")),
            limitations=_tuple_of_str(payload.get("limitations")),
            metadata={
                key: _json_safe(value)
                for key, value in payload.items()
                if key
                not in {
                    "engine",
                    "status",
                    "maturity",
                    "runtime",
                    "artifact_kind",
                    "requires_cuda",
                    "requires_calibration",
                    "requires_exportable_graph",
                    "available",
                    "supported",
                    "methods",
                    "model_families",
                    "primary_module_types",
                    "target_module_types",
                    "candidate_module_types",
                    "precisions",
                    "notes",
                    "limitations",
                }
            },
        )


@dataclass(frozen=True)
class BenchmarkMetricSchema:
    """Canonical benchmark metric groups used by XQT reports."""

    latency_fields: tuple[str, ...] = (
        "mean_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
    )
    throughput_fields: tuple[str, ...] = (
        "throughput",
        "throughput_items_per_s",
    )
    memory_fields: tuple[str, ...] = (
        "peak_memory_bytes",
        "delta_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    )
    compile_fields: tuple[str, ...] = (
        "compile_time_ms",
        "graph_count",
        "graph_break_count",
    )
    artifact_fields: tuple[str, ...] = (
        "artifact_size_bytes",
        "artifact_checksum",
    )
    llm_workload_fields: tuple[str, ...] = (
        "prompt_length",
        "output_length",
        "batch_size",
        "ttft_ms",
        "tpot_ms",
        "tokens_per_s",
        "request_throughput",
        "kv_memory_bytes",
        "prefix_cache_hit_rate",
    )

    def to_dict(self) -> dict[str, list[str]]:
        """Return metric groups as plain lists."""

        return {
            "latency": list(self.latency_fields),
            "throughput": list(self.throughput_fields),
            "memory": list(self.memory_fields),
            "compile": list(self.compile_fields),
            "artifact": list(self.artifact_fields),
            "llm_workload": list(self.llm_workload_fields),
        }


@dataclass(frozen=True)
class NumericDiffSchema:
    """Canonical numeric difference metric names used by XQT reports."""

    fields: tuple[str, ...] = (
        "allclose",
        "max_abs",
        "mean_abs",
        "cosine",
        "relative_error",
        "failed_tensor_count",
    )

    def to_dict(self) -> dict[str, list[str]]:
        """Return numeric diff fields as a plain dictionary."""

        return {"numeric_diff": list(self.fields)}


@dataclass(frozen=True)
class RuntimeFeatureSchema:
    """Runtime feature metadata reserved for deployment adapters."""

    fields: tuple[str, ...] = (
        "prefix_cache",
        "paged_kv",
        "kv_cache_quant",
        "chunked_prefill",
        "speculative_decode",
        "continuous_batching",
    )

    def to_dict(self) -> dict[str, list[str]]:
        """Return runtime feature fields as a plain dictionary."""

        return {"runtime_features": list(self.fields)}


@dataclass(frozen=True)
class StageReport:
    """Manifest-friendly report for one optimization workflow stage."""

    stage_name: str
    stage_kind: str
    status: str
    accepted: bool
    message: str
    engine: str | None = None
    target_module: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    capability: dict[str, Any] | None = None
    benchmark: dict[str, Any] = field(default_factory=dict)
    numeric_diff: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable stage report."""

        return {
            "stage_name": self.stage_name,
            "stage_kind": self.stage_kind,
            "status": self.status,
            "accepted": self.accepted,
            "message": self.message,
            "engine": self.engine,
            "target_module": self.target_module,
            "metrics": _json_safe(self.metrics),
            "artifacts": _json_safe(self.artifacts),
            "capability": _json_safe(self.capability),
            "benchmark": _json_safe(self.benchmark),
            "numeric_diff": _json_safe(self.numeric_diff),
            "execution": _json_safe(self.execution),
            "lineage": _json_safe(self.lineage),
            "metadata": _json_safe(self.metadata),
        }


def normalize_capability_status(status: str) -> str:
    """Normalize legacy status names to the shared capability status vocabulary."""

    text = str(status).lower()
    if text == "implemented":
        return "available"
    if text in CAPABILITY_STATUSES:
        return text
    if text in {"ok", "present"}:
        return "available"
    if text in {"missing", "failed", "error"}:
        return "unavailable"
    return "not_verified"


def normalize_capability_maturity(maturity: str) -> str:
    """Normalize capability maturity names to the shared maturity vocabulary."""

    text = str(maturity).lower()
    aliases = {
        "available": "executable",
        "implemented": "executable",
        "adapter": "executable",
        "runtime_kernel": "executable",
        "production": "executable",
        "reference": "reference_guarded",
        "reference_only": "reference_guarded",
        "reference-guarded": "reference_guarded",
        "design_extracted_reference_guarded": "reference_guarded",
        "metadata": "metadata_only",
        "capability_only": "metadata_only",
    }
    normalized = aliases.get(text, text)
    if normalized in CAPABILITY_MATURITIES:
        return normalized
    return "planned"


def default_benchmark_metric_schema() -> BenchmarkMetricSchema:
    """Return the canonical benchmark metric schema."""

    return BenchmarkMetricSchema()


def default_numeric_diff_schema() -> NumericDiffSchema:
    """Return the canonical numeric diff schema."""

    return NumericDiffSchema()


def default_runtime_feature_schema() -> RuntimeFeatureSchema:
    """Return runtime feature fields reserved for adapter/readiness reports."""

    return RuntimeFeatureSchema()


def reporting_schema_payload() -> dict[str, Any]:
    """Return all shared reporting schemas as one dictionary."""

    return {
        "benchmark": default_benchmark_metric_schema().to_dict(),
        "numeric_diff": default_numeric_diff_schema().to_dict(),
        "runtime_features": default_runtime_feature_schema().to_dict(),
    }


def normalize_benchmark_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Project arbitrary stage metrics onto the canonical benchmark fields."""

    schema = default_benchmark_metric_schema()
    fields = (
        schema.latency_fields
        + schema.throughput_fields
        + schema.memory_fields
        + schema.compile_fields
        + schema.artifact_fields
        + schema.llm_workload_fields
    )
    normalized = {field: _find_first_numeric(metrics, field) for field in fields}
    if normalized["peak_memory_bytes"] is None:
        normalized["peak_memory_bytes"] = _find_first_numeric(metrics, "peak_bytes")
    return normalized


def normalize_numeric_diff_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Project arbitrary stage metrics onto the canonical numeric diff fields."""

    schema = default_numeric_diff_schema()
    normalized: dict[str, Any] = {}
    for field_name in schema.fields:
        if field_name == "allclose":
            normalized[field_name] = _find_first_bool(metrics, field_name)
        else:
            normalized[field_name] = _find_first_numeric(metrics, field_name)
    return normalized


def normalize_stage_execution(
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    engine: str | None = None,
    device: str | None = None,
    shape: Any = None,
    warmup: int | None = None,
    iterations: int | None = None,
) -> dict[str, Any]:
    """Project every workflow stage onto a fixed runtime-reporting envelope."""

    metric_shape = _find_first_value(metrics, "shape_signature") or _find_first_value(
        metrics, "input_shapes"
    )
    fallback = (
        _find_first_text(metrics, "fallback_reason")
        or _find_first_text(metrics, "skip_reason")
        or _find_first_text(metrics, "fallback")
    )
    return {
        "backend": _find_first_text(metrics, "backend")
        or _find_first_text(metrics, "runtime")
        or engine
        or _find_first_text(metrics, "engine")
        or _find_first_text(metrics, "engines"),
        "engine": engine
        or _find_first_text(metrics, "engine")
        or _find_first_text(metrics, "engines"),
        "device": device or _find_first_text(metrics, "device"),
        "shape": _json_safe(metric_shape)
        if metric_shape is not None
        else _shape_signature(shape),
        "warmup": warmup
        if warmup is not None
        else _find_first_numeric(metrics, "warmup"),
        "iterations": (
            iterations
            if iterations is not None
            else _find_first_numeric(metrics, "iterations")
        ),
        "fallback": fallback,
        "artifact_kinds": _artifact_kinds(metrics, artifacts),
    }


def build_stage_report(
    *,
    stage_name: str,
    stage_kind: str,
    accepted: bool,
    message: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    engine: str | None = None,
    target_module: str | None = None,
    capability: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    device: str | None = None,
    shape: Any = None,
    warmup: int | None = None,
    iterations: int | None = None,
) -> StageReport:
    """Build a canonical stage report from workflow result data."""

    status = "accepted" if accepted else "rejected"
    return StageReport(
        stage_name=stage_name,
        stage_kind=stage_kind,
        status=status,
        accepted=accepted,
        message=message,
        engine=engine
        or _find_first_text(metrics, "engine")
        or _find_first_text(metrics, "engines"),
        target_module=target_module
        or _find_first_text(metrics, "target_module")
        or _find_first_text(metrics, "module_path")
        or _find_first_text(metrics, "target_path")
        or _find_first_text(metrics, "target_name"),
        metrics=_json_safe(dict(metrics)),
        artifacts=_json_safe(dict(artifacts)),
        capability=_json_safe(dict(capability)) if capability is not None else None,
        benchmark=normalize_benchmark_metrics(metrics),
        numeric_diff=normalize_numeric_diff_metrics(metrics),
        execution=normalize_stage_execution(
            metrics,
            artifacts,
            engine=engine,
            device=device,
            shape=shape,
            warmup=warmup,
            iterations=iterations,
        ),
        lineage=_json_safe(dict(lineage or {})),
        metadata=_json_safe(dict(metadata or {})),
    )


def add_stage_report_to_manifest(
    manifest: ArtifactManifest,
    report: StageReport,
) -> ArtifactManifest:
    """Attach a canonical stage report to an artifact manifest."""

    payload = report.to_dict()
    manifest.add_metric(
        MetricRecord(
            name=f"stage.{report.stage_name}.status",
            value=report.status,
            passed=report.accepted,
            metadata=payload,
        )
    )
    manifest.add_metric(
        MetricRecord(
            name=f"stage.{report.stage_name}.accepted",
            value=report.accepted,
            passed=report.accepted,
            metadata={
                "stage_name": report.stage_name,
                "stage_kind": report.stage_kind,
                "message": report.message,
            },
        )
    )
    if report.capability is not None:
        status = str(report.capability.get("status", "not_verified"))
        manifest.add_metric(
            MetricRecord(
                name=f"stage.{report.stage_name}.capability.status",
                value=status,
                passed=status in {"available", "adapter"},
                metadata=report.capability,
            )
        )
    return manifest


__all__ = [
    "CAPABILITY_MATURITIES",
    "CAPABILITY_STATUSES",
    "BenchmarkMetricSchema",
    "NumericDiffSchema",
    "OptimizationCapability",
    "RuntimeFeatureSchema",
    "StageReport",
    "add_stage_report_to_manifest",
    "build_stage_report",
    "default_benchmark_metric_schema",
    "default_numeric_diff_schema",
    "default_runtime_feature_schema",
    "normalize_benchmark_metrics",
    "normalize_capability_maturity",
    "normalize_capability_status",
    "normalize_numeric_diff_metrics",
    "normalize_stage_execution",
    "reporting_schema_payload",
]
