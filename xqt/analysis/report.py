"""Small report helpers for tests and recipe manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _extract_metric(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def flatten_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten nested metric mappings into dotted keys."""

    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        flat_key = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(
                flatten_metrics(value, prefix=flat_key, separator=separator)
            )
        else:
            flattened[flat_key] = value
    return flattened


def write_json_report(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write a mapping as a JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def write_csv_report(rows: list[Mapping[str, Any]], path: str | Path) -> Path:
    """Write row dictionaries to a CSV report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return output_path


def records_to_rows(records: list[object]) -> list[dict[str, Any]]:
    """Normalize analysis records into flat row dictionaries."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if hasattr(record, "to_dict"):
            raw = record.to_dict()
        elif isinstance(record, Mapping):
            raw = dict(record)
        else:
            raise TypeError("record must be a mapping or expose to_dict()")
        rows.append(flatten_metrics(raw))
    return rows


def records_to_dataframe(records: list[object]) -> pd.DataFrame:
    """Convert analysis records to a stable DataFrame for reporting."""

    return pd.DataFrame(records_to_rows(records))


def build_pareto_points(
    runs: list[Mapping[str, Any]],
    *,
    config_id_key: str = "config_id",
    latency_key: str = "benchmark.latency.p50_ms",
    memory_key: str = "benchmark.memory.delta_bytes",
    error_key: str = "analysis.records.0.diff.mean_abs",
    metric_delta_key: str = "baseline.metrics.top1",
) -> list[dict[str, Any]]:
    """Build stable Pareto rows from run metric snapshots."""

    points: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        point = {
            "config_id": run.get(config_id_key, f"run_{index}"),
            "latency_ms": _extract_metric(run, latency_key),
            "memory_bytes": _extract_metric(run, memory_key),
            "error": _extract_metric(run, error_key),
            "metric_delta": _extract_metric(run, metric_delta_key),
        }
        points.append(point)
    return points


def write_markdown_report(
    title: str,
    sections: Mapping[str, Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write simple metric sections as a Markdown report."""

    lines = [f"# {title}", ""]
    for section_name, values in sections.items():
        lines.extend([f"## {section_name}", "", "| Metric | Value |", "| --- | --- |"])
        for key, value in values.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = [
    "build_pareto_points",
    "flatten_metrics",
    "records_to_dataframe",
    "records_to_rows",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
]
