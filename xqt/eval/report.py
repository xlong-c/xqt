"""Small report helpers for tests and recipe manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


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
    "flatten_metrics",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
]
