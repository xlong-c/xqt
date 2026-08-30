"""Report-writing pass for XQT stage execution."""

from __future__ import annotations

from pathlib import Path

from xqt.analysis import write_json_report, write_markdown_report
from xqt.core.types import XQTContext


class WriteReportsPass:
    """Write JSON and Markdown reports for collected metrics."""

    name = "write_reports"

    def run(self, context: XQTContext) -> XQTContext:
        artifact_dir = Path(context.artifact_dir)
        json_path = write_json_report(context.metrics, artifact_dir / "metrics.json")
        markdown_path = write_markdown_report(
            "XQT Metrics",
            {
                key: value if isinstance(value, dict) else {"value": value}
                for key, value in context.metrics.items()
            },
            artifact_dir / "metrics.md",
        )
        context.artifacts["metrics_json"] = json_path
        context.artifacts["metrics_markdown"] = markdown_path
        return context


__all__ = ["WriteReportsPass"]
