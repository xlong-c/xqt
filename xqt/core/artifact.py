"""Artifact manifest helpers for XQT runs."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .errors import XQTArtifactError


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 checksum for a file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise XQTArtifactError(f"Artifact file not found: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_dependency_versions() -> Dict[str, Optional[str]]:
    """Collect lightweight runtime dependency versions."""

    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for package_name in ("torchao", "onnx", "onnxruntime", "tensorrt", "openvino"):
        try:
            module = __import__(package_name)
        except ImportError:
            versions[package_name] = None
            continue
        versions[package_name] = getattr(module, "__version__", "unknown")
    return versions


@dataclass
class ArtifactRecord:
    """A single generated artifact."""

    path: str
    format: str
    runtime: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        format: str,
        runtime: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ArtifactRecord":
        file_path = Path(path)
        return cls(
            path=str(file_path),
            format=format,
            runtime=runtime,
            checksum=file_sha256(file_path),
            metadata=dict(metadata or {}),
        )


@dataclass
class MetricRecord:
    """A named metric or validation result."""

    name: str
    value: Any
    threshold: Optional[Any] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactManifest:
    """XQT run manifest."""

    project_name: str
    created_at: str = field(default_factory=utc_timestamp)
    xqt_version: str = "0.1.0"
    source_checkpoint: Optional[str] = None
    source_checksum: Optional[str] = None
    compression_axes: List[str] = field(default_factory=list)
    passes: List[str] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Optional[str]] = field(default_factory=collect_dependency_versions)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    metrics: List[MetricRecord] = field(default_factory=list)
    operator_optimization: Optional[Dict[str, Any]] = None

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        """Append an artifact record."""

        self.artifacts.append(artifact)

    def add_metric(self, metric: MetricRecord) -> None:
        """Append a metric record."""

        self.metrics.append(metric)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to a JSON-serializable dictionary."""

        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        """Write the manifest to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load a manifest JSON file as a plain dictionary."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "MetricRecord",
    "collect_dependency_versions",
    "file_sha256",
    "load_manifest",
    "utc_timestamp",
]
