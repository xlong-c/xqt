"""Deprecated compatibility imports for :mod:`xqt.core.base.artifact`."""

from .base.artifact import (
    ArtifactManifest,
    ArtifactRecord,
    MetricRecord,
    collect_dependency_versions,
    file_sha256,
    load_manifest,
    utc_timestamp,
)


__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "MetricRecord",
    "collect_dependency_versions",
    "file_sha256",
    "load_manifest",
    "utc_timestamp",
]
