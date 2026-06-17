import hashlib
import json

import pytest

from xqt.core.artifact import (
    ArtifactManifest,
    ArtifactRecord,
    MetricRecord,
    collect_dependency_versions,
    file_sha256,
    load_manifest,
)
from xqt.core.errors import XQTArtifactError


def test_file_sha256_and_artifact_record_from_file(tmp_path) -> None:
    artifact_path = tmp_path / "model.onnx"
    artifact_path.write_bytes(b"xqt-artifact")

    expected = hashlib.sha256(b"xqt-artifact").hexdigest()

    assert file_sha256(artifact_path) == expected

    record = ArtifactRecord.from_file(
        artifact_path,
        format="onnx",
        runtime="onnxruntime",
        metadata={"opset": 18},
    )

    assert record.path == str(artifact_path)
    assert record.format == "onnx"
    assert record.runtime == "onnxruntime"
    assert record.checksum == expected
    assert record.metadata == {"opset": 18}


def test_file_sha256_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(XQTArtifactError, match="Artifact file not found"):
        file_sha256(tmp_path / "missing.pt")


def test_manifest_write_and_load_round_trip(tmp_path) -> None:
    manifest = ArtifactManifest(
        project_name="xqt_test",
        source_checkpoint="teacher.pt",
        compression_axes=["precision"],
        config_snapshot={"project": {"name": "xqt_test"}},
    )
    manifest.add_artifact(
        ArtifactRecord(
            path="model.onnx",
            format="onnx",
            checksum="abc",
            runtime="onnxruntime",
        )
    )
    manifest.add_metric(
        MetricRecord(
            name="output.max_abs",
            value=0.001,
            threshold=0.01,
            passed=True,
        )
    )

    output_path = manifest.write_json(tmp_path / "manifest.json")
    loaded = load_manifest(output_path)

    assert loaded["project_name"] == "xqt_test"
    assert loaded["compression_axes"] == ["precision"]
    assert loaded["artifacts"][0]["format"] == "onnx"
    assert loaded["metrics"][0]["passed"] is True

    json.dumps(loaded)


def test_collect_dependency_versions_contains_core_runtime_fields() -> None:
    versions = collect_dependency_versions()

    assert versions["python"]
    assert versions["torch"]
    assert "cuda" in versions
    assert "torchao" in versions
