"""Standard file-based model-package helpers for inference."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from xqt.core.artifact import file_sha256, utc_timestamp
from xqt.core.errors import XQTArtifactError, XQTBackendError
from xqt.core.serialization import json_safe_value
from xqt.export import create_onnxruntime_session
from xqt.export.input_utils import build_onnx_feed

MODEL_PACKAGE_SCHEMA_VERSION = "1.0"
MODEL_PACKAGE_ARTIFACT_TYPE = "xqt_model_package"


def _json_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    safe_value = json_safe_value(dict(value))
    if not isinstance(safe_value, dict):
        raise XQTArtifactError(f"{name} must serialize to a JSON object")
    return safe_value


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise XQTArtifactError(f"{name} file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise XQTArtifactError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise XQTArtifactError(f"{name} must contain a JSON object: {path}")
    return payload


def _resolve_package_file(
    package_dir: Path,
    relative_path: str,
    *,
    name: str,
) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise XQTArtifactError(f"{name} must be a relative path: {relative_path}")
    root = package_dir.resolve()
    resolved = (package_dir / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise XQTArtifactError(
            f"{name} escapes the model package root: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise XQTArtifactError(f"{name} file not found: {resolved}")
    return resolved


def _io_names(entries: Any) -> list[str]:
    if not isinstance(entries, list):
        return []
    names: list[str] = []
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


@dataclass
class ModelPackageManifest:
    """Stable runtime-facing package manifest."""

    schema_version: str = MODEL_PACKAGE_SCHEMA_VERSION
    artifact_type: str = MODEL_PACKAGE_ARTIFACT_TYPE
    package_version: str = "1.0"
    entrypoints: dict[str, str] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    io: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelPackageManifest":
        if payload.get("artifact_type") != MODEL_PACKAGE_ARTIFACT_TYPE:
            raise XQTArtifactError(
                "model package manifest has unsupported artifact_type: "
                f"{payload.get('artifact_type')!r}"
            )
        if str(payload.get("schema_version")) != MODEL_PACKAGE_SCHEMA_VERSION:
            raise XQTArtifactError(
                "model package manifest has unsupported schema_version: "
                f"{payload.get('schema_version')!r}"
            )
        entrypoints = payload.get("entrypoints")
        model = payload.get("model")
        runtime = payload.get("runtime")
        if not isinstance(entrypoints, Mapping):
            raise XQTArtifactError("model package manifest.entrypoints is required")
        if not isinstance(model, Mapping):
            raise XQTArtifactError("model package manifest.model is required")
        if not isinstance(runtime, Mapping):
            raise XQTArtifactError("model package manifest.runtime is required")
        for field_name in ("model", "runtime_config"):
            raw = entrypoints.get(field_name)
            if not isinstance(raw, str) or not raw:
                raise XQTArtifactError(
                    f"model package entrypoints.{field_name} must be a non-empty string"
                )
        model_format = model.get("format")
        if not isinstance(model_format, str) or not model_format:
            raise XQTArtifactError(
                "model package manifest.model.format must be a non-empty string"
            )
        io = payload.get("io", {})
        quantization = payload.get("quantization", {})
        metadata = payload.get("metadata", {})
        if not isinstance(io, Mapping):
            raise XQTArtifactError("model package manifest.io must be a JSON object")
        if not isinstance(quantization, Mapping):
            raise XQTArtifactError(
                "model package manifest.quantization must be a JSON object"
            )
        if not isinstance(metadata, Mapping):
            raise XQTArtifactError(
                "model package manifest.metadata must be a JSON object"
            )
        return cls(
            schema_version=str(payload["schema_version"]),
            artifact_type=str(payload["artifact_type"]),
            package_version=str(payload.get("package_version", "1.0")),
            entrypoints={str(key): str(value) for key, value in entrypoints.items()},
            model=dict(model),
            runtime=dict(runtime),
            io=dict(io),
            quantization=dict(quantization),
            metadata=dict(metadata),
        )


@dataclass
class LoadedModelPackage:
    """Resolved package view used by runtime consumers."""

    package_dir: Path
    manifest_path: Path
    manifest: ModelPackageManifest
    model_path: Path
    runtime_config_path: Path
    runtime_config: dict[str, Any]

    @property
    def model_format(self) -> str:
        return str(self.manifest.model.get("format", ""))

    @property
    def preferred_backend(self) -> str:
        runtime_name = self.runtime_config.get("runtime")
        if isinstance(runtime_name, str) and runtime_name:
            return runtime_name
        preferred = self.manifest.runtime.get("preferred_backend")
        if isinstance(preferred, str) and preferred:
            return preferred
        return "onnxruntime"


class ONNXRuntimeRunner:
    """Thin file-based runner backed by one ONNX Runtime session."""

    def __init__(
        self,
        package: LoadedModelPackage,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        if package.model_format != "onnx":
            raise XQTBackendError(
                f"ONNX Runtime runner requires an ONNX package, got {package.model_format!r}"
            )
        if package.preferred_backend != "onnxruntime":
            raise XQTBackendError(
                "ONNX Runtime runner requires runtime=onnxruntime in the package config"
            )
        configured_providers = package.runtime_config.get("providers")
        resolved_providers = list(providers or [])
        if not resolved_providers and isinstance(configured_providers, list):
            resolved_providers = [
                str(item) for item in configured_providers if isinstance(item, str)
            ]
        if not resolved_providers:
            resolved_providers = ["CPUExecutionProvider"]
        self.package = package
        self.providers = resolved_providers
        self.input_names = _io_names(package.manifest.io.get("inputs"))
        if not self.input_names:
            raw_names = package.manifest.model.get("input_names", [])
            if isinstance(raw_names, list):
                self.input_names = [
                    str(item) for item in raw_names if isinstance(item, str)
                ]
        self.session = create_onnxruntime_session(
            package.model_path,
            providers=self.providers,
        )

    def run(self, inputs: Any) -> list[Any]:
        input_names = self.input_names or ["input"]
        feeds = build_onnx_feed(inputs, input_names=input_names)
        return self.session.run(None, feeds)

    def __call__(self, inputs: Any) -> list[Any]:
        return self.run(inputs)


def write_model_package(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    model_format: str,
    runtime_name: str,
    runtime_config: Mapping[str, Any] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    io: Mapping[str, Any] | None = None,
    quantization: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    package_version: str = "1.0",
) -> Path:
    """Write one self-contained runtime package rooted by manifest.json."""

    source_model_path = Path(model_path)
    if not source_model_path.is_file():
        raise XQTArtifactError(f"model file not found for packaging: {source_model_path}")

    package_dir = Path(output_dir)
    model_dir = package_dir / "model"
    runtime_dir = package_dir / "runtime"
    model_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    packaged_model_path = model_dir / source_model_path.name
    if source_model_path.resolve() != packaged_model_path.resolve():
        shutil.copy2(source_model_path, packaged_model_path)

    runtime_payload = _json_mapping(runtime_config, name="runtime_config")
    runtime_payload["runtime"] = str(runtime_name)
    runtime_config_path = runtime_dir / "config.json"
    runtime_config_path.write_text(
        json.dumps(runtime_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest_model = _json_mapping(model_metadata, name="model_metadata")
    manifest_model["format"] = str(model_format)
    manifest_model["checksum"] = file_sha256(packaged_model_path)

    manifest_runtime = {
        "preferred_backend": str(runtime_name),
        "supported_backends": [str(runtime_name)],
    }
    providers = runtime_payload.get("providers")
    if isinstance(providers, list):
        manifest_runtime["providers"] = [
            str(item) for item in providers if isinstance(item, str)
        ]

    manifest_metadata = _json_mapping(metadata, name="metadata")
    manifest_metadata.setdefault("producer", "xqt")
    manifest_metadata.setdefault("created_at", utc_timestamp())
    manifest_metadata.setdefault("source_model_path", str(source_model_path))

    manifest = ModelPackageManifest(
        package_version=package_version,
        entrypoints={
            "model": packaged_model_path.relative_to(package_dir).as_posix(),
            "runtime_config": runtime_config_path.relative_to(package_dir).as_posix(),
        },
        model=manifest_model,
        runtime=manifest_runtime,
        io=_json_mapping(io, name="io"),
        quantization=_json_mapping(quantization, name="quantization"),
        metadata=manifest_metadata,
    )
    manifest.write_json(package_dir / "manifest.json")
    return package_dir


def load_model_package(path: str | Path) -> LoadedModelPackage:
    """Load and validate one runtime model package."""

    candidate = Path(path)
    manifest_path = candidate / "manifest.json" if candidate.is_dir() else candidate
    if manifest_path.name != "manifest.json":
        raise XQTArtifactError(
            "load_model_package expects a package directory or manifest.json path"
        )
    manifest_payload = _load_json_mapping(manifest_path, name="model package manifest")
    manifest = ModelPackageManifest.from_dict(manifest_payload)
    package_dir = manifest_path.parent.resolve()
    model_path = _resolve_package_file(
        package_dir,
        manifest.entrypoints["model"],
        name="entrypoints.model",
    )
    runtime_config_path = _resolve_package_file(
        package_dir,
        manifest.entrypoints["runtime_config"],
        name="entrypoints.runtime_config",
    )
    runtime_config = _load_json_mapping(
        runtime_config_path,
        name="runtime config",
    )
    return LoadedModelPackage(
        package_dir=package_dir,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        model_path=model_path,
        runtime_config_path=runtime_config_path,
        runtime_config=runtime_config,
    )


def create_inference_runner(
    package: str | Path | LoadedModelPackage,
    *,
    backend: str | None = None,
    providers: Sequence[str] | None = None,
) -> ONNXRuntimeRunner:
    """Create one runtime runner from the standard package contract."""

    loaded = package if isinstance(package, LoadedModelPackage) else load_model_package(package)
    resolved_backend = str(backend or loaded.preferred_backend)
    if resolved_backend != "onnxruntime":
        raise XQTBackendError(
            "create_inference_runner currently supports backend=onnxruntime only"
        )
    return ONNXRuntimeRunner(loaded, providers=providers)


__all__ = [
    "LoadedModelPackage",
    "MODEL_PACKAGE_ARTIFACT_TYPE",
    "MODEL_PACKAGE_SCHEMA_VERSION",
    "ModelPackageManifest",
    "ONNXRuntimeRunner",
    "create_inference_runner",
    "load_model_package",
    "write_model_package",
]
