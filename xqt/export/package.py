"""Industrial model package exporter for XQT (.xqtpkg).

Creates a standardized deployment artifact package containing:
- manifest.json: Package metadata, file integrity checksums (SHA256), and schema versions.
- config.json: Model architecture, tensor dimensions, and quantization layout configurations.
- compute.json: Hardware execution parameters, kernel schedules, and tilelang/runtime metadata.
- weights/model.safetensors: Compact Safetensors weight tensors.
- (Optional) archive into a single compressed .xqtpkg archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import nn

if TYPE_CHECKING:
    from xqt.kernels.timing.cache import UnifiedKernelTimingCache

from xqt.contracts.quantized import QuantizedModel
from xqt.contracts.runtime_manifest import RuntimeManifest
from xqt.core.errors import XQTArtifactError
from xqt.export.base import ExportResultBase
from xqt.export.hf_quant import _collect_packed_state, _quantization_config_payload


def _calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ModelPackageReport(ExportResultBase):
    """Execution report and diagnostic summary for an exported XQT model package."""

    package_path: str
    manifest_path: str
    weights_path: str
    config_path: str
    compute_path: str
    archive_path: str | None = None
    package_name: str = "xqt_model"
    version: str = "1.0.0"
    backend: str = "tilelang"
    module_count: int = 0
    tensor_count: int = 0
    total_size_bytes: int = 0
    files: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        paths = [Path(self.package_path)]
        if self.archive_path:
            paths.append(Path(self.archive_path))
        return tuple(paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_path": self.package_path,
            "manifest_path": self.manifest_path,
            "weights_path": self.weights_path,
            "config_path": self.config_path,
            "compute_path": self.compute_path,
            "archive_path": self.archive_path,
            "package_name": self.package_name,
            "version": self.version,
            "backend": self.backend,
            "module_count": int(self.module_count),
            "tensor_count": int(self.tensor_count),
            "total_size_bytes": int(self.total_size_bytes),
            "files": list(self.files),
            "metadata": dict(self.metadata),
        }


def export_model_package(
    model_or_quant: QuantizedModel | nn.Module,
    output_path: str | Path,
    *,
    package_name: str = "xqt_model",
    version: str = "1.0.0",
    backend: str = "tilelang",
    runtime_config: Mapping[str, Any] | None = None,
    compute_config: Mapping[str, Any] | None = None,
    runtime_manifest: RuntimeManifest | Mapping[str, Any] | None = None,
    timing_cache: UnifiedKernelTimingCache | None = None,
    optimal_kernel_routes: Mapping[str, Any] | None = None,
    archive: bool = False,
) -> ModelPackageReport:
    """Export a quantized model or standard PyTorch module into an industrial XQT package."""
    target_path = Path(output_path)
    is_archive_dest = target_path.name.endswith(".xqtpkg") or target_path.name.endswith(".tar.gz")

    if is_archive_dest:
        pkg_dir = target_path.parent / target_path.name.replace(".tar.gz", "").replace(".xqtpkg", "_dir")
        archive_dest: Path | None = target_path
    else:
        pkg_dir = target_path
        archive_dest = (target_path.parent / f"{target_path.name}.xqtpkg") if archive else None

    pkg_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = pkg_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Extract model and quantization specs
    if isinstance(model_or_quant, QuantizedModel):
        root_module = model_or_quant.model
        method = model_or_quant.method
        strategy = model_or_quant.strategy
        model_backend = backend if backend is not None else (model_or_quant.backend or "tilelang")
    elif isinstance(model_or_quant, nn.Module):
        root_module = model_or_quant
        method = None
        strategy = None
        model_backend = backend
    else:
        raise TypeError(
            f"export_model_package expects QuantizedModel or nn.Module; got {type(model_or_quant).__name__}"
        )

    # 1. Collect packed and unquantized weights and buffers
    state, packed_modules, kv_scales = _collect_packed_state(root_module)

    # Save Safetensors
    weights_file = weights_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(
            {k: (v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v) for k, v in state.items()},
            str(weights_file),
        )
        weights_format = "safetensors"
    except ImportError:
        weights_file = weights_dir / "model.pt"
        torch.save(state, weights_file)
        weights_format = "torch_state_dict"

    # 2. Build config.json
    bits = 4
    group_size = 128
    for mod in root_module.modules():
        if hasattr(mod, "bits") and hasattr(mod, "group_size"):
            bits = int(getattr(mod, "bits"))
            group_size = int(getattr(mod, "group_size"))
            break

    quant_payload = _quantization_config_payload(
        format_name="compressed-tensors",
        bits=bits,
        group_size=group_size,
        method=method,
        strategy=strategy,
        kv_cache_scheme={"type": "fp8", "num_bits": 8, "strategy": "tensor"} if kv_scales else None,
        extra=dict(runtime_config or {}),
    )
    config_dict = {
        "model_type": getattr(root_module, "__class__", type(root_module)).__name__,
        "architectures": [getattr(root_module, "__class__", type(root_module)).__name__],
        "quantization_config": quant_payload,
        "runtime_config": dict(runtime_config or {}),
    }
    config_file = pkg_dir / "config.json"
    config_file.write_text(json.dumps(config_dict, indent=2, sort_keys=True), encoding="utf-8")

    # 3. Build compute.json
    compute_dict: dict[str, Any] = {
        "producer": "xqt.export.package",
        "backend": model_backend,
        "compute_config": dict(compute_config or {}),
        "packed_modules": packed_modules,
        "kv_scales": kv_scales,
    }
    if optimal_kernel_routes is not None:
        compute_dict["optimal_kernel_routes"] = dict(optimal_kernel_routes)
    elif timing_cache is not None:
        # Extract optimal routes from timing cache
        routes = {}
        for k_str, rec in getattr(timing_cache, "_entries", {}).items():
            if getattr(rec, "verified_correct", True):
                routes[k_str] = {
                    "median_latency_us": rec.median_latency_us,
                    "preset_name": rec.preset_name,
                }
        if routes:
            compute_dict["optimal_kernel_routes"] = routes

    if runtime_manifest is not None:
        if isinstance(runtime_manifest, RuntimeManifest):
            compute_dict["runtime_manifest"] = runtime_manifest.to_dict()
        elif isinstance(runtime_manifest, Mapping):
            compute_dict["runtime_manifest"] = dict(runtime_manifest)
    compute_file = pkg_dir / "compute.json"
    compute_file.write_text(json.dumps(compute_dict, indent=2, sort_keys=True), encoding="utf-8")

    # 4. Build manifest.json with checksums
    file_rel_paths = [
        "config.json",
        "compute.json",
        f"weights/{weights_file.name}",
    ]
    checksums = {rel: _calculate_sha256(pkg_dir / rel) for rel in file_rel_paths}
    total_size_bytes = sum((pkg_dir / rel).stat().st_size for rel in file_rel_paths)

    manifest_dict: dict[str, Any] = {
        "schema_version": 1,
        "package_name": str(package_name),
        "version": str(version),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": model_backend,
        "weights_format": weights_format,
        "files": file_rel_paths,
        "checksums": checksums,
        "total_size_bytes": total_size_bytes,
        "tensor_count": len(state),
        "module_count": len(packed_modules),
    }
    manifest_file = pkg_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_dict, indent=2, sort_keys=True), encoding="utf-8")

    # 5. Build archive if requested
    archive_path_str: str | None = None
    if archive_dest is not None:
        archive_dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_dest, "w:gz") as tar:
            for rel in ["manifest.json", "config.json", "compute.json", f"weights/{weights_file.name}"]:
                tar.add(pkg_dir / rel, arcname=rel)
        archive_path_str = str(archive_dest)

    all_files = tuple(["manifest.json", "config.json", "compute.json", f"weights/{weights_file.name}"])
    return ModelPackageReport(
        package_path=str(pkg_dir),
        manifest_path=str(manifest_file),
        weights_path=str(weights_file),
        config_path=str(config_file),
        compute_path=str(compute_file),
        archive_path=archive_path_str,
        package_name=package_name,
        version=version,
        backend=model_backend,
        module_count=len(packed_modules),
        tensor_count=len(state),
        total_size_bytes=total_size_bytes,
        files=all_files,
        metadata={
            "weights_format": weights_format,
            "bits": bits,
            "group_size": group_size,
            "kv_scales": kv_scales,
        },
    )


def load_model_package_manifest(package_path: str | Path) -> dict[str, Any]:
    """Inspect and load manifest metadata from an .xqtpkg directory or archive."""
    pkg_path = Path(package_path)
    if not pkg_path.exists():
        raise XQTArtifactError(f"Package path does not exist: {pkg_path}")

    if pkg_path.is_file():
        # Read from tar archive directly
        try:
            with tarfile.open(pkg_path, "r:*") as tar:
                manifest_member = tar.extractfile("manifest.json")
                if manifest_member is None:
                    raise XQTArtifactError("Archive does not contain manifest.json")
                content = manifest_member.read().decode("utf-8")
                return json.loads(content)
        except Exception as exc:
            raise XQTArtifactError(f"Failed to read manifest from archive {pkg_path}: {exc}") from exc

    manifest_file = pkg_path / "manifest.json"
    if not manifest_file.is_file():
        raise XQTArtifactError(f"Package directory does not contain manifest.json: {pkg_path}")

    try:
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise XQTArtifactError(f"Failed to decode manifest.json from {manifest_file}: {exc}") from exc


__all__ = [
    "ModelPackageReport",
    "export_model_package",
    "load_model_package_manifest",
]
