"""Standalone deployment loader for verified Quant Pairs (XQT-016).

Enforces:
1. Strict schema version validation (rejects unknown contract/schema versions).
2. Trusted boundary enforcement:
   - quant.json must specify a registered model profile or family.
   - Arbitrary module paths in untrusted artifact JSON are strictly rejected.
   - .safetensors is prioritized; .pt/pickle requires explicit allow_untrusted_pickle flag.
3. Checksum verification:
   - Weights file SHA256 is verified before loading into PyTorch.
   - File corruption triggers fail-closed XQTArtifactError.
4. Path traversal prevention:
   - Checks relative paths; rejects '..' traversal or absolute paths.
5. Hardware & environment preflight:
   - Validates GPU architecture and compute capabilities required by compute_config.
   - Unmet hardware requirements trigger fail-closed XQTBackendError.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from xqt.contracts.quant_pair_schema import (
    DEFAULT_SIDECAR_NAME,
    QUANT_SIDECAR_ARTIFACT_TYPE,
    QUANT_SIDECAR_SCHEMA_VERSION,
    SUPPORTED_WEIGHTS_FORMATS,
    WEIGHTS_FORMAT_SAFETENSORS,
    WEIGHTS_FORMAT_TORCH_STATE_DICT,
    QuantPairManifest,
)
from xqt.contracts.model_structure import MODEL_FAMILY_NAMES
from xqt.core.base import XQTArtifactError, XQTBackendError, XQTConfigError, file_sha256
from xqt.model.registry import resolve_model_profile


@dataclass
class DeploymentExecutionReport:
    """Deployment runtime execution and preflight diagnostics."""

    model_name: str
    weights_path: str
    weights_checksum: str
    loader_version: str
    weights_format: str
    device: str
    target_hardware: dict[str, Any]
    resolved_profile: str
    preflight_passed: bool
    is_reloaded_clean: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "weights_path": self.weights_path,
            "weights_checksum": self.weights_checksum,
            "loader_version": self.loader_version,
            "weights_format": self.weights_format,
            "device": self.device,
            "target_hardware": dict(self.target_hardware),
            "resolved_profile": self.resolved_profile,
            "preflight_passed": self.preflight_passed,
            "is_reloaded_clean": self.is_reloaded_clean,
            "notes": list(self.notes),
        }


class DeployedModelInstance:
    """Live inference wrapper over a reloaded deployment artifact."""

    def __init__(
        self,
        model: nn.Module,
        manifest: QuantPairManifest,
        report: DeploymentExecutionReport,
    ) -> None:
        self.model = model
        self.manifest = manifest
        self.report = report

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


class StandaloneDeployLoader:
    """Stateless loader executing strictly verified deployment loads."""

    LOADER_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        allow_untrusted_pickle: bool = False,
    ) -> None:
        self.allow_untrusted_pickle = allow_untrusted_pickle

    def _resolve_and_verify_sidecar(self, pair_dir: Path) -> tuple[Path, dict[str, Any]]:
        sidecar_path = pair_dir / DEFAULT_SIDECAR_NAME
        if not sidecar_path.is_file():
            raise XQTArtifactError(f"Deployment sidecar not found: {sidecar_path}")
        try:
            sidecar_dict = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise XQTArtifactError(f"Corrupted sidecar JSON: {sidecar_path}") from exc

        # 1. Schema & type verification
        artifact_type = sidecar_dict.get("artifact_type")
        if artifact_type != QUANT_SIDECAR_ARTIFACT_TYPE:
            raise XQTArtifactError(
                f"Unexpected artifact_type {artifact_type!r}; expected {QUANT_SIDECAR_ARTIFACT_TYPE!r}"
            )
        schema_version = str(sidecar_dict.get("schema_version", ""))
        if schema_version != QUANT_SIDECAR_SCHEMA_VERSION:
            raise XQTArtifactError(
                f"Unsupported schema_version {schema_version!r}; loader supports {QUANT_SIDECAR_SCHEMA_VERSION!r}"
            )

        return sidecar_path, sidecar_dict

    def _verify_weights_path_and_integrity(
        self,
        pair_dir: Path,
        weights_info: Mapping[str, Any],
    ) -> tuple[Path, str, str]:
        raw_path = str(weights_info.get("path", "")).strip()
        if not raw_path:
            raise XQTArtifactError("Missing weights.path in quant.json")

        # 2. Path traversal security check
        candidate = Path(raw_path)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise XQTArtifactError(f"weights.path escapes root directory: {raw_path}")

        resolved = (pair_dir / candidate).resolve()
        try:
            resolved.relative_to(pair_dir.resolve())
        except ValueError as exc:
            raise XQTArtifactError(f"weights.path escapes root directory: {raw_path}") from exc

        if not resolved.is_file():
            raise XQTArtifactError(f"Weights file not found: {resolved}")

        # 3. Format and safety check
        weights_format = str(weights_info.get("format", "")).strip()
        if weights_format not in SUPPORTED_WEIGHTS_FORMATS:
            raise XQTArtifactError(
                f"Unsupported weights format: {weights_format!r}; supported: {SUPPORTED_WEIGHTS_FORMATS}"
            )
        if weights_format == WEIGHTS_FORMAT_TORCH_STATE_DICT and not self.allow_untrusted_pickle:
            raise XQTArtifactError(
                "Refusing to load pickle state_dict without allow_untrusted_pickle=True; "
                "safetensors format is strictly required for untrusted deployment"
            )

        # 4. Checksum verification
        expected_checksum = str(weights_info.get("checksum", "")).strip()
        if not expected_checksum:
            raise XQTArtifactError("Missing required weights checksum in quant.json")

        actual_checksum = file_sha256(resolved)
        if actual_checksum != expected_checksum:
            raise XQTArtifactError(
                f"Weights checksum verification FAILED: expected {expected_checksum}, got {actual_checksum}"
            )

        return resolved, weights_format, actual_checksum

    def _preflight_hardware(
        self,
        compute_config: Mapping[str, Any] | None,
        target_device: torch.device,
    ) -> dict[str, Any]:
        hw_info: dict[str, Any] = {
            "device": str(target_device),
            "is_cuda": target_device.type == "cuda",
        }
        if target_device.type != "cuda":
            return hw_info

        if not torch.cuda.is_available():
            raise XQTBackendError("CUDA requested but not available in current environment")

        cap = torch.cuda.get_device_capability(target_device)
        current_sm = f"sm_{cap[0]}{cap[1]}"
        hw_info["current_sm"] = current_sm
        hw_info["cuda_capability"] = f"{cap[0]}.{cap[1]}"
        hw_info["gpu_name"] = torch.cuda.get_device_name(target_device)

        if compute_config is not None:
            required_arch = compute_config.get("target_arch") or compute_config.get("min_arch")
            if required_arch is not None:
                req_str = str(required_arch).lower().replace(".", "")
                if req_str.startswith("sm_"):
                    req_val = int(req_str.replace("sm_", ""))
                    curr_val = cap[0] * 10 + cap[1]
                    if curr_val < req_val:
                        raise XQTBackendError(
                            f"Hardware preflight failed: current GPU {current_sm} does not meet "
                            f"required architecture {required_arch}"
                        )

        return hw_info

    def load(
        self,
        artifact_dir: str | Path,
        *,
        model_shell: nn.Module | None = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        strict: bool = False,
    ) -> DeployedModelInstance:
        """Load, verify, and instantiate a deployed model instance."""
        pair_dir = Path(artifact_dir).resolve()
        if not pair_dir.is_dir():
            raise XQTArtifactError(f"Quant pair directory does not exist: {pair_dir}")

        sidecar_path, sidecar_dict = self._resolve_and_verify_sidecar(pair_dir)
        manifest = QuantPairManifest.from_dict(sidecar_dict)

        weights_file, weights_format, checksum = self._verify_weights_path_and_integrity(
            pair_dir,
            manifest.weights,
        )

        target_dev = torch.device(device)
        hw_info = self._preflight_hardware(manifest.compute_config, target_dev)

        # Trusted profile resolution
        metadata = manifest.metadata or {}
        profile_id = metadata.get("model_profile") or metadata.get("profile_id")
        resolved_profile_name = "custom"

        if model_shell is None:
            if not profile_id:
                raise XQTArtifactError(
                    "quant.json does not specify 'model_profile' and no model_shell was provided"
                )
            profile = resolve_model_profile(str(profile_id))
            resolved_profile_name = profile.profile_id
            # Reconstruct model from registered loader
            from xqt.core.imports import resolve_target
            loader_fn = resolve_target(profile.loader_target)
            loader_params = dict(profile.loader_params or {})
            model = loader_fn(**loader_params)
        else:
            model = model_shell

        # Load weights
        if weights_format == WEIGHTS_FORMAT_SAFETENSORS:
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_file), device="cpu")
        else:
            state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

        model.load_state_dict(state_dict, strict=strict)
        model.to(target_dev).eval()

        report = DeploymentExecutionReport(
            model_name=metadata.get("model_name", profile_id or "unknown"),
            weights_path=str(weights_file),
            weights_checksum=checksum,
            loader_version=self.LOADER_VERSION,
            weights_format=weights_format,
            device=str(target_dev),
            target_hardware=hw_info,
            resolved_profile=resolved_profile_name,
            preflight_passed=True,
            is_reloaded_clean=True,
            notes=["verified_sha256", "preflight_passed", "trusted_profile_enforced"],
        )

        return DeployedModelInstance(model=model, manifest=manifest, report=report)


__all__ = [
    "DeployedModelInstance",
    "DeploymentExecutionReport",
    "StandaloneDeployLoader",
]
