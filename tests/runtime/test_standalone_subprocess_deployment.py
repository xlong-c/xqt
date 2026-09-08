"""Tests for standalone deployment loader and subprocess isolation (XQT-016)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from xqt.contracts.quant_pair import write_quant_pair
from xqt.contracts.quant_pair_schema import (
    DEFAULT_SIDECAR_NAME,
    WEIGHTS_FORMAT_SAFETENSORS,
    WEIGHTS_FORMAT_TORCH_STATE_DICT,
)
from xqt.core.base import XQTArtifactError, XQTBackendError
from xqt.runtime.deploy_loader import StandaloneDeployLoader


class _DeploymentToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


@pytest.fixture
def deployed_toy_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "toy_deployment_pair"
    model = _DeploymentToyModel().eval()
    with torch.no_grad():
        model.fc.weight.fill_(0.5)
        model.fc.bias.fill_(1.0)

    write_quant_pair(
        model,
        artifact_dir,
        weights_format=WEIGHTS_FORMAT_SAFETENSORS,
        metadata={
            "model_name": "toy_model",
            "model_profile": "custom",
            "model_family": "transformer",
        },
        compute_config={
            "schema_version": "1.0",
            "target_arch": "sm_80",
        },
    )
    return artifact_dir


def test_standalone_deployment_subprocess_forward(deployed_toy_artifact: Path) -> None:
    """Test loading and inference in a completely fresh, isolated Python subprocess."""
    script = f"""
import sys
import torch
from xqt.runtime.deploy_loader import StandaloneDeployLoader
from tests.xqt.runtime.test_standalone_subprocess_deployment import _DeploymentToyModel

loader = StandaloneDeployLoader()
shell = _DeploymentToyModel().eval()
instance = loader.load(r"{deployed_toy_artifact}", model_shell=shell, device="cpu")

inputs = torch.ones(1, 8)
out = instance(inputs)
assert out.shape == (1, 4)
# 8 * 0.5 + 1.0 = 5.0
assert torch.allclose(out, torch.full((1, 4), 5.0))
print("SUBPROCESS_SUCCESS")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "SUBPROCESS_SUCCESS" in result.stdout


def test_standalone_deployment_rejects_corrupted_checksum(deployed_toy_artifact: Path) -> None:
    """Tampering with weights file triggers fail-closed checksum verification."""
    weights_path = deployed_toy_artifact / "model.safetensors"
    data = bytearray(weights_path.read_bytes())
    # Corrupt a byte in the weight buffer
    data[-1] = (data[-1] + 1) % 256
    weights_path.write_bytes(data)

    loader = StandaloneDeployLoader()
    shell = _DeploymentToyModel().eval()
    with pytest.raises(XQTArtifactError, match="Weights checksum verification FAILED"):
        loader.load(deployed_toy_artifact, model_shell=shell, device="cpu")


def test_standalone_deployment_rejects_path_escape(deployed_toy_artifact: Path) -> None:
    """Path traversal in weights.path is strictly blocked."""
    sidecar_path = deployed_toy_artifact / DEFAULT_SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["weights"]["path"] = "../escaped.safetensors"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    loader = StandaloneDeployLoader()
    shell = _DeploymentToyModel().eval()
    with pytest.raises(XQTArtifactError, match="escapes root directory"):
        loader.load(deployed_toy_artifact, model_shell=shell, device="cpu")


def test_standalone_deployment_rejects_unsupported_schema(deployed_toy_artifact: Path) -> None:
    """Unknown or unsupported schema_version triggers fail-closed rejection."""
    sidecar_path = deployed_toy_artifact / DEFAULT_SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["schema_version"] = "99.0_unsupported"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    loader = StandaloneDeployLoader()
    shell = _DeploymentToyModel().eval()
    with pytest.raises(XQTArtifactError, match="Unsupported schema_version"):
        loader.load(deployed_toy_artifact, model_shell=shell, device="cpu")


def test_standalone_deployment_rejects_unmet_hardware_requirement(
    deployed_toy_artifact: Path,
) -> None:
    """Higher architecture requirement fails preflight check on lower hardware."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for hardware preflight test")

    sidecar_path = deployed_toy_artifact / DEFAULT_SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["compute_config"]["target_arch"] = "sm_99"  # Higher than any current GPU
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    loader = StandaloneDeployLoader()
    shell = _DeploymentToyModel().eval()
    with pytest.raises(XQTBackendError, match="Hardware preflight failed"):
        loader.load(deployed_toy_artifact, model_shell=shell, device="cuda")


def test_standalone_deployment_rejects_untrusted_pickle(tmp_path: Path) -> None:
    """Torch state_dict (.pt pickle) is rejected by default unless allow_untrusted_pickle=True."""
    artifact_dir = tmp_path / "pickle_pair"
    model = _DeploymentToyModel().eval()
    write_quant_pair(
        model,
        artifact_dir,
        weights_format=WEIGHTS_FORMAT_TORCH_STATE_DICT,
    )

    loader = StandaloneDeployLoader(allow_untrusted_pickle=False)
    shell = _DeploymentToyModel().eval()
    with pytest.raises(XQTArtifactError, match="Refusing to load pickle state_dict without allow_untrusted_pickle=True"):
        loader.load(artifact_dir, model_shell=shell, device="cpu")

    # Allowing untrusted pickle succeeds
    trusted_loader = StandaloneDeployLoader(allow_untrusted_pickle=True)
    instance = trusted_loader.load(artifact_dir, model_shell=shell, device="cpu")
    assert instance is not None
