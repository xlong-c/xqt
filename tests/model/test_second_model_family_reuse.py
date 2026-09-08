"""Automated unit and regression tests for second model family reuse (XQT-017)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from xqt.contracts.model_structure import (
    ModelStructureContract,
    resolve_and_validate_structure_contract,
    structure_contract_mismatches,
)
from xqt.contracts.quant_pair import write_quant_pair
from xqt.contracts.quant_pair_schema import (
    DEFAULT_SIDECAR_NAME,
    WEIGHTS_FORMAT_SAFETENSORS,
)
from xqt.core.base import XQTArtifactError, XQTBackendError
from xqt.model import (
    model_profile_names,
    resolve_model_adapter,
    resolve_model_profile,
)
from xqt.model.flux2_klein.adapter import (
    Flux2KleinBF16Adapter,
    Flux2KleinNVFP4Adapter,
)
from xqt.runtime.deploy_loader import StandaloneDeployLoader

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_second_model_profile_and_adapter_registration() -> None:
    names = model_profile_names()
    assert "diffusers.flux2-klein" in names
    assert "diffusers.flux2-klein-nvfp4" in names

    profile_first = resolve_model_profile("diffusers.flux2-klein")
    assert profile_first.family == "diffusion"
    adapter_first = resolve_model_adapter(profile_first)
    assert isinstance(adapter_first, Flux2KleinBF16Adapter)

    profile_second = resolve_model_profile("diffusers.flux2-klein-nvfp4")
    assert profile_second.family == "diffusion"
    assert profile_second.metadata.get("format") == "nvfp4"
    adapter_second = resolve_model_adapter(profile_second)
    assert isinstance(adapter_second, Flux2KleinNVFP4Adapter)


def test_zero_core_pollution_static_audit() -> None:
    core_dirs = [
        _REPO_ROOT / "xqt/core",
        _REPO_ROOT / "xqt/session",
        _REPO_ROOT / "xqt/contracts",
        _REPO_ROOT / "xqt/transforms",
        _REPO_ROOT / "xqt/runtime/deploy_loader.py",
        _REPO_ROOT / "xqt/compression/quant/transforms",
    ]
    forbidden_terms = ["flux", "flux2", "flux_2", "klein"]
    violations: list[str] = []

    for target in core_dirs:
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = list(target.rglob("*.py"))
        else:
            continue

        for py_file in files:
            lines = py_file.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines, start=1):
                clean = line.strip().lower()
                if clean.startswith("#") or clean.startswith('"""') or clean.startswith("'''"):
                    continue
                for term in forbidden_terms:
                    if f'"{term}"' in clean or f"'{term}'" in clean:
                        violations.append(
                            f"{py_file.relative_to(_REPO_ROOT)}:L{idx} -> {line.strip()}"
                        )

    assert not violations, f"Found hardcoded model names in generic core: {violations}"


def test_second_model_structure_contract_resolution() -> None:
    class DummyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(16, 16)
            self.norm = nn.LayerNorm(16)

    class DummyKleinTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList([DummyBlock() for _ in range(2)])
            self.proj_out = nn.Linear(16, 16)

    model = DummyKleinTransformer()
    adapter = Flux2KleinNVFP4Adapter()
    profile = resolve_model_profile("diffusers.flux2-klein-nvfp4")

    contract = resolve_and_validate_structure_contract(
        model,
        profile=profile,
        adapter=adapter,
        strict=True,
    )
    assert contract is not None
    assert contract.family == "diffusion"
    assert contract.topology_fingerprint is not None

    mismatches = structure_contract_mismatches(model, contract)
    assert mismatches.is_consistent


def test_quant_pair_export_and_negative_security_guards(tmp_path: Path) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = TinyModel().eval()
    pair_dir = write_quant_pair(
        model,
        tmp_path / "tiny_quant_pair",
        weights_format=WEIGHTS_FORMAT_SAFETENSORS,
        metadata={
            "model_name": "tiny-model",
            "model_profile": "diffusers.flux2-klein-nvfp4",
        },
        compute_config={
            "schema_version": "1.0",
            "target_arch": "sm_89",
        },
    )

    sidecar_path = pair_dir / DEFAULT_SIDECAR_NAME
    assert sidecar_path.is_file()
    original_sidecar = sidecar_path.read_text(encoding="utf-8")

    loader = StandaloneDeployLoader()

    # 1. Corrupted checksum guard
    sidecar_dict = json.loads(original_sidecar)
    sidecar_dict["weights"]["checksum"] = "bad" * 16
    sidecar_path.write_text(json.dumps(sidecar_dict), encoding="utf-8")
    with pytest.raises(XQTArtifactError, match="Weights checksum verification FAILED"):
        loader.load(pair_dir, model_shell=TinyModel().eval(), device="cpu")

    # 2. Path traversal guard
    sidecar_dict = json.loads(original_sidecar)
    sidecar_dict["weights"]["path"] = "../../escape.safetensors"
    sidecar_path.write_text(json.dumps(sidecar_dict), encoding="utf-8")
    with pytest.raises(XQTArtifactError, match="escapes root directory"):
        loader.load(pair_dir, model_shell=TinyModel().eval(), device="cpu")

    # 3. Unmet hardware requirement guard
    if torch.cuda.is_available():
        sidecar_dict = json.loads(original_sidecar)
        sidecar_dict["compute_config"]["target_arch"] = "sm_99"
        sidecar_path.write_text(json.dumps(sidecar_dict), encoding="utf-8")
        with pytest.raises(XQTBackendError, match="Hardware preflight failed"):
            loader.load(pair_dir, model_shell=TinyModel().eval(), device="cuda:0")
