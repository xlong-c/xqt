"""ModelAdapter implementations for the FLUX.2 klein family."""

from __future__ import annotations

from typing import Any
import torch
from torch import nn

from xqt.contracts.model_structure import ModelStructureContract
from xqt.kernels.nn.fixtures.families import build_structure_contract
from xqt.model.adapter import ModelAdapter

from .types import FLUX2_KLEIN_4B_REPO_ID
from .load import (
    load_flux2_klein_bf16_transformer,
    load_flux2_klein_nvfp4_transformer,
)


class Flux2KleinBF16Adapter(ModelAdapter):
    """Adapter for the official FLUX.2 klein BF16 base transformer."""

    def load(self, checkpoint: str | None = None, **params: Any) -> nn.Module:
        """Load the BF16 transformer instance."""
        repo_id = params.get("repo_id") or checkpoint
        subfolder = params.get("subfolder", "transformer")
        device = params.get("device")
        torch_dtype = params.get("torch_dtype", torch.bfloat16)
        local_files_only = params.get("local_files_only", False)
        return load_flux2_klein_bf16_transformer(
            pretrained_model_name_or_path=repo_id,
            subfolder=subfolder,
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    def adapt(self, model: nn.Module, **params: Any) -> nn.Module:
        """Adapt transformer for evaluation/optimization."""
        model.eval()
        device = params.get("device")
        if device is not None:
            model.to(device=device)
        return model

    def structure_contract(self, model: nn.Module) -> ModelStructureContract:
        """Derive and return the verified model structure contract."""
        return build_structure_contract(model, family="diffusion")

    def inference_contract(self, model: nn.Module) -> dict[str, Any]:
        """Return the standard input argument protocol."""
        del model
        return {
            "required_inputs": [
                "hidden_states",
                "encoder_hidden_states",
                "timestep",
                "img_ids",
                "txt_ids",
            ],
            "optional_inputs": ["guidance", "joint_attention_kwargs"],
            "output_format": "tuple",
        }


class Flux2KleinNVFP4Adapter(ModelAdapter):
    """Adapter for the official FLUX.2 klein NVFP4 4-bit transformer."""

    def load(self, checkpoint: str | None = None, **params: Any) -> nn.Module:
        """Load the NVFP4 transformer instance."""
        model_file = params.get("model_file") or checkpoint
        config = params.get("config") or FLUX2_KLEIN_4B_REPO_ID
        config_subfolder = params.get("config_subfolder", "transformer")
        device = params.get("device")
        dtype = params.get("dtype", torch.float16)
        local_files_only = params.get("local_files_only", False)
        return load_flux2_klein_nvfp4_transformer(
            model_file=model_file,
            config=config,
            config_subfolder=config_subfolder,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    def adapt(self, model: nn.Module, **params: Any) -> nn.Module:
        """Adapt NVFP4 transformer for evaluation/optimization."""
        model.eval()
        device = params.get("device")
        if device is not None:
            model.to(device=device)
        return model

    def structure_contract(self, model: nn.Module) -> ModelStructureContract:
        """Derive and return the verified model structure contract."""
        return build_structure_contract(model, family="diffusion")

    def inference_contract(self, model: nn.Module) -> dict[str, Any]:
        """Return the standard input argument protocol."""
        del model
        return {
            "required_inputs": [
                "hidden_states",
                "encoder_hidden_states",
                "timestep",
                "img_ids",
                "txt_ids",
            ],
            "optional_inputs": ["guidance", "joint_attention_kwargs"],
            "output_format": "tuple",
        }


__all__ = [
    "Flux2KleinBF16Adapter",
    "Flux2KleinNVFP4Adapter",
]
