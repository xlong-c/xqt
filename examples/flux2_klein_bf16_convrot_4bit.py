"""Minimal FLUX.2 klein 4B BF16 -> ConvRot W4A4 quantization example."""

from __future__ import annotations

from typing import Any

import torch

from examples.xqt_models.flux2_klein import (
    load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_4bit,
)


def build_example_config() -> dict[str, Any]:
    return {
        "repo_id": "black-forest-labs/FLUX.2-klein-4b",
        "dtype": torch.bfloat16,
        "device": None,
        "local_files_only": False,
        "policy": {
            "dtype": "int4",
            "scheme": "convrot_w4a4",
            "group_size": 128,
            "rot_size": 256,
            "mixed_precision_ratio": 0.2,
        },
        "calibration_inputs": None,
        "materialize_mixed_precision": True,
    }


def main() -> None:
    config = build_example_config()
    _, result = load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_4bit(
        repo_id=config["repo_id"],
        dtype=config["dtype"],
        device=config["device"],
        local_files_only=config["local_files_only"],
        policy=config["policy"],
        calibration_inputs=config["calibration_inputs"],
        materialize_mixed_precision=config["materialize_mixed_precision"],
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
