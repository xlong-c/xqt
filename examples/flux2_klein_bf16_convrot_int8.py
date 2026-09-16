"""Run FLUX.2 Klein 4B BF16 transformer with XQT ConvRot W8A8."""

from __future__ import annotations

import torch

from examples.xqt_models.flux2_klein import (
    FLUX2_KLEIN_4B_REPO_ID,
    load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_int8,
)


REPO_ID = FLUX2_KLEIN_4B_REPO_ID
DEVICE = "cuda"
PROMPT = "A small red kite over a quiet lake"


def main() -> None:
    pipeline, result = (
        load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_int8(
            repo_id=REPO_ID,
            dtype=torch.bfloat16,
            device=DEVICE,
            engine="triton",
            min_int8_rows=17,
        )
    )
    print(
        {
            "quantized_modules": len(result.quantized_modules),
            "engine": "triton",
            "min_int8_rows": 17,
            "model_family": result.metadata["model_family"],
        }
    )
    with torch.inference_mode():
        pipeline(prompt=PROMPT)


if __name__ == "__main__":
    main()
