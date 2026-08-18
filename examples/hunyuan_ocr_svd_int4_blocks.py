"""Quantize tencent/HunyuanOCR to INT4 and compile its transformer blocks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.xqt_models.hunyuan_ocr import (
    HUNYUAN_OCR_REPO_ID,
    load_hunyuan_ocr,
    optimize_hunyuan_ocr_svd_int4_blocks,
)


CONFIG: dict[str, Any] = {
    "model_id": HUNYUAN_OCR_REPO_ID,
    "revision": None,
    "local_files_only": False,
    "dtype": torch.bfloat16,
    "artifact_dir": "artifacts/xqt/hunyuan_ocr_svd_int4_blocks",
    # Provide a Tensor, positional tuple/list, or keyword mapping accepted by model.forward.
    "example_inputs": None,
    "quantization": {
        "rank": 32,
        "group_size": 128,
        "engine": "auto",
        "fallback_engine": "torch_int_mm",
        "block_engine": "inductor",
        "block_dynamic": True,
        "policy": {
            "include_module_types": ["Linear"],
            "exclude_name_patterns": [],
        },
    },
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    artifact_dir = Path(str(CONFIG["artifact_dir"]))
    example_inputs = CONFIG["example_inputs"]
    if example_inputs is None:
        raise ValueError(
            "CONFIG['example_inputs'] must contain one real HunyuanOCR model input "
            "for block compilation warmup"
        )
    quantization = dict(CONFIG["quantization"])
    model = load_hunyuan_ocr(
        repo_id=str(CONFIG["model_id"]),
        revision=CONFIG["revision"],
        dtype=CONFIG["dtype"],
        device=_device(),
        local_files_only=bool(CONFIG["local_files_only"]),
    )
    result = optimize_hunyuan_ocr_svd_int4_blocks(
        model,
        artifact_dir=artifact_dir,
        rank=int(quantization["rank"]),
        group_size=int(quantization["group_size"]),
        engine=str(quantization["engine"]),
        fallback_engine=str(quantization["fallback_engine"]),
        block_engine=str(quantization["block_engine"]),
        block_dynamic=bool(quantization["block_dynamic"]),
        policy=quantization["policy"],
        example_inputs=example_inputs,
        calibration_inputs=[example_inputs],
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "hunyuan_ocr_svd_int4_blocks.json"
    report_path.write_text(
        json.dumps(
            {
                "model_id": CONFIG["model_id"],
                "device": str(_device()),
                "quant_stage": result.stage.metrics,
                "compute_config": result.compute_config,
                "block_optimization": result.block_optimization.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
