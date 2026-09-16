"""Benchmark mixed-precision MiniCPM5-2B candidates with XQT."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from xqt.model.minicpm5 import (
    load_minicpm5,
    minicpm5_edge_protected_quantization_policy,
    minicpm5_mlp_only_quantization_policy,
)
from examples.xqt_models.minicpm5_2b_quant_benchmark import (
    CALIBRATION_PROMPTS,
    DEVICE,
    MAX_NEW_TOKENS,
    _forward_logits,
    _inputs,
    _run_candidate,
    _source,
)

OUTPUT = Path("artifacts/xqt/inference/minicpm5-2b/mixed_precision.json")


def main() -> None:
    """Run mixed-precision candidates sequentially."""

    source = _source()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    base_model = load_minicpm5(
        source,
        dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
        device="cpu",
        local_files_only=True,
    )
    inputs_cpu = {key: value.cpu() for key, value in _inputs(tokenizer).items()}
    calibration_inputs_cpu = [
        {key: value.cpu() for key, value in _inputs(tokenizer, prompt).items()}
        for prompt in CALIBRATION_PROMPTS
    ]
    base_model = base_model.to(DEVICE)
    inputs = {key: value.to(DEVICE) for key, value in inputs_cpu.items()}
    reference_logits = _forward_logits(base_model, inputs)
    base_model = base_model.to("cpu")

    candidates = (
        ("awq_mlp_only_g64", "w4a16_int4", "awq", 64, minicpm5_mlp_only_quantization_policy()),
        ("fp4_mlp_only_g64", "w4a16_fp4", None, 64, minicpm5_mlp_only_quantization_policy()),
        ("awq_middle_g64_edges2", "w4a16_int4", "awq", 64, minicpm5_edge_protected_quantization_policy(edge_layers=2)),
        ("awq_middle_g128_edges2", "w4a16_int4", "awq", 128, minicpm5_edge_protected_quantization_policy(edge_layers=2)),
    )
    reports = []
    for name, strategy, method, group_size, policy in candidates:
        try:
            reports.append(
                _run_candidate(
                    name,
                    strategy,
                    method,
                    base_model,
                    tokenizer,
                    inputs_cpu,
                    calibration_inputs_cpu,
                    reference_logits,
                    policy,
                    group_size,
                    0,
                )
            )
        except Exception as exc:
            reports.append(
                {"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "model_id": "openbmb/MiniCPM5-2B",
        "source": source,
        "device": str(DEVICE),
        "max_new_tokens": MAX_NEW_TOKENS,
        "candidates": reports,
        "limitations": [
            "The logit comparison uses one held-out prompt; calibration uses five prompts.",
            "MLP-only and edge-protected candidates trade compression coverage for quality; they are not full-model 4-bit formats.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
