"""Run an XQT benchmark against the MiniCPM5-2B BF16 Hugging Face checkpoint.

This example keeps tokenizer and text generation in Transformers, while XQT
owns the model-side latency and CUDA memory measurement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xqt.kernels.wrappers.bench.latency import benchmark_callable
from xqt.kernels.wrappers.bench.memory import benchmark_memory

MODEL_ID = "openbmb/MiniCPM5-2B"
LOCAL_MODEL_DIR = Path("downloads/MiniCPM5-2B-bf16")
ARTIFACT_DIR = Path("artifacts/xqt/inference/minicpm5-2b")
PROMPT = "用一句话解释什么是 GQA, 并说明它为什么能降低 KV cache 的显存占用."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def load_model_and_tokenizer() -> tuple[Any, Any, str]:
    """Load the official checkpoint, preferring the local HF snapshot."""

    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=source != MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        local_files_only=source != MODEL_ID,
    ).eval().to(DEVICE)
    return model, tokenizer, source


def build_inputs(tokenizer: Any) -> dict[str, torch.Tensor]:
    """Apply the model's official ChatML template for one text request."""

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    return {
        key: value.to(DEVICE)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }


def generate(model: Any, inputs: dict[str, torch.Tensor], *, max_new_tokens: int) -> Any:
    """Run one deterministic-shape generation request."""

    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            eos_token_id=[1, 130073],
            use_cache=True,
        )


def main() -> None:
    """Load, generate, benchmark, and write a JSON report."""

    model, tokenizer, source = load_model_and_tokenizer()
    inputs = build_inputs(tokenizer)
    max_new_tokens = 128

    output = generate(model, inputs, max_new_tokens=max_new_tokens)
    prompt_length = int(inputs["input_ids"].shape[-1])
    text = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=False)

    latency = benchmark_callable(
        lambda: generate(model, inputs, max_new_tokens=max_new_tokens),
        warmup=2,
        iterations=5,
        sync_cuda=True,
        device=DEVICE,
    )
    memory = benchmark_memory(
        lambda: generate(model, inputs, max_new_tokens=max_new_tokens),
        iterations=1,
        sync_cuda=True,
        device=DEVICE,
    )
    report = {
        "model_id": MODEL_ID,
        "source": source,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "prompt_tokens": prompt_length,
        "max_new_tokens": max_new_tokens,
        "generated_text": text,
        "xqt_benchmark": latency.to_dict(),
        "xqt_memory": memory.to_dict(),
        "note": "XQT benchmark wrappers measured Transformers generate; no XQT quantization or custom kernel was applied.",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
