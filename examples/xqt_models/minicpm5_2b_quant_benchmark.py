"""Compare MiniCPM5-2B BF16 and XQT quantization candidates.

This is an offline model-side experiment. Transformers owns tokenization and
text generation; XQT owns quantization and benchmark measurement.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from xqt.compression.quant import QuantizationPolicy
from xqt.kernels.wrappers.bench.latency import benchmark_callable
from xqt.kernels.wrappers.bench.memory import benchmark_memory
from xqt.model.minicpm5 import (
    MINICPM5_2B_REPO_ID,
    load_minicpm5,
    minicpm5_linear_summary,
    minicpm5_quantization_policy,
    quantize_minicpm5,
)

MODEL_DIR = Path("downloads/MiniCPM5-2B-bf16")
ARTIFACT_DIR = Path("artifacts/xqt/inference/minicpm5-2b")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT = "用一句话解释什么是 GQA, 并说明它为什么能降低 KV cache 的显存占用."
MAX_NEW_TOKENS = 64

CANDIDATES = (
    ("bf16", None, None, None, 0),
    ("awq_w4_group64", "w4a16_int4", "awq", 64, 0),
    ("awq_w4_group128", "w4a16_int4", "awq", 128, 0),
    ("gptq_w4_group64", "w4a16_gptq", "gptq", 64, 0),
    ("gptq_w4_group128", "w4a16_gptq", "gptq", 128, 0),
    ("fp4_w4_group64", "w4a16_fp4", None, 64, 0),
    ("fp4_w4_group128", "w4a16_fp4", None, 128, 0),
    ("int8_w8a8_prefill", "w8a8_int8", None, None, 0),
    ("int8_w8a8_small_fallback", "w8a8_int8", None, None, 32),
)

CALIBRATION_PROMPTS = (
    PROMPT,
    "写一个 Python 函数, 返回斐波那契数列的前 20 项, 并解释时间复杂度.",
    "如果一个三角形的底为 8, 高为 5, 面积是多少? 请给出计算过程.",
    "请比较数据库索引和哈希表的适用场景.",
    "设计一个可靠的 HTTP 重试策略, 需要考虑哪些因素?",
)


def _source() -> str:
    return str(MODEL_DIR) if MODEL_DIR.exists() else MINICPM5_2B_REPO_ID


def _inputs(tokenizer: Any, prompt: str = PROMPT) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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


def _generate(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=[1, 130073],
            use_cache=True,
        )


def _forward_logits(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        return model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()


def _parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _buffer_bytes(model: torch.nn.Module) -> int:
    return sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())


def _run_candidate(
    name: str,
    strategy: str | None,
    method: str | None,
    base_model: torch.nn.Module,
    tokenizer: Any,
    inputs_cpu: dict[str, torch.Tensor],
    calibration_inputs_cpu: list[dict[str, torch.Tensor]],
    reference_logits: torch.Tensor,
    policy: QuantizationPolicy,
    group_size: int | None,
    min_int8_rows: int,
) -> dict[str, Any]:
    """Quantize one candidate on CPU, then benchmark it on the target device."""

    if strategy is None:
        model = base_model
        quantization_metadata: dict[str, Any] = {"strategy": "bf16"}
    else:
        calibration = calibration_inputs_cpu
        result = quantize_minicpm5(
            base_model,
            strategy=strategy,
            policy=policy,
            calibration_inputs=calibration,
            inplace=False,
            engine="auto",
            group_size=group_size,
            min_int8_rows=min_int8_rows,
        )
        model = result.model
        quantization_metadata = {
            "strategy": strategy,
            "method": method,
            "quantized_module_count": len(result.quantized_modules),
            "metadata": dict(result.metadata),
        }

    model = model.eval().to(DEVICE)
    inputs = {key: value.to(DEVICE) for key, value in inputs_cpu.items()}
    logits = _forward_logits(model, inputs)
    diff = logits - reference_logits
    generated = _generate(model, inputs)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    text = tokenizer.decode(generated[0, prompt_tokens:], skip_special_tokens=False)
    latency = benchmark_callable(
        lambda: _generate(model, inputs),
        warmup=1,
        iterations=3,
        sync_cuda=True,
        device=str(DEVICE),
    )
    memory = benchmark_memory(
        lambda: _generate(model, inputs),
        iterations=1,
        sync_cuda=True,
        device=str(DEVICE),
    )
    execution_counts: dict[str, int] = {}
    for module in model.modules():
        metadata_fn = getattr(module, "execution_metadata", None)
        if not callable(metadata_fn):
            continue
        metadata = metadata_fn()
        engine_name = str(metadata.get("engine", "unknown"))
        execution_counts[engine_name] = execution_counts.get(engine_name, 0) + 1
    report = {
        "name": name,
        "quantization": quantization_metadata,
        "parameter_bytes": _parameter_bytes(model),
        "buffer_bytes": _buffer_bytes(model),
        "logit_diff": {
            "max_abs": float(diff.abs().max().item()),
            "mean_abs": float(diff.abs().mean().item()),
            "rmse": float(diff.square().mean().sqrt().item()),
        },
        "generated_text": text,
        "latency": latency.to_dict(),
        "memory": memory.to_dict(),
        "runtime_execution": execution_counts,
    }
    del model, inputs, logits, generated
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return report


def main() -> None:
    """Run all candidates sequentially and write the comparison report."""

    source = _source()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=source != MINICPM5_2B_REPO_ID)
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
    policy = minicpm5_quantization_policy()
    base_model = base_model.eval().to(DEVICE)
    inputs = {key: value.to(DEVICE) for key, value in inputs_cpu.items()}
    reference_logits = _forward_logits(base_model, inputs)
    baseline = _run_candidate(
        "bf16",
        None,
        None,
        base_model,
        tokenizer,
        inputs_cpu,
        calibration_inputs_cpu,
        reference_logits,
        policy,
        None,
        0,
    )
    base_model = base_model.to("cpu")
    reports = [baseline]
    for name, strategy, method, group_size, min_int8_rows in CANDIDATES[1:]:
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
                    min_int8_rows,
                )
            )
        except Exception as exc:
            reports.append(
                {
                    "name": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "model_id": MINICPM5_2B_REPO_ID,
        "source": source,
        "device": str(DEVICE),
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "linear_summary": minicpm5_linear_summary(base_model),
        "candidates": reports,
        "limitations": [
            "One prompt is used for smoke calibration and logit comparison, not task-level evaluation.",
            "Latency measures Transformers generate through XQT benchmark wrappers.",
            "Quantized candidates are XQT runtime modules and are not automatically equivalent to GGUF/GPTQ production formats.",
        ],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = ARTIFACT_DIR / "quantization_comparison.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
