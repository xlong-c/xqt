"""Benchmark zero-calibration (RTN) weight-only quantization for MiniCPM5-2B.

XQT 的 AWQ/GPTQ 入口在 ``calibration_inputs=None`` 时拿不到激活统计,
group multiplier 与 Hessian 修正全部退化为 per-group 对称 RTN. 本脚本量化
"零校准下界" 并回答三个问题:

1. 纯 RTN (无旋转) 的 w4/w8 精度与延迟在哪里.
2. QuaRot 离线旋转对零校准路径有多大帮助 (QuaRot 论文核心声明:
   旋转后 RTN 即可接近校准方法).
3. 5-prompt AWQ 校准 (见 quarot_comparison.json) 相对零校准值多少.

对照数据: awq_w4_g64 (5-prompt 校准) logit RMSE 1.32, latency 1061 ms,
bf16 基线 latency 1046 ms. fp4_w4 无校准路径 RMSE 1.58 (g64).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from xqt.compression.quant import quantize_with_awq_weight_only
from xqt.kernels.wrappers.bench.latency import benchmark_callable
from xqt.model.minicpm5 import load_minicpm5
from xqt.model.minicpm5_quarot import apply_quarot_minicpm5
from examples.xqt_models.minicpm5_2b_quant_benchmark import (
    DEVICE,
    MAX_NEW_TOKENS,
    _forward_logits,
    _inputs,
    _source,
)

OUTPUT = Path("artifacts/xqt/inference/minicpm5-2b/zero_calibration.json")
TEST_PROMPTS = (
    "用一句话解释什么是 GQA, 并说明它为什么能降低 KV cache 的显存占用.",
    "写一个 Python 函数, 返回斐波那契数列的前 20 项, 并解释时间复杂度.",
    "如果一个三角形的底为 8, 高为 5, 面积是多少? 请给出计算过程.",
)

# (name, bits, group_size)
PLAIN_CANDIDATES = (
    ("rtn_w4_g64", 4, 64),
    ("rtn_w4_g128", 4, 128),
    ("rtn_w8_g128", 8, 128),
)
QUAROT_CANDIDATES = (
    ("quarot_rtn_w4_g64", 4, 64),
    ("quarot_rtn_w4_g128", 4, 128),
)


def _logit_metrics(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict[str, float]:
    diffs = [right - left for left, right in zip(reference, candidate)]
    flat = torch.cat([item.reshape(-1) for item in diffs])
    return {
        "max_abs": float(flat.abs().max().item()),
        "mean_abs": float(flat.abs().mean().item()),
        "rmse": float(flat.square().mean().sqrt().item()),
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


def _run_candidate(
    base_model: Any,
    tokenizer: Any,
    name: str,
    bits: int,
    group_size: int,
    prompt_inputs: list[dict[str, torch.Tensor]],
    reference_logits: list[torch.Tensor],
) -> dict[str, Any]:
    """Quantize one zero-calibration candidate and measure it end to end."""

    result = quantize_with_awq_weight_only(
        base_model,
        policy={
            "include_module_types": ["Linear"],
            "exclude_module_types": ["LayerNorm", "Embedding"],
            "exclude_name_patterns": [r"(^|\.)lm_head$"],
            "bits": int(bits),
            "group_size": int(group_size),
        },
        calibration_inputs=None,
        strategy="w4a16_int4",
        inplace=False,
    )
    metadata = dict(result.metadata)
    model = result.model.to(DEVICE).eval()
    logits = [
        _forward_logits(model, {key: value.to(DEVICE) for key, value in inputs.items()})
        for inputs in prompt_inputs
    ]
    metrics = _logit_metrics(reference_logits, logits)

    first_inputs = {key: value.to(DEVICE) for key, value in prompt_inputs[0].items()}
    generated = _generate(model, first_inputs)
    prompt_tokens = int(first_inputs["input_ids"].shape[-1])
    text = tokenizer.decode(generated[0, prompt_tokens:], skip_special_tokens=False)
    latency = benchmark_callable(
        lambda: _generate(model, first_inputs),
        warmup=1,
        iterations=3,
        sync_cuda=True,
        device=str(DEVICE),
    )

    report = {
        "name": name,
        "bits": int(bits),
        "group_size": int(group_size),
        "calibrated_module_count": metadata.get("calibrated_module_count"),
        "calibration_sample_count": metadata.get("calibration_sample_count"),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "logit_diff": metrics,
        "generated_text": text,
        "latency": latency.to_dict(),
    }
    del model, logits, generated
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return report


def main() -> None:
    """Run zero-calibration candidates before and after QuaRot rotation."""

    source = _source()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    prompt_inputs = [
        {key: value.cpu() for key, value in _inputs(tokenizer, prompt).items()}
        for prompt in TEST_PROMPTS
    ]
    model = load_minicpm5(
        source,
        dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
        device="cpu",
        local_files_only=True,
    )
    model = model.to(DEVICE)
    reference_logits = [
        _forward_logits(model, {key: value.to(DEVICE) for key, value in inputs.items()})
        for inputs in prompt_inputs
    ]
    model = model.to("cpu")

    reports: list[dict[str, Any]] = []
    for name, bits, group_size in PLAIN_CANDIDATES:
        try:
            reports.append(
                _run_candidate(model, tokenizer, name, bits, group_size, prompt_inputs, reference_logits)
            )
        except Exception as exc:  # noqa: BLE001 - keep the sweep running
            reports.append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    transform_report = apply_quarot_minicpm5(model, seed=42)
    for name, bits, group_size in QUAROT_CANDIDATES:
        try:
            reports.append(
                _run_candidate(model, tokenizer, name, bits, group_size, prompt_inputs, reference_logits)
            )
        except Exception as exc:  # noqa: BLE001 - keep the sweep running
            reports.append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    result = {
        "model_id": "openbmb/MiniCPM5-2B",
        "source": source,
        "device": str(DEVICE),
        "calibration": "none (calibration_inputs=None; AWQ degrades to RTN)",
        "transform_after_plain_group": transform_report.to_dict(),
        "candidates": reports,
        "reference_points": {
            "bf16_latency_mean_ms": 1046.3,
            "awq_w4_g64_5prompt_logit_rmse": 1.32,
            "awq_w4_g64_5prompt_latency_mean_ms": 1061.5,
            "fp4_w4_g64_nocal_logit_rmse": 1.58,
        },
        "limitations": [
            "Three prompts measure logit divergence and one prompt drives generation; this is not a task benchmark.",
            "Zero-calibration means no activation statistics; AWQ/GPTQ metadata still labels the algorithm.",
            "The QuaRot group runs after the offline rotation was applied in place; plain and rotated candidates share one base model.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        if report.get("status") == "failed":
            print(f"{report['name']}: FAILED {report['error']}")
            continue
        print(
            f"{report['name']}: rmse={report['logit_diff']['rmse']:.4f} "
            f"mean={report['logit_diff']['mean_abs']:.4f} "
            f"latency={report['latency']['mean_ms']:.1f}ms"
        )
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
