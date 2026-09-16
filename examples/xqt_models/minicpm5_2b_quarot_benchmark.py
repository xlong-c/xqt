"""Benchmark QuaRot-style MiniCPM5-2B rotation and rotated AWQ."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from xqt.compression.quant import quantize_with_awq_weight_only
from xqt.contracts.weight_only import AWQGPTQWeightOnlyLinear
from xqt.kernels.wrappers.bench.latency import benchmark_callable
from xqt.model.minicpm5 import (
    load_minicpm5,
    materialize_minicpm5_weight_only_runtime,
)
from xqt.model.minicpm5_quarot import apply_quarot_minicpm5
from examples.xqt_models.minicpm5_2b_quant_benchmark import (
    CALIBRATION_PROMPTS,
    DEVICE,
    MAX_NEW_TOKENS,
    _forward_logits,
    _inputs,
    _source,
)

OUTPUT = Path("artifacts/xqt/inference/minicpm5-2b/quarot_comparison.json")
TEST_PROMPTS = (
    "用一句话解释什么是 GQA, 并说明它为什么能降低 KV cache 的显存占用.",
    "写一个 Python 函数, 返回斐波那契数列的前 20 项, 并解释时间复杂度.",
    "如果一个三角形的底为 8, 高为 5, 面积是多少? 请给出计算过程.",
)


def _model_inputs(tokenizer: Any, prompt: str) -> dict[str, torch.Tensor]:
    return {key: value.cpu() for key, value in _inputs(tokenizer, prompt).items()}


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


def _benchmark_generation(model: Any, inputs_cpu: dict[str, torch.Tensor], tokenizer: Any) -> dict[str, Any]:
    inputs = {key: value.to(DEVICE) for key, value in inputs_cpu.items()}
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
    return {"text": text, "latency": latency.to_dict()}


def _tilelang_packed_kernel_feasible(module: Any, *, target_arch: str) -> bool:
    """Check the packed TileLang dequant kernel fits this module on this GPU."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if f"sm_{major}{minor}" != target_arch:
        return False
    from xqt.kernels.ops._impl.tilelang.gemm_builder import (
        build_tilelang_fp4_fused_dequant_gemm_kernel,
    )

    try:
        build_tilelang_fp4_fused_dequant_gemm_kernel(
            m=16,
            n=module.output_features,
            input_features=module.input_features,
            group_size=module.group_size,
            block_m=16,
            block_n=64,
            block_k=128,
            num_stages=3,
            threads=256,
            target_arch=target_arch,
        )
    except (ValueError, Exception) as exc:  # noqa: BLE001 - report any infeasibility
        print(f"[minicpm5] packed TileLang kernel infeasible: {exc}")
        return False
    return True


def main() -> None:
    """Validate offline rotation invariance and benchmark rotated AWQ."""

    source = _source()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    prompt_inputs = [_model_inputs(tokenizer, prompt) for prompt in TEST_PROMPTS]
    calibration_inputs = [
        _model_inputs(tokenizer, prompt) for prompt in CALIBRATION_PROMPTS
    ]
    dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

    model = load_minicpm5(source, dtype=dtype, device="cpu", local_files_only=True)
    model = model.to(DEVICE)
    reference_logits = [
        _forward_logits(model, {key: value.to(DEVICE) for key, value in inputs.items()})
        for inputs in prompt_inputs
    ]
    model = model.to("cpu")
    transform_report = apply_quarot_minicpm5(model, seed=42)
    model = model.to(DEVICE)
    rotated_logits = [
        _forward_logits(model, {key: value.to(DEVICE) for key, value in inputs.items()})
        for inputs in prompt_inputs
    ]
    rotation_metrics = _logit_metrics(reference_logits, rotated_logits)
    model = model.to("cpu")

    quantized = quantize_with_awq_weight_only(
        model,
        policy={
            "include_module_types": ["Linear"],
            "exclude_module_types": ["LayerNorm", "Embedding"],
            "exclude_name_patterns": [r"(^|\.)lm_head$"],
            "bits": 4,
            "group_size": 64,
        },
        calibration_inputs=calibration_inputs,
        strategy="w4a16_int4",
        inplace=False,
    )
    first_packed = next(
        module
        for module in quantized.model.modules()
        if isinstance(module, AWQGPTQWeightOnlyLinear)
    )
    packed_feasible = _tilelang_packed_kernel_feasible(
        first_packed,
        target_arch="sm_89",
    )
    print(
        f"[minicpm5] tilelang packed kernel feasible: {packed_feasible}; "
        "using native SM89 CUDA hybrid decode route"
    )
    # Native SM89 CUDA decode path: bf16-safe, no dtype cast needed. Decode
    # rows (M <= 8) route to the FasterTransformer-style W4A16 GEMV; larger
    # batches fall back to the dense cache.
    runtime_model = quantized.model.to(DEVICE)
    runtime_model = materialize_minicpm5_weight_only_runtime(
        runtime_model,
        engine="cuda",
    )
    quantized_model = runtime_model.eval()
    quantized_logits = [
        _forward_logits(
            quantized_model,
            {key: value.to(DEVICE) for key, value in inputs.items()},
        )
        for inputs in prompt_inputs
    ]
    quant_metrics = _logit_metrics(reference_logits, quantized_logits)
    generation = _benchmark_generation(quantized_model, prompt_inputs[0], tokenizer)

    result = {
        "model_id": "openbmb/MiniCPM5-2B",
        "source": source,
        "device": str(DEVICE),
        "dtype": str(dtype),
        "transform": transform_report.to_dict(),
        "rotation_invariance": {
            "prompts": list(TEST_PROMPTS),
            "logit_diff": rotation_metrics,
        },
        "rotated_awq": {
            "strategy": "w4a16_int4",
            "group_size": 64,
            "quantized_module_count": len(quantized.quantized_modules),
            "logit_diff_against_original_bf16": quant_metrics,
            "generation": generation,
            "runtime": {
                "decode_engine": "native_sm89_awq_w4a16_gemv_hybrid",
                "runtime_dtype": "bfloat16",
                "execution_metadata": [
                    {
                        "name": name,
                        "metadata": module.execution_metadata(),
                    }
                    for name, module in quantized_model.named_modules()
                    if callable(getattr(module, "execution_metadata", None))
                ][:8],
            },
        },
        "limitations": [
            "This is the QuaRot offline hidden-coordinate transform; online KV/value and MLP-down Hadamard nodes are intentionally disabled.",
            "Three prompts are used for rotation invariance and five prompts for AWQ calibration; this is not a task benchmark.",
            "The AWQ runtime result is an XQT artifact and is not an official QuaRot packed CUDA format.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rotation_invariance": rotation_metrics,
        "rotated_awq": {
            "quantized_module_count": len(quantized.quantized_modules),
            "logit_diff": quant_metrics,
            "latency": generation["latency"],
        },
    }, ensure_ascii=False, indent=2))
    del quantized_model, model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
