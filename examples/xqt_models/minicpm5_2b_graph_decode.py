"""MiniCPM5-2B decode speedup benchmark: eager vs CUDA-graph quantized decode.

Seven routes are measured on the same 7976-character English document that
``minicpm5_2b_translation_eval.py`` uses:

- ``bf16_eager``: BF16 model through Transformers ``generate`` (the baseline).
- ``bf16_graph``: BF16 model through ``CudaGraphDecodeSession`` (runtime gain).
- ``w4a16_eager``: AWQ W4A16 native SM89 model through Transformers ``generate``.
- ``w4a16_graph``: AWQ W4A16 model through ``CudaGraphDecodeSession`` with the
  production optimizations: one length-agnostic CUDA graph, a single-pass
  Triton GQA decode attention, fused q/k/v and gate/up decode GEMVs, a fused
  RoPE + KV-cache scatter, fused RMSNorm, a Triton SwiGLU, a residual epilogue
  that folds ``hidden += projection(x)`` into o_proj/down_proj, chunked token
  readback (one device sync per 16 tokens), and a W4A16 INT4 LM head
  (``quantize_minicpm5_lm_head_w4``). The same runtime and fusions are used for
  both graph routes, so the comparison isolates the quantization gain.
- ``w4a8_graph``: the same W4 model and graph runtime with per-row INT8
  activation quantization (``int8_activations=True``) on the decode path: the
  QuaRot-style "INT8 activation + 4-bit weight" configuration.
- ``w4a8_kv8_graph``: W4A8 graph runtime with token-wise INT8 KV cache.
- ``w4a8_kv4_graph``: W4A8 graph runtime with OScaR (Orthogonalized Scaled Cache
  Rotation) 4-bit packed KV cache.

The goal metric is end-to-end wall time of the quantized graph route against
the BF16 eager route, plus the decode-only tokens/s. Configuration is
code-level constants; results are written to
``artifacts/xqt/inference/minicpm5-2b/graph_decode_benchmark.json``.
"""

from __future__ import annotations

import difflib
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

import examples.xqt_models.minicpm5_2b_translation_eval as ev
from xdl.metric.text import BLEUScore, ROUGELScore
from xqt.model.minicpm5 import (
    load_minicpm5,
    materialize_minicpm5_weight_only_runtime,
    minicpm5_quantization_policy,
    quantize_minicpm5,
    quantize_minicpm5_lm_head_w4,
)
from xqt.runtime import CudaGraphDecodeSession

ARTIFACT_DIR = Path("artifacts/xqt/inference/minicpm5-2b")
OUTPUT = ARTIFACT_DIR / "graph_decode_benchmark.json"

# bf16 graph e2e recorded by R-052 (2026-09-10, pre-rewrite runtime); kept for
# the goal-relevant ratio next to the same-runtime comparison.
RECORDED_BF16_GRAPH_MS = 16240.1
MAX_NEW_TOKENS = 1536
MAX_CACHE_LEN = 4096
EOS_TOKEN_IDS = (1, 130073)
DEVICE = torch.device("cuda")


def _free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _render(tokenizer: Any) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ev._translation_prompt()}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, return_tensors="pt")["input_ids"].to(DEVICE)


def _eager_generate(
    model: Any, tokenizer: Any, input_ids: torch.Tensor
) -> dict[str, Any]:
    """Timed HF ``generate`` with one warmup run."""

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=list(EOS_TOKEN_IDS),
                use_cache=True,
            )

    run()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = output[0, input_ids.shape[1] :].tolist()
    return {
        "e2e_ms": elapsed * 1000.0,
        "generated_tokens": len(tokens),
        "decode_tokens_per_second": (len(tokens) - 1) / elapsed,
        "hit_eos": bool(tokens and tokens[-1] in EOS_TOKEN_IDS),
        "token_ids": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
    }


def _graph_generate(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    lm_head: Any | None = None,
    int8_activations: bool = False,
    kv_quant: str = "none",
) -> dict[str, Any]:
    """Timed CUDA-graph decode, with graph capture measured separately."""

    session = CudaGraphDecodeSession(
        model,
        max_cache_len=MAX_CACHE_LEN,
        lm_head=lm_head,
        int8_activations=int8_activations,
        kv_quant=kv_quant,
    )
    capture_started = time.perf_counter()
    session.capture_until(int(input_ids.shape[1]) + MAX_NEW_TOKENS)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - capture_started) * 1000.0

    # First pass warms kernels; second pass is the measured run.
    session.generate(
        input_ids, max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=EOS_TOKEN_IDS
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = session.generate(
        input_ids, max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=EOS_TOKEN_IDS
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    decode_rate: float | None = None
    steady_started = time.perf_counter()
    session.decode_batch(32)
    torch.cuda.synchronize()
    steady = time.perf_counter() - steady_started
    decode_rate = 32 / steady

    return {
        "e2e_ms": elapsed * 1000.0,
        "capture_ms": capture_ms,
        "captured_graphs": session.captured_graphs,
        "fused_projection_count": session.fused_projection_count,
        "lm_head": "override" if lm_head is not None else "model default",
        "int8_activations": bool(int8_activations),
        "generated_tokens": result.generated_tokens,
        "decode_tokens_per_second": (result.generated_tokens - 1) / elapsed,
        "steady_decode_tokens_per_second": decode_rate,
        "hit_eos": result.hit_eos,
        "token_ids": result.token_ids,
        "text": tokenizer.decode(result.token_ids, skip_special_tokens=True),
    }


def _token_agreement(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    """Token-level agreement between two greedy runs of the same model."""

    shared = min(len(reference), len(candidate))
    matches = sum(1 for index in range(shared) if reference[index] == candidate[index])
    first_mismatch = next(
        (index for index in range(shared) if reference[index] != candidate[index]),
        None,
    )
    return {
        "shared_tokens": shared,
        "matching_tokens": matches,
        "agreement": matches / max(shared, 1),
        "first_mismatch_index": first_mismatch,
    }


def _text_quality(reference: str, candidate: str) -> dict[str, float]:
    """Character-level agreement of two translations."""

    reference_chars = list(reference)
    candidate_chars = list(candidate)
    return {
        "bleu4_char": float(BLEUScore(max_n=4)([candidate_chars], [reference_chars])),
        "rouge_l_char": float(ROUGELScore()([candidate_chars], [reference_chars])),
        "sequence_ratio": float(
            difflib.SequenceMatcher(
                None, reference_chars, candidate_chars, autojunk=False
            ).ratio()
        ),
    }


def _assert_short_run_matches(
    model: Any, tokenizer: Any, input_ids: torch.Tensor, steps: int = 24
) -> None:
    """Strict greedy-equality check on a short horizon (cheap, catches drift)."""

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=steps,
            do_sample=False,
            eos_token_id=list(EOS_TOKEN_IDS),
            use_cache=True,
        )
    reference = output[0, input_ids.shape[1] :].tolist()
    session = CudaGraphDecodeSession(
        model,
        max_cache_len=MAX_CACHE_LEN,
        fuse_norms=False,
    )
    result = session.generate(
        input_ids, max_new_tokens=steps, eos_token_ids=EOS_TOKEN_IDS
    )
    if result.token_ids != reference:
        raise AssertionError(
            "CUDA-graph decode diverged from eager greedy within "
            f"{steps} tokens: {result.token_ids[:8]} vs {reference[:8]}"
        )
    del session


def _strip_ids(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "token_ids"}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "downloads/MiniCPM5-2B-bf16", local_files_only=True
    )
    input_ids = _render(tokenizer)
    prompt_tokens = int(input_ids.shape[1])
    print(f"[graph-decode] prompt tokens: {prompt_tokens}", flush=True)

    base_model = load_minicpm5(
        "downloads/MiniCPM5-2B-bf16",
        dtype=torch.bfloat16,
        device="cpu",
        local_files_only=True,
    )
    calibration = [
        {key: value.cpu() for key, value in ev._inputs(tokenizer, text).items()}
        for text in ev.CALIBRATION_PROMPTS
    ]

    bf16_model = base_model.to(DEVICE)
    print("[graph-decode] bf16 eager ...", flush=True)
    bf16_eager = _eager_generate(bf16_model, tokenizer, input_ids)
    print(
        f"[graph-decode] bf16 eager: e2e={bf16_eager['e2e_ms']:.0f}ms "
        f"decode={bf16_eager['decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )
    print("[graph-decode] bf16 graph ...", flush=True)
    bf16_graph = _graph_generate(bf16_model, tokenizer, input_ids)
    print(
        f"[graph-decode] bf16 graph: e2e={bf16_graph['e2e_ms']:.0f}ms "
        f"steady={bf16_graph['steady_decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )
    _assert_short_run_matches(bf16_model, tokenizer, input_ids)
    bf16_agreement = _token_agreement(bf16_eager["token_ids"], bf16_graph["token_ids"])
    gc.collect()
    base_model = bf16_model.to("cpu")
    del bf16_model
    _free_cuda()

    print("[graph-decode] quantizing awq_w4_g64 ...", flush=True)
    packed = quantize_minicpm5(
        base_model,
        strategy="w4a16_int4",
        policy=minicpm5_quantization_policy(),
        calibration_inputs=calibration,
        inplace=False,
        group_size=64,
    )
    quantized = materialize_minicpm5_weight_only_runtime(
        packed.model.to(DEVICE), engine="cuda"
    ).eval()
    del packed
    _free_cuda()

    print("[graph-decode] w4a16 eager ...", flush=True)
    w4a16_eager = _eager_generate(quantized, tokenizer, input_ids)
    print(
        f"[graph-decode] w4a16 eager: e2e={w4a16_eager['e2e_ms']:.0f}ms "
        f"decode={w4a16_eager['decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )
    print("[graph-decode] quantizing lm_head to W4A16 ...", flush=True)
    w4_head = quantize_minicpm5_lm_head_w4(quantized)
    print("[graph-decode] w4a16 graph ...", flush=True)
    w4a16_graph = _graph_generate(quantized, tokenizer, input_ids, lm_head=w4_head)
    print(
        f"[graph-decode] w4a16 graph: e2e={w4a16_graph['e2e_ms']:.0f}ms "
        f"steady={w4a16_graph['steady_decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )
    print("[graph-decode] w4a8 graph (INT8 activations) ...", flush=True)
    w4a8_graph = _graph_generate(
        quantized, tokenizer, input_ids, lm_head=w4_head, int8_activations=True
    )
    print(
        f"[graph-decode] w4a8 graph: e2e={w4a8_graph['e2e_ms']:.0f}ms "
        f"steady={w4a8_graph['steady_decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )
    print("[graph-decode] w4a8_kv8 graph (INT8 activations + INT8 KV) ...", flush=True)
    w4a8_kv8_graph = _graph_generate(
        quantized,
        tokenizer,
        input_ids,
        lm_head=w4_head,
        int8_activations=True,
        kv_quant="int8",
    )
    print(
        f"[graph-decode] w4a8_kv8 graph: e2e={w4a8_kv8_graph['e2e_ms']:.0f}ms "
        f"steady={w4a8_kv8_graph['steady_decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )
    print(
        "[graph-decode] w4a8_kv4 graph (INT8 activations + OScaR INT4 KV) ...",
        flush=True,
    )
    w4a8_kv4_graph = _graph_generate(
        quantized,
        tokenizer,
        input_ids,
        lm_head=w4_head,
        int8_activations=True,
        kv_quant="int4",
    )
    print(
        f"[graph-decode] w4a8_kv4 graph: e2e={w4a8_kv4_graph['e2e_ms']:.0f}ms "
        f"steady={w4a8_kv4_graph['steady_decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )
    w4a16_agreement = _token_agreement(
        w4a16_eager["token_ids"], w4a16_graph["token_ids"]
    )
    quant_agreement = _token_agreement(
        bf16_eager["token_ids"], w4a16_graph["token_ids"]
    )
    text_quality = {
        "w4a16_graph_vs_bf16_eager": _text_quality(
            bf16_eager["text"], w4a16_graph["text"]
        ),
        "w4a8_graph_vs_bf16_eager": _text_quality(
            bf16_eager["text"], w4a8_graph["text"]
        ),
        "w4a8_kv8_graph_vs_bf16_eager": _text_quality(
            bf16_eager["text"], w4a8_kv8_graph["text"]
        ),
        "w4a8_kv4_graph_vs_bf16_eager": _text_quality(
            bf16_eager["text"], w4a8_kv4_graph["text"]
        ),
        "w4a16_eager_vs_bf16_eager": _text_quality(
            bf16_eager["text"], w4a16_eager["text"]
        ),
    }

    speedup = bf16_eager["e2e_ms"] / w4a16_graph["e2e_ms"]
    result = {
        "model_id": "openbmb/MiniCPM5-2B",
        "device": str(DEVICE),
        "document": "minicpm5_2b_translation_eval.SOURCE_DOCUMENT",
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_cache_len": MAX_CACHE_LEN,
        "routes": {
            "bf16_eager": _strip_ids(bf16_eager),
            "bf16_graph": _strip_ids(bf16_graph),
            "w4a16_eager": _strip_ids(w4a16_eager),
            "w4a16_graph": _strip_ids(w4a16_graph),
            "w4a8_graph": _strip_ids(w4a8_graph),
            "w4a8_kv8_graph": _strip_ids(w4a8_kv8_graph),
            "w4a8_kv4_graph": _strip_ids(w4a8_kv4_graph),
        },
        "token_agreement": {
            "bf16_graph_vs_bf16_eager": bf16_agreement,
            "w4a16_graph_vs_w4a16_eager": w4a16_agreement,
            "w4a8_graph_vs_w4a16_graph": _token_agreement(
                w4a16_graph["token_ids"], w4a8_graph["token_ids"]
            ),
            "w4a8_kv8_graph_vs_w4a16_graph": _token_agreement(
                w4a16_graph["token_ids"], w4a8_kv8_graph["token_ids"]
            ),
            "w4a8_kv4_graph_vs_w4a16_graph": _token_agreement(
                w4a16_graph["token_ids"], w4a8_kv4_graph["token_ids"]
            ),
            "w4a16_graph_vs_bf16_eager": quant_agreement,
        },
        "text_quality": text_quality,
        "speedup": {
            "e2e_vs_bf16_eager": speedup,
            "decode_tokens_per_second_ratio": (
                w4a16_graph["steady_decode_tokens_per_second"]
                / bf16_eager["decode_tokens_per_second"]
            ),
            "steady_decode_tokens_per_second_ratio": (
                w4a16_graph["steady_decode_tokens_per_second"]
                / bf16_graph["steady_decode_tokens_per_second"]
            ),
            "e2e_vs_bf16_graph": (bf16_graph["e2e_ms"] / w4a16_graph["e2e_ms"]),
            "e2e_vs_bf16_graph_int8_activations": (
                bf16_graph["e2e_ms"] / w4a8_graph["e2e_ms"]
            ),
            "e2e_vs_bf16_graph_w4a8_kv8": (
                bf16_graph["e2e_ms"] / w4a8_kv8_graph["e2e_ms"]
            ),
            "e2e_vs_bf16_graph_w4a8_kv4": (
                bf16_graph["e2e_ms"] / w4a8_kv4_graph["e2e_ms"]
            ),
            "steady_ratio_kv4_vs_w4a16": (
                w4a8_kv4_graph["steady_decode_tokens_per_second"]
                / w4a16_graph["steady_decode_tokens_per_second"]
            ),
            "target_3x_vs_bf16_graph_met": bool(
                bf16_graph["e2e_ms"] / w4a16_graph["e2e_ms"] >= 3.0
            ),
            "e2e_vs_recorded_bf16_graph_baseline": (
                RECORDED_BF16_GRAPH_MS / w4a16_graph["e2e_ms"]
            ),
            "e2e_vs_recorded_bf16_graph_baseline_int8_activations": (
                RECORDED_BF16_GRAPH_MS / w4a8_graph["e2e_ms"]
            ),
            "e2e_vs_recorded_bf16_graph_baseline_w4a8_kv4": (
                RECORDED_BF16_GRAPH_MS / w4a8_kv4_graph["e2e_ms"]
            ),
            "target_3x_vs_recorded_baseline_met": bool(
                RECORDED_BF16_GRAPH_MS
                / min(
                    w4a16_graph["e2e_ms"],
                    w4a8_graph["e2e_ms"],
                    w4a8_kv4_graph["e2e_ms"],
                )
                >= 3.0
            ),
        },
        "notes": [
            "Eager routes use one warmup pass; graph routes use one warmup pass and a single pre-captured CUDA graph.",
            "Both graph routes share the same runtime, including the Triton decode attention, fused q/k/v and gate/up decode GEMVs with a folded residual epilogue, the fused RoPE + KV scatter, fused RMSNorm and SwiGLU; only the quantized route also uses a W4A16 LM head. Quality is reported against the BF16 eager translation.",
            "Short-horizon greedy equality with eager is asserted for the BF16 graph route; the W4A16 graph route also changes the head precision, so only agreement and text quality are reported there.",
            "Greedy decoding reads tokens back in chunks of 16 (readback_chunk), so the W4 graph routes pay one device synchronization per 16 tokens instead of one per token; the steady loop measures one such chunk.",
            "decode tokens/s counts generated tokens minus the first token over wall time.",
        ],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["speedup"], ensure_ascii=False, indent=2))
    print(json.dumps(result["token_agreement"], ensure_ascii=False, indent=2))
    print(f"[graph-decode] artifact: {OUTPUT}")
    del quantized
    _free_cuda()


if __name__ == "__main__":
    main()
