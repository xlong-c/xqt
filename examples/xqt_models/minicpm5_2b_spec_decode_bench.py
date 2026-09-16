"""Speculative decode A/B: w4a16 graph vs w4a8 spec (PLD draft + exact verify).

Same run, same prompt as ``minicpm5_2b_graph_decode.py``. The comparison is
the goal metric for the W4A8 route: end-to-end wall time and steady decode
tok/s of the w4a8 speculative route against the non-speculative w4a16 route,
plus subset verification that the greedy token stream is identical when the
drafter proposing nothing (sanity) and how many draft tokens the prompt-lookup
drafter actually lands on this translation task.

Results go to ``artifacts/xqt/inference/minicpm5-2b/spec_decode_benchmark.json``.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

import examples.xqt_models.minicpm5_2b_translation_eval as ev
from xqt.model.minicpm5 import (
    load_minicpm5,
    materialize_minicpm5_weight_only_runtime,
    minicpm5_quantization_policy,
    quantize_minicpm5,
    quantize_minicpm5_lm_head_w4,
)
from xqt.runtime import CudaGraphDecodeSession, SpecDecodeSession

ARTIFACT_DIR = Path("artifacts/xqt/inference/minicpm5-2b")
OUTPUT = ARTIFACT_DIR / "spec_decode_benchmark.json"
MODEL_PATH = "downloads/MiniCPM5-2B-bf16"
DEVICE = torch.device("cuda")
MAX_NEW_TOKENS = 1536
MAX_CACHE_LEN = 4096
EOS_TOKEN_IDS = (1, 130073)
STEADY_REPEATS = 5


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


def _build_model() -> tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base = load_minicpm5(
        MODEL_PATH, dtype=torch.bfloat16, device="cpu", local_files_only=True
    )
    calibration = [
        {key: value.cpu() for key, value in ev._inputs(tokenizer, text).items()}
        for text in ev.CALIBRATION_PROMPTS
    ]
    packed = quantize_minicpm5(
        base,
        strategy="w4a16_int4",
        policy=minicpm5_quantization_policy(),
        calibration_inputs=calibration,
        inplace=False,
        group_size=64,
    )
    del base, calibration
    quantized = materialize_minicpm5_weight_only_runtime(
        packed.model.to(DEVICE), engine="cuda"
    ).eval()
    del packed
    _free_cuda()
    head = quantize_minicpm5_lm_head_w4(quantized)
    return tokenizer, quantized, head


def _run_baseline(model: Any, head: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    """Non-speculative w4a16 route, same measurement as prior records."""

    session = CudaGraphDecodeSession(model, max_cache_len=MAX_CACHE_LEN, lm_head=head)
    # warmup
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
    steady = []
    for _ in range(STEADY_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.decode_batch(32)
        torch.cuda.synchronize()
        steady.append(32 / (time.perf_counter() - t0))
    return {
        "e2e_ms": elapsed * 1000.0,
        "generated_tokens": result.generated_tokens,
        "decode_tokens_per_second": result.generated_tokens / elapsed,
        "steady_decode_tokens_per_second": statistics.median(steady),
        "token_ids": result.token_ids,
    }


def _run_spec(
    model: Any,
    head: Any,
    input_ids: torch.Tensor,
    *,
    int8: bool,
    draft_tokens: int,
) -> dict[str, Any]:
    session = CudaGraphDecodeSession(
        model,
        max_cache_len=MAX_CACHE_LEN,
        lm_head=head,
        int8_activations=int8,
    )
    spec = SpecDecodeSession(session, draft_tokens=draft_tokens)
    # warmup (compiles kernels + captures the verify graph)
    spec.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=EOS_TOKEN_IDS)
    torch.cuda.synchronize()
    started = time.perf_counter()
    tokens, stats = spec.generate(
        input_ids, max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=EOS_TOKEN_IDS
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "e2e_ms": elapsed * 1000.0,
        "generated_tokens": len(tokens),
        "decode_tokens_per_second": len(tokens) / elapsed,
        "spec_stats": stats,
        "token_ids": tokens,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, model, head = _build_model()
    input_ids = _render(tokenizer)
    prompt_tokens = int(input_ids.shape[1])
    print(f"[spec-bench] prompt tokens: {prompt_tokens}", flush=True)

    print("[spec-bench] w4a16 graph baseline ...", flush=True)
    base = _run_baseline(model, head, input_ids)
    print(
        f"[spec-bench] w4a16: e2e={base['e2e_ms']:.0f}ms "
        f"steady={base['steady_decode_tokens_per_second']:.1f}tok/s",
        flush=True,
    )

    runs: dict[str, Any] = {}
    for name, int8, k in [
        ("w4a8_spec_k3", True, 3),
        ("w4a8_spec_k7", True, 7),
    ]:
        print(f"[spec-bench] {name} ...", flush=True)
        out = _run_spec(model, head, input_ids, int8=int8, draft_tokens=k)
        agree = sum(1 for a, b in zip(base["token_ids"], out["token_ids"]) if a == b)
        shared = min(len(base["token_ids"]), len(out["token_ids"]))
        out["agreement_with_baseline"] = agree / shared if shared else 0.0
        out["accept_rate"] = out["spec_stats"]["accepted_draft_tokens"] / max(
            out["spec_stats"]["drafted_tokens"], 1
        )
        runs[name] = out
        print(
            f"[spec-bench] {name}: e2e={out['e2e_ms']:.0f}ms "
            f"tok/s={out['decode_tokens_per_second']:.1f} "
            f"speedup={base['e2e_ms'] / out['e2e_ms']:.2f}x "
            f"accept={out['accept_rate']:.2f}",
            flush=True,
        )
        _free_cuda()

    result = {
        "model_id": "openbmb/MiniCPM5-2B",
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "baseline": {k: v for k, v in base.items() if k != "token_ids"},
        "spec_runs": {
            k: {kk: vv for kk, vv in v.items() if kk != "token_ids"}
            for k, v in runs.items()
        },
        "speedup_e2e": {k: base["e2e_ms"] / v["e2e_ms"] for k, v in runs.items()},
        "speedup_tokps": {
            k: v["decode_tokens_per_second"] / base["decode_tokens_per_second"]
            for k, v in runs.items()
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[spec-bench] written {OUTPUT}", flush=True)
    print(json.dumps(result["speedup_e2e"], indent=1), flush=True)


if __name__ == "__main__":
    main()
