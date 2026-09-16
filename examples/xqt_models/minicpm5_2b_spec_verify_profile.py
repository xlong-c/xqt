"""Decompose the spec-decode verify step cost on the real MiniCPM5-2B W4 model.

Isolates: graph replay, per-kernel budget (torch profiler), and the host-side
readback (argmax/tolist) so the 20 ms/step seen in the k=3 PLD run can be
attributed.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

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

OUT = Path("artifacts/xqt/inference/minicpm5-2b/spec_verify_profile.json")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "downloads/MiniCPM5-2B-bf16", local_files_only=True
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ev._translation_prompt()}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"].cuda()
    base = load_minicpm5(
        "downloads/MiniCPM5-2B-bf16",
        dtype=torch.bfloat16,
        device="cpu",
        local_files_only=True,
    )
    calib = [
        {k: v.cpu() for k, v in ev._inputs(tokenizer, t).items()}
        for t in ev.CALIBRATION_PROMPTS
    ]
    packed = quantize_minicpm5(
        base,
        strategy="w4a16_int4",
        policy=minicpm5_quantization_policy(),
        calibration_inputs=calib,
        inplace=False,
        group_size=64,
    )
    del base, calib
    model = materialize_minicpm5_weight_only_runtime(
        packed.model.cuda(), engine="cuda"
    ).eval()
    del packed
    gc.collect()
    torch.cuda.empty_cache()
    head = quantize_minicpm5_lm_head_w4(model)

    session = CudaGraphDecodeSession(
        model, max_cache_len=4096, lm_head=head, int8_activations=True
    )
    spec = SpecDecodeSession(session, draft_tokens=3)
    session.prefill(input_ids)
    spec._pos.fill_(int(input_ids.shape[1]) - 1)
    spec._valid_lens.copy_(
        torch.tensor([int(input_ids.shape[1])] * spec._rows, dtype=torch.int32).cuda()
    )
    spec._window.copy_(session.token.reshape(1, 1).repeat(spec._rows, 1).to(torch.long))
    # warm + capture
    with torch.inference_mode():
        spec._verify_body()
        torch.cuda.synchronize()
        spec._capture()
    torch.cuda.synchronize()

    result: dict[str, object] = {"rows": spec._rows, "draft_tokens": spec._k}

    # 1) pure replay
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        spec._graphs[0].replay()
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - t0) / 20 * 1000
    result["replay_ms"] = replay_ms

    # 2) replay + argmax device work
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        spec._graphs[0].replay()
        spec._logits_out.argmax(-1)
    torch.cuda.synchronize()
    result["replay_plus_argmax_ms"] = (time.perf_counter() - t0) / 20 * 1000

    # 3) replay + argmax + host readback
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        spec._graphs[0].replay()
        spec._logits_out.argmax(-1).tolist()
    torch.cuda.synchronize()
    result["replay_plus_readback_ms"] = (time.perf_counter() - t0) / 20 * 1000

    # 4) full iteration (window/valid/pos fill + replay + readback)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        spec._window.fill_(3)
        spec._valid_lens.fill_(2000)
        spec._pos.fill_(1000)
        spec._graphs[0].replay()
        spec._logits_out.argmax(-1).tolist()
    torch.cuda.synchronize()
    result["full_iteration_ms"] = (time.perf_counter() - t0) / 20 * 1000

    # 5) compare: single-token graph replay (production w4a8 route)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        session.decode_batch(1)
    torch.cuda.synchronize()
    result["single_token_decode_batch1_ms"] = (time.perf_counter() - t0) / 20 * 1000

    # 6) profiler budget of the verify body (eager)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            for _ in range(10):
                spec._verify_body()
        torch.cuda.synchronize()
    rows = []
    for ev_ in prof.key_averages():
        if ev_.device_time_total > 0:
            rows.append(
                {
                    "name": ev_.key[:100],
                    "count": int(ev_.count),
                    "us_per_call": round(ev_.device_time_total / 10, 1),
                }
            )
    rows.sort(key=lambda r: -r["us_per_call"])
    result["verify_kernel_budget_us"] = rows[:15]
    result["verify_kernel_total_us"] = round(sum(r["us_per_call"] for r in rows), 1)

    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1)[:3000])


if __name__ == "__main__":
    main()
