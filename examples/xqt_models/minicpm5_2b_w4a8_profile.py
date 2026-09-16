"""W4A8 vs W4A16 same-run baseline, per-step kernel budget, and ncu counters.

One script, three evidence streams for R-055:

1. Same-run A/B baseline: w4a16 and w4a8 sessions built from the same
   quantized model in one process; steady decode measured interleaved
   (median of repeated batches) so run-to-run drift cannot fake a ratio.
2. torch profiler per-step kernel budget for each mode (decode step replayed
   under ``torch.profiler`` with CUDA activity, aggregated by kernel name).
3. ncu artifact hooks: ``--ncu-step`` wraps one ``decode_batch`` call in a
   ``torch.cuda.profiler`` range so the replayed graph kernels can be
   attributed; run the script itself under ncu with ``--replay-mode application``
   is too heavy, so the intended flow is documented in R-055 instead.

Prompt and model match ``minicpm5_2b_graph_decode.py`` exactly. Results are
written to ``artifacts/xqt/inference/minicpm5-2b/w4a8_baseline_profile.json``.

Configuration is code-level constants per workspace rules.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
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
from xqt.runtime import CudaGraphDecodeSession

ARTIFACT_DIR = Path("artifacts/xqt/inference/minicpm5-2b")
OUTPUT = ARTIFACT_DIR / "w4a8_baseline_profile.json"
MODEL_PATH = "downloads/MiniCPM5-2B-bf16"
DEVICE = torch.device("cuda")
MAX_CACHE_LEN = 4096
EOS_TOKEN_IDS = (1, 130073)

# Steady measurement: interleaved batches, median reported. R-054 saw 265-311
# tok/s spread across runs; only same-run interleaved comparisons count.
STEADY_BATCH = 64
STEADY_REPEATS = 7
WARMUP_STEPS = 32
PROFILER_STEPS = 16


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


def _ncu_oneshot() -> None:
    """Short eager window for ncu: prefill then three eager decode steps."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    input_ids = _render(tokenizer)
    base_model = load_minicpm5(
        MODEL_PATH, dtype=torch.bfloat16, device="cpu", local_files_only=True
    )
    calibration = [
        {key: value.cpu() for key, value in ev._inputs(tokenizer, text).items()}
        for text in ev.CALIBRATION_PROMPTS
    ]
    packed = quantize_minicpm5(
        base_model,
        strategy="w4a16_int4",
        policy=minicpm5_quantization_policy(),
        calibration_inputs=calibration,
        inplace=False,
        group_size=64,
    )
    del base_model, calibration
    quantized = materialize_minicpm5_weight_only_runtime(
        packed.model.to(DEVICE), engine="cuda"
    ).eval()
    w4_head = quantize_minicpm5_lm_head_w4(quantized)
    session = CudaGraphDecodeSession(
        quantized,
        max_cache_len=MAX_CACHE_LEN,
        lm_head=w4_head,
        int8_activations=os.environ.get("W4A8_INT8", "0") == "1",
    )
    del packed
    _free_cuda()
    session.prefill(input_ids)
    # Eager decode steps: not captured, so ncu sees them as regular launches,
    # and wrapped in an NVTX range so ncu can profile just the decode step
    # (``--nvtx --nvtx-include eager_decode/``) instead of the prefill noise.
    with torch.inference_mode():
        for _ in range(3):
            session._fill_step_state(session.length + 1)
            with torch.cuda.nvtx.range("eager_decode"):
                session._decode_body()
            session.length += 1
    session.prefill(input_ids)
    with torch.inference_mode():
        for _ in range(3):
            session._fill_step_state(session.length + 1)
            with torch.cuda.nvtx.range("eager_decode"):
                session._decode_body()
            session.length += 1
    torch.cuda.synchronize()
    print("[ncu-oneshot] done", flush=True)


def _build_sessions(quantized: Any, w4_head: Any) -> tuple[Any, torch.Tensor]:
    """Build both sessions (w4a16 / w4a8) over the SAME quantized model."""

    session = CudaGraphDecodeSession(
        quantized,
        max_cache_len=MAX_CACHE_LEN,
        lm_head=w4_head,
        int8_activations=False,
    )
    session8 = CudaGraphDecodeSession(
        quantized,
        max_cache_len=MAX_CACHE_LEN,
        lm_head=w4_head,
        int8_activations=True,
    )
    return session, session8


def _prepare(
    sessions: dict[str, CudaGraphDecodeSession], input_ids: torch.Tensor
) -> None:
    """Prefill every session to the same state (${"`"}WARMUP_STEPS warmup`), then sync."""

    for session in sessions.values():
        session.prefill(input_ids)
        session.decode_batch(WARMUP_STEPS)
    torch.cuda.synchronize()


def _steady_interleaved(
    sessions: dict[str, CudaGraphDecodeSession], repeats: int = STEADY_REPEATS
) -> dict[str, dict[str, float]]:
    """Interleaved steady measurement, median over batches per route."""

    rates: dict[str, list[float]] = {name: [] for name in sessions}
    step_ms: dict[str, list[float]] = {name: [] for name in sessions}
    for _ in range(repeats):
        for name, session in sessions.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            session.decode_batch(64)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rates[name].append(64.0 / elapsed)
            step_ms[name].append(elapsed * 1000.0 / 64.0)
    return {
        name: {
            "steady_tok_s_median": statistics.median(rates[name]),
            "steady_tok_s_min": min(rates[name]),
            "steady_tok_s_max": max(rates[name]),
            "step_ms_median": statistics.median(step_ms[name]),
            "step_ms_all": [round(v, 3) for v in step_ms[name]],
        }
        for name in sessions
    }


def _profile_step(session: CudaGraphDecodeSession) -> list[dict[str, Any]]:
    """One profiler-traced ``decode_batch``; aggregate CUDA kernels by name."""

    from torch.profiler import ProfilerActivity, profile

    session.decode_batch(WARMUP_STEPS)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        session.decode_batch(PROFILER_STEPS)
        torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    for ev_ in prof.key_averages():
        if ev_.device_time_total > 0 and not ev_.key.startswith("cpu"):
            rows.append(
                {
                    "name": ev_.key[:120],
                    "count": int(ev_.count),
                    "cuda_us_total": round(ev_.device_time_total, 1),
                    "cuda_us_per_step": round(
                        ev_.device_time_total / PROFILER_STEPS, 3
                    ),
                    "calls_per_step": ev_.count / PROFILER_STEPS,
                }
            )
    rows.sort(key=lambda r: -r["cuda_us_total"])
    total = sum(r["cuda_us_total"] for r in rows)
    return {"rows": rows, "total_cuda_us": round(total, 1)}


def main() -> None:
    # ``ncu-oneshot``: replay a fixed token, run three decode steps outside any
    # CUDA graph, and print kernel names so ncu can attach (ncu profiles CUDA
    # graphs in node mode by default; replaying the whole session under ncu is
    # too heavy, so this mode gives it a short eager window instead).
    if os.environ.get("W4A8_PROFILE_MODE") == "ncu-oneshot":
        _ncu_oneshot()
        return
    torch.manual_seed(0)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "model_id": "openbmb/MiniCPM5-2B",
        "device": torch.cuda.get_device_name(0),
        "steady_batch": STEADY_BATCH,
        "steady_repeats": STEADY_REPEATS,
        "warmup_steps": WARMUP_STEPS,
        "profiler_steps": PROFILER_STEPS,
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    input_ids = _render(tokenizer)
    base_model = load_minicpm5(
        MODEL_PATH, dtype=torch.bfloat16, device="cpu", local_files_only=True
    )
    calibration = [
        {key: value.cpu() for key, value in ev._inputs(tokenizer, text).items()}
        for text in ev.CALIBRATION_PROMPTS
    ]
    packed = quantize_minicpm5(
        base_model,
        strategy="w4a16_int4",
        policy=minicpm5_quantization_policy(),
        calibration_inputs=calibration,
        inplace=False,
        group_size=64,
    )
    del base_model
    quantized = materialize_minicpm5_weight_only_runtime(
        packed.model.to(DEVICE), engine="cuda"
    ).eval()
    del packed
    w4_head = quantize_minicpm5_lm_head_w4(quantized)
    del calibration
    _free_cuda()

    s16, s8 = _build_sessions(quantized, w4_head)
    for session in (s16, s8):
        session.capture_until(int(input_ids.shape[1]) + 2048)
    print(
        f"[profile] sessions ready, prompt_tokens={int(input_ids.shape[1])}",
        flush=True,
    )

    # Prefill latency: warm up twice (first call compiles Triton kernels), then
    # take the mean of repeats. Both sessions share the model, so kernel
    # compile happens once; the second session's first prefill just runs.
    for name, session in (("w4a16", s16), ("w4a8", s8)):
        session.prefill(input_ids)
    prefill_samples: dict[str, list[float]] = {"w4a16": [], "w4a8": []}
    for _ in range(5):
        for name, session in (("w4a16", s16), ("w4a8", s8)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            session.prefill(input_ids)
            torch.cuda.synchronize()
            prefill_samples[name].append((time.perf_counter() - started) * 1000.0)
    result["prefill_ms"] = {
        name: {
            "median": round(statistics.median(v), 2),
            "min": round(min(v), 2),
            "max": round(max(v), 2),
        }
        for name, v in prefill_samples.items()
    }
    print(
        f"[profile] prefill median: w4a16 {result['prefill_ms']['w4a16']['median']:.1f} ms, "
        f"w4a8 {result['prefill_ms']['w4a8']['median']:.1f} ms",
        flush=True,
    )

    both = {"w4a16": s16, "w4a8": s8}
    _prepare(both, input_ids)
    steady = _steady_interleaved(both)
    result["steady"] = steady
    ratio = steady["w4a16"]["step_ms_median"] / steady["w4a8"]["step_ms_median"]
    result["steady_step_ratio_w4a16_over_w4a8"] = round(ratio, 4)
    print(
        f"[profile] interleaved steady: w4a16 "
        f"{steady['w4a16']['steady_tok_s_median']:.1f} tok/s "
        f"({steady['w4a16']['step_ms_median']:.3f} ms/step), w4a8 "
        f"{steady['w4a8']['steady_tok_s_median']:.1f} tok/s "
        f"({steady['w4a8']['step_ms_median']:.3f} ms/step), ratio={ratio:.4f}",
        flush=True,
    )

    result["kernel_budget_w4a16"] = _profile_step(s16)
    result["kernel_budget_w4a8"] = _profile_step(s8)
    for name in ("w4a16", "w4a8"):
        budget = result[f"kernel_budget_{name}"]
        print(
            f"[profile] {name} kernel total {budget['total_cuda_us'] / PROFILER_STEPS:.3f} us/step",
            flush=True,
        )
        for row in budget["rows"][:8]:
            print(
                f"    {row['cuda_us_per_step']:8.3f} us/step x{row['calls_per_step']:6.1f}  {row['name'][:80]}",
                flush=True,
            )

    OUTPUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[profile] written {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
