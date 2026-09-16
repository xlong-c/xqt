"""SIMT vs tensor-core decode attention: same-run A/B on MiniCPM5-2B.

The tensor-core variant (``attention_impl="tc"``) has already been validated
against SDPA in the kernel tests; this script decides whether it is faster on
the real route. One process, one quantized W4 model, three runtime
configurations interleaved with a per-round alternating order (a fixed order
plus sequential profiling let GPU state drift fake a 44% difference on kernels
that are bit-identical between routes). Steady decode is measured over
``STEADY_REPEATS`` interleaved batches and reported as medians; GPU
clock/temperature samples are recorded per round. A short greedy run records
token agreement against the SIMT control, and the torch profiler budget is
taken twice per route to expose its own variance.

Results are written to
``artifacts/xqt/inference/minicpm5-2b/attn_impl_ab.json``. Configuration is
code-level constants per workspace rules.
"""

from __future__ import annotations

import gc
import json
import statistics
import subprocess
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
OUTPUT = ARTIFACT_DIR / "attn_impl_ab.json"
MODEL_PATH = "downloads/MiniCPM5-2B-bf16"
DEVICE = torch.device("cuda")
MAX_CACHE_LEN = 4096
EOS_TOKEN_IDS = (1, 130073)
AGREEMENT_TOKENS = 64

# (attention_impl, attention_splits) per route; the first route is the control.
ROUTES: tuple[tuple[str, int], ...] = (
    ("simt", 16),
    ("tc", 16),
    ("tc", 64),
)

# Steady measurement: interleaved batches, median reported (R-054: only
# same-run interleaved comparisons count, absolute rates drift across runs).
STEADY_BATCH = 64
STEADY_REPEATS = 15
ORDER_FLIP_ROUNDS = True
WARMUP_STEPS = 32
PROFILER_STEPS = 16
PROFILER_ROUNDS = 2


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


def _route_name(impl: str, splits: int) -> str:
    return f"{impl}_s{splits}"


def _gpu_state() -> str:
    """One nvidia-smi sample so clock/power drift is visible in the artifact."""

    try:
        sample = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return sample.stdout.strip()
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"unavailable: {exc}"


def _build_sessions(quantized: Any, w4_head: Any) -> dict[str, Any]:
    """Build every route over the SAME quantized model.

    Routing only through production constructor arguments keeps the A/B honest:
    the only difference between the tc routes and the control is the attention
    implementation (and the split count for the sweep rows).
    """

    sessions: dict[str, CudaGraphDecodeSession] = {}
    for impl, splits in ROUTES:
        sessions[_route_name(impl, splits)] = CudaGraphDecodeSession(
            quantized,
            max_cache_len=MAX_CACHE_LEN,
            lm_head=w4_head,
            attention_splits=splits,
            attention_impl=impl,
        )
    return sessions


def _prepare(
    sessions: dict[str, CudaGraphDecodeSession], input_ids: torch.Tensor
) -> None:
    for session in sessions.values():
        session.prefill(input_ids)
        session.decode_batch(WARMUP_STEPS)
    torch.cuda.synchronize()


def _steady_interleaved(
    sessions: dict[str, CudaGraphDecodeSession], repeats: int = STEADY_REPEATS
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Interleaved steady measurement with a per-round order flip.

    Every route is advanced by the same number of steps at the same cache
    length; the visit order alternates so no route is systematically measured on
    hotter hardware than another.
    """

    names = list(sessions)
    rates: dict[str, list[float]] = {name: [] for name in names}
    step_ms: dict[str, list[float]] = {name: [] for name in names}
    states: list[str] = []
    for round_index in range(repeats):
        order = (
            names if (round_index % 2 == 0 or not ORDER_FLIP_ROUNDS) else names[::-1]
        )
        for name in order:
            session = sessions[name]
            torch.cuda.synchronize()
            started = time.perf_counter()
            session.decode_batch(STEADY_BATCH)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rates[name].append(STEADY_BATCH / elapsed)
            step_ms[name].append(elapsed * 1000.0 / STEADY_BATCH)
        states.append(_gpu_state())
    return (
        {
            name: {
                "steady_tok_s_median": statistics.median(rates[name]),
                "steady_tok_s_min": min(rates[name]),
                "steady_tok_s_max": max(rates[name]),
                "step_ms_median": statistics.median(step_ms[name]),
                "step_ms_min": min(step_ms[name]),
                "step_ms_all": [round(value, 3) for value in step_ms[name]],
            }
            for name in names
        },
        states,
    )


def _profile_step(session: CudaGraphDecodeSession) -> dict[str, Any]:
    """One profiler-traced ``decode_batch``; aggregate CUDA kernels by name."""

    from torch.profiler import ProfilerActivity, profile

    session.decode_batch(WARMUP_STEPS)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        session.decode_batch(PROFILER_STEPS)
        torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    for event in prof.key_averages():
        if event.device_time_total > 0 and not event.key.startswith("cpu"):
            rows.append(
                {
                    "name": event.key[:120],
                    "count": int(event.count),
                    "cuda_us_per_step": round(
                        event.device_time_total / PROFILER_STEPS, 3
                    ),
                    "calls_per_step": event.count / PROFILER_STEPS,
                }
            )
    rows.sort(key=lambda row: -row["cuda_us_per_step"])
    total = sum(row["cuda_us_per_step"] for row in rows)
    attention = [row for row in rows if "attn" in row["name"]]
    return {
        "total_cuda_us_per_step": round(total, 3),
        "attention_rows": attention,
        "rows": rows,
    }


def _agreement_profile(reference: list[int], candidate: list[int]) -> dict[str, float]:
    """Token agreement over the first 8 / 32 / all generated tokens."""

    shared = min(len(reference), len(candidate))

    def agreement(limit: int) -> float:
        span = min(shared, limit)
        matches = sum(
            1 for index in range(span) if reference[index] == candidate[index]
        )
        return round(matches / max(span, 1), 4)

    return {
        "first_8": agreement(8),
        "first_32": agreement(32),
        "all": agreement(shared),
    }


def main() -> None:
    torch.manual_seed(0)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "model_id": "openbmb/MiniCPM5-2B",
        "device": torch.cuda.get_device_name(0),
        "routes": [{"impl": impl, "splits": splits} for impl, splits in ROUTES],
        "steady_batch": STEADY_BATCH,
        "steady_repeats": STEADY_REPEATS,
        "order_flip_rounds": ORDER_FLIP_ROUNDS,
        "warmup_steps": WARMUP_STEPS,
        "profiler_steps": PROFILER_STEPS,
        "gpu_state_start": _gpu_state(),
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    input_ids = _render(tokenizer)
    prompt_tokens = int(input_ids.shape[1])
    print(f"[attn-ab] prompt tokens: {prompt_tokens}", flush=True)

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

    sessions = _build_sessions(quantized, w4_head)
    for session in sessions.values():
        session.capture_until(prompt_tokens + 2048)
    print(f"[attn-ab] {len(sessions)} sessions captured", flush=True)

    # Short greedy runs: agreement against the SIMT control. Greedy drift over
    # this prompt is ~0.01-0.04 even between same-precision runtime variants
    # (graph_decode_benchmark.json), so this is recorded, not gated.
    baseline_tokens: list[int] | None = None
    result["token_agreement"] = {}
    for name, session in sessions.items():
        session.prefill(input_ids)
        run = session.generate(
            input_ids, max_new_tokens=AGREEMENT_TOKENS, eos_token_ids=EOS_TOKEN_IDS
        )
        if baseline_tokens is None:
            baseline_tokens = run.token_ids
        result["token_agreement"][name] = _agreement_profile(
            baseline_tokens, run.token_ids
        )
    print(f"[attn-ab] token agreement: {result['token_agreement']}", flush=True)

    _prepare(sessions, input_ids)
    steady, states = _steady_interleaved(sessions)
    result["steady"] = steady
    result["gpu_state_per_round"] = states
    result["gpu_state_end"] = _gpu_state()
    control = _route_name(*ROUTES[0])
    result["ratios_vs_control"] = {
        name: round(
            steady[name]["step_ms_median"] / steady[control]["step_ms_median"], 4
        )
        for name in sessions
    }
    for name in sessions:
        print(
            f"[attn-ab] {name}: {steady[name]['steady_tok_s_median']:.1f} tok/s "
            f"({steady[name]['step_ms_median']:.3f} ms/step, min "
            f"{steady[name]['step_ms_min']:.3f}), "
            f"ratio={result['ratios_vs_control'][name]:.4f}",
            flush=True,
        )

    result["kernel_budget"] = {}
    for round_index in range(PROFILER_ROUNDS):
        for name, session in sessions.items():
            budget = _profile_step(session)
            result["kernel_budget"].setdefault(name, []).append(budget)
            print(
                f"[attn-ab] profiler round {round_index + 1} {name}: total "
                f"{budget['total_cuda_us_per_step']:.1f} us/step",
                flush=True,
            )
            for row in budget["attention_rows"]:
                print(
                    f"    {row['cuda_us_per_step']:8.3f} us/step x{row['calls_per_step']:6.1f}  "
                    f"{row['name'][:80]}",
                    flush=True,
                )

    OUTPUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[attn-ab] written {OUTPUT}", flush=True)
    del sessions, quantized
    _free_cuda()


if __name__ == "__main__":
    main()
