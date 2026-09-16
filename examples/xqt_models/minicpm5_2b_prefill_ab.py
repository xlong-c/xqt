"""Prefill attention backend A/B on MiniCPM5-2B: SDPA vs Triton vs TileLang.

R-056 recorded that the previous prefill A/B was a session-local ``/tmp``
script that was never committed. This is the committed real-route harness. It
answers one question: does routing prefill attention away from
``F.scaled_dot_product_attention`` actually change prompt-processing latency on
the real model, and do the backends compute the same thing?

Discipline follows ``minicpm5_2b_attn_ab.py``:

- one process, one quantized W4 model, every backend interleaved per round with
  a per-round alternating order (R-054: only same-run interleaved comparisons
  count; absolute rates drift across runs),
- ``torch.cuda.synchronize`` around every timed prefill, median of rounds,
- GPU clock/temperature sampled per round,
- first-token and full logit agreement plus greedy token agreement against the
  SDPA control, to prove the backends compute the same thing,
- a recorded, explicit reason when a backend cannot run on the real model.

``tilelang`` does not implement GQA, so MiniCPM5-2B (16 q heads / 2 kv heads)
is expected to be rejected at session construction; the script records that and
continues with ``sdpa`` vs ``triton``. Results are written to
``artifacts/xqt/inference/minicpm5-2b/prefill_ab.json``. Configuration is
code-level constants per workspace rules.
"""

from __future__ import annotations

import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoTokenizer

import examples.xqt_models.minicpm5_2b_translation_eval as ev
from xqt.core.errors import XQTBackendError
from xqt.model.minicpm5 import (
    load_minicpm5,
    materialize_minicpm5_weight_only_runtime,
    minicpm5_quantization_policy,
    quantize_minicpm5,
    quantize_minicpm5_lm_head_w4,
)
from xqt.runtime import CudaGraphDecodeSession

ARTIFACT_DIR = Path("artifacts/xqt/inference/minicpm5-2b")
OUTPUT = ARTIFACT_DIR / "prefill_ab.json"
MODEL_PATH = "downloads/MiniCPM5-2B-bf16"
DEVICE = torch.device("cuda")
MAX_CACHE_LEN = 4096
EOS_TOKEN_IDS = (1, 130073)

# Backends offered by CudaGraphDecodeSession.prefill_attention_impl.
BACKENDS: tuple[str, ...] = ("sdpa", "triton", "tilelang")
CONTROL = "sdpa"

# Prompt-processing lengths; the base prompt is 1715 tokens, so longer targets
# tile the same document. 4096 is MAX_CACHE_LEN.
PROMPT_LENGTHS: tuple[int, ...] = (128, 512, 1024, 2048, 4096)

# Interleaved steady measurement, order flipped every round (R-054).
ROUNDS = 7
ORDER_FLIP_ROUNDS = True
AGREEMENT_TOKENS = 16


def _free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


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


def _base_prompt_ids(tokenizer: Any) -> torch.Tensor:
    """The same 7976-char document the other MiniCPM5 scripts render."""

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ev._translation_prompt()}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, return_tensors="pt")["input_ids"].to(DEVICE)


def _prompt_for_length(base_ids: torch.Tensor, length: int) -> torch.Tensor:
    """Slice the base prompt, or tile it when the target exceeds its length."""

    base_len = int(base_ids.shape[1])
    if length <= base_len:
        return base_ids[:, :length].contiguous()
    repeats = math.ceil(length / base_len)
    return base_ids.repeat(1, repeats)[:, :length].contiguous()


def _logit_module(session: CudaGraphDecodeSession) -> torch.nn.Module:
    """The module ``_logits`` actually calls, so a forward hook sees logits."""

    override = getattr(session, "_lm_head_override", None)
    if override is not None:
        return cast(torch.nn.Module, override)
    return cast(torch.nn.Module, session.model.lm_head)


def _prefill_with_logits(
    session: CudaGraphDecodeSession, input_ids: torch.Tensor
) -> tuple[int, torch.Tensor]:
    """Run prefill and capture the final-position logits through a hook.

    Uses the public ``register_forward_hook`` API; no private method is called.
    """

    captured: dict[str, torch.Tensor] = {}

    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["logits"] = output.detach().float().reshape(-1).cpu()

    handle = _logit_module(session).register_forward_hook(hook)
    try:
        first = session.prefill(input_ids)
    finally:
        handle.remove()
    if "logits" not in captured:
        raise RuntimeError("prefill did not invoke the LM head; no logits captured")
    return first, captured["logits"]


def _build_sessions(
    quantized: Any, w4_head: Any
) -> tuple[dict[str, CudaGraphDecodeSession], dict[str, str]]:
    """Build one session per backend over the SAME quantized model.

    A backend that cannot serve this model's head configuration is recorded in
    ``failures`` with the exact reason instead of being silently dropped.
    """

    sessions: dict[str, CudaGraphDecodeSession] = {}
    failures: dict[str, str] = {}
    for impl in BACKENDS:
        try:
            sessions[impl] = CudaGraphDecodeSession(
                quantized,
                max_cache_len=MAX_CACHE_LEN,
                lm_head=w4_head,
                prefill_attention_impl=impl,
            )
        except (XQTBackendError, ValueError) as exc:
            failures[impl] = f"{type(exc).__name__}: {exc}"
    return sessions, failures


def _probe_backends(
    sessions: dict[str, CudaGraphDecodeSession],
    failures: dict[str, str],
    base_ids: torch.Tensor,
) -> dict[str, CudaGraphDecodeSession]:
    """Drop any backend whose prefill raises on the real model, with reason."""

    probe = base_ids[:, : min(128, int(base_ids.shape[1]))].contiguous()
    active: dict[str, CudaGraphDecodeSession] = {}
    for impl, session in sessions.items():
        try:
            session.prefill(probe)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - record any real-route failure
            failures[impl] = f"{type(exc).__name__}: {exc}"
            continue
        active[impl] = session
    return active


def _timed_prefill(session: CudaGraphDecodeSession, input_ids: torch.Tensor) -> float:
    """One synchronized prefill; returns milliseconds."""

    torch.cuda.synchronize()
    started = time.perf_counter()
    session.prefill(input_ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _measure_length(
    sessions: dict[str, CudaGraphDecodeSession],
    input_ids: torch.Tensor,
    rounds: int = ROUNDS,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Interleaved per-round prefill timing with a per-round order flip."""

    names = list(sessions)
    samples: dict[str, list[float]] = {name: [] for name in names}
    states: list[str] = []
    # Warm up every backend at this length so JIT/autotune is off the clock.
    for name in names:
        _timed_prefill(sessions[name], input_ids)
    for round_index in range(rounds):
        order = (
            names if (round_index % 2 == 0 or not ORDER_FLIP_ROUNDS) else names[::-1]
        )
        for name in order:
            samples[name].append(_timed_prefill(sessions[name], input_ids))
        states.append(_gpu_state())
    metrics: dict[str, dict[str, Any]] = {}
    for name in names:
        values = samples[name]
        median = statistics.median(values)
        metrics[name] = {
            "prefill_ms_median": median,
            "prefill_ms_min": min(values),
            "prefill_ms_max": max(values),
            "prefill_ms_all": [round(value, 4) for value in values],
            "prefill_us_per_token": median * 1000.0 / int(input_ids.shape[1]),
        }
    return metrics, states


def _token_agreement(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    """Greedy token agreement over the shared horizon."""

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


def _logit_agreement(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    """Numeric agreement of the final-position logits from two backends."""

    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    difference = (reference - candidate).abs()
    ref_norm = float(reference.norm())
    cand_norm = float(candidate.norm())
    cosine = float(torch.dot(reference, candidate) / (ref_norm * cand_norm + 1e-12))
    top_ref = torch.topk(reference, k=min(5, reference.numel())).indices.tolist()
    top_cand = torch.topk(candidate, k=min(5, candidate.numel())).indices.tolist()
    overlap = len(set(top_ref) & set(top_cand))
    return {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "cosine": cosine,
        "top1_match": bool(int(reference.argmax()) == int(candidate.argmax())),
        "top5_overlap": overlap,
    }


def main() -> None:
    torch.manual_seed(0)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "model_id": "openbmb/MiniCPM5-2B",
        "device": torch.cuda.get_device_name(0),
        "backends": list(BACKENDS),
        "control": CONTROL,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "rounds": ROUNDS,
        "order_flip_rounds": ORDER_FLIP_ROUNDS,
        "max_cache_len": MAX_CACHE_LEN,
        "gpu_state_start": _gpu_state(),
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base_ids = _base_prompt_ids(tokenizer)
    base_tokens = int(base_ids.shape[1])
    print(f"[prefill-ab] base prompt tokens: {base_tokens}", flush=True)

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

    sessions, failures = _build_sessions(quantized, w4_head)
    sessions = _probe_backends(sessions, failures, base_ids)
    result["backend_failures"] = failures
    if CONTROL not in sessions:
        raise RuntimeError(
            f"control backend {CONTROL!r} failed: {failures.get(CONTROL)}"
        )
    print(
        f"[prefill-ab] active backends: {list(sessions)}; failures: {failures}",
        flush=True,
    )

    # Per-length latency and numeric agreement against the control.
    per_length: dict[str, Any] = {}
    for length in PROMPT_LENGTHS:
        input_ids = _prompt_for_length(base_ids, length)
        metrics, states = _measure_length(sessions, input_ids)
        control_ms = metrics[CONTROL]["prefill_ms_median"]
        for name in metrics:
            metrics[name]["ratio_vs_control"] = round(
                metrics[name]["prefill_ms_median"] / control_ms, 4
            )
        reference_first, reference_logits = _prefill_with_logits(
            sessions[CONTROL], input_ids
        )
        agreement: dict[str, Any] = {}
        for name, session in sessions.items():
            first, logits = _prefill_with_logits(session, input_ids)
            agreement[name] = {
                "first_token": first,
                "first_token_matches_control": bool(first == reference_first),
                "logits_vs_control": _logit_agreement(reference_logits, logits),
            }
        # Short greedy agreement per length, where the budget allows. Greedy
        # argmax amplifies a near-tied logit gap into a token flip, so this is
        # recorded next to the logit agreement, not read as a correctness gate.
        if length + AGREEMENT_TOKENS <= MAX_CACHE_LEN:
            greedy: dict[str, Any] = {}
            reference_tokens = (
                sessions[CONTROL]
                .generate(
                    input_ids,
                    max_new_tokens=AGREEMENT_TOKENS,
                    eos_token_ids=EOS_TOKEN_IDS,
                )
                .token_ids
            )
            for name, session in sessions.items():
                tokens = session.generate(
                    input_ids,
                    max_new_tokens=AGREEMENT_TOKENS,
                    eos_token_ids=EOS_TOKEN_IDS,
                ).token_ids
                greedy[name] = _token_agreement(reference_tokens, tokens)
            per_length_greedy: dict[str, Any] = greedy
        else:
            per_length_greedy = {
                "skipped": (
                    f"length {length} + {AGREEMENT_TOKENS} agreement tokens "
                    f"exceeds max_cache_len {MAX_CACHE_LEN}"
                )
            }
        per_length[str(length)] = {
            "tokens": length,
            "metrics": metrics,
            "agreement_vs_control": agreement,
            "greedy_token_agreement_vs_control": per_length_greedy,
            "gpu_state_per_round": states,
        }
        summary = ", ".join(
            f"{name}={metrics[name]['prefill_ms_median']:.2f}ms "
            f"({metrics[name]['ratio_vs_control']:.3f}x)"
            for name in metrics
        )
        print(f"[prefill-ab] L={length}: {summary}", flush=True)
    result["per_length"] = per_length

    # Greedy token agreement at the real (untiled) prompt length. The control
    # is also generated twice: any run-to-run drift there is decode-path
    # nondeterminism, not a prefill-backend difference, and must be separated
    # before the cross-backend token agreement is read as a correctness signal.
    agreement_ids = _prompt_for_length(base_ids, base_tokens)

    def _greedy_tokens(session: CudaGraphDecodeSession) -> list[int]:
        return session.generate(
            agreement_ids, max_new_tokens=AGREEMENT_TOKENS, eos_token_ids=EOS_TOKEN_IDS
        ).token_ids

    reference_tokens = _greedy_tokens(sessions[CONTROL])
    result["token_agreement_vs_control"] = {}
    for name, session in sessions.items():
        result["token_agreement_vs_control"][name] = _token_agreement(
            reference_tokens, _greedy_tokens(session)
        )
    result["control_self_agreement"] = _token_agreement(
        reference_tokens, _greedy_tokens(sessions[CONTROL])
    )
    print(
        f"[prefill-ab] token agreement: {result['token_agreement_vs_control']}; "
        f"control self {result['control_self_agreement']}",
        flush=True,
    )

    result["gpu_state_end"] = _gpu_state()
    result["notes"] = [
        "One process, one quantized W4A16 model, all backends interleaved per round with a per-round order flip; median of rounds.",
        "Only the prefill attention implementation differs between sessions; the MLP and decode paths are shared and unchanged.",
        "Agreement is reported against the SDPA control as first-token match, final-position logit cosine/top-1/top-5 overlap, and short greedy token agreement.",
        "Longer-than-prompt targets tile the same 7976-char document; 4096 equals max_cache_len.",
    ]
    OUTPUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[prefill-ab] written {OUTPUT}", flush=True)
    del sessions, quantized
    _free_cuda()


if __name__ == "__main__":
    main()
