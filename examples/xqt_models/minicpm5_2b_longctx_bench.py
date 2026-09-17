"""Long-context English->Chinese spec-decode benchmark with full provenance.

Why this exists: every decode benchmark in this repo so far ran at a 1715-token
prompt with ``max_cache_len=4096``, and the one attempt to reach 4096 tokens
tiled a single 7976-character document. Neither says anything about 8k/16k
behaviour, and several numbers in the project roadmap were asserted rather than
measured. This script is the measurement base that closes that gap.

What it measures, per (document, route):

- ``ttft_ms``            device-synchronised ``session.prefill`` on a warmed
                         session; excludes tokenization and graph capture
- ``e2e_ms``             the full generation (prefill + decode), capture excluded
- ``generated_tokens``   includes the token prefill emits
- ``decode_tokens_per_second``  ``(generated_tokens - 1) / (decode_ms/1000)``
- ``steady_tokens_per_second``  ``32 / time(decode_batch(32))``, for continuity
                         with the recorded 347.5 tok/s baseline
- ``peak_vram_bytes`` / ``reserved_vram_bytes``
- spec-only: ``drafted``/``accepted``/``accept_rate``/``tau``/``round_ms``

It also records two things the roadmap got wrong, so they can be settled with
numbers instead of prose:

- ``round_profile``: the verify round cost as a function of cache position,
  measured from ONE captured graph (the graph is length-agnostic, so this is a
  measured curve rather than four re-captures).
- ``bandwidth``: weight and KV bytes per round from live tensors, divided by a
  peak-bandwidth figure measured in the same process. The roadmap's "87% of the
  physical wall" used 504 GB/s, which is the 192-bit RTX 4070 Ti figure; this
  board is 256-bit GDDR6X.

Microbenchmarks (``round_profile``, ``bandwidth``) are recorded for
*explanation only* and must never be reported as an end-to-end speedup.

Routes
------
``w4a8_kv8_graph``  the target baseline: true-INT8 activations + INT8 KV
``w4a8_kv0_graph``  true-INT8 activations + bf16 KV. This is the apples-to-apples
                    partner for the spec route, which can only read bf16 KV
                    (``decode_attention_rows_forward_triton`` takes no
                    k_scale/v_scale -- see ``SpecDecodeSession.__init__``)
``w4a8_spec_k5``    DSpark speculative decode, 8-row verify, bf16 KV

The KV-quantization asymmetry between the baseline and the spec route is real
and is recorded in ``notes`` rather than hidden; it favours the baseline, so the
spec speedup reported here is conservative.

Results go to ``artifacts/xqt/inference/minicpm5-2b/longctx_benchmark.json``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

import examples.xqt_models.minicpm5_2b_graph_decode as gd
from xqt.model.minicpm5 import (
    load_minicpm5,
    materialize_minicpm5_weight_only_runtime,
    minicpm5_quantization_policy,
    quantize_minicpm5,
    quantize_minicpm5_lm_head_w4,
)
from xqt.model.minicpm5_chat import render_translation_input_ids
from xqt.runtime import CudaGraphDecodeSession, DSparkDrafter, SpecDecodeSession

XQT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = XQT_ROOT.parent
XDL_ROOT = WORKSPACE_ROOT / "xdl"
MODEL_PATH = WORKSPACE_ROOT / "downloads" / "MiniCPM5-2B-bf16"
EVIDENCE_DIR = WORKSPACE_ROOT / "data" / "long_context_eval"
DSPARK_CKPT = WORKSPACE_ROOT / "artifacts" / "dspark_minicpm5_2b" / "dspark_drafter.pt"
ARTIFACT_DIR = WORKSPACE_ROOT / "artifacts" / "xqt" / "inference" / "minicpm5-2b"
OUTPUT = ARTIFACT_DIR / "longctx_benchmark.json"

DEVICE = torch.device("cuda")
MAX_CACHE_LEN = 16384
MAX_NEW_TOKENS = 2048
EOS_TOKEN_IDS = (1, 130073)
SPLIT_K = 5
DOCUMENTS = ("en_1715", "en_8192", "en_16384", "book_8192")
ROUND_CONTEXTS = (1715, 4096, 8192, 16384)
ROUND_REPEATS = 20
STEADY_BATCH = 32

NOTES = [
    "ttft_ms is device-synchronised prefill wall time on a warmed session; it "
    "excludes tokenization and CUDA-graph capture.",
    "e2e_ms is the whole generation including its own prefill, capture excluded.",
    "decode_tokens_per_second excludes the prefill-emitted token; "
    "steady_tokens_per_second is a 32-step replay with no sampling or readback "
    "and exists only for continuity with the recorded 347.5 tok/s figure.",
    "round_profile and bandwidth blocks are microbenchmarks kept for "
    "explanation; they are not end-to-end speedups.",
    "The baseline reads an INT8 KV cache while the spec route can only read "
    "bf16 KV, so the baseline is measured in its strongest configuration. The "
    "spec speedup is therefore conservative.",
    "Prompts are rendered by xqt.model.minicpm5_chat, which fixes a double-BOS "
    "bug in the pre-2026-09-17 scripts (the ChatML template already emits BOS "
    "and the tokenizer added a second one). The legacy prompt is 1715 tokens "
    "for the short document; the canonical one is 1714.",
    "minicpm5_2b_prefill_ab.py reaches 4096 tokens by tiling one document and "
    "is not long-context evidence; the documents here use distinct sources and "
    "the builder asserts it.",
]


def _free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except Exception:
        return None


def _command_line() -> str:
    """Reproduce the documented invocation.

    ``examples`` is an implicit namespace package, so the script must be run as
    a module from the repo root for ``examples.*`` imports to resolve. Recording
    the runnable form matters: a command that cannot be replayed is not evidence.
    """

    relative = Path(__file__).resolve().relative_to(XQT_ROOT).with_suffix("")
    module = ".".join(relative.parts)
    parts = [sys.executable, "-m", module] + sys.argv[1:]
    return " ".join(parts)


def _load_document(name: str) -> tuple[str, dict[str, Any]]:
    manifest = json.loads((EVIDENCE_DIR / "manifest.json").read_text("utf-8"))
    entries = {entry["name"]: entry for entry in manifest["documents"]}
    if name not in entries:
        raise SystemExit(f"{name} not in {EVIDENCE_DIR / 'manifest.json'}")
    entry = entries[name]
    text = (WORKSPACE_ROOT / entry["path"]).read_text("utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(
            f"{name}: sha256 mismatch (file {digest} vs manifest {entry['sha256']})"
        )
    if len(set(entry["source_ids"])) != len(entry["source_ids"]):
        raise SystemExit(f"{name}: duplicate source ids -- tiled document rejected")
    reference = ""
    if entry.get("reference_path"):
        reference = (WORKSPACE_ROOT / entry["reference_path"]).read_text("utf-8")
    return text, {**entry, "reference": reference}


def _build_model() -> tuple[Any, Any, Any]:
    """Build the W4A16 native runtime exactly as the recorded baseline did.

    The quantization calibration is copied verbatim from
    ``minicpm5_2b_spec_decode_bench.py`` so the weights are bit-identical to the
    ones behind the recorded 347.5 tok/s figure; any difference here would put
    the two artifacts on different models.
    """

    import examples.xqt_models.minicpm5_2b_translation_eval as ev

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    base = load_minicpm5(
        str(MODEL_PATH), dtype=torch.bfloat16, device="cpu", local_files_only=True
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


def _weight_bytes(model: torch.nn.Module) -> tuple[int, dict[str, int]]:
    total = 0
    parts: dict[str, int] = {}
    for name, module in model.named_modules():
        for attr in ("qweight", "weight_scale", "weight_zero_point", "qzeros", "scales"):
            tensor = getattr(module, attr, None)
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
                parts[attr] = parts.get(attr, 0) + tensor.numel() * tensor.element_size()
    return total, parts


def _measure_peak_bandwidth() -> float:
    """Measure device bandwidth in this process, so the denominator is not assumed."""

    size = 256 * 1024 * 1024
    src = torch.empty(size // 4, dtype=torch.float32, device=DEVICE)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    repeats = 20
    started = time.perf_counter()
    for _ in range(repeats):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # copy_ reads and writes, so the moved byte count is twice the buffer size.
    return (2 * src.numel() * src.element_size() * repeats) / elapsed / 1e9


def _timed_prefill(session: CudaGraphDecodeSession, input_ids: torch.Tensor) -> list[float]:
    times: list[float] = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        session.prefill(input_ids)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def _run_baseline(
    session: CudaGraphDecodeSession,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    session.capture()
    session.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_ids=EOS_TOKEN_IDS)

    ttft_ms = statistics.median(_timed_prefill(session, input_ids))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    started = time.perf_counter()
    first = session.prefill(input_ids)
    tokens: list[int] = [first]
    hit_eos = first in EOS_TOKEN_IDS
    while len(tokens) < max_new_tokens and not hit_eos:
        take = min(session._readback_chunk, max_new_tokens - len(tokens))
        for token in session.decode_batch(take).tolist():
            tokens.append(token)
            if token in EOS_TOKEN_IDS:
                hit_eos = True
                break
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - started) * 1000.0
    decode_seconds = max(e2e_ms - ttft_ms, 1e-9) / 1000.0

    steady_started = time.perf_counter()
    session.decode_batch(STEADY_BATCH)
    torch.cuda.synchronize()
    steady_ms = (time.perf_counter() - steady_started) * 1000.0

    return {
        "token_ids": tokens,
        "hit_eos": hit_eos,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "generated_tokens": len(tokens),
        "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds,
        "steady_tokens_per_second": STEADY_BATCH / (steady_ms / 1000.0),
        "steady_batch_ms": steady_ms,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(DEVICE),
        "reserved_vram_bytes": torch.cuda.max_memory_reserved(DEVICE),
    }


def _run_spec(
    session: CudaGraphDecodeSession,
    input_ids: torch.Tensor,
    drafter: DSparkDrafter,
    *,
    k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    spec = SpecDecodeSession(session, draft_tokens=k, drafter=drafter)
    spec.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_ids=EOS_TOKEN_IDS)

    ttft_ms = statistics.median(_timed_prefill(session, input_ids))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    started = time.perf_counter()
    tokens, stats = spec.generate(
        input_ids, max_new_tokens=max_new_tokens, eos_token_ids=EOS_TOKEN_IDS
    )
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - started) * 1000.0
    decode_seconds = max(e2e_ms - ttft_ms, 1e-9) / 1000.0

    steps = max(int(stats["verify_steps"]), 1)
    accepted = int(stats["accepted_draft_tokens"])
    drafted = int(stats["drafted_tokens"])
    return {
        "token_ids": tokens,
        "hit_eos": EOS_TOKEN_IDS[-1] in tokens,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "generated_tokens": len(tokens),
        "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds,
        "round_ms": (e2e_ms - ttft_ms) / steps,
        "spec": {
            "k": k,
            "rows": spec._rows,
            "verify_steps": steps,
            "drafted_tokens": drafted,
            "accepted_draft_tokens": accepted,
            "accept_rate": accepted / drafted if drafted else 0.0,
            "tau": 1.0 + accepted / steps,
            "mean_accepted_per_step": accepted / steps,
        },
        "peak_vram_bytes": torch.cuda.max_memory_allocated(DEVICE),
        "reserved_vram_bytes": torch.cuda.max_memory_reserved(DEVICE),
    }


def _round_profile(
    spec: SpecDecodeSession, contexts: tuple[int, ...]
) -> dict[str, Any]:
    """Verify-round cost vs cache position, from the single captured graph.

    Microbenchmark for explanation only: it replays the verify graph at a given
    cache position with no sampling and no readback, which is exactly how the
    roadmap's round-time claims should be read.
    """

    graph = spec._graphs[0]
    rows = spec._rows
    out: dict[str, Any] = {}
    for ctx in contexts:
        base = ctx - rows
        if base < 0:
            continue
        spec._pos.fill_(base)
        spec._valid_lens.copy_(spec._lens_offsets)
        spec._valid_lens.add_(base)
        torch.cuda.synchronize()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(ROUND_REPEATS):
            graph.replay()
        torch.cuda.synchronize()
        replay_ms = (time.perf_counter() - started) / ROUND_REPEATS * 1000.0

        started = time.perf_counter()
        for _ in range(ROUND_REPEATS):
            graph.replay()
            spec._logits_out.argmax(-1).tolist()
        torch.cuda.synchronize()
        replay_readback_ms = (time.perf_counter() - started) / ROUND_REPEATS * 1000.0

        out[str(ctx)] = {
            "replay_ms": replay_ms,
            "replay_plus_readback_ms": replay_readback_ms,
            "kv_bytes": 2 * spec._layers * spec._kv_heads * ctx * spec._head_dim * 2,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default=",".join(DOCUMENTS))
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-cache-len", type=int, default=MAX_CACHE_LEN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--skip-spec",
        action="store_true",
        help="run baselines only (useful when no drafter checkpoint exists)",
    )
    parser.add_argument(
        "--skip-round-profile",
        action="store_true",
        help="skip the T_round(ctx) microbenchmark",
    )
    args = parser.parse_args()

    documents = [name for name in args.documents.split(",") if name]
    for name in documents:
        if name not in DOCUMENTS:
            raise SystemExit(f"unknown document {name!r}; known: {list(DOCUMENTS)}")
    if args.max_cache_len < args.max_new_tokens + 64:
        raise SystemExit("max_cache_len must leave room for the generation")

    provenance = {
        "schema": "xqt.bench.v2",
        "command": _command_line(),
        "cwd": str(Path.cwd()),
        "script": str(Path(__file__).resolve().relative_to(XQT_ROOT)),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "xqt_commit": _git_commit(XQT_ROOT),
        "xdl_commit": _git_commit(XDL_ROOT),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": _triton_version(),
        "gpu": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
    }

    print(f"[longctx] building model from {MODEL_PATH} ...", flush=True)
    tokenizer, runtime, head = _build_model()
    _free_cuda()

    cfg = runtime.config
    hidden = int(cfg.hidden_size)
    layers = int(cfg.num_hidden_layers)
    q_heads = int(cfg.num_attention_heads)
    kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", hidden // q_heads))
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 2
    weight_bytes, weight_parts = _weight_bytes(runtime)
    head_bytes, _ = _weight_bytes(head)
    bandwidth_peak = _measure_peak_bandwidth()
    print(
        f"[longctx] weights={weight_bytes / 1e9:.3f} GB "
        f"lm_head={head_bytes / 1e9:.3f} GB "
        f"measured_peak_bw={bandwidth_peak:.0f} GB/s",
        flush=True,
    )

    result: dict[str, Any] = {
        "provenance": provenance,
        "config": {
            "documents": documents,
            "max_cache_len": args.max_cache_len,
            "max_new_tokens": args.max_new_tokens,
            "spec_k": SPLIT_K,
            "eos_token_ids": list(EOS_TOKEN_IDS),
            "steady_batch": STEADY_BATCH,
            "round_contexts": list(ROUND_CONTEXTS),
        },
        "model": {
            "weight_bytes": weight_bytes,
            "weight_bytes_by_tensor": weight_parts,
            "lm_head_bytes": head_bytes,
            "num_layers": layers,
            "num_q_heads": q_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden,
        },
        "bandwidth": {
            "measured_peak_bytes_per_second": bandwidth_peak * 1e9,
            "kv_bytes_per_token": kv_bytes_per_token,
        },
        "documents": {},
        "runs": {},
        "round_profile": {},
        "notes": list(NOTES),
    }
    baselines: dict[str, list[int]] = {}

    drafter = None
    if not args.skip_spec:
        if DSPARK_CKPT.exists():
            print(f"[longctx] loading drafter {DSPARK_CKPT}", flush=True)
            drafter = DSparkDrafter.from_checkpoint(
                DSPARK_CKPT, device=DEVICE, dtype=torch.bfloat16, conf_threshold=0.05
            )
        else:
            print(f"[longctx] no drafter at {DSPARK_CKPT}; spec route skipped", flush=True)

    input_ids_by_doc: dict[str, torch.Tensor] = {}
    for name in documents:
        document, entry = _load_document(name)
        ids = render_translation_input_ids(tokenizer, document)
        needed = len(ids) + args.max_new_tokens + 8
        if needed > args.max_cache_len:
            raise SystemExit(
                f"{name}: prompt {len(ids)} + {args.max_new_tokens} + rows 8 exceeds "
                f"max_cache_len {args.max_cache_len}"
            )
        input_ids_by_doc[name] = torch.tensor(
            ids, dtype=torch.long, device=DEVICE
        ).unsqueeze(0)
        result["documents"][name] = {
            key: entry[key]
            for key in (
                "sha256",
                "chars",
                "prompt_tokens",
                "source_kind",
                "source_count",
                "distinct_sources",
                "truncated_chars",
                "has_chinese_reference",
            )
        }
        result["documents"][name]["rendered_prompt_tokens"] = len(ids)
        print(f"[longctx] {name}: {len(ids)} prompt tokens", flush=True)

    spec_session = None
    for name in documents:
        input_ids = input_ids_by_doc[name]
        for route, kv_quant in (("w4a8_kv8_graph", "int8"), ("w4a8_kv0_graph", "none")):
            print(f"[longctx] {name} :: {route} ...", flush=True)
            session = CudaGraphDecodeSession(
                runtime,
                max_cache_len=args.max_cache_len,
                lm_head=head,
                int8_activations=True,
                kv_quant=kv_quant,
                attention_splits=16,
            )
            out = _run_baseline(
                session, input_ids, max_new_tokens=args.max_new_tokens
            )
            result["runs"][f"{name}::{route}"] = _finalize(out, baselines, name)
            if route == "w4a8_kv0_graph":
                spec_session = session
            else:
                del session
            _free_cuda()
            print(
                f"[longctx]   ttft={out['ttft_ms']:.1f}ms e2e={out['e2e_ms']:.0f}ms "
                f"decode={out['decode_tokens_per_second']:.1f} tok/s "
                f"steady={out['steady_tokens_per_second']:.1f} tok/s "
                f"peak={out['peak_vram_bytes'] / 1e9:.2f}GB",
                flush=True,
            )

        if drafter is not None and spec_session is not None:
            print(f"[longctx] {name} :: w4a8_spec_k{SPLIT_K} ...", flush=True)
            out = _run_spec(
                spec_session,
                input_ids,
                drafter,
                k=SPLIT_K,
                max_new_tokens=args.max_new_tokens,
            )
            result["runs"][f"{name}::w4a8_spec_k{SPLIT_K}"] = _finalize(
                out, baselines, name
            )
            print(
                f"[longctx]   ttft={out['ttft_ms']:.1f}ms e2e={out['e2e_ms']:.0f}ms "
                f"decode={out['decode_tokens_per_second']:.1f} tok/s "
                f"tau={out['spec']['tau']:.3f} acc={out['spec']['accept_rate']:.2%} "
                f"round={out['round_ms']:.3f}ms",
                flush=True,
            )
        del spec_session
        spec_session = None
        _free_cuda()

    if not args.skip_round_profile and drafter is not None:
        print("[longctx] round profile ...", flush=True)
        profile_doc = documents[0]
        session = CudaGraphDecodeSession(
            runtime,
            max_cache_len=args.max_cache_len,
            lm_head=head,
            int8_activations=True,
            kv_quant="none",
            attention_splits=16,
        )
        probe = SpecDecodeSession(session, draft_tokens=SPLIT_K, drafter=drafter)
        probe.generate(
            input_ids_by_doc[profile_doc],
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=EOS_TOKEN_IDS,
        )
        result["round_profile"] = {
            "document": profile_doc,
            "rows": probe._rows,
            "repeats": ROUND_REPEATS,
            "contexts": _round_profile(probe, ROUND_CONTEXTS),
            "weight_bytes_per_round": weight_bytes + head_bytes,
        }
        for ctx, entry in result["round_profile"]["contexts"].items():
            total = weight_bytes + head_bytes + entry["kv_bytes"]
            entry["total_bytes"] = total
            entry["implied_gigabytes_per_second"] = (
                total / (entry["replay_ms"] / 1000.0) / 1e9
            )
    _free_cuda()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[longctx] wrote {args.output}", flush=True)


def _finalize(
    out: dict[str, Any], baselines: dict[str, list[int]], name: str
) -> dict[str, Any]:
    """Attach token-level agreement against the first route run for this document."""

    record = {key: value for key, value in out.items() if key != "token_ids"}
    tokens = out["token_ids"]
    reference = baselines.get(name)
    if reference is None:
        baselines[name] = tokens
    else:
        record.update(gd._token_agreement(reference, tokens))
    record["output_sha256"] = hashlib.sha256(
        ",".join(str(token) for token in tokens).encode("ascii")
    ).hexdigest()
    return record


if __name__ == "__main__":
    main()
