"""MiniCPM5-2B English-to-Chinese document translation end-to-end evaluation.

This example extends the MiniCPM5-2B model-side experiments with a real task
chain: one ~8000-character English document is translated with greedy decoding
under eight precision/runtime routes, and each route is measured from
tokenization through prefill/decode to detokenization.

The document is translated in one greedy generation per candidate, and each
candidate is measured from chat-template rendering through prefill/decode to
detokenization. The prompt must be rendered with ``apply_chat_template``:
feeding the raw instruction text straight into the tokenizer drops the model
into continuation mode, which produces English continuations with heavy
n-gram repetition instead of a translation (measured, not assumed).

Candidates:
- ``bf16``: unquantized baseline and quality reference.
- ``awq_w4_g64_dense`` / ``awq_w4_g64_native``: one AWQ W4A16 g64 checkpoint
  measured through the dense-cache view and the native SM89 GEMV hybrid view.
- ``w8a8_int8``: dynamic W8A8 runtime view (``engine="auto"``).
- ``awq_w4_mlp_only_native`` / ``awq_w4_edges2_native``: mixed-precision AWQ
  W4A16 policies through the native hybrid view.
- ``convrot_w8a8_int8``: ConvRot online Hadamard rotation + dynamic per-token
  W8A8 INT8 through the execution view (native SM89 path when shape-supported).
- ``quarot_w8a8_int8``: offline QuaRot hidden-coordinate rotation + W8A8 INT8.
  On sm_89 INT8 is the strongest low-precision hardware path, so these two
  rotation routes are the primary candidates; the W4A16 routes stay for
  context.

Quality is reported as agreement against the BF16 translation (character-level
BLEU/ROUGE-L plus sequence ratios). There is no human reference translation, so
these numbers describe quantization fidelity, not absolute translation quality.

Configuration is code-level constants; results are written to
``artifacts/xqt/inference/minicpm5-2b/translation_eval.json`` and a side-by-side
Markdown file. Transformers owns tokenization and generation; XQT owns
quantization, runtime routing, and the benchmark wrappers.
"""

from __future__ import annotations

import difflib
import gc
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from xdl.metric.text import BLEUScore, ROUGELScore
from xqt.compression.quant import quantize_with_convrot_int8
from xqt.model.minicpm5 import (
    MINICPM5_2B_REPO_ID,
    load_minicpm5,
    materialize_minicpm5_weight_only_runtime,
    minicpm5_edge_protected_quantization_policy,
    minicpm5_mlp_only_quantization_policy,
    minicpm5_quantization_policy,
    quantize_minicpm5,
)
from xqt.model.minicpm5_quarot import apply_quarot_minicpm5
from xqt.runtime import materialize_convrot_execution_views
from examples.xqt_models.minicpm5_2b_quant_benchmark import (
    CALIBRATION_PROMPTS,
    DEVICE,
    _inputs,
    _source,
)

MODEL_DIR = Path("downloads/MiniCPM5-2B-bf16")
ARTIFACT_DIR = Path("artifacts/xqt/inference/minicpm5-2b")
JSON_OUTPUT = ARTIFACT_DIR / "translation_eval.json"
MARKDOWN_OUTPUT = ARTIFACT_DIR / "translation_eval.md"

MAX_NEW_TOKENS = 2048
WARMUP_NEW_TOKENS = 8
EOS_TOKEN_IDS = (1, 130073)
INPUT_KEYS = frozenset({"input_ids", "attention_mask"})

# The prompt is rendered through the model's ChatML template with
# enable_thinking=False; the BF16 translation of the document is about 1500
# tokens, so MAX_NEW_TOKENS leaves headroom for the quantized candidates.
INSTRUCTION = (
    "Translate the following English text into Chinese. "
    "Output only the translation, preserving technical terms when appropriate.\n\n"
)

# A self-authored, neutral multi-domain English document (narrative, technical,
# argumentative, procedural, travel, Q&A, reflection) used as the translation
# input. Character count is asserted in ``main``.
SOURCE_DOCUMENT = """## The Harbor at First Light

The fishing boats came back just before dawn, their engines ticking as they slid past the breakwater. On the pier, Marta counted the crates as they came off the *Estrella*: forty-two of ice, eleven of rope, and one small wooden box that nobody claimed. The gulls had already learned the schedule. They circled above the sorting tables in loose gray spirals, waiting for the scraps that would come. By six o'clock the auction would begin, and the price of hake would be set for the entire coast. For now, the harbor belonged to the quiet: to the men coiling lines, to the woman writing numbers in a salt-stained ledger, and to the fog that still clung to the hills above the town like a blanket nobody wanted to fold.

Her father had fished these waters for thirty-one years, and he had taught her three rules. Never argue with the weather. Never sell the first crate before you have smelled the second. And never, under any circumstances, trust a man who is cheerful at four in the morning. Marta had broken the third rule only once, and the memory of that mistake still made her smile.

## How a Search Index Remembers

A search index is not a library. A library keeps books on shelves and hopes that readers will find them; an index keeps a map of every word that ever appeared in every book, and it keeps that map in a form that a machine can read in milliseconds. When you type a query, the engine does not read the documents. It reads the map.

The oldest version of this idea is called an inverted index. Imagine a notebook with one page for each distinct word. On the page for "harbor" you would find a list: document 12, page 3; document 47, page 91; document 108, page 6. To answer a query, the engine walks the pages for each word and intersects the lists. If a word appears in ten million documents, its list is ten million entries long, and the intersection can still finish in under a second, because the lists are sorted and the machine only moves forward.

Modern systems add two refinements. First, they compress the lists, because a sorted list of integers is full of patterns, and patterns can be stored in fewer bytes. Second, they score the results, because a match is not the same thing as an answer. A document in which "harbor" appears once in a footnote is probably less useful than a document in which it appears in the title. The scoring function is a guess about human attention, and like every guess, it is sometimes wrong.

## In Defense of Unfinished Projects

We are taught from childhood that finishing matters. Clean your plate. Complete your homework. Cross the last item off the list. The lesson is useful, but it is also incomplete, and the older I get, the more convinced I am that the unfinished project deserves a better reputation than it has.

Consider what an unfinished thing actually is. It is a place where curiosity has not yet been replaced by obligation. The half-built bookshelf in my garage has taught me more about wood grain than any completed cabinet ever did, because I was still experimenting when I built it, and experiments are allowed to fail. A finished project has to defend itself against every critic. An unfinished one is still asking questions.

This is not a defense of laziness. There is a difference between a project that is unfinished because it is still alive and one that is unfinished because it is dead. The first kind keeps a notebook. It accumulates sketches, corrections, and small discoveries. The second kind accumulates dust. The trick is to tell them apart early, and to be honest about which one you are feeding.

## A Field Guide to Winter Pruning

Late winter is the best time to prune most deciduous trees, because the tree is dormant and the shape of its skeleton is visible. Work on a dry day when the temperature is above minus five degrees Celsius; frozen wood splinters, and wet wood invites disease.

Begin by removing the three D's: dead, damaged, and diseased branches. Cut them back to healthy wood, leaving the branch collar intact. Do not paint the wounds. A clean cut heals faster in open air than under a layer of tar.

Next, address the structure. Look for branches that cross, rub, or grow inward toward the trunk. Choose the stronger of any two rivals and remove the other completely, rather than shortening it. A stub of ten centimeters will die back and become an entry point for rot, while a full removal lets the tree seal the wound with new growth.

Finally, step back. Walk around the tree, crouch, and look up. A good pruning job looks as if nothing was done at all, only tidier, like a room that has been straightened rather than rearranged. If you can see every cut from the street, you have probably taken too much. Limit yourself to removing a quarter of the living canopy in a single season, and let the tree spend its spring on the branches that remain.

## Notes from the Night Train

The sleeper car smelled of diesel, oranges, and clean linen. I had the upper berth, which meant I could watch the landscape without being watched in return. Somewhere after midnight we crossed a river so wide that the clicking of the rails softened for nearly a minute, and I imagined the water below us, black and patient.

At three in the morning the train stopped in a town whose name I never learned. A woman boarded with two baskets and a sleeping child. She sat across the aisle, arranged the baskets with great care, and fell asleep sitting perfectly upright, the way only parents can. When I woke again, the child was awake and watching me through the gap between the seats. We studied each other for a long time. Then he smiled, and I smiled back, and neither of us said a word, because there was no language we shared except the obvious one.

In the morning the porter brought tea in a glass with a metal holder. The tea was too hot and too sweet, and it was exactly right.

## Six Questions About Sleep

Why do we dream? Nobody knows for certain. The leading theories say that dreams help us rehearse threats, sort memories, or simply keep the visual system busy while the body repairs itself. All three may be true on different nights.

How much sleep do adults need? Most studies point to seven to nine hours, but the range is wide. The better question is whether you wake up without an alarm and stay alert through the afternoon.

Is caffeine in the evening always a bad idea? Caffeine leaves the body slowly, with a half-life of roughly five hours. A cup at four in the afternoon still leaves a quarter of its caffeine in your blood at midnight.

Should you exercise before bed? Moderate exercise usually improves sleep, as long as it ends two or three hours before you lie down. Intense training late at night can push the body into a state of alertness that takes hours to fade.

Why do some people wake at three in the morning? The sleep cycle naturally lightens around that hour, and any unresolved worry finds the opening. Writing the worry down on paper helps more than most people expect.

What about naps? A nap of twenty minutes restores attention without leaving you groggy. Anything longer risks the strange fog that settles in when you wake from deep sleep in the middle of an afternoon.

## What the Garden Taught Us

The garden does not care about our schedules. It cares about light, water, and time, in amounts that we can measure but never fully control. Every spring we plant more than we can eat, and every autumn we learn the same lesson: abundance is not the same as success. The zucchini that grew larger than a forearm, the tomatoes that split after a sudden rain, the herbs that went to flower before we remembered to harvest them, all of them were generous in ways we did not plan for.

Perhaps that is why people keep gardens, even when the supermarket is closer and cheaper. A garden is a conversation with something that does not speak our language, and every season it answers in a vocabulary of leaves."""


class _DecodeTimeline(StoppingCriteria):
    """Record time to the first decoded token and the number of decode steps."""

    def __init__(self) -> None:
        self.first_token_time: float | None = None
        self.steps = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.steps += 1
        return False


def _free_cuda() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def _translation_prompt() -> str:
    return INSTRUCTION + SOURCE_DOCUMENT


def _generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run one greedy generation and decompose the full pipeline latency."""

    started = time.perf_counter()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    device_inputs = {
        key: value.to(DEVICE) for key, value in encoded.items() if key in INPUT_KEYS
    }
    prepared = time.perf_counter()
    prompt_tokens = int(device_inputs["input_ids"].shape[-1])
    timeline = _DecodeTimeline()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
    generate_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **device_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=list(EOS_TOKEN_IDS),
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([timeline]),
        )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    generate_finished = time.perf_counter()
    generated_ids = [int(token) for token in generated[0, prompt_tokens:].tolist()]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finished = time.perf_counter()

    first_token_time = timeline.first_token_time or generate_finished
    decode_seconds = max(generate_finished - first_token_time, 1e-9)
    speed = {
        "tokenize_transfer_ms": (prepared - started) * 1000.0,
        "generate_ms": (generate_finished - generate_started) * 1000.0,
        "ttft_ms": (first_token_time - generate_started) * 1000.0,
        "decode_ms": (generate_finished - first_token_time) * 1000.0,
        "detokenize_ms": (finished - generate_finished) * 1000.0,
        "e2e_ms": (finished - started) * 1000.0,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_ids),
        "decode_steps": timeline.steps,
        "decode_tokens_per_second": timeline.steps / decode_seconds,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(DEVICE))
            if DEVICE.type == "cuda"
            else 0
        ),
    }
    return {
        "speed": speed,
        "text": text,
        "token_ids": generated_ids,
        "finished_with_eos": bool(generated_ids and generated_ids[-1] in EOS_TOKEN_IDS),
        "truncated": len(generated_ids) >= max_new_tokens
        and not (generated_ids and generated_ids[-1] in EOS_TOKEN_IDS),
    }


def _weight_bytes(model: torch.nn.Module) -> int:
    parameters = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffers = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return int(parameters + buffers)


def _execution_counts(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in model.modules():
        metadata_fn = getattr(module, "execution_metadata", None)
        if not callable(metadata_fn):
            continue
        metadata = cast(dict[str, Any], metadata_fn())
        name = str(
            metadata.get("engine")
            or metadata.get("decode_engine")
            or metadata.get("implementation")
            or "unknown"
        )
        counts[name] = counts.get(name, 0) + 1
    return counts


def _curated_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "implementation",
        "bits",
        "group_size",
        "calibrated_module_count",
        "calibration_sample_count",
        "quantization_nature",
        "activation_encoding",
        "weight_encoding",
        "engine_preference",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _quality_metrics(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Character-level agreement against the BF16 translation."""

    reference_text = str(reference["text"])
    candidate_text = str(candidate["text"])
    reference_chars = list(reference_text)
    candidate_chars = list(candidate_text)
    return {
        "exact_match": bool(reference_text == candidate_text),
        "bleu2_char": float(BLEUScore(max_n=2)([candidate_chars], [reference_chars])),
        "bleu4_char": float(BLEUScore(max_n=4)([candidate_chars], [reference_chars])),
        "rouge_l_char": float(ROUGELScore()([candidate_chars], [reference_chars])),
        "char_sequence_ratio": float(
            difflib.SequenceMatcher(
                None, reference_chars, candidate_chars, autojunk=False
            ).ratio()
        ),
        "token_sequence_ratio": float(
            difflib.SequenceMatcher(
                None, list(reference["token_ids"]), list(candidate["token_ids"]),
                autojunk=False,
            ).ratio()
        ),
        "char_length_ratio": float(
            len(candidate_chars) / max(len(reference_chars), 1)
        ),
    }


def _run_candidate(
    name: str,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    quantization: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Measure one candidate with a short warmup plus one full generation."""

    model = model.eval().to(DEVICE)
    _generate_once(model, tokenizer, prompt, max_new_tokens=WARMUP_NEW_TOKENS)
    measured = _generate_once(
        model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS
    )
    report: dict[str, Any] = {
        "name": name,
        "status": "ok",
        "quantization": dict(quantization),
        "weight_bytes": _weight_bytes(model),
        "runtime_execution": _execution_counts(model),
        "speed": measured["speed"],
        "output": {
            "text": measured["text"],
            "token_ids": measured["token_ids"],
            "token_count": len(measured["token_ids"]),
            "finished_with_eos": measured["finished_with_eos"],
            "truncated": measured["truncated"],
        },
    }
    if reference is None:
        report["quality_vs_bf16"] = {"exact_match": True, "note": "reference"}
    else:
        report["quality_vs_bf16"] = _quality_metrics(reference, measured)
    return report


def _failed_report(name: str, error: Exception) -> dict[str, Any]:
    return {"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}"}


def _write_markdown(result: Mapping[str, Any]) -> None:
    lines: list[str] = [
        "# MiniCPM5-2B English-to-Chinese translation end-to-end evaluation",
        "",
        f"- Document: {result['document']['characters']} characters, "
        f"{result['document']['section_count']} sections, "
        f"sha256 `{result['document']['sha256'][:12]}`",
        f"- Pipeline: one greedy generation, max_new_tokens={MAX_NEW_TOKENS}, "
        f"prompt tokens={result['reference_speed']['prompt_tokens']}",
        "- Quality reference: the BF16 translation (agreement, not absolute quality)",
        "",
        "## Speed and agreement",
        "",
        "| candidate | weight MB | e2e ms | TTFT ms | decode tok/s | BLEU-4 (char) | "
        "ROUGE-L (char) | token ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for report in result["candidates"]:
        if report.get("status") != "ok":
            lines.append(
                f"| {report['name']} | failed: {report.get('error', '')} "
                "| | | | | | |"
            )
            continue
        speed = report["speed"]
        quality = report["quality_vs_bf16"]
        lines.append(
            f"| {report['name']} | {report['weight_bytes'] / 1e6:.0f} "
            f"| {speed['e2e_ms']:.0f} | {speed['ttft_ms']:.0f} "
            f"| {speed['decode_tokens_per_second']:.1f} "
            f"| {quality.get('bleu4_char', 1.0):.3f} "
            f"| {quality.get('rouge_l_char', 1.0):.3f} "
            f"| {quality.get('token_sequence_ratio', 1.0):.3f} |"
        )
    lines.extend(["", "## Source document", "", SOURCE_DOCUMENT, ""])
    for report in result["candidates"]:
        lines.extend(
            [
                f"## Translation: {report['name']}",
                "",
                report.get("output", {}).get("text", "(failed)"),
                "",
            ]
        )
    MARKDOWN_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def _build_result(
    source: str,
    dtype: torch.dtype,
    document_chars: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the report dict; also used for incremental JSON flushes."""

    return {
        "model_id": MINICPM5_2B_REPO_ID,
        "source": source,
        "device": str(DEVICE),
        "dtype": str(dtype),
        "document": {
            "characters": document_chars,
            "section_count": SOURCE_DOCUMENT.count("## "),
            "sha256": hashlib.sha256(SOURCE_DOCUMENT.encode("utf-8")).hexdigest(),
        },
        "pipeline": {
            "instruction": INSTRUCTION,
            "generation": "one greedy generation over the full document",
            "chat_template": "model default, enable_thinking=False",
            "enable_thinking": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        },
        "candidates": candidates,
        "limitations": [
            "There is no human reference translation; quality is agreement against the BF16 translation, not an absolute translation score.",
            "BLEU/ROUGE-L are computed over character sequences because Chinese has no whitespace token boundaries here.",
            "One document and one measured run per candidate; greedy decoding keeps the outputs deterministic.",
            "The prompt must be rendered with the model chat template; raw-text tokenization drops the model into continuation mode and produces English continuations with heavy repetition.",
            "The W8A8 INT8 routes use the current XQT runtime views; the realized engine is recorded in runtime_execution and no native INT8 speedup is claimed.",
            "The ConvRot W8A8 sm_89 execution view uses its small-M native path at decode; the small-M kernel is used, not merely the M-padded prefill path.",
            "The QuaRot transform is the offline hidden-coordinate rotation only; online KV/value and MLP-down Hadamard nodes are disabled.",
            "The mixed-precision candidates quantize fewer modules and are not full-model 4-bit formats.",
        ],
    }


def main() -> None:
    """Run all candidates sequentially and write the JSON/Markdown reports."""

    document_chars = len(SOURCE_DOCUMENT)
    if not 7600 <= document_chars <= 8400:
        raise ValueError(f"source document must be about 8000 chars, got {document_chars}")
    source = _source()
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=source != MINICPM5_2B_REPO_ID
    )
    prompt = _translation_prompt()
    calibration_inputs = [
        {key: value.cpu() for key, value in _inputs(tokenizer, text).items()}
        for text in CALIBRATION_PROMPTS
    ]
    dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    base_model = load_minicpm5(
        source, dtype=dtype, device="cpu", local_files_only=True
    )

    print(
        f"[translation-eval] {document_chars} chars, {SOURCE_DOCUMENT.count('## ')} sections",
        flush=True,
    )

    def _flush() -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        JSON_OUTPUT.write_text(
            json.dumps(
                _build_result(source, dtype, document_chars, candidates),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("[translation-eval] running bf16 ...", flush=True)
    candidates: list[dict[str, Any]] = []
    bf16_report = _run_candidate(
        "bf16",
        base_model,
        tokenizer,
        prompt,
        quantization={"strategy": "bf16"},
        reference=None,
    )
    candidates.append(bf16_report)
    reference: dict[str, Any] = {
        "text": bf16_report["output"]["text"],
        "token_ids": bf16_report["output"]["token_ids"],
    }
    print(
        f"[translation-eval] bf16: e2e={bf16_report['speed']['e2e_ms']:.0f}ms "
        f"decode={bf16_report['speed']['decode_tokens_per_second']:.1f}tok/s "
        f"generated={bf16_report['speed']['generated_tokens']} "
        f"eos={bf16_report['output']['finished_with_eos']}",
        flush=True,
    )
    _flush()
    gc.collect()
    base_model = base_model.to("cpu")
    _free_cuda()

    def _attempt(
        name: str,
        built: torch.nn.Module,
        quantization: Mapping[str, Any],
    ) -> None:
        print(f"[translation-eval] running {name} ...", flush=True)
        try:
            report = _run_candidate(
                name,
                built,
                tokenizer,
                prompt,
                quantization=quantization,
                reference=reference,
            )
            candidates.append(report)
            print(
                f"[translation-eval] {name}: e2e={report['speed']['e2e_ms']:.0f}ms "
                f"decode={report['speed']['decode_tokens_per_second']:.1f}tok/s "
                f"bleu4={report['quality_vs_bf16']['bleu4_char']:.3f} "
                f"eos={report['output']['finished_with_eos']} "
                f"truncated={report['output']['truncated']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep the sweep running
            candidates.append(_failed_report(name, exc))
            print(
                f"[translation-eval] {name}: FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            _flush()
            _free_cuda()

    # One AWQ W4A16 g64 checkpoint, measured through both runtime views.
    try:
        print("[translation-eval] quantizing awq_w4_g64 ...", flush=True)
        quantized = quantize_minicpm5(
            base_model,
            strategy="w4a16_int4",
            policy=minicpm5_quantization_policy(),
            calibration_inputs=calibration_inputs,
            inplace=False,
            group_size=64,
        )
    except Exception as exc:  # noqa: BLE001 - record both views as failed
        for failed_name in ("awq_w4_g64_dense", "awq_w4_g64_native"):
            candidates.append(_failed_report(failed_name, exc))
    else:
        awq_quantization: dict[str, Any] = {
            "strategy": "w4a16_int4",
            "method": "awq",
            "group_size": 64,
            "quantized_module_count": len(quantized.quantized_modules),
            "metadata": _curated_metadata(quantized.metadata),
        }
        dense_view = materialize_minicpm5_weight_only_runtime(
            quantized.model, engine="dense"
        )
        _attempt("awq_w4_g64_dense", dense_view, awq_quantization)
        # The native decode view must be materialized while the weights are
        # CUDA-resident (SM89 prepack requirement); _attempt leaves the model
        # on the device.
        native_view = materialize_minicpm5_weight_only_runtime(dense_view, engine="cuda")
        _attempt("awq_w4_g64_native", native_view, awq_quantization)
        native_view.to("cpu")
        del native_view, dense_view, quantized
        _free_cuda()

    # W8A8 INT8 runtime view.
    try:
        print("[translation-eval] quantizing w8a8_int8 ...", flush=True)
        int8 = quantize_minicpm5(
            base_model,
            strategy="w8a8_int8",
            policy=minicpm5_quantization_policy(),
            inplace=False,
            engine="auto",
            materialize_runtime=True,
        )
    except Exception as exc:  # noqa: BLE001 - record the candidate as failed
        candidates.append(_failed_report("w8a8_int8", exc))
    else:
        _attempt(
            "w8a8_int8",
            int8.model,
            {
                "strategy": "w8a8_int8",
                "quantized_module_count": len(int8.quantized_modules),
                "metadata": _curated_metadata(int8.metadata),
            },
        )
        int8.model.to("cpu")
        del int8
        _free_cuda()

    # Mixed-precision AWQ W4A16 policies through the native hybrid view.
    try:
        print("[translation-eval] quantizing awq_w4_mlp_only ...", flush=True)
        mlp_only = quantize_minicpm5(
            base_model,
            strategy="w4a16_int4",
            policy=minicpm5_mlp_only_quantization_policy(),
            calibration_inputs=calibration_inputs,
            inplace=False,
            group_size=64,
        )
    except Exception as exc:  # noqa: BLE001 - record the candidate as failed
        candidates.append(_failed_report("awq_w4_mlp_only_native", exc))
    else:
        # Materialize on CUDA: the SM89 AWQ prepack requires CUDA-resident
        # weights.
        mlp_only_model = materialize_minicpm5_weight_only_runtime(
            mlp_only.model.to(DEVICE), engine="cuda"
        )
        _attempt(
            "awq_w4_mlp_only_native",
            mlp_only_model,
            {
                "strategy": "w4a16_int4",
                "method": "awq",
                "group_size": 64,
                "policy": "mlp_only",
                "quantized_module_count": len(mlp_only.quantized_modules),
                "metadata": _curated_metadata(mlp_only.metadata),
            },
        )
        mlp_only_model.to("cpu")
        del mlp_only_model, mlp_only
        _free_cuda()

    try:
        print("[translation-eval] quantizing awq_w4_edges2 ...", flush=True)
        edge_protected = quantize_minicpm5(
            base_model,
            strategy="w4a16_int4",
            policy=minicpm5_edge_protected_quantization_policy(edge_layers=2),
            calibration_inputs=calibration_inputs,
            inplace=False,
            group_size=64,
        )
    except Exception as exc:  # noqa: BLE001 - record the candidate as failed
        candidates.append(_failed_report("awq_w4_edges2_native", exc))
    else:
        edge_protected_model = materialize_minicpm5_weight_only_runtime(
            edge_protected.model.to(DEVICE), engine="cuda"
        )
        _attempt(
            "awq_w4_edges2_native",
            edge_protected_model,
            {
                "strategy": "w4a16_int4",
                "method": "awq",
                "group_size": 64,
                "policy": "edge_protected_edges2",
                "quantized_module_count": len(edge_protected.quantized_modules),
                "metadata": _curated_metadata(edge_protected.metadata),
            },
        )
        edge_protected_model.to("cpu")
        del edge_protected_model, edge_protected
        _free_cuda()

    # ConvRot W8A8 INT8: online Hadamard rotation + dynamic per-token activation
    # quant. On sm_89 this is the native ConvRot path when the execution view is
    # materialized; INT8 is the strongest low-precision route on this GPU.
    try:
        print("[translation-eval] quantizing convrot_w8a8_int8 ...", flush=True)
        convrot = quantize_with_convrot_int8(
            base_model,
            policy={
                "dtype": "int8",
                "scheme": "convrot_w8a8",
                "include_module_types": ["Linear"],
                "exclude_module_types": ["LayerNorm", "Embedding"],
                "exclude_name_patterns": [r"(^|\.)lm_head$"],
                "rot_size": 256,
                "activation_scale_mode": "dynamic",
            },
            calibration_inputs=calibration_inputs,
            inplace=False,
            engine="auto",
            fallback_engine="torch_int_mm",
        )
        convrot_model = materialize_convrot_execution_views(convrot.model, inplace=True)
    except Exception as exc:  # noqa: BLE001 - record the candidate as failed
        candidates.append(_failed_report("convrot_w8a8_int8", exc))
    else:
        _attempt(
            "convrot_w8a8_int8",
            convrot_model,
            {
                "strategy": "w8a8_int8",
                "method": "convrot",
                "rot_size": 256,
                "quantized_module_count": len(convrot.quantized_modules),
                "metadata": _curated_metadata(convrot.metadata),
            },
        )
        convrot_model.to("cpu")
        del convrot_model, convrot
        _free_cuda()

    # QuaRot offline hidden-coordinate rotation + W8A8 INT8. The rotation
    # mutates the base model in place, so it runs last.
    try:
        print("[translation-eval] rotating + quantizing quarot_w8a8_int8 ...", flush=True)
        transform_report = apply_quarot_minicpm5(base_model, seed=42)
        quarot = quantize_minicpm5(
            base_model,
            strategy="w8a8_int8",
            policy=minicpm5_quantization_policy(),
            inplace=True,
            engine="auto",
            materialize_runtime=True,
        )
    except Exception as exc:  # noqa: BLE001 - record the candidate as failed
        candidates.append(_failed_report("quarot_w8a8_int8", exc))
    else:
        _attempt(
            "quarot_w8a8_int8",
            quarot.model,
            {
                "strategy": "w8a8_int8",
                "method": "quarot_offline_rotation",
                "transform": transform_report.to_dict(),
                "quantized_module_count": len(quarot.quantized_modules),
                "metadata": _curated_metadata(quarot.metadata),
            },
        )
        quarot.model.to("cpu")
        del quarot
        _free_cuda()

    result = _build_result(source, dtype, document_chars, candidates)
    result["reference_speed"] = candidates[0]["speed"]
    _flush()
    _write_markdown(result)
    summary = [
        {
            "name": report["name"],
            "status": report["status"],
            **(
                {
                    "e2e_ms": round(report["speed"]["e2e_ms"], 1),
                    "decode_tps": round(report["speed"]["decode_tokens_per_second"], 2),
                    "bleu4_char": report["quality_vs_bf16"].get("bleu4_char"),
                    "rouge_l_char": report["quality_vs_bf16"].get("rouge_l_char"),
                    "generated_tokens": report["speed"]["generated_tokens"],
                    "finished_with_eos": report["output"]["finished_with_eos"],
                    "truncated": report["output"]["truncated"],
                }
                if report["status"] == "ok"
                else {"error": report.get("error")}
            ),
        }
        for report in candidates
    ]
    print(json.dumps({"document": result["document"], "candidates": summary}, ensure_ascii=False, indent=2))
    del base_model
    _free_cuda()


if __name__ == "__main__":
    main()
