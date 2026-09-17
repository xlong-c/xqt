#!/usr/bin/env python
"""Build long-context English->Chinese translation evaluation fixtures.

Why this exists: the project's only long-context "evidence" so far came from
tiling a single 7976-character document to reach 4096 tokens, which measures
nothing about long-context behaviour -- every position sees the same text as
every other. This builder assembles documents from *distinct* sources, records
the source ids, and asserts `len(set(source_ids)) == len(ids)` so tiling cannot
creep back in.

Sources
-------
- ``downloads/aaai2026``: 4920 AAAI-2026 paper title/abstract pairs with human
  Chinese translations. Used for both the English prompt document and the
  Chinese reference used to score the model's output.
- Project Gutenberg public-domain prose: a distribution shift away from
  technical abstracts. There is no Chinese reference for the book, so runs on
  it report agreement against the baseline only -- never BLEU.

Outputs go to ``data/long_context_eval`` (a shared, gitignored directory that
both ``xdl`` and ``xqt`` see through their ``data -> ../data`` symlinks):

- ``en_*.txt``          English source document, fed to the model
- ``zh_*.txt``          Chinese reference, used for scoring only
- ``manifest.json``     per-document provenance: sha256, token counts, source
                        ids, builder hash, normalization recipe
- ``split.json``        authoritative disjointness record: eval / val / train
                        source ids. The Phase-1 training data builder must read
                        this and hard-fail if a training document contains an
                        eval id.

Prompt token targets are chosen so that ``prompt + 512 generated`` lands on a
round context length, with the small margin the verify window needs:

===========  ================  ================  ==================
document     prompt target     +512 generated    context
===========  ================  ================  ==================
en_2k        1528              2040              2k
en_4k        3576              4088              4k
en_8k        7672              8184              8k
en_16k       15864             16376             16k  (cache 16384)
===========  ================  ================  ==================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
AAAI_DIR = REPO_ROOT / "downloads" / "aaai2026"
BOOK_DIR = REPO_ROOT / "downloads" / "gutenberg"
MODEL_DIR = REPO_ROOT / "downloads" / "MiniCPM5-2B-bf16"
OUT_DIR = REPO_ROOT / "data" / "long_context_eval"

# Prompt-only token targets; the benchmark adds 512 generated tokens on top.
# 8 is the verify window height, hence the extra margin at the top end.
GENERATED_TOKENS = 512
VERIFY_ROWS = 8
TARGETS: dict[str, int] = {
    "en_2k": 2048 - GENERATED_TOKENS - VERIFY_ROWS,
    "en_4k": 4096 - GENERATED_TOKENS - VERIFY_ROWS,
    "en_8k": 8192 - GENERATED_TOKENS - VERIFY_ROWS,
    "en_16k": 16384 - GENERATED_TOKENS - VERIFY_ROWS,
}
BOOK_SOURCE = {
    "name": "book_2k",
    "target": TARGETS["en_2k"],
    "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "pg_id": 2701,
    "title": "Moby Dick; Or, The Whale",
}
# Filtered-AAAI start offsets keep the assembled spans disjoint by construction.
START_OFFSETS: dict[str, int] = {
    "en_2k": 0,
    "en_4k": 100,
    "en_8k": 300,
    "en_16k": 500,
}
# Reserved for the Phase-1 held-out validation documents.
VAL_BLOCK_START = 900
VAL_BLOCK_SIZE = 24
MIN_ABSTRACT_CHARS = 400
MAX_NON_ASCII_RATIO = 0.02
TOLERANCE = 0.02


def _normalize(text: str) -> str:
    """Normalize a raw source string; the exact steps are recorded in the manifest."""

    text = text.encode("utf-8", errors="strict").decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def load_aaai_pairs() -> tuple[list[dict], list[dict]]:
    """Return (accepted, rejected) AAAI pairs in deterministic file order."""

    master = json.loads((AAAI_DIR / "aaai2026_master.json").read_text("utf-8"))
    translations = json.loads(
        (AAAI_DIR / "aaai2026_translations.json").read_text("utf-8")
    )
    accepted: list[dict] = []
    rejected: list[dict] = []
    for index, entry in enumerate(master):
        key = str(index)
        record = translations.get(key)
        if record is None:
            rejected.append({"source_id": index, "reason": "no_translation"})
            continue
        abstract = _normalize(entry.get("abstract") or "")
        title = _normalize(entry.get("title") or "")
        abstract_zh = _normalize(record.get("abstract_zh") or "")
        title_zh = _normalize(record.get("title_zh") or "")
        if not title or not abstract or not abstract_zh or not title_zh:
            rejected.append({"source_id": index, "reason": "empty_field"})
            continue
        if len(abstract) < MIN_ABSTRACT_CHARS:
            rejected.append({"source_id": index, "reason": "abstract_too_short"})
            continue
        if _non_ascii_ratio(abstract) > MAX_NON_ASCII_RATIO or "\ufffd" in abstract:
            rejected.append({"source_id": index, "reason": "abstract_not_english"})
            continue
        accepted.append(
            {
                "source_id": index,
                "track": entry.get("track") or "",
                "en": f"## {title}\n\n{abstract}\n\n",
                "zh": f"## {title_zh}\n\n{abstract_zh}\n\n",
            }
        )
    return accepted, rejected


def _fit_to_target(
    units: Sequence[dict],
    target: int,
    count_fn: Callable[[str], int],
) -> tuple[str, list[int], int, int]:
    """Grow ``units`` into one document whose prompt token count hits ``target``.

    Returns ``(document, source_ids, prompt_tokens, truncated_chars)``. The last
    unit may be cut mid-way so the count lands inside tolerance; that is
    recorded rather than hidden, because it makes no pretense of being a
    natural document boundary.
    """

    full = "".join(unit["en"] for unit in units)
    if count_fn(full) <= target:
        return full, [unit["source_id"] for unit in units], count_fn(full), 0

    best_cut = 0
    best_delta = abs(count_fn("") - target)
    lo, hi = 0, len(full)
    while lo <= hi:
        mid = (lo + hi) // 2
        count = count_fn(full[:mid])
        delta = abs(count - target)
        if delta < best_delta:
            best_delta = delta
            best_cut = mid
        if count < target:
            lo = mid + 1
        else:
            hi = mid - 1

    document = full[:best_cut]
    used: list[int] = []
    cursor = 0
    for unit in units:
        if cursor >= best_cut:
            break
        used.append(unit["source_id"])
        cursor += len(unit["en"])
    truncated = sum(len(unit["en"]) for unit in units[: len(used)]) - best_cut
    return document, used, count_fn(document), max(truncated, 0)


def _fit_interleaved(
    units: Sequence[dict],
    target: int,
    count_fn: Callable[[str], int],
) -> tuple[str, str, list[int], int, int]:
    """Fit an English document and keep the Chinese reference unit-aligned."""

    document, used, tokens, truncated = _fit_to_target(units, target, count_fn)
    used_set = set(used)
    zh = "".join(unit["zh"] for unit in units if unit["source_id"] in used_set)
    return document, zh, used, tokens, truncated


def fetch_book(url: str, pg_id: int) -> str:
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    cached = BOOK_DIR / f"pg{pg_id}.txt"
    if not cached.exists():
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        cached.write_bytes(payload)
    raw = cached.read_text("utf-8", errors="strict")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    if start is None or end is None:
        raise SystemExit(f"Gutenberg markers not found in {cached}")
    return raw[start.end() : end.start()]


def _build_book_document(tokenizer) -> dict:
    from xqt.model.minicpm5_chat import render_translation_input_ids

    body = _normalize(fetch_book(BOOK_SOURCE["url"], BOOK_SOURCE["pg_id"]))

    def count_fn(document: str) -> int:
        return len(render_translation_input_ids(tokenizer, document))

    target = int(BOOK_SOURCE["target"])
    best = ("", abs(count_fn("") - target))
    lo, hi = 0, len(body)
    while lo <= hi:
        mid = (lo + hi) // 2
        count = count_fn(body[:mid])
        delta = abs(count - target)
        if delta < best[1]:
            best = (body[:mid], delta)
        if count < target:
            lo = mid + 1
        else:
            hi = mid - 1
    document = best[0]
    return {
        "name": BOOK_SOURCE["name"],
        "en": document,
        "zh": "",
        "source_ids": [f"gutenberg:{BOOK_SOURCE['pg_id']}"],
        "source_kind": "gutenberg",
        "prompt_tokens": count_fn(document),
        "truncated_chars": len(body) - len(document),
        "provenance": {
            "pg_id": BOOK_SOURCE["pg_id"],
            "title": BOOK_SOURCE["title"],
            "url": BOOK_SOURCE["url"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR,
        help="tokenizer directory used to measure prompt token counts",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    import transformers
    from transformers import AutoTokenizer

    from xqt.model.minicpm5_chat import render_translation_input_ids

    # The book is ~1.2M characters, so the tokenizer's advisory "longer than
    # the model's max length" warning fires on every probe. This tool only
    # counts tokens and never runs the sequence through a model.
    transformers.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    accepted, rejected = load_aaai_pairs()
    print(f"AAAI: {len(accepted)} usable pairs, {len(rejected)} rejected")

    def count_fn(document: str) -> int:
        return len(render_translation_input_ids(tokenizer, document))

    builder_sha = _sha256_file(Path(__file__).resolve())
    args.out.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    consumed: list[int] = []
    for name, target in TARGETS.items():
        offset = START_OFFSETS[name]
        window = accepted[offset:]
        if not window:
            raise SystemExit(f"not enough filtered pairs for {name}")
        en, zh, used, tokens, truncated = _fit_interleaved(window, target, count_fn)
        consumed.extend(used)
        documents.append(
            {
                "name": name,
                "en": en,
                "zh": zh,
                "source_ids": used,
                "source_kind": "aaai2026",
                "prompt_tokens": tokens,
                "truncated_chars": truncated,
                "provenance": {"start_offset": offset, "target_tokens": target},
            }
        )
    documents.append(_build_book_document(tokenizer))

    reserved = accepted[VAL_BLOCK_START : VAL_BLOCK_START + VAL_BLOCK_SIZE]
    val_ids = [unit["source_id"] for unit in reserved]

    consumed_set = set(consumed)
    val_set = set(val_ids)
    overlap = consumed_set & val_set
    if overlap:
        raise SystemExit(f"eval and val blocks overlap: {sorted(overlap)[:5]}")
    train_ids = [
        unit["source_id"]
        for unit in accepted
        if unit["source_id"] not in consumed_set and unit["source_id"] not in val_set
    ]

    manifest = {
        "schema": "xqt.long_context_eval.v1",
        "builder_sha256": builder_sha,
        "model_dir": str(args.model),
        "normalization": [
            "utf-8 strict decode",
            "NFC",
            "CRLF|CR -> LF",
            "collapse 3+ newlines to 2",
            "rstrip each line",
            "strip",
        ],
        "filters": {
            "min_abstract_chars": MIN_ABSTRACT_CHARS,
            "max_non_ascii_ratio": MAX_NON_ASCII_RATIO,
            "require_translation": True,
        },
        "token_measurement": (
            "len(render_translation_input_ids(tokenizer, document)) using the "
            "canonical ChatML renderer in xqt.model.minicpm5_chat"
        ),
        "tolerance": TOLERANCE,
        "documents": [],
        "rejected_count": len(rejected),
        "notes": [
            "Source ids within one document are distinct by construction; the "
            "builder asserts this before writing.",
            "documents built from downloads/aaai2026 (human EN/ZH pairs) and "
            "Project Gutenberg public-domain prose.",
            "minicpm5_2b_prefill_ab.py reaches 4096 tokens by TILING one "
            "7976-char document and is NOT long-context evidence; it is never "
            "used here.",
            "The book document has no Chinese reference; runs on it report "
            "agreement against the baseline only, never BLEU.",
        ],
    }

    for record in documents:
        ids = record["source_ids"]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{record['name']}: duplicate source ids (tiling)")
        prompt_tokens = int(record["prompt_tokens"])
        target = int(record["provenance"].get("target_tokens", BOOK_SOURCE["target"]))
        drift = abs(prompt_tokens - target) / target
        if record["source_kind"] == "aaai2026" and drift > TOLERANCE:
            raise SystemExit(
                f"{record['name']}: prompt_tokens {prompt_tokens} is {drift:.1%} "
                f"off target {target} (tolerance {TOLERANCE:.0%})"
            )
        en_path = args.out / f"{record['name']}.txt"
        en_path.write_text(record["en"], encoding="utf-8")
        entry = {
            "name": record["name"],
            "path": str(en_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_text(record["en"]),
            "chars": len(record["en"]),
            "prompt_tokens": prompt_tokens,
            "target_tokens": target,
            "source_kind": record["source_kind"],
            "source_ids": ids,
            "distinct_sources": len(set(ids)),
            "source_count": len(ids),
            "truncated_chars": int(record["truncated_chars"]),
            "provenance": record["provenance"],
            "has_chinese_reference": bool(record["zh"]),
        }
        if record["zh"]:
            zh_path = args.out / f"{record['name']}_zh.txt"
            zh_path.write_text(record["zh"], encoding="utf-8")
            entry["reference_path"] = str(zh_path.relative_to(REPO_ROOT))
            entry["reference_sha256"] = _sha256_text(record["zh"])
        manifest["documents"].append(entry)
        print(
            f"{record['name']:<12} prompt_tokens={prompt_tokens:<6} "
            f"sources={len(ids):<4} chars={len(record['en'])}"
        )

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "split.json").write_text(
        json.dumps(
            {
                "schema": "xqt.long_context_eval.split.v1",
                "eval_source_ids": sorted(consumed_set),
                "val_source_ids": sorted(val_set),
                "train_source_ids": sorted(train_ids),
                "book": {"pg_id": BOOK_SOURCE["pg_id"], "usage": "eval_only"},
                "rules": [
                    "Phase-1 training data must contain no eval_source_ids and "
                    "no val_source_ids; the data builder hard-fails on a hit.",
                    "eval documents are never used for training or validation.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"\nwrote {len(manifest['documents'])} documents, "
        f"eval sources={len(consumed_set)}, val sources={len(val_set)}, "
        f"train sources={len(train_ids)}"
    )


if __name__ == "__main__":
    main()
