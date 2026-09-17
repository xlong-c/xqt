#!/usr/bin/env python
"""Build translation-drafter *training prompts* from the held-out training pool.

Why this exists: the drafter being trained here only ever runs on the
translation workload, so the corpus it learns from has to be the same
distribution the benchmark measures. The benchmark documents in
``data/long_context_eval`` are AAAI-2026 title/abstract concatenations at
2k/4k/8k/16k; this builder draws from the *same* accepted-pair pool, so the
train and eval documents differ only in which papers they contain.

Three invariants are enforced rather than trusted:

1. **Disjointness.** ``split.json`` is authoritative for which source ids are
   eval, val and train. A training document that contains an eval or val id
   aborts the build. There is no flag to override this.
2. **No tiling.** Source ids inside one document are distinct; the eval
   builder was written to kill a tiling bug and the same check applies here.
3. **Canonical prompt.** Ids come from
   ``xqt.model.minicpm5_chat.render_translation_input_ids``, the same renderer
   the benchmark and the runtime use. Prompt drift is silent -- a template
   mismatch shows up only as an acceptance collapse with no assertion firing
   anywhere -- so the ids are stored here and re-checked at train time.

What this does NOT do: generate the responses. Responses must come from the
target model in non-thinking mode, so this stage is tokenizer-only and runs in
seconds; response generation is a separate GPU stage.

Outputs go to ``data/drafter_translation``:

- ``prompts_train.jsonl``  one record per training document
- ``prompts_val.jsonl``    one record per held-out validation document
- ``manifest.json``        provenance: bucket targets, per-bucket counts,
                           source ids consumed, builder hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AAAI_DIR = REPO_ROOT / "downloads" / "aaai2026"
MODEL_DIR = REPO_ROOT / "downloads" / "MiniCPM5-2B-bf16"
SPLIT_PATH = REPO_ROOT / "data" / "long_context_eval" / "split.json"
OUT_DIR = REPO_ROOT / "data" / "drafter_translation"

# Prompt token targets, identical to the eval builder: prompt + 512 generated
# lands exactly on the benchmark's context length (2k/4k/8k), so a training
# sequence and an eval sequence have the same shape.
PROMPT_TARGETS: dict[str, int] = {
    "2k": 2048 - 512 - 8,
    "4k": 4096 - 512 - 8,
    "8k": 8192 - 512 - 8,
}
# Documents per bucket. 8k first because it consumes the most papers per
# document, so the long bucket is never starved by the short one. The pool is
# 4775 papers at ~283 prompt tokens each = 1.35M tokens; this plan spends 1.22M
# of it (roughly 4635 papers) and leaves the rest as slack for the truncated
# last paper of each document.
BUCKET_PLAN: list[tuple[str, int]] = [("8k", 40), ("4k", 115), ("2k", 330)]
# Held-out validation: two documents drawn from the reserved val pool only.
VAL_PLAN: list[tuple[str, int]] = [("2k", 1), ("4k", 1)]
TOLERANCE = 0.02


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_eval_builder():
    """Import the sibling eval builder by path.

    ``tools/`` holds standalone scripts rather than an installable package, so
    the shared normalisation and document-fitting helpers are loaded from the
    file next to this one instead of being copied. Copying would let the train
    and eval documents drift apart in how they are assembled, which is exactly
    the distribution gap this corpus exists to close.
    """

    import importlib.util

    path = Path(__file__).resolve().parent / "build_long_context_eval.py"
    spec = importlib.util.spec_from_file_location("_lc_eval_builder", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the eval builder at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_split() -> dict:
    if not SPLIT_PATH.exists():
        raise SystemExit(
            f"{SPLIT_PATH} is missing; run tools/dataset/build_long_context_eval.py first"
        )
    return json.loads(SPLIT_PATH.read_text("utf-8"))


def build_records(
    units: list[dict],
    plan: list[tuple[str, int]],
    *,
    count_fn,
    id_prefix: str,
    fit_to_target,
) -> tuple[list[dict], set[int]]:
    """Consume ``units`` in order, producing ``plan`` documents.

    Returns ``(records, consumed_source_ids)``. Callers must verify the
    consumed set against the eval/val sets; this function has no opinion about
    which pool ``units`` came from.
    """

    records: list[dict] = []
    cursor = 0
    consumed: set[int] = set()
    for bucket, wanted in plan:
        target = PROMPT_TARGETS[bucket]
        for index in range(wanted):
            window = units[cursor:]
            if not window:
                print(
                    f"  [warn] paper pool exhausted at {bucket} "
                    f"document {index}/{wanted}"
                )
                break
            document, used, tokens, truncated = fit_to_target(
                window, target, count_fn
            )
            if not used:
                break
            if len(set(used)) != len(used):
                raise SystemExit(f"{id_prefix}_{bucket}: duplicate source ids (tiling)")
            records.append(
                {
                    "doc_id": f"{id_prefix}_{bucket}_{index:04d}",
                    "bucket": bucket,
                    "target_tokens": target,
                    "prompt_ids": count_fn.ids(document),
                    "prompt_tokens": tokens,
                    "en_chars": len(document),
                    "truncated_chars": truncated,
                    "source_ids": used,
                    "source_count": len(used),
                    "source_kind": "aaai2026",
                    "text": document,
                }
            )
            consumed.update(used)
            cursor += len(used)
            print(
                f"  {id_prefix}_{bucket}_{index:04d}  tokens={tokens:<6} "
                f"papers={len(used):<3} chars={len(document)}"
            )
    return records, consumed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="clamp every bucket to N documents (smoke testing only)",
    )
    args = parser.parse_args()

    import transformers
    from transformers import AutoTokenizer

    from xqt.model.minicpm5_chat import render_translation_input_ids

    eval_builder = _load_eval_builder()
    load_aaai_pairs = eval_builder.load_aaai_pairs
    fit_to_target = eval_builder._fit_to_target

    transformers.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))

    class CountFn:
        """Token counter that memoises the last document, because the fitting
        routine probes the same prefix repeatedly and tokenising an 8k-token
        string is not free."""

        def __init__(self) -> None:
            self._cache: dict[str, int] = {}

        def __call__(self, document: str) -> int:
            cached = self._cache.get(document)
            if cached is None:
                cached = len(render_translation_input_ids(tokenizer, document))
                if len(self._cache) < 64:
                    self._cache[document] = cached
            return cached

        def ids(self, document: str) -> list[int]:
            return render_translation_input_ids(tokenizer, document)

    count_fn = CountFn()

    split = _load_split()
    eval_ids = set(split["eval_source_ids"])
    val_ids = set(split["val_source_ids"])
    train_ids = set(split["train_source_ids"])
    if eval_ids & val_ids or (eval_ids | val_ids) & train_ids:
        raise SystemExit("split.json pools overlap; refusing to build")

    accepted, rejected = load_aaai_pairs()
    by_id = {unit["source_id"]: unit for unit in accepted}
    missing = (eval_ids | val_ids | train_ids) - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} split ids are not in the accepted pool")

    train_units = [by_id[i] for i in sorted(train_ids)]
    val_units = [by_id[i] for i in sorted(val_ids)]
    print(
        f"pool: {len(accepted)} accepted, {len(rejected)} rejected | "
        f"train={len(train_units)} val={len(val_units)} eval={len(eval_ids)}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    builder_sha = _sha256_file(Path(__file__).resolve())

    bucket_plan = BUCKET_PLAN
    val_plan = VAL_PLAN
    if args.max_docs is not None:
        bucket_plan = [(bucket, min(n, args.max_docs)) for bucket, n in BUCKET_PLAN]
        val_plan = [(bucket, min(n, args.max_docs)) for bucket, n in VAL_PLAN]

    print("\n[train]")
    train_records, consumed = build_records(
        train_units, bucket_plan, count_fn=count_fn, id_prefix="train",
        fit_to_target=fit_to_target,
    )
    leak = consumed & (eval_ids | val_ids)
    if leak:
        raise SystemExit(
            f"REFUSING TO WRITE: training documents contain eval/val source ids "
            f"{sorted(leak)[:5]}"
        )

    print("\n[val]")
    val_records, val_consumed = build_records(
        val_units, val_plan, count_fn=count_fn, id_prefix="val",
        fit_to_target=fit_to_target,
    )
    val_leak = val_consumed & eval_ids
    if val_leak:
        raise SystemExit(
            f"REFUSING TO WRITE: validation documents contain eval ids "
            f"{sorted(val_leak)[:5]}"
        )

    for name, records in (("prompts_train.jsonl", train_records),
                          ("prompts_val.jsonl", val_records)):
        path = args.out / name
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    per_bucket: dict[str, dict[str, int]] = {}
    for record in train_records:
        entry = per_bucket.setdefault(
            record["bucket"], {"documents": 0, "prompt_tokens": 0}
        )
        entry["documents"] += 1
        entry["prompt_tokens"] += record["prompt_tokens"]

    manifest = {
        "schema": "xqt.drafter_translation.prompts.v1",
        "builder_sha256": builder_sha,
        "model_dir": str(args.model),
        "split_path": "data/long_context_eval/split.json",
        "split_schema": split.get("schema"),
        "prompt_renderer": "xqt.model.minicpm5_chat.render_translation_input_ids",
        "non_thinking": True,
        "prompt_targets": PROMPT_TARGETS,
        "tolerance": TOLERANCE,
        "train_documents": len(train_records),
        "val_documents": len(val_records),
        "train_papers_consumed": len(consumed),
        "train_papers_available": len(train_units),
        "per_bucket": per_bucket,
        "disjointness": {
            "eval_ids_used_in_train": 0,
            "val_ids_used_in_train": 0,
            "eval_ids_used_in_val": 0,
            "enforced": "hard abort on any hit",
        },
        "notes": [
            "Documents are built from the same accepted AAAI pair pool as the "
            "benchmark, restricted to split.json train_source_ids.",
            "Responses are NOT in this file. They must be generated by the "
            "target model in non-thinking mode; see generate_drafter_targets.py.",
            "prompt_ids are stored so the training loop can assert that the "
            "renderer still produces identical ids.",
        ],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nwrote {len(train_records)} train / {len(val_records)} val documents "
        f"to {args.out}"
    )
    print(
        f"papers consumed: train {len(consumed)}/{len(train_units)}, "
        f"val {len(val_consumed)}/{len(val_units)}"
    )
    for bucket, entry in sorted(per_bucket.items()):
        print(
            f"  {bucket}: {entry['documents']} docs, "
            f"{entry['prompt_tokens']} prompt tokens"
        )


if __name__ == "__main__":
    main()
