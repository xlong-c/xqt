"""Benchmark baidu/Unlimited-OCR Half inference against ConvRot W8A8.

This is a fixed-input, end-to-end benchmark.  It owns neither preprocessing nor
generation: each measurement calls the checkpoint's official ``model.infer``
API.  The benchmark reports the real ConvRot execution engines so a result with
small-batch Half fallbacks is never presented as an all-INT8 speedup.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.xqt_models.unlimited_ocr import (
    UNLIMITED_OCR_REPO_ID,
    UnlimitedOcrConvRotInt8Result,
    load_unlimited_ocr,
    quantize_unlimited_ocr_convrot_int8,
)
from xqt.quant.quantizers.convrot_int8 import ConvRotInt8Linear


CONFIG: dict[str, Any] = {
    "model_id": UNLIMITED_OCR_REPO_ID,
    "revision": None,
    "local_files_only": True,
    "artifact_dir": "artifacts/xqt/examples/unlimited_ocr_convrot_int8",
    "input": {
        "pdf_path": "others/resume/resume.pdf",
        "render_dpi": 200,
        "max_pages": 3,
        "prompt": "<image>document parsing.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "max_length": 8192,
        "no_repeat_ngram_size": 35,
        "ngram_window": 128,
    },
    "benchmark": {
        "warmup_calls": 1,
        "save_results": False,
    },
    "quantization": {
        "engine": "cuda_sm89",
        "fallback_engine": "torch_int_mm",
        "rot_size": 256,
        "activation_scale_mode": "static",
        "mse_clip": True,
        # Decode has M=1..8 for most projections.  The cached Half fallback is
        # faster there, and metadata makes the split visible in the report.
        "min_int8_rows": 32,
        "only_static_int8_eligible_modules": True,
        "policy": {
            "dtype": "int8",
            "scheme": "convrot_w8a8",
            "include_module_types": ["Linear"],
            "exclude_name_patterns": [
                r"^model\.vision_model(?:\.|$)",
                r"^model\.sam_model(?:\.|$)",
                r"^model\.projector(?:\.|$)",
                r"lm_head$",
                r"\.gate$",
            ],
            "min_parameters": 65536,
        },
    },
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _render_pdf_pages(config: dict[str, Any], artifact_dir: Path) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render the benchmark PDF") from exc

    input_config = dict(config["input"])
    source = Path(str(input_config["pdf_path"]))
    if not source.is_file():
        raise FileNotFoundError(f"Unlimited-OCR benchmark PDF does not exist: {source}")
    page_dir = artifact_dir / "rendered_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    max_pages = max(1, int(input_config["max_pages"]))
    scale = float(input_config["render_dpi"]) / 72.0
    document = fitz.open(source)
    page_paths: list[Path] = []
    try:
        for index, page in enumerate(document):
            if index >= max_pages:
                break
            page_path = page_dir / f"page-{index + 1}.png"
            page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(
                str(page_path)
            )
            page_paths.append(page_path)
    finally:
        document.close()
    if not page_paths:
        raise RuntimeError(f"No pages rendered from {source}")
    return page_paths


def _infer_page(
    model: nn.Module,
    tokenizer: Any,
    *,
    page_path: Path,
    output_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    input_config = dict(config["input"])
    output_path.mkdir(parents=True, exist_ok=True)
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        text = model.infer(
            tokenizer,
            prompt=str(input_config["prompt"]),
            image_file=str(page_path),
            output_path=str(output_path),
            base_size=int(input_config["base_size"]),
            image_size=int(input_config["image_size"]),
            crop_mode=bool(input_config["crop_mode"]),
            max_length=int(input_config["max_length"]),
            no_repeat_ngram_size=int(input_config["no_repeat_ngram_size"]),
            ngram_window=int(input_config["ngram_window"]),
            save_results=bool(config["benchmark"]["save_results"]),
            eval_mode=True,
        )
    _sync(device)
    seconds = time.perf_counter() - start
    rendered = str(text)
    return {
        "seconds": seconds,
        "text": rendered,
        "characters": len(rendered),
        "characters_per_second": len(rendered) / seconds if seconds > 0.0 else None,
    }


def _warmup(
    model: nn.Module,
    tokenizer: Any,
    *,
    page_path: Path,
    artifact_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    for index in range(max(0, int(config["benchmark"]["warmup_calls"]))):
        _infer_page(
            model,
            tokenizer,
            page_path=page_path,
            output_path=artifact_dir / f"warmup-{index + 1}",
            config=config,
            device=device,
        )


def _run_pages(
    model: nn.Module,
    tokenizer: Any,
    *,
    page_paths: list[Path],
    label: str,
    artifact_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, page_path in enumerate(page_paths, start=1):
        print(f"{label} page {index}/{len(page_paths)}", flush=True)
        result = _infer_page(
            model,
            tokenizer,
            page_path=page_path,
            output_path=artifact_dir / f"{label}-page-{index}",
            config=config,
            device=device,
        )
        result["page"] = index
        results.append(result)
    return results


def _similarity(reference: str, candidate: str) -> float:
    normalized_reference = " ".join(reference.split())
    normalized_candidate = " ".join(candidate.split())
    if not normalized_reference and not normalized_candidate:
        return 1.0
    return SequenceMatcher(None, normalized_reference, normalized_candidate).ratio()


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_seconds = sum(float(item["seconds"]) for item in results)
    total_characters = sum(int(item["characters"]) for item in results)
    return {
        "total_seconds": total_seconds,
        "mean_seconds_per_page": total_seconds / len(results),
        "total_characters": total_characters,
        "characters_per_second": (
            total_characters / total_seconds if total_seconds > 0.0 else None
        ),
        "pages": [
            {
                "page": item["page"],
                "seconds": item["seconds"],
                "characters": item["characters"],
                "characters_per_second": item["characters_per_second"],
            }
            for item in results
        ],
    }


def _execution_summary(model: nn.Module) -> dict[str, Any]:
    engines: Counter[str] = Counter()
    activation_quant_engines: Counter[str] = Counter()
    true_int8_mma_modules = 0
    executed_modules = 0
    for module in model.modules():
        if not isinstance(module, ConvRotInt8Linear):
            continue
        metadata = module.execution_metadata()
        if metadata.get("engine") == "not_run":
            continue
        executed_modules += 1
        engines[str(metadata.get("engine"))] += 1
        activation_quant_engines[str(metadata.get("activation_quant_engine"))] += 1
        true_int8_mma_modules += int(bool(metadata.get("true_int8_mma")))
    return {
        "executed_module_count": executed_modules,
        "true_int8_mma_module_count": true_int8_mma_modules,
        "engines": dict(engines),
        "activation_quant_engines": dict(activation_quant_engines),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def _quantize(
    model: nn.Module,
    tokenizer: Any,
    *,
    calibration_page: Path,
    artifact_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> UnlimitedOcrConvRotInt8Result:
    quantization = dict(config["quantization"])

    def calibration_call(current: nn.Module) -> None:
        _infer_page(
            current,
            tokenizer,
            page_path=calibration_page,
            output_path=artifact_dir / "calibration",
            config=config,
            device=device,
        )

    return quantize_unlimited_ocr_convrot_int8(
        model,
        calibration_call=calibration_call,
        policy=dict(quantization["policy"]),
        activation_scale_mode=str(quantization["activation_scale_mode"]),
        rot_size=int(quantization["rot_size"]),
        engine=str(quantization["engine"]),
        fallback_engine=str(quantization["fallback_engine"]),
        mse_clip=bool(quantization["mse_clip"]),
        min_int8_rows=int(quantization["min_int8_rows"]),
        only_static_int8_eligible_modules=bool(
            quantization["only_static_int8_eligible_modules"]
        ),
        inplace=True,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Unlimited-OCR ConvRot benchmark requires CUDA")
    device = torch.device("cuda")
    artifact_dir = Path(str(CONFIG["artifact_dir"]))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    page_paths = _render_pdf_pages(CONFIG, artifact_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(CONFIG["model_id"]),
        revision=CONFIG["revision"],
        trust_remote_code=True,
        local_files_only=bool(CONFIG["local_files_only"]),
    )
    model = load_unlimited_ocr(
        repo_id=str(CONFIG["model_id"]),
        revision=CONFIG["revision"],
        dtype=torch.float16,
        device=device,
        local_files_only=bool(CONFIG["local_files_only"]),
    )

    _warmup(
        model,
        tokenizer,
        page_path=page_paths[0],
        artifact_dir=artifact_dir,
        config=CONFIG,
        device=device,
    )
    baseline_memory = torch.cuda.memory_allocated(device)
    baseline = _run_pages(
        model,
        tokenizer,
        page_paths=page_paths,
        label="half",
        artifact_dir=artifact_dir,
        config=CONFIG,
        device=device,
    )

    quantize_start = time.perf_counter()
    quantized_result = _quantize(
        model,
        tokenizer,
        calibration_page=page_paths[0],
        artifact_dir=artifact_dir,
        config=CONFIG,
        device=device,
    )
    quantize_seconds = time.perf_counter() - quantize_start
    model = quantized_result.model.eval().to(device)
    _warmup(
        model,
        tokenizer,
        page_path=page_paths[0],
        artifact_dir=artifact_dir,
        config=CONFIG,
        device=device,
    )
    quantized_memory = torch.cuda.memory_allocated(device)
    quantized = _run_pages(
        model,
        tokenizer,
        page_paths=page_paths,
        label="convrot_int8",
        artifact_dir=artifact_dir,
        config=CONFIG,
        device=device,
    )

    baseline_summary = _aggregate(baseline)
    quantized_summary = _aggregate(quantized)
    similarities = [
        _similarity(str(reference["text"]), str(candidate["text"]))
        for reference, candidate in zip(baseline, quantized)
    ]
    execution = _execution_summary(model)
    speedup = baseline_summary["total_seconds"] / quantized_summary["total_seconds"]
    summary = {
        "model_id": CONFIG["model_id"],
        "input": {
            "pdf_path": CONFIG["input"]["pdf_path"],
            "pages": len(page_paths),
            "generation": dict(CONFIG["input"]),
        },
        "environment": _environment(device),
        "half": {"dtype": "float16", **baseline_summary},
        "convrot_w8a8": {
            "dtype": "float16 output, int8 rotated weights and activations",
            "quantize_seconds": quantize_seconds,
            "quantized_module_count": len(quantized_result.quantized_modules),
            "calibration": (
                None
                if quantized_result.calibration is None
                else quantized_result.calibration.to_dict()
            ),
            "execution": execution,
            **quantized_summary,
        },
        "memory_bytes": {
            "half_allocated": baseline_memory,
            "convrot_allocated": quantized_memory,
            "saved": baseline_memory - quantized_memory,
        },
        "speedup_vs_half": {
            "total": speedup,
            "per_page": [
                float(reference["seconds"]) / float(candidate["seconds"])
                for reference, candidate in zip(baseline, quantized)
            ],
            "convrot_is_faster": speedup > 1.0,
        },
        "text_similarity": {
            "per_page": similarities,
            "mean": sum(similarities) / len(similarities),
        },
    }
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Half total: {baseline_summary['total_seconds']:.3f}s")
    print(f"ConvRot total: {quantized_summary['total_seconds']:.3f}s")
    print(f"ConvRot speedup vs Half: {speedup:.4f}x")
    print(f"Mean text similarity: {summary['text_similarity']['mean']:.6f}")
    print(f"Execution: {execution}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
