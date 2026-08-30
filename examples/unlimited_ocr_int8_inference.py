"""INT8 MMA inference path for baidu/Unlimited-OCR.

The script keeps all settings in CONFIG instead of command-line arguments.  It
first proves the local INT8 path with a TileLang PTX/kernel probe, then runs an
Unlimited-OCR-shaped Linear benchmark.  Full checkpoint OCR inference is gated
behind CONFIG["run"]["full_ocr"] because the checkpoint is about 6.7 GB.
"""

from __future__ import annotations

import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.kernels.ops._impl.tilelang.int8_mma import (  # noqa: E402
    build_tilelang_int8_linear_static_activation_kernel,
    build_tilelang_int8_mma_kernel,
    int8_linear_static_activation_tilelang,
    int8_linear_tilelang,
    int8_mma_tilelang,
    pad_rows_to_block,
    static_activation_quantize_tilelang,
)
from xqt.compression.quant.quantizers.int8_mma import (  # noqa: E402
    Int8MmaLinear,
    quantize_with_int8_mma,
)

MappingLike = dict[str, Any]


CONFIG: dict[str, Any] = {
    "model_id": "baidu/Unlimited-OCR",
    "revision": None,
    "artifact_dir": "artifacts/xqt/examples/unlimited_ocr_int8",
    "run": {
        "tilelang_probe": True,
        "linear_benchmark": True,
        "full_ocr": False,
    },
    "quantization": {
        "engine": "tilelang",
        "fallback_engine": "torch_int_mm",
        "block_m": 32,
        "block_n": 128,
        "block_k": 128,
        "threads": 128,
        "num_stages": 2,
        "activation_quant_block_size": 256,
        "policy": {
            "dtype": "int8",
            "scheme": "dynamic_mma",
            "include_module_types": ["Linear"],
            "selection_mode": "include_only",
            "include_name_patterns": [
                r"^model\.layers\.[1-9][0-9]*\.mlp\.experts\.[0-9]+\.down_proj$",
            ],
            "exclude_name_patterns": [],
            "min_parameters": 4096,
        },
    },
    "tilelang_probe": {
        "m": 128,
        "n": 128,
        "k": 128,
        "warmup": 10,
        "iterations": 50,
    },
    "linear_benchmark": {
        "m": 1,
        "in_features": 896,
        "out_features": 1280,
        "dtype": "bfloat16",
        "activation_scale_mode": "static",
        "warmup": 20,
        "iterations": 100,
        "seed": 0,
    },
    "full_ocr": {
        "enabled": False,
        "compare_bf16": True,
        "calibrate_static_scales": True,
        "measure_steady_state": True,
        "image_file": "",
        "prompt": "<image>document parsing.",
        "output_path": "artifacts/xqt/examples/unlimited_ocr_int8/ocr_output",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "max_length": 2048,
        "no_repeat_ngram_size": 35,
        "ngram_window": 128,
        "save_results": True,
        "quality_gate": {
            "min_text_similarity": 0.98,
            "required_keywords": [
                "Unlimited-OCR INT8 MMA smoke image",
                "This file is generated under artifacts",
            ],
        },
    },
}


def _artifact_dir() -> Path:
    path = Path(str(CONFIG["artifact_dir"]))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _target_arch(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(int(warmup)):
        fn()
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(iterations)):
            fn()
        end.record()
        _sync(device)
        return float(start.elapsed_time(end) / int(iterations))
    start_time = time.perf_counter()
    for _ in range(int(iterations)):
        fn()
    return float((time.perf_counter() - start_time) * 1000.0 / int(iterations))


def _tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    diff = (ref - cand).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "allclose_1e_2": bool(torch.allclose(ref, cand, atol=1e-2, rtol=1e-2)),
        "shape": list(ref.shape),
    }


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _ptx_int8_mma_evidence(path: Path, ptx_text: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "has_mma_sync": "mma.sync" in ptx_text,
        "has_m16n8k32": "m16n8k32" in ptx_text,
        "has_s8": ".s8" in ptx_text or "s8." in ptx_text,
        "signature": (
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
            in ptx_text
        ),
    }


def _speedup_gate(speedup: float | None) -> dict[str, Any]:
    if speedup is None:
        return {
            "passed": False,
            "reason": "missing comparable BF16 and INT8 latency",
        }
    if speedup <= 1.0:
        return {
            "passed": False,
            "reason": f"INT8 path is not faster than BF16 baseline: {speedup:.4f}x",
        }
    return {
        "passed": True,
        "reason": f"INT8 path is faster than BF16 baseline: {speedup:.4f}x",
    }


def _true_mma_gate(
    *,
    executed_true_mma_module_count: int,
    executed_tilelang_module_count: int,
    executed_tilelang_fused_static_module_count: int,
    executed_int8_module_count: int,
) -> dict[str, Any]:
    if executed_int8_module_count <= 0:
        return {"passed": False, "reason": "no INT8 modules executed"}
    if executed_true_mma_module_count <= 0:
        return {"passed": False, "reason": "no executed module reported true INT8 MMA"}
    if executed_tilelang_module_count <= 0:
        return {"passed": False, "reason": "no executed module used TileLang"}
    if executed_tilelang_fused_static_module_count <= 0:
        return {"passed": False, "reason": "no executed module used fused static TileLang INT8 MMA"}
    return {
        "passed": True,
        "reason": (
            f"{executed_tilelang_fused_static_module_count}/{executed_int8_module_count} executed "
            "INT8 modules used fused static TileLang INT8 MMA"
        ),
    }


def _linear_acceleration_gate(
    *,
    dynamic_speedup: float | None,
    static_speedup: float | None,
    prequantized_speedup: float | None,
    fused_static_speedup: float | None,
    static_activation_quant_engine: str,
) -> dict[str, Any]:
    static_gate = _speedup_gate(static_speedup)
    prequantized_gate = _speedup_gate(prequantized_speedup)
    fused_static_gate = _speedup_gate(fused_static_speedup)
    return {
        "passed": bool(fused_static_gate["passed"] and prequantized_gate["passed"]),
        "dynamic_speedup": dynamic_speedup,
        "static_speedup": static_speedup,
        "prequantized_speedup": prequantized_speedup,
        "fused_static_speedup": fused_static_speedup,
        "static_activation_quant_engine": static_activation_quant_engine,
        "static": static_gate,
        "prequantized": prequantized_gate,
        "fused_static": fused_static_gate,
    }


def _normalize_ocr_text(text: str) -> str:
    return " ".join(str(text).split())


def _compact_keyword_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower())


def _ocr_text_quality_gate(
    *,
    reference: str,
    candidate: str,
    min_text_similarity: float,
    required_keywords: tuple[str, ...],
) -> dict[str, Any]:
    reference_normalized = _normalize_ocr_text(reference)
    candidate_normalized = _normalize_ocr_text(candidate)
    similarity = (
        SequenceMatcher(None, reference_normalized, candidate_normalized).ratio()
        if reference_normalized or candidate_normalized
        else 1.0
    )
    similarity_passed = (
        True if not reference_normalized else similarity >= float(min_text_similarity)
    )
    candidate_compact = _compact_keyword_text(candidate_normalized)
    missing_keywords = [
        keyword
        for keyword in required_keywords
        if _compact_keyword_text(keyword) not in candidate_compact
    ]
    keywords_passed = not missing_keywords
    passed = bool(similarity_passed and keywords_passed)
    return {
        "passed": passed,
        "reference_available": bool(reference_normalized),
        "exact_match": reference_normalized == candidate_normalized,
        "text_similarity": float(similarity),
        "min_text_similarity": float(min_text_similarity),
        "similarity_passed": bool(similarity_passed),
        "required_keywords": list(required_keywords),
        "missing_keywords": missing_keywords,
        "keywords_passed": bool(keywords_passed),
    }


def _summarize_execution_metadata(metadata_items: list[dict[str, Any]]) -> dict[str, Any]:
    input_rows: dict[str, int] = {}
    padded_rows: dict[str, int] = {}
    shapes: dict[str, int] = {}
    engines: dict[str, int] = {}
    quant_engines: dict[str, int] = {}
    for metadata in metadata_items:
        input_row = str(metadata.get("input_rows"))
        padded_row = str(metadata.get("padded_rows"))
        shape = (
            f"{metadata.get('input_rows')}x{metadata.get('input_features')}"
            f"->{metadata.get('output_features')}"
        )
        engine = str(metadata.get("engine"))
        quant_engine = str(metadata.get("activation_quant_engine"))
        input_rows[input_row] = input_rows.get(input_row, 0) + 1
        padded_rows[padded_row] = padded_rows.get(padded_row, 0) + 1
        shapes[shape] = shapes.get(shape, 0) + 1
        engines[engine] = engines.get(engine, 0) + 1
        quant_engines[quant_engine] = quant_engines.get(quant_engine, 0) + 1
    return {
        "input_rows": input_rows,
        "padded_rows": padded_rows,
        "shapes": shapes,
        "engines": engines,
        "activation_quant_engines": quant_engines,
    }


def _clone_ocr_call_kwargs(full: MappingLike, image_path: Path, output_path: Path) -> dict[str, Any]:
    return {
        "prompt": str(full["prompt"]),
        "image_file": str(image_path),
        "output_path": str(output_path),
        "base_size": int(full["base_size"]),
        "image_size": int(full["image_size"]),
        "crop_mode": bool(full["crop_mode"]),
        "max_length": int(full["max_length"]),
        "no_repeat_ngram_size": int(full["no_repeat_ngram_size"]),
        "ngram_window": int(full["ngram_window"]),
        "save_results": bool(full["save_results"]),
        "eval_mode": True,
    }


def _measure_ocr_infer(
    model: nn.Module,
    tokenizer: object,
    *,
    device: torch.device,
    full: MappingLike,
    image_path: Path,
    output_path: Path,
) -> tuple[str, float]:
    output_path.mkdir(parents=True, exist_ok=True)
    _sync(device)
    infer_start = time.perf_counter()
    with torch.inference_mode():
        output_text = model.infer(
            tokenizer,
            **_clone_ocr_call_kwargs(full, image_path, output_path),
        )
    _sync(device)
    return str(output_text), time.perf_counter() - infer_start


def _collect_linear_input_scales(
    model: nn.Module,
    tokenizer: object,
    *,
    full: MappingLike,
    image_path: Path,
    output_path: Path,
    policy: MappingLike,
    eps: float = 1e-6,
) -> tuple[dict[str, float], dict[str, Any]]:
    from xqt.compression.quant.policy import QuantizationPolicy, should_quantize_module

    quant_policy = QuantizationPolicy(
        dtype=str(policy.get("dtype", "int8")),
        scheme=str(policy.get("scheme", "dynamic_mma")),
        include_module_types=tuple(str(item) for item in policy.get("include_module_types", ("Linear",))),
        exclude_module_types=tuple(str(item) for item in policy.get("exclude_module_types", ())),
        include_name_patterns=tuple(str(item) for item in policy.get("include_name_patterns", ())),
        exclude_name_patterns=tuple(str(item) for item in policy.get("exclude_name_patterns", ())),
        include_module_names=tuple(str(item) for item in policy.get("include_module_names", ())),
        exclude_module_names=tuple(str(item) for item in policy.get("exclude_module_names", ())),
        min_parameters=int(policy.get("min_parameters", 0)),
    )
    accumulators: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    matched = 0
    selection_mode = str(policy.get("selection_mode", "default"))
    include_patterns = tuple(str(item) for item in policy.get("include_name_patterns", ()))
    include_names = tuple(str(item) for item in policy.get("include_module_names", ()))

    def should_collect(module_name: str, module: nn.Module) -> bool:
        if selection_mode == "include_only":
            included = module_name in include_names or any(
                re.search(pattern, module_name) for pattern in include_patterns
            )
            return included and should_quantize_module(module_name, module, quant_policy)
        return should_quantize_module(module_name, module, quant_policy)

    def make_hook(module_name: str) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        def hook(_module: nn.Module, inputs: tuple[object, ...], _output: object) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            tensor = inputs[0].detach()
            if tensor.numel() == 0:
                return
            max_abs = tensor.float().abs().amax()
            previous = accumulators.get(module_name)
            accumulators[module_name] = max_abs if previous is None else torch.maximum(previous, max_abs)

        return hook

    try:
        for name, module in model.named_modules():
            if not name or not isinstance(module, nn.Linear):
                continue
            if not should_collect(name, module):
                continue
            matched += 1
            handles.append(module.register_forward_hook(make_hook(name)))
        _measure_ocr_infer(
            model,
            tokenizer,
            device=next(model.parameters()).device,
            full=full,
            image_path=image_path,
            output_path=output_path,
        )
    finally:
        while handles:
            handles.pop().remove()

    scales = {
        name: float(max(value.detach().float().cpu().item(), float(eps)) / 127.0)
        for name, value in accumulators.items()
    }
    return scales, {
        "matched_linear_modules": matched,
        "calibrated_linear_modules": len(scales),
        "missing_scale_modules": matched - len(scales),
        "scale_mode": "static_signed_int8_per_tensor",
    }


def _environment_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "target_arch": _target_arch(device),
    }
    if device.type == "cuda":
        report["device_name"] = torch.cuda.get_device_name(device)
        report["capability"] = list(torch.cuda.get_device_capability(device))
    for module_name in ("tilelang", "triton", "tensorrt", "transformers"):
        try:
            module = __import__(module_name)
            report[module_name] = str(getattr(module, "__version__", "imported"))
        except Exception as exc:
            report[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    return report


def run_tilelang_probe(device: torch.device, artifact_dir: Path) -> dict[str, Any]:
    if device.type != "cuda":
        return {"status": "skipped", "reason": "CUDA is required for TileLang INT8 MMA"}
    settings = dict(CONFIG["tilelang_probe"])
    quant = dict(CONFIG["quantization"])
    m, n, k = int(settings["m"]), int(settings["n"]), int(settings["k"])
    target_arch = _target_arch(device)
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (m, k), device=device, dtype=torch.int8)
    b = torch.randint(-127, 128, (k, n), device=device, dtype=torch.int8)
    kernel = build_tilelang_int8_mma_kernel(
        m,
        n,
        k,
        block_m=int(quant["block_m"]),
        block_n=int(quant["block_n"]),
        block_k=int(quant["block_k"]),
        threads=int(quant["threads"]),
        num_stages=int(quant["num_stages"]),
        target_arch=target_arch,
    )
    ptx_path = artifact_dir / "tilelang_int8_mma_probe.ptx"
    ptx_evidence: dict[str, Any]
    try:
        kernel.export_ptx(str(ptx_path))
        ptx_text = ptx_path.read_text(errors="ignore")
        ptx_evidence = _ptx_int8_mma_evidence(ptx_path, ptx_text)
    except Exception as exc:
        ptx_evidence = {
            "path": str(ptx_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    fused_ptx_path = artifact_dir / "tilelang_int8_linear_static_activation_probe.ptx"
    fused_ptx_evidence: dict[str, Any]
    try:
        fused_kernel = build_tilelang_int8_linear_static_activation_kernel(
            m,
            n,
            k,
            input_dtype="bfloat16",
            output_dtype="bfloat16",
            has_bias=False,
            block_m=int(quant["block_m"]),
            block_n=int(quant["block_n"]),
            block_k=int(quant["block_k"]),
            threads=int(quant["threads"]),
            num_stages=int(quant["num_stages"]),
            target_arch=target_arch,
        )
        fused_kernel.export_ptx(str(fused_ptx_path))
        fused_ptx_text = fused_ptx_path.read_text(errors="ignore")
        fused_ptx_evidence = _ptx_int8_mma_evidence(fused_ptx_path, fused_ptx_text)
    except Exception as exc:
        fused_ptx_evidence = {
            "path": str(fused_ptx_path),
            "error": f"{type(exc).__name__}: {exc}",
        }

    output = int8_mma_tilelang(
        a,
        b,
        block_m=int(quant["block_m"]),
        block_n=int(quant["block_n"]),
        block_k=int(quant["block_k"]),
        threads=int(quant["threads"]),
        num_stages=int(quant["num_stages"]),
        target_arch=target_arch,
    )
    reference = torch._int_mm(a, b)
    _sync(device)
    correctness = {
        "max_abs": int((output - reference).abs().max().item()),
        "exact": bool(torch.equal(output, reference)),
        "output_dtype": str(output.dtype),
    }
    tilelang_ms = _benchmark(
        lambda: int8_mma_tilelang(
            a,
            b,
            block_m=int(quant["block_m"]),
            block_n=int(quant["block_n"]),
            block_k=int(quant["block_k"]),
            threads=int(quant["threads"]),
            num_stages=int(quant["num_stages"]),
            target_arch=target_arch,
        ),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    torch_int_mm_ms = _benchmark(
        lambda: torch._int_mm(a, b),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    return {
        "status": "ok",
        "shape": {"m": m, "n": n, "k": k},
        "engine": "tilelang",
        "target_arch": target_arch,
        "ptx_evidence": ptx_evidence,
        "fused_static_linear_ptx_evidence": fused_ptx_evidence,
        "correctness": correctness,
        "latency_ms": {
            "tilelang_int8_mma": tilelang_ms,
            "torch_int_mm": torch_int_mm_ms,
            "tilelang_vs_torch_int_mm": torch_int_mm_ms / tilelang_ms
            if tilelang_ms > 0
            else None,
        },
    }


def run_linear_benchmark(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"status": "skipped", "reason": "CUDA is required for true INT8 MMA"}
    settings = dict(CONFIG["linear_benchmark"])
    quant = dict(CONFIG["quantization"])
    m = int(settings["m"])
    in_features = int(settings["in_features"])
    out_features = int(settings["out_features"])
    dtype = _dtype_from_name(str(settings["dtype"]))
    block_m = int(quant["block_m"])
    block_n = int(quant["block_n"])
    block_k = int(quant["block_k"])
    threads = int(quant["threads"])
    num_stages = int(quant["num_stages"])
    target_arch = _target_arch(device)
    torch.manual_seed(int(settings["seed"]))
    source = torch.nn.Linear(
        in_features,
        out_features,
        bias=False,
        dtype=dtype,
        device=device,
    ).eval()
    qlinear = Int8MmaLinear.from_linear(
        source,
        engine=str(quant["engine"]),
        fallback_engine=str(quant["fallback_engine"]),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
    ).to(device=device).eval()
    static_qlinear = Int8MmaLinear.from_linear(
        source,
        engine=str(quant["engine"]),
        fallback_engine=str(quant["fallback_engine"]),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
        activation_scale_mode="static",
    ).to(device=device).eval()
    inputs = torch.randn(m, in_features, device=device, dtype=dtype)

    def run_prequantized(
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor,
        qlinear_module: Int8MmaLinear,
    ) -> torch.Tensor:
        padded, original_rows = pad_rows_to_block(qactivation, block_m)
        output = int8_linear_tilelang(
            padded,
            qlinear_module.qweight_t,
            activation_scale.reshape(1),
            qlinear_module.weight_scale,
            qlinear_module.bias,
            output_dtype=dtype,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            target_arch=target_arch,
        )
        return output[:original_rows]

    def run_fused_static() -> torch.Tensor:
        padded, original_rows = pad_rows_to_block(inputs.reshape(-1, in_features), block_m)
        output = int8_linear_static_activation_tilelang(
            padded,
            static_qlinear.qweight_t,
            static_activation_scale.reshape(1),
            static_qlinear.weight_scale,
            static_qlinear.bias,
            output_dtype=dtype,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            target_arch=target_arch,
        )
        return output[:original_rows]

    with torch.no_grad():
        static_scale = static_qlinear.calibrate_static_activation_scale(inputs)
        reference = source(inputs)
        dynamic_candidate = qlinear(inputs)
        static_candidate = static_qlinear(inputs)
        qactivation, activation_scale, _ = qlinear._quantize_activation(inputs)
        static_qactivation, static_activation_scale, static_activation_quant_engine = (
            static_qlinear._quantize_activation(
                inputs,
                prefer_tilelang=True,
            )
        )
        prequantized_candidate = run_prequantized(
            qactivation,
            activation_scale.reshape(1),
            qlinear,
        )
        static_prequantized_candidate = run_prequantized(
            static_qactivation,
            static_activation_scale.reshape(1),
            static_qlinear,
        )
        fused_static_candidate = run_fused_static()
    _sync(device)
    baseline_ms = _benchmark(
        lambda: F.linear(inputs, source.weight, source.bias),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    int8_ms = _benchmark(
        lambda: qlinear(inputs),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    static_int8_ms = _benchmark(
        lambda: static_qlinear(inputs),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    static_activation_quant_ms = _benchmark(
        lambda: static_activation_quantize_tilelang(
            inputs.reshape(-1, in_features),
            static_scale,
            block_size=int(quant.get("activation_quant_block_size", 256)),
            target_arch=_target_arch(device),
        ),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    prequantized_ms = _benchmark(
        lambda: run_prequantized(qactivation, activation_scale.reshape(1), qlinear),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    static_prequantized_ms = _benchmark(
        lambda: run_prequantized(
            static_qactivation,
            static_activation_scale.reshape(1),
            static_qlinear,
        ),
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    fused_static_ms = _benchmark(
        run_fused_static,
        device=device,
        warmup=int(settings["warmup"]),
        iterations=int(settings["iterations"]),
    )
    dynamic_speedup = baseline_ms / int8_ms if int8_ms > 0 else None
    static_speedup = baseline_ms / static_int8_ms if static_int8_ms > 0 else None
    prequantized_speedup = (
        baseline_ms / static_prequantized_ms if static_prequantized_ms > 0 else None
    )
    fused_static_speedup = baseline_ms / fused_static_ms if fused_static_ms > 0 else None
    acceleration_gate = _linear_acceleration_gate(
        dynamic_speedup=dynamic_speedup,
        static_speedup=static_speedup,
        prequantized_speedup=prequantized_speedup,
        fused_static_speedup=fused_static_speedup,
        static_activation_quant_engine=static_activation_quant_engine,
    )
    return {
        "status": "ok",
        "shape": {
            "m": m,
            "in_features": in_features,
            "out_features": out_features,
        },
        "dtype": str(dtype),
        "latency_ms": {
            "bf16_or_fp16_linear": baseline_ms,
            "dynamic_int8_mma_linear": int8_ms,
            "dynamic_speedup": dynamic_speedup,
            "static_int8_mma_linear": static_int8_ms,
            "static_speedup": static_speedup,
            "static_activation_quantize": static_activation_quant_ms,
            "dynamic_prequantized_int8_mma_linear": prequantized_ms,
            "dynamic_prequantized_speedup": baseline_ms / prequantized_ms
            if prequantized_ms > 0
            else None,
            "static_prequantized_int8_mma_linear": static_prequantized_ms,
            "static_prequantized_speedup": prequantized_speedup,
            "fused_static_int8_mma_linear": fused_static_ms,
            "fused_static_speedup": fused_static_speedup,
            "dynamic_quantization_overhead_ms": int8_ms - prequantized_ms,
            "static_quantization_overhead_ms": static_int8_ms - static_prequantized_ms,
        },
        "acceleration_verified": acceleration_gate["passed"],
        "acceleration_gate": acceleration_gate,
        "dynamic_accuracy": _tensor_diff(reference, dynamic_candidate),
        "static_accuracy": _tensor_diff(reference, static_candidate),
        "prequantized_accuracy": _tensor_diff(reference, prequantized_candidate),
        "static_prequantized_accuracy": _tensor_diff(reference, static_prequantized_candidate),
        "fused_static_accuracy": _tensor_diff(reference, fused_static_candidate),
        "dynamic_execution_metadata": qlinear.execution_metadata(),
        "static_execution_metadata": static_qlinear.execution_metadata(),
        "static_activation_quant_engine": static_activation_quant_engine,
        "static_activation_scale": float(static_scale.detach().cpu().item()),
        "note": (
            "The INT8 path is true W8A8 MMA. End-to-end speedup also depends on "
            "activation quantization and scale epilogue overhead. Dynamic mode computes "
            "activation range every call; static mode uses a calibrated scale; the "
            "prequantized measurement isolates the MMA plus dequant epilogue path."
        ),
    }


def _ensure_sample_image(path: Path) -> Path:
    if path.is_file():
        return path
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to create a sample OCR image") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    draw.text((96, 128), "Unlimited-OCR INT8 MMA smoke image", fill="black")
    draw.text((96, 192), "This file is generated under artifacts/.", fill="black")
    image.save(path)
    return path


def run_full_ocr(device: torch.device, artifact_dir: Path) -> dict[str, Any]:
    full = dict(CONFIG["full_ocr"])
    if not bool(CONFIG["run"].get("full_ocr")) and not bool(full.get("enabled")):
        return {
            "status": "skipped",
            "reason": "set CONFIG['run']['full_ocr'] or CONFIG['full_ocr']['enabled'] to True",
        }
    if device.type != "cuda":
        return {"status": "skipped", "reason": "CUDA is required for Unlimited-OCR"}
    from transformers import AutoModel, AutoTokenizer

    image_file = str(full.get("image_file") or "")
    if image_file:
        image_path = Path(image_file)
    else:
        image_path = artifact_dir / "sample_ocr_input.png"
    image_path = _ensure_sample_image(image_path)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if CONFIG["revision"] is not None:
        model_kwargs["revision"] = CONFIG["revision"]
    tokenizer_kwargs = {"trust_remote_code": True}
    if CONFIG["revision"] is not None:
        tokenizer_kwargs["revision"] = CONFIG["revision"]
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], **tokenizer_kwargs)
    model = AutoModel.from_pretrained(CONFIG["model_id"], **model_kwargs).eval()
    quant = dict(CONFIG["quantization"])
    model = model.to(device=device)
    load_seconds = time.perf_counter() - load_start
    output_path = Path(str(full["output_path"]))
    output_path.mkdir(parents=True, exist_ok=True)
    baseline_output_text = ""
    baseline_seconds: float | None = None
    calibration_scales: dict[str, float] | None = None
    calibration_metadata: dict[str, Any] | None = None
    calibration_seconds: float | None = None
    if bool(full.get("compare_bf16", True)):
        baseline_output_path = output_path / "bf16_baseline"
        baseline_output_text, baseline_seconds = _measure_ocr_infer(
            model,
            tokenizer,
            device=device,
            full=full,
            image_path=image_path,
            output_path=baseline_output_path,
        )
    if bool(full.get("calibrate_static_scales", True)):
        calibration_start = time.perf_counter()
        calibration_output_path = output_path / "static_scale_calibration"
        calibration_scales, calibration_metadata = _collect_linear_input_scales(
            model,
            tokenizer,
            full=full,
            image_path=image_path,
            output_path=calibration_output_path,
            policy=quant["policy"],
        )
        calibration_seconds = time.perf_counter() - calibration_start
    quantize_start = time.perf_counter()
    quant_result = quantize_with_int8_mma(
        model,
        policy=quant["policy"],
        strategy="tilelang_int8_mma",
        inplace=True,
        engine=str(quant["engine"]),
        fallback_engine=str(quant["fallback_engine"]),
        block_m=int(quant["block_m"]),
        block_n=int(quant["block_n"]),
        block_k=int(quant["block_k"]),
        threads=int(quant["threads"]),
        num_stages=int(quant["num_stages"]),
        activation_scale_mode="static" if calibration_scales else "dynamic",
        activation_scales=calibration_scales,
        activation_quant_block_size=int(quant.get("activation_quant_block_size", 256)),
    )
    model = quant_result.model.eval().to(device=device)
    quantize_seconds = time.perf_counter() - quantize_start
    int8_output_path = output_path / "int8_static" if calibration_scales else output_path / "int8_dynamic"
    cold_output_text, cold_infer_seconds = _measure_ocr_infer(
        model,
        tokenizer,
        device=device,
        full=full,
        image_path=image_path,
        output_path=int8_output_path,
    )
    steady_output_text = cold_output_text
    steady_infer_seconds = cold_infer_seconds
    if bool(full.get("measure_steady_state", True)):
        steady_output_path = (
            output_path / "int8_static_steady_state"
            if calibration_scales
            else output_path / "int8_dynamic_steady_state"
        )
        steady_output_text, steady_infer_seconds = _measure_ocr_infer(
            model,
            tokenizer,
            device=device,
            full=full,
            image_path=image_path,
            output_path=steady_output_path,
        )
    output_text = steady_output_text
    infer_seconds = steady_infer_seconds
    (output_path / "int8_output.txt").write_text(str(output_text), encoding="utf-8")
    if baseline_output_text:
        (output_path / "bf16_output.txt").write_text(baseline_output_text, encoding="utf-8")
    int8_modules = [
        module.execution_metadata()
        for module in model.modules()
        if isinstance(module, Int8MmaLinear) and module.execution_metadata().get("engine") != "not_run"
    ]
    true_mma_modules = sum(1 for metadata in int8_modules if metadata.get("true_int8_mma"))
    tilelang_modules = sum(1 for metadata in int8_modules if metadata.get("engine") == "tilelang")
    tilelang_static_quant_modules = sum(
        1 for metadata in int8_modules if metadata.get("activation_quant_engine") == "tilelang_static"
    )
    tilelang_fused_static_modules = sum(
        1 for metadata in int8_modules if metadata.get("activation_quant_engine") == "tilelang_fused_static"
    )
    cold_speedup = (
        baseline_seconds / cold_infer_seconds
        if baseline_seconds is not None and cold_infer_seconds > 0
        else None
    )
    speedup = baseline_seconds / infer_seconds if baseline_seconds is not None and infer_seconds > 0 else None
    true_mma_status = _true_mma_gate(
        executed_true_mma_module_count=true_mma_modules,
        executed_tilelang_module_count=tilelang_modules,
        executed_tilelang_fused_static_module_count=tilelang_fused_static_modules,
        executed_int8_module_count=len(int8_modules),
    )
    speedup_status = _speedup_gate(speedup)
    acceleration_verified = bool(true_mma_status["passed"] and speedup_status["passed"])
    quality_config = dict(full.get("quality_gate", {}))
    required_keywords = tuple(
        str(keyword) for keyword in quality_config.get("required_keywords", ())
    )
    min_text_similarity = float(quality_config.get("min_text_similarity", 0.98))
    cold_text_quality = _ocr_text_quality_gate(
        reference=baseline_output_text,
        candidate=str(cold_output_text),
        min_text_similarity=min_text_similarity,
        required_keywords=required_keywords,
    )
    text_quality = _ocr_text_quality_gate(
        reference=baseline_output_text,
        candidate=str(output_text),
        min_text_similarity=min_text_similarity,
        required_keywords=required_keywords,
    )
    quality_verified = bool(text_quality["passed"])
    deployment_verified = bool(acceleration_verified and quality_verified)
    return {
        "status": "ok",
        "model_id": CONFIG["model_id"],
        "recommended_strategy": "moe_down_only_accuracy_safe_int8_mma",
        "load_seconds": load_seconds,
        "calibration_seconds": calibration_seconds,
        "quantize_seconds": quantize_seconds,
        "load_and_quantize_seconds": load_seconds + quantize_seconds,
        "bf16_infer_seconds": baseline_seconds,
        "int8_cold_infer_seconds": cold_infer_seconds,
        "int8_cold_speedup_vs_bf16": cold_speedup,
        "int8_infer_seconds": infer_seconds,
        "int8_steady_state_infer_seconds": steady_infer_seconds,
        "int8_speedup_vs_bf16": speedup,
        "int8_steady_state_speedup_vs_bf16": speedup,
        "acceleration_verified": acceleration_verified,
        "quality_verified": quality_verified,
        "deployment_verified": deployment_verified,
        "acceleration_gate": {
            "passed": acceleration_verified,
            "true_int8_mma": true_mma_status,
            "speedup": speedup_status,
        },
        "quality_gate": {
            "passed": quality_verified,
            "cold": cold_text_quality,
            "steady_state": text_quality,
        },
        "deployment_gate": {
            "passed": deployment_verified,
            "acceleration": acceleration_verified,
            "quality": quality_verified,
        },
        "image_file": str(image_path),
        "output_path": str(output_path),
        "quantized_module_count": len(quant_result.quantized_modules),
        "first_quantized_modules": quant_result.quantized_modules[:20],
        "quantization_metadata": quant_result.metadata,
        "calibration_metadata": calibration_metadata,
        "executed_int8_module_count": len(int8_modules),
        "executed_tilelang_module_count": tilelang_modules,
        "executed_true_mma_module_count": true_mma_modules,
        "executed_tilelang_static_quant_module_count": tilelang_static_quant_modules,
        "executed_tilelang_fused_static_module_count": tilelang_fused_static_modules,
        "execution_shape_summary": _summarize_execution_metadata(int8_modules),
        "first_execution_metadata": int8_modules[:20],
        "bf16_output_preview": baseline_output_text[:1000],
        "cold_output_preview": str(cold_output_text)[:1000],
        "output_preview": str(output_text)[:1000],
    }


def main() -> None:
    artifact_dir = _artifact_dir()
    device = _runtime_device()
    report: dict[str, Any] = {
        "objective": "baidu/Unlimited-OCR true INT8 MMA inference path",
        "model_id": CONFIG["model_id"],
        "environment": _environment_report(device),
        "config": CONFIG,
    }
    if CONFIG["run"].get("tilelang_probe", True):
        report["tilelang_probe"] = run_tilelang_probe(device, artifact_dir)
    if CONFIG["run"].get("linear_benchmark", True):
        report["linear_benchmark"] = run_linear_benchmark(device)
    report["full_ocr"] = run_full_ocr(device, artifact_dir)
    report_path = artifact_dir / "unlimited_ocr_int8_report.json"
    _write_json(report_path, report)
    print(f"Report: {report_path}")
    linear = report.get("linear_benchmark", {})
    if isinstance(linear, dict) and linear.get("status") == "ok":
        dynamic_speedup = linear.get("latency_ms", {}).get("dynamic_speedup")
        static_speedup = linear.get("latency_ms", {}).get("static_speedup")
        if dynamic_speedup:
            print(f"Dynamic Linear INT8 speedup: {dynamic_speedup:.4f}x")
        if static_speedup:
            print(f"Static Linear INT8 speedup: {static_speedup:.4f}x")
    full_status = report.get("full_ocr", {}).get("status")
    print(f"Full OCR status: {full_status}")
    full = report.get("full_ocr", {})
    if isinstance(full, dict) and full.get("status") == "ok":
        speedup = full.get("int8_steady_state_speedup_vs_bf16")
        quality = full.get("quality_gate", {}).get("steady_state", {})
        if speedup:
            print(f"Full OCR INT8 steady-state speedup: {speedup:.4f}x")
        if isinstance(quality, dict):
            print(
                "Full OCR quality: "
                f"passed={quality.get('passed')} "
                f"similarity={quality.get('text_similarity')}"
            )


if __name__ == "__main__":
    main()
