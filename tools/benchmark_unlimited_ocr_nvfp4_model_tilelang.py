"""Benchmark real Unlimited-OCR NVFP4 model routing through XQT TileLang.

This script intentionally uses explicit in-file settings instead of command-line
arguments, matching the workspace script convention.  It measures a text-only
language-prefill path so the benchmark focuses on NVFP4 Linear routing rather
than OCR image preprocessing.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.benchmark import LatencyReport, measure_callable_ms
from xqt.operator_opt import OperatorOptimizationTargetPlan, materialize_operator_candidate_models
from xqt.quant import infer_nvfp4_tensor_layout


REPO_ID = "sahilchachra/Unlimited-OCR-NVFP4"
HF_CACHE_ROOT = Path.home() / ".cache/huggingface/hub/models--sahilchachra--Unlimited-OCR-NVFP4"
SNAPSHOT_REVISION = "2ac0aa28a3509c7056648ffc0d4f9fdb79a454d0"
DEFAULT_ARTIFACT_DIR = Path("artifacts/xqt/benchmarks/unlimited_ocr_nvfp4_model_tilelang")
DEFAULT_OCR_IMAGE_PATH = REPO_ROOT / "research/unlimited-ocr/outputs/resume_page1_for_nvfp4.png"
DEFAULT_OCR_PROMPT = "<image>\nExtract the text in the image. "
DEFAULT_OCR_BASE_SIZE = 1024
DEFAULT_OCR_IMAGE_SIZE = 640
DEFAULT_OCR_CROP_MODE = True

BENCHMARK_SEED = 0
SEQ_LEN = 128
WARMUP = 1
ITERATIONS = 3
MAX_TARGETS: int | None = None
TARGET_ARCH: str | None = None
BENCHMARK_DTYPE = torch.float16


def _force_offline_hf() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules_xdl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-xdl")


def _find_snapshot_path() -> Path:
    explicit = HF_CACHE_ROOT / "snapshots" / SNAPSHOT_REVISION
    if explicit.exists():
        return explicit
    snapshots_dir = HF_CACHE_ROOT / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"local Hugging Face snapshot directory does not exist: {snapshots_dir}"
        )
    candidates = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no local snapshots found under {snapshots_dir}")
    return candidates[-1]


def _cuda_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"sm_{major}{minor}"


def _has_usable_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
    except Exception:
        return False
    return True


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _latency_report(samples_ms: list[float]) -> LatencyReport:
    sorted_samples = sorted(samples_ms)
    return LatencyReport(
        iterations=len(samples_ms),
        warmup=WARMUP,
        mean_ms=sum(samples_ms) / len(samples_ms),
        p50_ms=_percentile(sorted_samples, 50),
        p90_ms=_percentile(sorted_samples, 90),
        p99_ms=_percentile(sorted_samples, 99),
        samples_ms=samples_ms,
    )


def _benchmark_sequential(
    fn: Any,
    *,
    device: str,
) -> dict[str, Any]:
    samples_ms: list[float] = []

    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        _sync_cuda()

        for _ in range(ITERATIONS):
            samples_ms.append(
                measure_callable_ms(
                    fn,
                    sync_cuda=device == "cuda",
                    device=device,
                )
            )

    return _latency_report(samples_ms).to_dict()


def _speedup_report(
    official_report: dict[str, Any],
    xqt_report: dict[str, Any],
) -> dict[str, Any]:
    speedup_report = {
        "mean_ms": (
            float(official_report["mean_ms"]) / float(xqt_report["mean_ms"])
            if float(xqt_report["mean_ms"]) > 0.0
            else None
        ),
        "p50_ms": (
            float(official_report["p50_ms"]) / float(xqt_report["p50_ms"])
            if float(xqt_report["p50_ms"]) > 0.0
            else None
        ),
    }
    return speedup_report


def _load_model(snapshot_path: Path, *, device: str) -> torch.nn.Module:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(snapshot_path),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=BENCHMARK_DTYPE if device == "cuda" else torch.float32,
        local_files_only=True,
    )
    model.eval()
    model.to(device=device)
    if device == "cuda":
        model.to(dtype=BENCHMARK_DTYPE)
    return model


def _vocab_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return value
    language_config = getattr(config, "language_config", None)
    value = getattr(language_config, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return value
    return 129280


def _make_inputs(model: torch.nn.Module, *, device: str) -> dict[str, torch.Tensor]:
    vocab_size = _vocab_size(model)
    input_ids = torch.randint(
        low=2,
        high=max(3, vocab_size - 1),
        size=(1, SEQ_LEN),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.detach().clone()
            continue
        if isinstance(value, list):
            cloned[name] = [
                tuple(item.detach().clone() if isinstance(item, torch.Tensor) else item for item in value_tuple)
                if isinstance(value_tuple, tuple)
                else value_tuple
                for value_tuple in value
            ]
            continue
        cloned[name] = value
    return cloned


def _model_runtime_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return BENCHMARK_DTYPE


def _forward_model(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    call_inputs = dict(inputs)
    call_inputs.setdefault("use_cache", False)
    call_inputs.setdefault("return_dict", True)
    runtime_dtype = _model_runtime_dtype(model)
    with torch.no_grad():
        if inputs["input_ids"].is_cuda and runtime_dtype in {torch.float16, torch.bfloat16}:
            with torch.autocast("cuda", dtype=runtime_dtype):
                outputs = model(**call_inputs)
        else:
            outputs = model(**call_inputs)
    logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, tuple) else outputs)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model forward did not return tensor logits")
    return logits[:, -1, :].detach()


def _prepare_ocr_prefill_inputs(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    image_path: Path,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
) -> dict[str, Any]:
    module_impl = sys.modules[model.__class__.__module__]
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [str(image_path)],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    formatted_prompt = module_impl.format_messages(
        conversations=conversation,
        sft_format="plain",
        system_prompt="",
    )
    patch_size = 16
    downsample_ratio = 4
    images = module_impl.load_pil_images(conversation)
    image_transform = module_impl.BasicImageTransform(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        normalize=True,
    )
    image_token = "<image>"
    image_token_id = 128815
    text_splits = formatted_prompt.split(image_token)
    images_list: list[torch.Tensor] = []
    images_crop_list: list[torch.Tensor] = []
    images_seq_mask: list[bool] = []
    tokenized_str: list[int] = []
    images_spatial_crop: list[list[int]] = []

    for text_sep, image in zip(text_splits, images):
        tokenized_sep = module_impl.text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)
        if not crop_mode:
            raise RuntimeError("OCR benchmark currently assumes crop_mode=True")
        if image.size[0] <= 640 and image.size[1] <= 640:
            crop_ratio = [1, 1]
            images_crop_raw: list[Any] = []
        else:
            images_crop_raw, crop_ratio = module_impl.dynamic_preprocess(image)
        global_view = module_impl.ImageOps.pad(
            image,
            (base_size, base_size),
            color=tuple(int(x * 255) for x in image_transform.mean),
        )
        images_list.append(image_transform(global_view).to(torch.bfloat16))
        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])
        if width_crop_num > 1 or height_crop_num > 1:
            for crop in images_crop_raw:
                images_crop_list.append(image_transform(crop).to(torch.bfloat16))
        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
        tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
        tokenized_image += [image_token_id]
        if width_crop_num > 1 or height_crop_num > 1:
            tokenized_image += (
                ([image_token_id] * (num_queries * width_crop_num) + [image_token_id])
                * (num_queries * height_crop_num)
            )
        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = module_impl.text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    bos_id = 0
    tokenized_str = [bos_id] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask_tensor = torch.tensor(images_seq_mask, dtype=torch.bool)
    images_ori = torch.stack(images_list, dim=0)
    images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
    if images_crop_list:
        images_crop = torch.stack(images_crop_list, dim=0)
    else:
        images_crop = torch.zeros((1, 3, base_size, base_size))

    return {
        "input_ids": input_ids.unsqueeze(0).cuda(),
        "images": [(images_crop.cuda(), images_ori.cuda())],
        "images_seq_mask": images_seq_mask_tensor.unsqueeze(0).cuda(),
        "images_spatial_crop": images_spatial_crop_tensor,
        "use_cache": False,
        "return_dict": True,
    }


def _benchmark_ocr_prefill(
    *,
    snapshot_path: Path,
    targets: list[OperatorOptimizationTargetPlan],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if not DEFAULT_OCR_IMAGE_PATH.exists():
        return {
            "status": "skipped_missing_image",
            "image_path": str(DEFAULT_OCR_IMAGE_PATH),
        }

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    official_model = _load_model(snapshot_path, device="cuda").to(dtype=torch.bfloat16)
    ocr_inputs = _prepare_ocr_prefill_inputs(
        official_model,
        tokenizer,
        image_path=DEFAULT_OCR_IMAGE_PATH,
        prompt=DEFAULT_OCR_PROMPT,
        base_size=DEFAULT_OCR_BASE_SIZE,
        image_size=DEFAULT_OCR_IMAGE_SIZE,
        crop_mode=DEFAULT_OCR_CROP_MODE,
    )
    _forward_model(official_model, ocr_inputs)
    _sync_cuda()
    official_latency = _benchmark_sequential(
        lambda: _forward_model(official_model, ocr_inputs),
        device="cuda",
    )
    official_logits = _forward_model(official_model, ocr_inputs).float().cpu()
    del official_model
    _sync_cuda()
    gc.collect()
    torch.cuda.empty_cache()

    xqt_model = _load_model(snapshot_path, device="cuda").to(dtype=torch.bfloat16)
    xqt_model = materialize_operator_candidate_models(xqt_model, targets, inplace=True)
    _forward_model(xqt_model, ocr_inputs)
    _sync_cuda()
    xqt_latency = _benchmark_sequential(
        lambda: _forward_model(xqt_model, ocr_inputs),
        device="cuda",
    )
    xqt_logits = _forward_model(xqt_model, ocr_inputs).float().cpu()
    diff = (official_logits - xqt_logits).abs()
    speedup = _speedup_report(official_latency, xqt_latency)
    return {
        "status": "ok",
        "image_path": str(DEFAULT_OCR_IMAGE_PATH),
        "prompt": DEFAULT_OCR_PROMPT,
        "base_size": DEFAULT_OCR_BASE_SIZE,
        "image_size": DEFAULT_OCR_IMAGE_SIZE,
        "crop_mode": DEFAULT_OCR_CROP_MODE,
        "official_latency": official_latency,
        "xqt_latency": xqt_latency,
        "speedup": speedup,
        "numeric_diff": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "allclose": bool(torch.allclose(official_logits, xqt_logits, atol=1e-2, rtol=1e-2)),
            "shape": list(official_logits.shape),
        },
    }


def _collect_nvfp4_targets(
    model: torch.nn.Module,
    *,
    target_arch: str | None,
) -> tuple[list[OperatorOptimizationTargetPlan], list[dict[str, Any]]]:
    targets: list[OperatorOptimizationTargetPlan] = []
    samples: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        layout = infer_nvfp4_tensor_layout(module)
        if layout is None:
            continue
        targets.append(
            OperatorOptimizationTargetPlan(
                name=f"{name}_tilelang",
                backend="tilelang",
                target_path=name,
                patterns=["dequant_gemm_epilogue"],
                fallback="eager",
                min_speedup=1.000001,
                validate={"atol": 1e-3, "rtol": 1e-3},
                tilelang={
                    "target": "cuda",
                    "target_arch": target_arch,
                    "linear_runtime": "auto",
                    "linear_fastpath": "auto",
                },
            )
        )
        if len(samples) < 20:
            samples.append(
                {
                    "name": name,
                    "module_type": type(module).__name__,
                    "in_features": getattr(module, "in_features", None),
                    "out_features": getattr(module, "out_features", None),
                    "source_module_type": type(module).__name__,
                    "input_features": layout.input_features,
                    "output_features": layout.output_features,
                    "group_size": layout.group_size,
                }
            )
        if MAX_TARGETS is not None and len(targets) >= MAX_TARGETS:
            break
    return targets, samples


def _tilelang_target_plan_dict(
    *,
    name: str,
    target_arch: str | None,
) -> dict[str, Any]:
    return OperatorOptimizationTargetPlan(
        name=f"{name}_tilelang",
        backend="tilelang",
        target_path=name,
        patterns=["dequant_gemm_epilogue"],
        fallback="eager",
        min_speedup=1.000001,
        validate={"atol": 1e-3, "rtol": 1e-3},
        tilelang={
            "target": "cuda",
            "target_arch": target_arch,
            "linear_runtime": "auto",
            "linear_fastpath": "auto",
        },
    ).to_dict()


def _snapshot_nvfp4_target_manifest(
    snapshot_path: Path,
    *,
    target_arch: str | None,
) -> list[dict[str, Any]]:
    safetensors_path = snapshot_path / "model-00001-of-000001.safetensors"
    if not safetensors_path.exists():
        return []
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        module_names = sorted(
            key.removesuffix(".weight_packed")
            for key in keys
            if key.endswith(".weight_packed")
        )
        manifest: list[dict[str, Any]] = []
        for name in module_names:
            packed = handle.get_tensor(f"{name}.weight_packed")
            scale = handle.get_tensor(f"{name}.weight_scale")
            input_features = int(packed.shape[1]) * 2
            scale_groups = int(scale.shape[1])
            group_size = input_features // scale_groups if scale_groups > 0 else None
            manifest.append(
                {
                    "name": name,
                    "target_path": name,
                    "packed_shape": list(packed.shape),
                    "scale_shape": list(scale.shape),
                    "input_features": input_features,
                    "output_features": int(packed.shape[0]),
                    "group_size": group_size,
                    "target_plan": _tilelang_target_plan_dict(
                        name=name,
                        target_arch=target_arch,
                    ),
                }
            )
    return manifest


def _count_wrapped_modules(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if type(module).__name__ in {
            "_TileLangDequantGemmWrapper",
            "_TileLangEagerDenseLinearModule",
        }:
            count += 1
    return count


def _write_summary(summary: dict[str, Any]) -> Path:
    DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_ARTIFACT_DIR / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def _write_artifact_json(filename: str, data: dict[str, Any] | list[dict[str, Any]]) -> Path:
    DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_ARTIFACT_DIR / filename
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    _force_offline_hf()
    torch.manual_seed(BENCHMARK_SEED)

    snapshot_path = _find_snapshot_path()
    device = "cuda" if _has_usable_cuda() else "cpu"
    target_arch = TARGET_ARCH or _cuda_arch()

    if device != "cuda":
        target_manifest = _snapshot_nvfp4_target_manifest(
            snapshot_path,
            target_arch=target_arch,
        )
        manifest_path = _write_artifact_json("target_manifest.json", target_manifest)
        summary = {
            "repo_id": REPO_ID,
            "snapshot_path": str(snapshot_path),
            "status": "skipped_no_cuda",
            "reason": "model-level latency benchmark requires a usable CUDA device; CPU official forward hits mixed float32/bfloat16 compressed-tensors Linear paths",
            "device": device,
            "target_arch": target_arch,
            "seq_len": SEQ_LEN,
            "target_count_from_safetensors": len(target_manifest),
            "target_manifest_path": str(manifest_path),
            "target_samples": target_manifest[:20],
            "route": {
                "linear_runtime": "auto",
                "linear_fastpath": "auto",
                "sm_89_behavior": "packed NVFP4 dequantizes once into dense cache, then uses native half linear",
                "future_blackwell_behavior": "packed FP4/NVFP4 interfaces remain available for FP4 MMA expansion",
            },
        }
        output_path = _write_summary(summary)
        print(json.dumps(summary, indent=2))
        print(f"wrote {output_path}")
        return

    official_model = _load_model(snapshot_path, device=device)
    targets, target_samples = _collect_nvfp4_targets(
        official_model,
        target_arch=target_arch,
    )
    if not targets:
        raise RuntimeError("no bridgeable NVFP4 Linear modules found in the loaded model")

    official_inputs = _make_inputs(official_model, device=device)
    reference_inputs = _clone_inputs(official_inputs)
    official_last_logits = _forward_model(official_model, official_inputs).float().cpu()

    target_count = len(targets)
    manifest_path = _write_artifact_json(
        "target_manifest.json",
        [target.to_dict() for target in targets],
    )
    del official_model
    del official_inputs
    _sync_cuda()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xqt_model = _load_model(snapshot_path, device=device)
    xqt_model = materialize_operator_candidate_models(xqt_model, targets, inplace=True)
    wrapped_count = _count_wrapped_modules(xqt_model)

    xqt_last_logits = _forward_model(xqt_model, reference_inputs).float().cpu()
    diff = (official_last_logits - xqt_last_logits).abs()
    numeric_diff = {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "allclose": bool(
            torch.allclose(
                official_last_logits,
                xqt_last_logits,
                atol=1e-2,
                rtol=1e-2,
            )
        ),
        "shape": list(official_last_logits.shape),
    }

    del xqt_model
    del xqt_last_logits
    _sync_cuda()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    official_model = _load_model(snapshot_path, device=device)
    benchmark_inputs = _clone_inputs(reference_inputs)
    official_latency = _benchmark_sequential(
        lambda: _forward_model(official_model, benchmark_inputs),
        device=device,
    )
    del official_model
    _sync_cuda()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xqt_model = _load_model(snapshot_path, device=device)
    xqt_model = materialize_operator_candidate_models(xqt_model, targets, inplace=True)
    xqt_latency = _benchmark_sequential(
        lambda: _forward_model(xqt_model, benchmark_inputs),
        device=device,
    )
    speedup = _speedup_report(official_latency, xqt_latency)
    ocr_prefill = _benchmark_ocr_prefill(
        snapshot_path=snapshot_path,
        targets=targets,
    )

    summary = {
        "repo_id": REPO_ID,
        "snapshot_path": str(snapshot_path),
        "device": device,
        "target_arch": target_arch,
        "dtype": str(BENCHMARK_DTYPE if device == "cuda" else torch.float32),
        "seq_len": SEQ_LEN,
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "max_targets": MAX_TARGETS,
        "target_count": target_count,
        "wrapped_count": wrapped_count,
        "target_manifest_path": str(manifest_path),
        "target_samples": target_samples,
        "numeric_diff": numeric_diff,
        "official_latency": official_latency,
        "xqt_latency": xqt_latency,
        "speedup": speedup,
        "ocr_prefill": ocr_prefill,
        "route": {
            "linear_runtime": "auto",
            "linear_fastpath": "auto",
            "sm_89_behavior": "packed NVFP4 dequantizes once into dense cache, then uses native half/bfloat16 linear",
            "future_blackwell_behavior": "packed FP4/NVFP4 interfaces remain available for FP4 MMA expansion",
        },
    }
    output_path = _write_summary(summary)
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
