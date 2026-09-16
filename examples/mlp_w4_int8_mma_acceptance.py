"""Simulated-network W4/FP4 storage -> INT8 MMA sweep for pre-Blackwell GPUs.

This is an execution-oriented acceptance script, not a training loop. It keeps
the model synthetic but benchmarks whole-network latency instead of isolated
GEMM calls, and it compares the FP4 dequantized weights against the INT8 MMA
compute view created from the same packed W4 storage.
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xqt.compression.quant.quantizers.fp4_weight_only import (
    FP4WeightOnlyLinear,
    quantize_with_fp4_weight_only,
)
from xqt.compression.quant.quantizers.w4_storage_int8_mma import (
    W4StorageInt8MmaLinear,
    quantize_with_w4_storage_int8_mma,
)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    engine: str
    meaning: str
    block_m: int = 64
    block_n: int = 64
    block_k: int = 64
    threads: int = 128
    num_stages: int = 2
    activation_quant_block_size: int = 256
    min_rows: int = 1
    only_sm: tuple[str, ...] = ()


DEFAULT_CANDIDATES: tuple[CandidateConfig, ...] = (
    CandidateConfig(
        name="torch_int_mm",
        engine="torch_int_mm",
        meaning="W4 storage -> INT8 compute view -> torch._int_mm vendor path",
    ),
    CandidateConfig(
        name="tilelang_64x64x64",
        engine="tilelang",
        meaning="W4 storage -> TileLang fused static activation INT8 MMA",
        block_m=64,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
    ),
    CandidateConfig(
        name="ptx_sm89_prepacked",
        engine="ptx_sm89",
        meaning="W4 storage -> hand-written sm_89 PTX INT8 MMA with prepacked B",
        block_m=128,
        block_n=64,
        block_k=64,
        threads=128,
        num_stages=2,
        min_rows=1,
        only_sm=("sm_89",),
    ),
)


@dataclass(frozen=True)
class BenchConfig:
    depth: int = 10
    width: int = 10240
    num_classes: int = 16
    batch_size: int = 64
    num_batches: int = 8
    seed: int = 11
    group_size: int = 128
    warmup: int = 15
    repeats: int = 60
    timing_trials: int = 5
    max_accuracy_drop_pp: float = 5.0
    min_speedup_vs_fp4: float = 1.15
    max_weight_mean_abs: float = 1e-3
    weight_chunk_rows: int = 512
    artifact_path: str = "artifacts/xqt/w4_int8_mma_sweep.json"
    candidates: tuple[CandidateConfig, ...] = DEFAULT_CANDIDATES


class DeepMLP(nn.Module):
    def __init__(self, *, depth: int, width: int, num_classes: int) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        layers: list[nn.Module] = []
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(width, num_classes, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sm_name(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _sm_number(sm: str) -> int:
    if not sm.startswith("sm_"):
        raise ValueError(f"invalid SM name: {sm}")
    return int(sm.removeprefix("sm_"))


def _exclude_head(depth: int) -> list[str]:
    return [rf"net\.{2 * (depth - 1)}$"]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _predict_batches(model: nn.Module, feats: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    model.eval()
    preds: list[torch.Tensor] = []
    for start in range(0, feats.shape[0], batch_size):
        end = min(start + batch_size, feats.shape[0])
        logits = model(feats[start:end].to(dtype=torch.float16))
        preds.append(logits.argmax(dim=1).detach().cpu())
    return torch.cat(preds, dim=0)


@torch.no_grad()
def _collect_logits(model: nn.Module, feats: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, feats.shape[0], batch_size):
        end = min(start + batch_size, feats.shape[0])
        outputs.append(model(feats[start:end].to(dtype=torch.float16)).detach().float().cpu())
    return torch.cat(outputs, dim=0)


def _agreement(pred: torch.Tensor, ref: torch.Tensor) -> float:
    if pred.numel() == 0:
        return 0.0
    return 100.0 * float((pred == ref).sum().item()) / float(pred.numel())


@torch.no_grad()
def _latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> float:
    model.eval()
    for _ in range(warmup):
        _ = model(sample)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(sample)
    _sync(device)
    return 1000.0 * (time.perf_counter() - t0) / float(repeats)


def _median_latency_ms(model: nn.Module, sample: torch.Tensor, *, cfg: BenchConfig) -> float:
    device = sample.device
    samples = [
        _latency_ms(
            model,
            sample,
            warmup=cfg.warmup,
            repeats=cfg.repeats,
            device=device,
        )
        for _ in range(cfg.timing_trials)
    ]
    return float(statistics.median(samples))


def _counts(model: nn.Module) -> dict[str, int]:
    counts = {"Linear": 0, "FP4WeightOnlyLinear": 0, "W4StorageInt8MmaLinear": 0}
    for module in model.modules():
        name = type(module).__name__
        if name in counts:
            counts[name] += 1
    return counts


def _other_bytes(model: nn.Module, skip: tuple[type, ...]) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, skip):
            continue
        if any(isinstance(child, skip) for child in module.children()):
            continue
        tensors = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
        for tensor in tensors:
            if tensor is not None:
                total += int(tensor.nbytes)
    return total


def _storage_bytes(model: nn.Module) -> int:
    w4_int8 = [m for m in model.modules() if isinstance(m, W4StorageInt8MmaLinear)]
    fp4 = [m for m in model.modules() if isinstance(m, FP4WeightOnlyLinear)]
    if w4_int8:
        return sum(module.storage_nbytes() for module in w4_int8) + _other_bytes(
            model,
            (W4StorageInt8MmaLinear,),
        )
    if fp4:
        total = 0
        for module in fp4:
            total += int(module.packed_weight.nbytes)
            total += int(module.weight_scale.nbytes)
            if module.bias is not None:
                total += int(module.bias.nbytes)
        return total + _other_bytes(model, (FP4WeightOnlyLinear,))
    return sum(int(tensor.nbytes) for tensor in list(model.parameters()) + list(model.buffers()))


def _pack_w4(source: nn.Module, cfg: BenchConfig) -> nn.Module:
    return quantize_with_fp4_weight_only(
        copy.deepcopy(source),
        policy={
            "include_module_types": ["Linear"],
            "exclude_name_patterns": _exclude_head(cfg.depth),
            "group_size": cfg.group_size,
        },
        inplace=True,
    ).model


def _w4_modules(model: nn.Module) -> dict[str, W4StorageInt8MmaLinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, W4StorageInt8MmaLinear)
    }


def _fp4_modules(model: nn.Module) -> dict[str, FP4WeightOnlyLinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, FP4WeightOnlyLinear)
    }


def _configure_w4_module(module: W4StorageInt8MmaLinear, candidate: CandidateConfig) -> None:
    module.output_dtype = torch.float16
    module.block_m = candidate.block_m
    module.block_n = candidate.block_n
    module.block_k = candidate.block_k
    module.threads = candidate.threads
    module.num_stages = candidate.num_stages
    module.activation_quant_block_size = candidate.activation_quant_block_size
    module.release_int8_compute_view()


@torch.no_grad()
def _calibrate_static_scales(
    model: nn.Module,
    sample: torch.Tensor,
    candidate: CandidateConfig,
) -> dict[str, float]:
    activation_max: dict[str, float] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def hook_factory(name: str) -> Any:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            value = float(inputs[0].detach().float().abs().amax().item())
            activation_max[name] = max(activation_max.get(name, 0.0), value)

        return hook

    for name, module in _w4_modules(model).items():
        hooks.append(module.register_forward_hook(hook_factory(name)))
    try:
        for _ in range(5):
            _ = model(sample)
    finally:
        for hook in hooks:
            hook.remove()

    scales: dict[str, float] = {}
    for name, module in _w4_modules(model).items():
        scale = max(activation_max.get(name, 1.0) / 127.0, 1e-6)
        scales[name] = scale
        module.activation_scale_mode = "static"
        module._activation_scale = scale
        _configure_w4_module(module, candidate)
        compute = module._ensure_compute_view()
        compute.activation_scale_mode = "static"
        compute.set_static_activation_scale(scale)
        compute.block_m = candidate.block_m
        compute.block_n = candidate.block_n
        compute.block_k = candidate.block_k
        compute.threads = candidate.threads
        compute.num_stages = candidate.num_stages
        compute.activation_quant_block_size = candidate.activation_quant_block_size
    for _ in range(5):
        _ = model(sample)
    return scales


def _prepare_candidate(
    packed_w4: nn.Module,
    cfg: BenchConfig,
    candidate: CandidateConfig,
    sample: torch.Tensor,
) -> tuple[nn.Module, dict[str, float]]:
    model = quantize_with_w4_storage_int8_mma(
        copy.deepcopy(packed_w4),
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": _exclude_head(cfg.depth)},
        engine=candidate.engine,
        fallback_engine="torch_int_mm",
        source="fp4_weight_only",
        inplace=True,
        group_size=cfg.group_size,
        cache_int8_compute_view=True,
        activation_scale_mode="dynamic",
        block_m=candidate.block_m,
        block_n=candidate.block_n,
        block_k=candidate.block_k,
        threads=candidate.threads,
        num_stages=candidate.num_stages,
        activation_quant_block_size=candidate.activation_quant_block_size,
    ).model
    for module in _w4_modules(model).values():
        _configure_w4_module(module, candidate)
    static_scales = _calibrate_static_scales(model, sample, candidate)
    return model, static_scales


def _candidate_skip_reason(
    candidate: CandidateConfig,
    *,
    device: torch.device,
    rows: int,
    sm: str,
) -> str | None:
    if device.type != "cuda":
        return "CUDA is required for pre-Blackwell INT8 MMA timing."
    if _sm_number(sm) >= 100:
        return "This acceptance targets pre-Blackwell GPUs; use native FP4/NVFP4 paths on sm_100+."
    if candidate.only_sm and sm not in candidate.only_sm:
        return f"{candidate.name} supports {candidate.only_sm}, got {sm}."
    if rows < candidate.min_rows:
        return f"{candidate.name} requires at least {candidate.min_rows} rows, got {rows}."
    if candidate.engine == "ptx_sm89":
        try:
            from xqt.kernels.ops._impl.cute.int8mma_binding import int8mma_available
        except Exception as exc:
            return f"ptx_sm89 binding import failed: {exc}"
        if not int8mma_available():
            return "ptx_sm89 shared library is not built."
    if candidate.engine == "tilelang":
        try:
            import tilelang  # noqa: F401
        except Exception as exc:
            return f"TileLang is not importable: {exc}"
    return None


def _int8_compute_weight(module: W4StorageInt8MmaLinear, *, start: int, end: int) -> torch.Tensor:
    compute = module._ensure_compute_view()
    qweight = compute.qweight_t[:, start:end].t().to(torch.float32)
    scale = compute.weight_scale[start:end].to(torch.float32).reshape(-1, 1)
    return qweight * scale


@torch.no_grad()
def _weight_retarget_error_for_pair(
    fp4: FP4WeightOnlyLinear,
    w4_int8: W4StorageInt8MmaLinear,
    *,
    chunk_rows: int,
) -> dict[str, float | int]:
    fp4_weight = fp4.dequantize_weight().to(device=w4_int8.packed_weight.device, dtype=torch.float32)
    rows = int(fp4_weight.shape[0])
    total_abs = 0.0
    total_sq = 0.0
    total_numel = 0
    max_abs = 0.0
    max_rel = 0.0
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        lhs = fp4_weight[start:end]
        rhs = _int8_compute_weight(w4_int8, start=start, end=end)
        diff = (lhs - rhs).abs()
        denom = lhs.abs().clamp_min(1e-6)
        total_abs += float(diff.sum().item())
        total_sq += float((diff * diff).sum().item())
        total_numel += int(diff.numel())
        max_abs = max(max_abs, float(diff.max().item()))
        max_rel = max(max_rel, float((diff / denom).max().item()))
    mean_abs = total_abs / float(max(total_numel, 1))
    rmse = (total_sq / float(max(total_numel, 1))) ** 0.5
    return {
        "numel": total_numel,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_rel": max_rel,
    }


def _merge_error_stats(layer_stats: list[dict[str, float | int]]) -> dict[str, float | int]:
    total_numel = sum(int(item["numel"]) for item in layer_stats)
    total_abs = sum(float(item["mean_abs"]) * int(item["numel"]) for item in layer_stats)
    total_sq = sum((float(item["rmse"]) ** 2) * int(item["numel"]) for item in layer_stats)
    return {
        "layers": len(layer_stats),
        "numel": total_numel,
        "max_abs": max((float(item["max_abs"]) for item in layer_stats), default=0.0),
        "mean_abs": total_abs / float(max(total_numel, 1)),
        "rmse": (total_sq / float(max(total_numel, 1))) ** 0.5,
        "max_rel": max((float(item["max_rel"]) for item in layer_stats), default=0.0),
    }


def weight_retarget_error(
    fp4_model: nn.Module,
    w4_int8_model: nn.Module,
    *,
    chunk_rows: int = 512,
) -> dict[str, Any]:
    fp4_modules = _fp4_modules(fp4_model)
    w4_modules = _w4_modules(w4_int8_model)
    missing = sorted(set(fp4_modules) - set(w4_modules))
    if missing:
        raise RuntimeError(f"missing W4 INT8 modules for FP4 layers: {missing}")
    per_layer: dict[str, dict[str, float | int]] = {}
    for name, fp4_module in fp4_modules.items():
        per_layer[name] = _weight_retarget_error_for_pair(
            fp4_module,
            w4_modules[name],
            chunk_rows=int(chunk_rows),
        )
    return {
        "aggregate": _merge_error_stats(list(per_layer.values())),
        "per_layer": per_layer,
    }


def _output_error(candidate_logits: torch.Tensor, reference_logits: torch.Tensor) -> dict[str, float]:
    diff = (candidate_logits.float() - reference_logits.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
    }


def _execution_summary(model: nn.Module) -> dict[str, Any]:
    engines: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for module in _w4_modules(model).values():
        metadata = module.execution_metadata()
        engine = str(metadata.get("engine", "unknown"))
        reason = str(metadata.get("reason", "unknown"))
        engines[engine] = engines.get(engine, 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"engines": engines, "reasons": reasons}


def _select_best_path(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    runnable = [item for item in candidates if item.get("status") == "ok"]
    if not runnable:
        return None
    passing = [item for item in runnable if item["gates"]["overall"]]
    pool = passing if passing else runnable
    return min(pool, key=lambda item: float(item["latency_ms"]))


def _benchmark_candidate(
    packed_w4: nn.Module,
    reference_model: nn.Module,
    reference_logits: torch.Tensor,
    reference_preds: torch.Tensor,
    sample: torch.Tensor,
    feats: torch.Tensor,
    cfg: BenchConfig,
    candidate: CandidateConfig,
    *,
    sm: str,
) -> dict[str, Any]:
    skip_reason = _candidate_skip_reason(
        candidate,
        device=sample.device,
        rows=int(sample.shape[0]),
        sm=sm,
    )
    payload: dict[str, Any] = {
        "name": candidate.name,
        "engine": candidate.engine,
        "meaning": candidate.meaning,
        "config": asdict(candidate),
    }
    if skip_reason is not None:
        payload.update({"status": "skipped", "reason": skip_reason})
        return payload

    try:
        model, static_scales = _prepare_candidate(packed_w4, cfg, candidate, sample)
        model = model.to(sample.device)
        _ = model(sample)
        logits = _collect_logits(model, feats, batch_size=cfg.batch_size)
        preds = logits.argmax(dim=1)
        accuracy = _agreement(preds, reference_preds)
        latency = _median_latency_ms(model, sample, cfg=cfg)
        weight_error = weight_retarget_error(
            reference_model,
            model,
            chunk_rows=cfg.weight_chunk_rows,
        )
        output_error = _output_error(logits, reference_logits)
    except Exception as exc:
        payload.update({"status": "failed", "reason": str(exc)})
        return payload

    drop_pp = 100.0 - accuracy
    speedup = payload_baseline_speedup(reference_model, sample, cfg, latency)
    gates = {
        "accuracy": drop_pp <= cfg.max_accuracy_drop_pp,
        "speedup": speedup >= cfg.min_speedup_vs_fp4,
        "weight_mean_abs": float(weight_error["aggregate"]["mean_abs"]) <= cfg.max_weight_mean_abs,
    }
    gates["overall"] = all(bool(value) for value in gates.values())
    payload.update(
        {
            "status": "ok",
            "accuracy_vs_fp4_top1_pct": accuracy,
            "drop_vs_fp4_pp": drop_pp,
            "latency_ms": latency,
            "speedup_vs_fp4": speedup,
            "storage_bytes": _storage_bytes(model),
            "linears": _counts(model),
            "static_scale_count": len(static_scales),
            "weight_retarget_error": weight_error,
            "output_error_vs_fp4": output_error,
            "execution": _execution_summary(model),
            "gates": gates,
        },
    )
    return payload


def payload_baseline_speedup(
    reference_model: nn.Module,
    sample: torch.Tensor,
    cfg: BenchConfig,
    candidate_latency_ms: float,
) -> float:
    cached = getattr(reference_model, "_xqt_cached_latency_ms", None)
    if cached is None:
        cached = _median_latency_ms(reference_model, sample, cfg=cfg)
        setattr(reference_model, "_xqt_cached_latency_ms", float(cached))
    return float(cached) / max(float(candidate_latency_ms), 1e-9)


def run_benchmark(cfg: BenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BenchConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    _set_seed(cfg.seed)
    sm = _sm_name(device)
    model = DeepMLP(
        depth=cfg.depth,
        width=cfg.width,
        num_classes=cfg.num_classes,
    ).to(device, dtype=torch.float16).eval()
    packed_w4 = _pack_w4(model, cfg).to(device).eval()
    reference_model = copy.deepcopy(packed_w4).to(device).eval()
    feats = torch.randn(
        cfg.batch_size * cfg.num_batches,
        cfg.width,
        device=device,
        dtype=torch.float16,
    )
    sample = feats[: cfg.batch_size]
    reference_logits = _collect_logits(reference_model, feats, batch_size=cfg.batch_size)
    reference_preds = reference_logits.argmax(dim=1)
    reference_latency = _median_latency_ms(reference_model, sample, cfg=cfg)
    setattr(reference_model, "_xqt_cached_latency_ms", float(reference_latency))
    candidates = [
        _benchmark_candidate(
            packed_w4,
            reference_model,
            reference_logits,
            reference_preds,
            sample,
            feats,
            cfg,
            candidate,
            sm=sm,
        )
        for candidate in cfg.candidates
    ]
    best = _select_best_path(candidates)
    report = {
        "device": torch.cuda.get_device_name(device),
        "sm": sm,
        "scope": "pre_blackwell_w4_storage_to_int8_mma",
        "simulated_network": {
            "type": "DeepMLP",
            "depth": cfg.depth,
            "width": cfg.width,
            "num_classes": cfg.num_classes,
            "batch_size": cfg.batch_size,
            "num_batches": cfg.num_batches,
            "group_size": cfg.group_size,
            "no_qat": True,
        },
        "baseline_fp4": {
            "meaning": "packed W4/FP4 storage, dequantized weight, FP16 Linear",
            "latency_ms": reference_latency,
            "accuracy_vs_self_top1_pct": 100.0,
            "storage_bytes": _storage_bytes(reference_model),
            "linears": _counts(reference_model),
        },
        "candidate_paths": candidates,
        "best_path": None
        if best is None
        else {
            "name": best["name"],
            "engine": best["engine"],
            "latency_ms": best["latency_ms"],
            "speedup_vs_fp4": best["speedup_vs_fp4"],
            "accuracy_vs_fp4_top1_pct": best["accuracy_vs_fp4_top1_pct"],
            "overall_gate": best["gates"]["overall"],
        },
        "gates": {
            "max_accuracy_drop_pp": cfg.max_accuracy_drop_pp,
            "min_speedup_vs_fp4": cfg.min_speedup_vs_fp4,
            "max_weight_mean_abs": cfg.max_weight_mean_abs,
        },
    }
    return report


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _print_report(report: dict[str, Any]) -> None:
    baseline = report["baseline_fp4"]
    print("=== W4/FP4 storage -> INT8 MMA simulated-network sweep ===")
    print(
        f"device={report['device']} {report['sm']} "
        f"baseline_fp4={baseline['latency_ms']:.3f}ms"
    )
    for candidate in report["candidate_paths"]:
        if candidate["status"] != "ok":
            print(
                f"- {candidate['name']}: {str(candidate['status']).upper()} "
                f"{candidate['reason']}"
            )
            continue
        error = candidate["weight_retarget_error"]["aggregate"]
        gates = candidate["gates"]
        print(
            f"- {candidate['name']}: {candidate['latency_ms']:.3f}ms "
            f"speedup={candidate['speedup_vs_fp4']:.3f}x "
            f"top1={candidate['accuracy_vs_fp4_top1_pct']:.3f}% "
            f"weight_mean_abs={float(error['mean_abs']):.6g} "
            f"overall={'PASS' if gates['overall'] else 'FAIL'}"
        )
    best = report["best_path"]
    if best is None:
        print("best_path: none")
    else:
        print(
            f"best_path: {best['name']} engine={best['engine']} "
            f"{best['latency_ms']:.3f}ms speedup={best['speedup_vs_fp4']:.3f}x "
            f"gate={'PASS' if best['overall_gate'] else 'FAIL'}"
        )


def main() -> None:
    cfg = BenchConfig()
    report = run_benchmark(cfg)
    report_path = write_report(report, cfg.artifact_path)
    _print_report(report)
    print(f"report={report_path}")
    best = report["best_path"]
    if best is None or not bool(best["overall_gate"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
