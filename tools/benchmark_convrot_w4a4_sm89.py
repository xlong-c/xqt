"""Benchmark fused ConvRot W4A4 CUDA paths on Ada ``sm_89``."""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.operator_opt.kernels.cute.convrot_w4a4_rowwise_sm89 import (
    allocate_convrot_w4a4_rowwise_workspace,
    bind_convrot_w4a4_rowwise_linear,
    convrot_w4a4_rowwise_linear,
    native_rowwise_convrot_w4a4_version,
)
from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
    allocate_w4a4_workspace,
    bind_convrot_w4a4_linear,
    native_w4a4_version,
    pack_w4a4_linear,
    w4a4_linear,
)
from xqt.quant.quantizers.convrot_4bit import (
    ConvRotMixedPrecisionLinear,
    _apply_groupwise_rotation,
    _normalized_regular_hadamard,
)
from xqt.runtime import ConvRotW4A4ExecutionView

DEFAULT_ARTIFACT_DIR = "artifacts/xqt/benchmarks/convrot_w4a4_sm89"
DEFAULT_WARMUP = 30
DEFAULT_ITERATIONS = 1000
DEFAULT_SAMPLES = 15
DEFAULT_ROWS = (64, 256, 1024)
DEFAULT_FEATURES = (1024, 2048)
DEFAULT_DTYPES = (torch.float16, torch.bfloat16)


def _load_comfy_kitchen_cuda() -> tuple[Any, str] | None:
    try:
        import comfy_kitchen.backends.cuda as cuda_backend
    except Exception:
        return None
    if not bool(getattr(cuda_backend, "_EXT_AVAILABLE", False)):
        return None
    try:
        package_version = version("comfy-kitchen")
    except PackageNotFoundError:
        package_version = "unknown"
    return cuda_backend, package_version


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency sample list must not be empty")
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _benchmark_cuda(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples_us: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples_us.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return {
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "p90_us": _percentile(samples_us, 0.90),
        "max_us": max(samples_us),
        "samples_us": samples_us,
    }


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).square().mean().sqrt()
    denominator = expected.float().square().mean().sqrt().clamp(min=1.0e-12)
    return float((difference / denominator).item())


def benchmark_convrot_w4a4_sm89(
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    samples: int = DEFAULT_SAMPLES,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ConvRot W4A4 benchmark")
    if torch.cuda.get_device_capability() != (8, 9):
        major, minor = torch.cuda.get_device_capability()
        raise RuntimeError(f"ConvRot W4A4 benchmark targets sm_89, got sm_{major}{minor}")

    torch.manual_seed(2028)
    device = torch.device("cuda")
    results: list[dict[str, Any]] = []
    official_baseline = _load_comfy_kitchen_cuda()
    official_cuda = None if official_baseline is None else official_baseline[0]
    official_version = None if official_baseline is None else official_baseline[1]

    for dtype in DEFAULT_DTYPES:
        for features in DEFAULT_FEATURES:
            source = torch.nn.Linear(
                features,
                features,
                bias=True,
                device=device,
                dtype=dtype,
            ).eval()
            module = ConvRotW4A4ExecutionView.from_storage(
                ConvRotMixedPrecisionLinear.from_linear(
                    source,
                    rot_size=256,
                    group_size=128,
                    compute_precision="w4a4",
                    activation_scale_mode="dynamic",
                    w4a4_runtime_backend="rowwise",
                )
            ).eval()
            artifact_weight = module.dequantized_weight(
                dtype=dtype,
                device=device,
                include_padding=True,
            )
            artifact_bias = module._bias_for(dtype=dtype, device=device)
            nunchaku_packed = pack_w4a4_linear(artifact_weight, artifact_bias)
            rotation = _normalized_regular_hadamard(256, device=device)

            for rows in DEFAULT_ROWS:
                inputs = torch.randn(rows, features, device=device, dtype=dtype)
                wrapper_output = module(inputs)
                module(inputs)
                torch.cuda.synchronize()
                runner_cache = module._rowwise_w4a4_runner_cache
                if runner_cache is None:
                    raise RuntimeError("rowwise ConvRot wrapper did not materialize its runner")
                _, _, _, rowwise_packed, dynamic_runner = runner_cache

                rowwise_workspace = allocate_convrot_w4a4_rowwise_workspace(
                    rows,
                    rowwise_packed,
                )
                rowwise_bound = bind_convrot_w4a4_rowwise_linear(
                    rowwise_packed,
                    rowwise_workspace,
                    rows=rows,
                )
                rowwise_direct = lambda: convrot_w4a4_rowwise_linear(
                    inputs,
                    rowwise_packed,
                    workspace=rowwise_workspace,
                )
                rowwise_bound_call = lambda: rowwise_bound(inputs)
                dynamic_call = lambda: dynamic_runner(inputs)
                wrapper_call = lambda: module(inputs)

                def comfy_kitchen_call() -> torch.Tensor:
                    if official_cuda is None:
                        raise RuntimeError("comfy-kitchen CUDA baseline is unavailable")
                    return official_cuda.convrot_w4a4_linear(
                        inputs,
                        rowwise_packed.qweight,
                        rowwise_packed.weight_scales,
                        rowwise_packed.bias,
                        convrot_groupsize=256,
                        quant_group_size=64,
                        linear_dtype="int4",
                    )

                nunchaku_workspace = allocate_w4a4_workspace(rows, nunchaku_packed)
                nunchaku_bound = bind_convrot_w4a4_linear(
                    nunchaku_packed,
                    nunchaku_workspace,
                    rows=rows,
                    logical_input_features=features,
                    rotated_input_features=features,
                    rot_size=256,
                )
                nunchaku_bound_call = lambda: nunchaku_bound(inputs)
                split_workspace = allocate_w4a4_workspace(rows, nunchaku_packed)

                def split_call() -> torch.Tensor:
                    rotated = _apply_groupwise_rotation(
                        inputs,
                        rot_size=256,
                        rotation_matrix=rotation,
                        return_padded=True,
                    )
                    return w4a4_linear(
                        rotated,
                        nunchaku_packed,
                        workspace=split_workspace,
                    )

                rowwise_reference = rowwise_bound(inputs)
                dynamic_reference = dynamic_runner(inputs)
                official_reference = (
                    None if official_cuda is None else comfy_kitchen_call()
                )
                nunchaku_reference = nunchaku_bound(inputs)
                rotated_reference = _apply_groupwise_rotation(
                    inputs,
                    rot_size=256,
                    rotation_matrix=rotation,
                    return_padded=True,
                ).to(dtype)
                dense_reference = F.linear(
                    rotated_reference,
                    artifact_weight,
                    artifact_bias,
                )
                torch.cuda.synchronize()
                if not torch.equal(wrapper_output, dynamic_reference):
                    raise RuntimeError("wrapper and dynamic rowwise runner outputs differ")
                torch.testing.assert_close(
                    rowwise_reference,
                    dynamic_reference,
                    rtol=0.0,
                    atol=1.0e-2,
                )

                timings = {
                    "rowwise_direct": _benchmark_cuda(
                        rowwise_direct,
                        warmup=warmup,
                        iterations=iterations,
                        samples=samples,
                    ),
                    "rowwise_bound_floor": _benchmark_cuda(
                        rowwise_bound_call,
                        warmup=warmup,
                        iterations=iterations,
                        samples=samples,
                    ),
                    "rowwise_dynamic_runner": _benchmark_cuda(
                        dynamic_call,
                        warmup=warmup,
                        iterations=iterations,
                        samples=samples,
                    ),
                    "xqt_wrapper": _benchmark_cuda(
                        wrapper_call,
                        warmup=warmup,
                        iterations=iterations,
                        samples=samples,
                    ),
                    "nunchaku_bound": _benchmark_cuda(
                        nunchaku_bound_call,
                        warmup=warmup,
                        iterations=iterations,
                        samples=samples,
                    ),
                    "split_rotation_then_nunchaku": _benchmark_cuda(
                        split_call,
                        warmup=warmup,
                        iterations=iterations,
                        samples=samples,
                    ),
                }
                if official_reference is not None:
                    timings["comfy_kitchen_official"] = _benchmark_cuda(
                        comfy_kitchen_call,
                        warmup=warmup,
                        iterations=iterations,
                        samples=samples,
                    )
                wrapper_us = float(timings["xqt_wrapper"]["median_us"])
                floor_us = float(timings["rowwise_bound_floor"]["median_us"])
                nunchaku_us = float(timings["nunchaku_bound"]["median_us"])
                split_us = float(
                    timings["split_rotation_then_nunchaku"]["median_us"]
                )
                official_us = (
                    None
                    if official_reference is None
                    else float(timings["comfy_kitchen_official"]["median_us"])
                )
                results.append(
                    {
                        "dtype": str(dtype).removeprefix("torch."),
                        "shape": {"m": rows, "n": features, "k": features},
                        "timings": timings,
                        "speedup": {
                            "wrapper_vs_nunchaku": nunchaku_us / wrapper_us,
                            "wrapper_vs_split": split_us / wrapper_us,
                            "nunchaku_fused_vs_split": split_us / nunchaku_us,
                            "wrapper_over_bound_floor": wrapper_us / floor_us,
                            "wrapper_vs_comfy_kitchen_official": (
                                None
                                if official_us is None
                                else official_us / wrapper_us
                            ),
                        },
                        "numeric": {
                            "wrapper_equals_dynamic": torch.equal(
                                wrapper_output,
                                dynamic_reference,
                            ),
                            "rowwise_max_abs_vs_bound": float(
                                (dynamic_reference - rowwise_reference)
                                .abs()
                                .max()
                                .item()
                            ),
                            "rowwise_relative_rmse_vs_dense_artifact": _relative_rmse(
                                dynamic_reference,
                                dense_reference,
                            ),
                            "nunchaku_relative_rmse_vs_dense_artifact": _relative_rmse(
                                nunchaku_reference,
                                dense_reference,
                            ),
                            "rowwise_relative_rmse_vs_comfy_kitchen_official": (
                                None
                                if official_reference is None
                                else _relative_rmse(
                                    dynamic_reference,
                                    official_reference,
                                )
                            ),
                            "rowwise_max_abs_vs_comfy_kitchen_official": (
                                None
                                if official_reference is None
                                else float(
                                    (dynamic_reference - official_reference)
                                    .abs()
                                    .max()
                                    .item()
                                )
                            ),
                        },
                        "metadata": module.execution_metadata(),
                    }
                )

    output = {
        "benchmark": "convrot_w4a4_sm89",
        "measurement_mode": "cuda_event_steady_state_batched_calls",
        "warmup": warmup,
        "iterations_per_sample": iterations,
        "samples": samples,
        "target": {
            "operator": "ConvRot W4A4 linear",
            "backend": "custom_cuda_cutlass",
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": "sm_89",
            "rot_size": 256,
            "norm_fused": False,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "rowwise_backend_version": native_rowwise_convrot_w4a4_version(),
            "nunchaku_backend_version": native_w4a4_version(),
            "comfy_kitchen_version": official_version,
            "ncu_status": "counter_permission_denied",
        },
        "official_comparison": {
            "status": "measured" if official_cuda is not None else "unavailable",
            "package": "comfy-kitchen",
            "version": official_version,
            "comparison_layer": "full_python_operator_steady_state",
            "weight_contract": (
                "identical packed signed INT4 weight, row scale, and bias supplied "
                "to both implementations"
            ),
        },
        "source_references": [
            "https://github.com/Comfy-Org/comfy-kitchen/blob/b72e6dfa79b79a7aee33a9c7608b5d9b3005b7af/comfy_kitchen/backends/cuda/ops/convrot_w4a4.cu",
            "https://github.com/TheLegendOfKitty/ComfyUI-AnimaTurbo/blob/master/warp_fht/convrot_warp_quantize.cu",
        ],
        "results": results,
    }
    artifact_path = Path(artifact_dir) / "summary.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["artifact_path"] = str(artifact_path)
    return output


def main() -> None:
    print(json.dumps(benchmark_convrot_w4a4_sm89(), indent=2))


if __name__ == "__main__":
    main()
