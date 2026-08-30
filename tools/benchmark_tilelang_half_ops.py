"""Benchmark TileLang half operator paths for conv, linear, attention, and norm."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.kernels.wrappers.bench import benchmark_callable
from xqt.kernels.ops._impl.tilelang.attention import (
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)
from xqt.kernels.ops._impl.tilelang.conv import (
    conv2d_reference,
    conv2d_tilelang,
)
from xqt.kernels.ops._impl.tilelang.linear import (
    half_linear_reference,
    half_linear_tilelang,
)
from xqt.kernels.ops._impl.tilelang.norm import (
    layer_norm_reference,
    layer_norm_tilelang,
)


DEFAULT_ARTIFACT_DIR = "artifacts/xqt/benchmarks/tilelang_half_ops"
DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 50
DEFAULT_INNER_ITERATIONS = 100


def _cuda_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _repeat_callable(fn: Callable[[], torch.Tensor], count: int) -> Callable[[], torch.Tensor]:
    if count <= 0:
        raise ValueError("inner_iterations must be positive")

    def repeated() -> torch.Tensor:
        output = fn()
        for _ in range(count - 1):
            output = fn()
        return output

    return repeated


def _cuda_graph_replay_callable(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
) -> Callable[[], torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Graph capture requires CUDA")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn()

    def replay() -> torch.Tensor:
        graph.replay()
        return static_output

    return replay


def _per_call_latency(report: dict[str, Any], inner_iterations: int) -> dict[str, Any]:
    scaled = dict(report)
    for key in ("mean_ms", "p50_ms", "p90_ms", "p99_ms"):
        scaled[key] = float(report[key]) / float(inner_iterations)
    scaled["samples_ms"] = [
        float(sample_ms) / float(inner_iterations)
        for sample_ms in report["samples_ms"]
    ]
    scaled["inner_iterations"] = inner_iterations
    return scaled


def _benchmark_pair(
    *,
    name: str,
    reference_fn: Callable[[], torch.Tensor],
    tilelang_fn: Callable[[], torch.Tensor],
    details: dict[str, Any] | None,
    warmup: int,
    iterations: int,
    inner_iterations: int,
    device: str,
) -> dict[str, Any]:
    # Prime both paths once so the reported latency reflects steady-state
    # execution instead of first-call setup or JIT compilation.
    reference_fn()
    tilelang_fn()
    if device == "cuda":
        torch.cuda.synchronize()

    batched_reference_fn = _repeat_callable(reference_fn, inner_iterations)
    batched_tilelang_fn = _repeat_callable(tilelang_fn, inner_iterations)
    reference_report = _per_call_latency(benchmark_callable(
        batched_reference_fn,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=device == "cuda",
        device=device,
    ).to_dict(), inner_iterations)
    tilelang_report = _per_call_latency(benchmark_callable(
        batched_tilelang_fn,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=device == "cuda",
        device=device,
    ).to_dict(), inner_iterations)
    mean_speedup = (
        float(reference_report["mean_ms"]) / float(tilelang_report["mean_ms"])
        if float(tilelang_report["mean_ms"]) > 0.0
        else None
    )
    p50_speedup = (
        float(reference_report["p50_ms"]) / float(tilelang_report["p50_ms"])
        if float(tilelang_report["p50_ms"]) > 0.0
        else None
    )
    graph_reference_report: dict[str, Any] | None = None
    graph_tilelang_report: dict[str, Any] | None = None
    graph_speedup_mean: float | None = None
    graph_speedup_p50: float | None = None
    if device == "cuda":
        graph_reference_fn = _cuda_graph_replay_callable(
            batched_reference_fn,
            warmup=warmup,
        )
        graph_tilelang_fn = _cuda_graph_replay_callable(
            batched_tilelang_fn,
            warmup=warmup,
        )
        graph_reference_report = _per_call_latency(benchmark_callable(
            graph_reference_fn,
            warmup=warmup,
            iterations=iterations,
            sync_cuda=True,
            device=device,
        ).to_dict(), inner_iterations)
        graph_tilelang_report = _per_call_latency(benchmark_callable(
            graph_tilelang_fn,
            warmup=warmup,
            iterations=iterations,
            sync_cuda=True,
            device=device,
        ).to_dict(), inner_iterations)
        graph_speedup_mean = (
            float(graph_reference_report["mean_ms"]) / float(graph_tilelang_report["mean_ms"])
            if float(graph_tilelang_report["mean_ms"]) > 0.0
            else None
        )
        graph_speedup_p50 = (
            float(graph_reference_report["p50_ms"]) / float(graph_tilelang_report["p50_ms"])
            if float(graph_tilelang_report["p50_ms"]) > 0.0
            else None
        )
    return {
        "name": name,
        "backend": "tilelang",
        "reference_backend": "torch",
        "reference": reference_report,
        "tilelang": tilelang_report,
        "graph_reference": graph_reference_report,
        "graph_tilelang": graph_tilelang_report,
        "speedup": {
            "mean_ms": mean_speedup,
            "p50_ms": p50_speedup,
            "graph_mean_ms": graph_speedup_mean,
            "graph_p50_ms": graph_speedup_p50,
        },
        "details": dict(details or {}),
    }


def benchmark_tilelang_half_ops(
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    inner_iterations: int = DEFAULT_INNER_ITERATIONS,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TileLang half-op benchmark")
    if importlib.util.find_spec("tilelang") is None:
        raise RuntimeError("tilelang package is required for TileLang half-op benchmark")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    target_arch = _cuda_arch()

    conv_x = torch.randn(1, 64, 8, 8, device=device, dtype=dtype)
    conv_weight = torch.randn(64, 64, 1, 1, device=device, dtype=dtype)
    conv_bias = torch.randn(64, device=device, dtype=dtype)

    linear_x = torch.randn(64, 64, device=device, dtype=dtype)
    linear_weight = torch.randn(64, 64, device=device, dtype=dtype)
    linear_bias = torch.randn(64, device=device, dtype=dtype)

    attn_q = torch.randn(1, 4, 64, 32, device=device, dtype=dtype)
    attn_k = torch.randn(1, 4, 64, 32, device=device, dtype=dtype)
    attn_v = torch.randn(1, 4, 64, 32, device=device, dtype=dtype)

    norm_x = torch.randn(32, 64, device=device, dtype=dtype)
    norm_weight = torch.randn(64, device=device, dtype=dtype)
    norm_bias = torch.randn(64, device=device, dtype=dtype)

    results = [
        _benchmark_pair(
            name="conv",
            reference_fn=lambda: conv2d_reference(
                conv_x,
                conv_weight,
                conv_bias,
                stride=(1, 1),
                padding=(0, 0),
            ),
            tilelang_fn=lambda: conv2d_tilelang(
                conv_x,
                conv_weight,
                conv_bias,
                stride=(1, 1),
                padding=(0, 0),
                block_m=64,
                block_n=64,
                block_k=64,
                threads=128,
                num_stages=2,
                target_arch=target_arch,
            ),
            details={
                "pattern": "conv",
                "implementation": "tilelang_conv1x1_nchw_direct_or_unfold_half_gemm",
                "fastpath": "tilelang_conv1x1_nchw_direct",
                "fallback": "torch_unfold_plus_tilelang_half_gemm",
                "input_shapes": {
                    "x": list(conv_x.shape),
                    "weight": list(conv_weight.shape),
                    "bias": list(conv_bias.shape),
                },
                "stride": [1, 1],
                "padding": [0, 0],
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "threads": 128,
                "num_stages": 2,
            },
            warmup=warmup,
            iterations=iterations,
            inner_iterations=inner_iterations,
            device=device,
        ),
        _benchmark_pair(
            name="linear",
            reference_fn=lambda: half_linear_reference(
                linear_x,
                linear_weight,
                linear_bias,
            ),
            tilelang_fn=lambda: half_linear_tilelang(
                linear_x,
                linear_weight,
                linear_bias,
                block_m=64,
                block_n=64,
                block_k=32,
                threads=128,
                num_stages=2,
                target_arch=target_arch,
            ),
            details={
                "pattern": "linear",
                "implementation": "tilelang_dense_half_gemm",
                "input_shapes": {
                    "x": list(linear_x.shape),
                    "weight": list(linear_weight.shape),
                    "bias": list(linear_bias.shape),
                },
                "block_m": 64,
                "block_n": 64,
                "block_k": 32,
                "threads": 128,
                "num_stages": 2,
            },
            warmup=warmup,
            iterations=iterations,
            inner_iterations=inner_iterations,
            device=device,
        ),
        _benchmark_pair(
            name="attention",
            reference_fn=lambda: fused_attention_forward_reference(
                attn_q,
                attn_k,
                attn_v,
                causal=False,
                dropout_p=0.0,
            ),
            tilelang_fn=lambda: fused_attention_forward_tilelang(
                attn_q,
                attn_k,
                attn_v,
                causal=False,
                dropout_p=0.0,
                block_m=64,
                block_n=64,
                threads=128,
                num_stages=2,
            ),
            details={
                "pattern": "attention",
                "implementation": "tilelang_flash_attention",
                "input_shapes": {
                    "q": list(attn_q.shape),
                    "k": list(attn_k.shape),
                    "v": list(attn_v.shape),
                },
                "causal": False,
                "dropout_p": 0.0,
                "block_m": 64,
                "block_n": 64,
                "threads": 128,
                "num_stages": 2,
            },
            warmup=warmup,
            iterations=iterations,
            inner_iterations=inner_iterations,
            device=device,
        ),
        _benchmark_pair(
            name="norm",
            reference_fn=lambda: layer_norm_reference(
                norm_x,
                norm_weight,
                norm_bias,
                eps=1e-5,
            ),
            tilelang_fn=lambda: layer_norm_tilelang(
                norm_x,
                norm_weight,
                norm_bias,
                eps=1e-5,
                threads=64,
            ),
            details={
                "pattern": "norm",
                "implementation": "tilelang_reduce_sum_layer_norm",
                "input_shapes": {
                    "x": list(norm_x.shape),
                    "weight": list(norm_weight.shape),
                    "bias": list(norm_bias.shape),
                },
                "eps": 1e-5,
                "threads": 64,
            },
            warmup=warmup,
            iterations=iterations,
            inner_iterations=inner_iterations,
            device=device,
        ),
    ]

    output = {
        "benchmark": "tilelang_half_ops",
        "device": device,
        "dtype": str(dtype),
        "target_arch": target_arch,
        "measurement_mode": "steady_state_after_prime",
        "warmup": warmup,
        "iterations": iterations,
        "inner_iterations": inner_iterations,
        "results": results,
    }
    artifact_path = Path(artifact_dir) / "summary.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["artifact_path"] = str(artifact_path)
    return output


def main() -> None:
    result = benchmark_tilelang_half_ops()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
