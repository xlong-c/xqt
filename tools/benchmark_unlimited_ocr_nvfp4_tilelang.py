"""Benchmark a real Unlimited-OCR NVFP4 linear path against XQT TileLang."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt import XQTOptimizationSession
from xqt.kernels.wrappers.bench import benchmark_callable
from xqt.compression.quant import NVFP4LinearBridge, bridge_module_to_nvfp4_linear


REPO_ID = "sahilchachra/Unlimited-OCR-NVFP4"
DEFAULT_MODULE_NAME = "model.layers.0.self_attn.q_proj"
DEFAULT_GROUP_SIZE = 16
DEFAULT_ARTIFACT_DIR = "artifacts/xqt/benchmarks/unlimited_ocr_nvfp4_tilelang"
DEFAULT_SWEEP_MODULE_NAMES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.experts.0.gate_proj",
    "model.layers.1.mlp.experts.0.up_proj",
    "model.layers.1.mlp.experts.0.down_proj",
    "model.layers.10.self_attn.q_proj",
    "model.layers.10.mlp.experts.0.gate_proj",
    "model.layers.10.mlp.experts.0.up_proj",
    "model.layers.10.mlp.experts.0.down_proj",
]


@dataclass(frozen=True)
class NVFP4LinearSource:
    """Metadata for the source implementation used in this layer benchmark."""

    repo_id: str
    module_name: str
    safetensors_path: str
    implementation: str


class _WrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.module(inputs)


class _DenseSourceNVFP4Linear(torch.nn.Module):
    """Packed NVFP4 module with a source forward that uses load-time dequantized weight."""

    def __init__(
        self,
        *,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        source: NVFP4LinearSource,
    ) -> None:
        super().__init__()
        self.in_features = int(input_features)
        self.out_features = int(output_features)
        self.group_size = int(group_size)
        self.source = source
        self.register_buffer("weight_packed", packed_weight.detach().clone().to(torch.uint8))
        self.register_buffer("weight_scale", weight_scale.detach().clone())
        if weight_global_scale is None:
            self.register_buffer("weight_global_scale", None)
        else:
            self.register_buffer("weight_global_scale", weight_global_scale.detach().clone())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))

        bridge = bridge_module_to_nvfp4_linear(self)
        if bridge is None:
            raise RuntimeError("failed to create NVFP4 bridge")
        self._bridge = bridge
        self.register_buffer("dense_weight", bridge.dequantize_weight().detach().clone())

    def tilelang_packed_nvfp4_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None]:
        return self._bridge.tilelang_packed_nvfp4_dequant_gemm_args(dtype=dtype, device=device)

    def tilelang_dense_linear_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        bias = None if self.bias is None else self.bias.to(device=device, dtype=dtype)
        return self.dense_weight.to(device=device, dtype=dtype), bias, None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight.to(device=inputs.device, dtype=inputs.dtype)
        bias = None if self.bias is None else self.bias.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(inputs, weight, bias)


class _FakeCompressedNVFP4Linear(_DenseSourceNVFP4Linear):
    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 64,
        group_size: int = 16,
    ) -> None:
        if in_features % group_size != 0:
            raise ValueError("in_features must be divisible by group_size for fake benchmark fixture")
        packed_k = in_features // 2
        groups = in_features // group_size
        super().__init__(
            packed_weight=torch.full((out_features, packed_k), 0x21, dtype=torch.uint8),
            weight_scale=torch.ones((out_features, groups), dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            weight_global_scale=torch.tensor([1.0], dtype=torch.float32),
            bias=torch.zeros(out_features, dtype=torch.float32),
            input_features=in_features,
            output_features=out_features,
            group_size=group_size,
            source=NVFP4LinearSource(
                repo_id="synthetic",
                module_name="fake_nvfp4_linear",
                safetensors_path="",
                implementation="predequantized_dense_linear",
            ),
        )


def load_unlimited_ocr_nvfp4_linear(
    *,
    repo_id: str = REPO_ID,
    module_name: str = DEFAULT_MODULE_NAME,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> _DenseSourceNVFP4Linear:
    """Load one real packed NVFP4 linear layer from the Unlimited-OCR safetensors file."""

    from huggingface_hub import hf_hub_download

    safetensors_path = hf_hub_download(
        repo_id,
        filename="model-00001-of-000001.safetensors",
    )
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        packed_weight = handle.get_tensor(f"{module_name}.weight_packed")
        weight_scale = handle.get_tensor(f"{module_name}.weight_scale")
        weight_global_scale = handle.get_tensor(f"{module_name}.weight_global_scale")
        bias_key = f"{module_name}.bias"
        bias = handle.get_tensor(bias_key) if bias_key in handle.keys() else None

    input_features = int(packed_weight.shape[1]) * 2
    output_features = int(packed_weight.shape[0])
    expected_groups = input_features // int(group_size)
    if int(weight_scale.shape[1]) != expected_groups:
        raise ValueError(
            f"{module_name} weight_scale groups={int(weight_scale.shape[1])} does not match "
            f"input_features={input_features}, group_size={group_size}"
        )
    return _DenseSourceNVFP4Linear(
        packed_weight=packed_weight,
        weight_scale=weight_scale,
        weight_global_scale=weight_global_scale,
        bias=bias,
        input_features=input_features,
        output_features=output_features,
        group_size=group_size,
        source=NVFP4LinearSource(
            repo_id=repo_id,
            module_name=module_name,
            safetensors_path=str(safetensors_path),
            implementation="compressed_tensors_load_dequantized_dense_linear",
        ),
    )


def _sync_bridge_after_to(module: torch.nn.Module) -> None:
    bridge = getattr(module, "_bridge", None)
    if not isinstance(bridge, NVFP4LinearBridge):
        return
    bridge.to(device=module.weight_packed.device)


def _move_module_for_benchmark(
    module: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    module = module.eval().to(device=device)
    module = module.to(dtype=dtype)
    _sync_bridge_after_to(module)
    return module


def _cuda_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _latency_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_ms": report.get("mean_ms"),
        "p50_ms": report.get("p50_ms"),
        "p90_ms": report.get("p90_ms"),
        "p99_ms": report.get("p99_ms"),
        "iterations": report.get("iterations"),
        "warmup": report.get("warmup"),
    }


def _run_xqt_operator_benchmark(
    module: torch.nn.Module,
    *,
    example_inputs: torch.Tensor,
    artifact_dir: str,
    target_arch: str | None,
    patterns: list[str],
    warmup: int,
    iterations: int,
    min_speedup: float,
    tilelang_options: dict[str, Any] | None = None,
    target_name: str = "module_tilelang",
) -> dict[str, Any]:
    session = XQTOptimizationSession(
        project={
            "name": "unlimited_ocr_nvfp4_tilelang_benchmark",
            "artifact_dir": artifact_dir,
        },
        model=_WrappedModule(module).eval(),
        device=str(example_inputs.device),
        example_inputs=example_inputs,
    )
    stage = session.operator(
        name="tilelang_nvfp4_operator",
        benchmark={
            "warmup": int(warmup),
            "iterations": int(iterations),
            "sync_cuda": bool(example_inputs.is_cuda),
            "measure_memory": False,
        },
        targets=[
            {
                "name": target_name,
                "target": "module",
                "backend": "tilelang",
                "patterns": list(patterns),
                "min_speedup": float(min_speedup),
                "tilelang": {
                    "target_arch": target_arch,
                    **dict(tilelang_options or {}),
                },
            }
        ],
    )
    return stage.metrics["targets"][0]


def benchmark_nvfp4_linear_pair(
    module: torch.nn.Module,
    *,
    batch_size: int = 64,
    warmup: int = 20,
    iterations: int = 100,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    target_arch: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Benchmark source forward and both XQT auto/paked paths on the same module."""

    torch_device = torch.device(device)
    module = _move_module_for_benchmark(module, device=torch_device, dtype=dtype)
    example_inputs = torch.randn(
        batch_size,
        int(getattr(module, "in_features")),
        device=torch_device,
        dtype=dtype,
    )

    direct_source_latency = benchmark_callable(
        lambda: module(example_inputs),
        warmup=warmup,
        iterations=iterations,
        sync_cuda=torch_device.type == "cuda",
        device=device,
    ).to_dict()
    auto_target = _run_xqt_operator_benchmark(
        module,
        example_inputs=example_inputs,
        artifact_dir=f"{artifact_dir}/auto_native",
        target_arch=target_arch,
        patterns=["dequant_gemm_epilogue"],
        warmup=warmup,
        iterations=iterations,
        min_speedup=1.000001,
        tilelang_options={
            "linear_runtime": "auto",
            "linear_fastpath": "auto",
        },
        target_name="module_tilelang_auto",
    )
    packed_target = _run_xqt_operator_benchmark(
        module,
        example_inputs=example_inputs,
        artifact_dir=f"{artifact_dir}/packed_tilelang",
        target_arch=target_arch,
        patterns=["nvfp4_packed_dequant_gemm_epilogue"],
        warmup=warmup,
        iterations=iterations,
        min_speedup=1.000001,
        tilelang_options={
            "linear_runtime": "tilelang",
            "linear_fastpath": "packed",
        },
        target_name="module_tilelang_packed",
    )
    source_latency = auto_target.get("latency_before") or direct_source_latency
    auto_latency_after = auto_target.get("latency_after", {})
    packed_latency_after = packed_target.get("latency_after", {})
    source = getattr(module, "source", None)
    return {
        "source": asdict(source) if isinstance(source, NVFP4LinearSource) else None,
        "shape": {
            "batch_size": batch_size,
            "in_features": int(getattr(module, "in_features")),
            "out_features": int(getattr(module, "out_features")),
            "group_size": int(getattr(module, "group_size")),
        },
        "device": str(torch_device),
        "dtype": str(dtype),
        "target_arch": target_arch,
        "direct_source_latency": direct_source_latency,
        "direct_source_latency_summary": _latency_summary(direct_source_latency),
        "source_latency": source_latency,
        "source_latency_summary": _latency_summary(source_latency),
        "xqt_operator_auto": auto_target,
        "xqt_latency_summary_auto": _latency_summary(auto_latency_after),
        "source_vs_xqt_speedup_auto": (
            float(source_latency["mean_ms"]) / float(auto_latency_after["mean_ms"])
            if auto_latency_after.get("mean_ms")
            else None
        ),
        "xqt_operator_packed": packed_target,
        "xqt_latency_summary_packed": _latency_summary(packed_latency_after),
        "source_vs_xqt_speedup_packed": (
            float(source_latency["mean_ms"]) / float(packed_latency_after["mean_ms"])
            if packed_latency_after.get("mean_ms")
            else None
        ),
    }


def benchmark_nvfp4_layer_sweep(
    *,
    module_names: list[str],
    repo_id: str = REPO_ID,
    group_size: int = DEFAULT_GROUP_SIZE,
    batch_size: int = 64,
    warmup: int = 5,
    iterations: int = 20,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    target_arch: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Benchmark a representative set of Unlimited-OCR NVFP4 linear layers."""

    results: list[dict[str, Any]] = []
    auto_applied = 0
    packed_applied = 0
    auto_speedups: list[float] = []
    packed_speedups: list[float] = []

    for module_name in module_names:
        module = load_unlimited_ocr_nvfp4_linear(
            repo_id=repo_id,
            module_name=module_name,
            group_size=group_size,
        )
        result = benchmark_nvfp4_linear_pair(
            module,
            batch_size=batch_size,
            warmup=warmup,
            iterations=iterations,
            dtype=dtype,
            device=device,
            target_arch=target_arch,
            artifact_dir=f"{artifact_dir}/{module_name}",
        )
        results.append(result)
        if result["xqt_operator_auto"].get("applied"):
            auto_applied += 1
        if result["xqt_operator_packed"].get("applied"):
            packed_applied += 1
        auto_speedup = result.get("source_vs_xqt_speedup_auto")
        packed_speedup = result.get("source_vs_xqt_speedup_packed")
        if isinstance(auto_speedup, (float, int)):
            auto_speedups.append(float(auto_speedup))
        if isinstance(packed_speedup, (float, int)):
            packed_speedups.append(float(packed_speedup))

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "repo_id": repo_id,
        "device": device,
        "dtype": str(dtype),
        "target_arch": target_arch,
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "module_names": list(module_names),
        "results": results,
        "summary": {
            "module_count": len(results),
            "auto_applied_count": auto_applied,
            "packed_applied_count": packed_applied,
            "auto_applied_ratio": (auto_applied / len(results)) if results else None,
            "packed_applied_ratio": (packed_applied / len(results)) if results else None,
            "auto_mean_source_vs_xqt_speedup": _mean(auto_speedups),
            "packed_mean_source_vs_xqt_speedup": _mean(packed_speedups),
        },
    }


def main() -> None:
    torch.manual_seed(0)
    target_arch = _cuda_arch()
    benchmark_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    benchmark_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        result = benchmark_nvfp4_layer_sweep(
            module_names=DEFAULT_SWEEP_MODULE_NAMES,
            batch_size=64,
            warmup=5,
            iterations=20,
            dtype=benchmark_dtype,
            device=benchmark_device,
            target_arch=target_arch,
        )
        output_path = Path(DEFAULT_ARTIFACT_DIR) / "sweep_summary.json"
    except Exception as sweep_exc:
        print(f"failed to run real Unlimited-OCR NVFP4 sweep, falling back to single synthetic fixture: {sweep_exc}")
        module = _FakeCompressedNVFP4Linear()
        result = benchmark_nvfp4_linear_pair(
            module,
            batch_size=64,
            warmup=5,
            iterations=20,
            dtype=benchmark_dtype,
            device=benchmark_device,
            target_arch=target_arch,
        )
        output_path = Path(DEFAULT_ARTIFACT_DIR) / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
