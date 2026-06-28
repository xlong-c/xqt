"""Benchmark one NVFP4 linear path against the XQT TileLang NVFP4 bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from xqt import XQTOptimizationSession
from xqt.benchmark import benchmark_callable
from xqt.quant import bridge_module_to_nvfp4_linear


class _WrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.module(inputs)


class _FakeCompressedNVFP4Linear(torch.nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 64, group_size: int = 16) -> None:
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError("in_features must be divisible by group_size for fake benchmark fixture")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        packed_k = in_features // 2
        groups = in_features // group_size
        self.register_buffer("qweight", torch.full((out_features, packed_k), 0x21, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.ones((out_features, groups, 1), dtype=torch.float32))
        self.register_buffer("weight_global_scale", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        self._bridge = bridge_module_to_nvfp4_linear(self)
        if self._bridge is None:
            raise RuntimeError("failed to create fake NVFP4 bridge")

    def tilelang_packed_nvfp4_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None]:
        assert self._bridge is not None
        return self._bridge.tilelang_packed_nvfp4_dequant_gemm_args(dtype=dtype, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self._bridge is not None
        return self._bridge(inputs)


def _run_xqt_operator_benchmark(
    module: torch.nn.Module,
    *,
    example_inputs: torch.Tensor,
    artifact_dir: str,
    target_arch: str | None,
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
        targets=[
            {
                "name": "module_tilelang",
                "target": "module",
                "backend": "tilelang",
                "patterns": ["nvfp4_packed_dequant_gemm_epilogue"],
                "min_speedup": 1.000001,
                "tilelang": {
                    "target_arch": target_arch,
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
    artifact_dir: str = "artifacts/xqt/benchmarks/unlimited_ocr_nvfp4_tilelang",
) -> dict[str, Any]:
    """Benchmark source forward and XQT TileLang operator path on the same module."""

    torch_device = torch.device(device)
    module = module.eval().to(torch_device)
    example_inputs = torch.randn(batch_size, int(getattr(module, "in_features")), device=torch_device, dtype=dtype)

    source_latency = benchmark_callable(
        lambda: module(example_inputs),
        warmup=warmup,
        iterations=iterations,
        sync_cuda=torch_device.type == "cuda",
        device=device,
    ).to_dict()
    operator_target = _run_xqt_operator_benchmark(
        module,
        example_inputs=example_inputs,
        artifact_dir=artifact_dir,
        target_arch=target_arch,
    )
    return {
        "shape": {
            "batch_size": batch_size,
            "in_features": int(getattr(module, "in_features")),
            "out_features": int(getattr(module, "out_features")),
        },
        "source_latency": source_latency,
        "xqt_operator": operator_target,
        "source_vs_xqt_speedup": (
            float(source_latency["mean_ms"]) / float(operator_target["latency_after"]["mean_ms"])
            if operator_target.get("latency_after", {}).get("mean_ms")
            else None
        ),
    }


def main() -> None:
    torch.manual_seed(0)
    target_arch = "sm_89" if torch.cuda.is_available() else None
    module = _FakeCompressedNVFP4Linear()
    result = benchmark_nvfp4_linear_pair(
        module,
        batch_size=64,
        warmup=5,
        iterations=20,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_arch=target_arch,
    )
    output_path = Path("artifacts/xqt/benchmarks/unlimited_ocr_nvfp4_tilelang/summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
