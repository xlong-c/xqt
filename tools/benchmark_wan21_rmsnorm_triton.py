from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

from xqt.benchmark.latency import benchmark_callable
from xqt.operator_opt.executor import _TritonRMSNormWrapper
from xqt.operator_opt.kernels.triton.pointwise import fused_rmsnorm_triton


@dataclass
class BenchmarkCase:
    name: str
    dim: int
    shape: tuple[int, ...]
    images: bool


def _legacy_wrapper_forward(
    norm: WanRMS_norm,
    x: torch.Tensor,
    *,
    eps: float,
    block_size: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    weight = norm.gamma.to(device=x.device, dtype=x.dtype).reshape(-1).contiguous() * float(norm.scale)
    x_last = x.movedim(1, -1).contiguous() if x.ndim > 2 else x
    out_last = fused_rmsnorm_triton(
        x_last,
        weight,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_last.movedim(-1, 1).contiguous() if x.ndim > 2 else out_last


def _run_case(case: BenchmarkCase) -> dict[str, float | str | tuple[int, ...]]:
    norm = WanRMS_norm(case.dim, channel_first=True, images=case.images).eval().cuda().half()
    wrapper = _TritonRMSNormWrapper(
        norm,
        fallback="eager",
        settings={
            "eps": 1e-12,
            "block_size": 1024,
            "sites_per_program": 8,
            "num_warps": 4,
            "num_stages": 4,
        },
    ).eval().cuda()
    x = torch.randn(*case.shape, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        eager_out = norm(x)
        legacy_out = _legacy_wrapper_forward(
            norm,
            x,
            eps=1e-12,
            block_size=1024,
            num_warps=4,
            num_stages=4,
        )
        fast_out = wrapper(x)
        torch.cuda.synchronize()
    legacy_diff = (legacy_out.float() - eager_out.float()).abs()
    fast_diff = (fast_out.float() - eager_out.float()).abs()
    eager_report = benchmark_callable(
        lambda: norm(x),
        warmup=20,
        iterations=100,
        sync_cuda=True,
        device="cuda",
    ).to_dict()
    legacy_report = benchmark_callable(
        lambda: _legacy_wrapper_forward(
            norm,
            x,
            eps=1e-12,
            block_size=1024,
            num_warps=4,
            num_stages=4,
        ),
        warmup=20,
        iterations=100,
        sync_cuda=True,
        device="cuda",
    ).to_dict()
    fast_report = benchmark_callable(
        lambda: wrapper(x),
        warmup=20,
        iterations=100,
        sync_cuda=True,
        device="cuda",
    ).to_dict()
    return {
        "case": case.name,
        "shape": case.shape,
        "eager_mean_ms": float(eager_report["mean_ms"]),
        "legacy_mean_ms": float(legacy_report["mean_ms"]),
        "fast_mean_ms": float(fast_report["mean_ms"]),
        "fast_vs_eager_speedup": float(eager_report["mean_ms"] / fast_report["mean_ms"]),
        "fast_vs_legacy_speedup": float(legacy_report["mean_ms"] / fast_report["mean_ms"]),
        "legacy_max_abs": float(legacy_diff.max().item()),
        "fast_max_abs": float(fast_diff.max().item()),
        "legacy_mean_abs": float(legacy_diff.mean().item()),
        "fast_mean_abs": float(fast_diff.mean().item()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cases = [
        BenchmarkCase("cf_96_t4_h64_w64", 96, (1, 96, 4, 64, 64), False),
        BenchmarkCase("cf_192_t2_h32_w32", 192, (1, 192, 2, 32, 32), False),
        BenchmarkCase("cf_384_t1_h16_w16", 384, (1, 384, 1, 16, 16), False),
        BenchmarkCase("cf_384_hw8_attn", 384, (1, 384, 8, 8), True),
    ]
    results = [_run_case(case) for case in cases]
    print(json.dumps({"cases": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
