"""Hopper SM90 WGMMA and TMA scheduling contracts and reference guards.

Provides hardware-aligned schedule resolution for NVIDIA Hopper (SM90) architecture,
leveraging Warpgroup Matrix Multiply Accumulate (WGMMA) and Tensor Memory Accelerator (TMA).
Adheres strictly to the XQT reference_guarded discipline: on non-SM90 hardware, it
cleanly reports readiness blockers and falls back to safe reference paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm.reference import dense_gemm_reference
from xqt.kernels.jit.utils.arch import cuda_target_arch, target_arch_mismatch


@dataclass(frozen=True)
class SM90WgmmaSchedule:
    """Resolved Hopper SM90 execution schedule."""

    block_m: int
    block_n: int
    block_k: int
    threads: int
    num_stages: int
    use_tma: bool
    use_wgmma: bool
    target_arch: str = "sm_90"
    preset: str = "sm90_wgmma_default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "threads": self.threads,
            "num_stages": self.num_stages,
            "use_tma": self.use_tma,
            "use_wgmma": self.use_wgmma,
            "target_arch": self.target_arch,
            "preset": self.preset,
        }


def check_sm90_execution_readiness(
    target: torch.Tensor | torch.device | str,
) -> tuple[bool, str | None]:
    """Verify whether the runtime target is a genuine NVIDIA Hopper (SM90) device."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available on current host"

    if isinstance(target, torch.Tensor):
        if not target.is_cuda:
            return False, "tensor is not resident on a CUDA device"
        device = target.device
    elif isinstance(target, torch.device):
        device = target
    else:
        device = torch.device(target)

    if device.type != "cuda":
        return False, f"device {device} is not a CUDA device"

    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    if arch != "sm_90":
        return False, f"hardware capability {arch} does not match Hopper sm_90 requirement"

    return True, None


def resolve_sm90_wgmma_schedule(
    m: int,
    n: int,
    k: int,
    *,
    input_dtype: str = "bfloat16",
    has_bias: bool = False,
    activation: str | None = None,
) -> SM90WgmmaSchedule:
    """Resolve Hopper-optimized tile dimensions, stage depth, and warpgroup parameters."""
    # SM90 WGMMA operates at warpgroup granularity (128 threads)
    threads = 128

    if m <= 4:
        # Decode: low M favors smaller block_m to minimize tail latency
        return SM90WgmmaSchedule(
            block_m=16,
            block_n=128,
            block_k=64 if k % 64 == 0 else 32,
            threads=threads,
            num_stages=3,
            use_tma=True,
            use_wgmma=True,
            target_arch="sm_90",
            preset="sm90_wgmma_decode",
        )
    elif m <= 64:
        # Balanced batch: 64x128x64 with 3-stage asynchronous pipeline
        return SM90WgmmaSchedule(
            block_m=64,
            block_n=128,
            block_k=64 if k % 64 == 0 else 32,
            threads=threads,
            num_stages=3,
            use_tma=True,
            use_wgmma=True,
            target_arch="sm_90",
            preset="sm90_wgmma_medium",
        )
    else:
        # Prefill: large 128x128x64 tiles maximize compute throughput on SM90
        return SM90WgmmaSchedule(
            block_m=128,
            block_n=128,
            block_k=64 if k % 64 == 0 else 32,
            threads=threads,
            num_stages=4,
            use_tma=True,
            use_wgmma=True,
            target_arch="sm_90",
            preset="sm90_wgmma_prefill",
        )


def sm90_wgmma_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Numerical reference for SM90 WGMMA linear operations."""
    return dense_gemm_reference(
        x,
        weight.to(dtype=x.dtype, device=x.device),
        None if bias is None else bias.to(dtype=x.dtype, device=x.device),
        activation=activation,
        transpose_b=True,
    )


__all__ = [
    "SM90WgmmaSchedule",
    "check_sm90_execution_readiness",
    "resolve_sm90_wgmma_schedule",
    "sm90_wgmma_linear_reference",
]
