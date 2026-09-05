"""TileLang Linear operator references and guarded entry points."""

from dataclasses import dataclass

import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm.reference import dense_gemm_reference

from xqt.kernels.ops._impl.tilelang._common import (
    require_cuda_tensors,
    require_fp16_or_bf16_tensors,
    require_tilelang,
)
from xqt.kernels.ops._impl.tilelang.gemm_builder import build_tilelang_gemm_kernel
from xqt.kernels.ops._impl.tilelang.tuning_cache import (
    TimingCacheKey,
    get_tilelang_timing_cache,
)
from xqt.kernels.jit.utils.arch import target_arch_mismatch


@dataclass(frozen=True)
class TileLangLinearSchedule:
    """Resolved direct TileLang Linear launch schedule."""

    block_m: int
    block_n: int
    block_k: int
    threads: int
    num_stages: int
    target_arch: str | None
    preset: str

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "threads": self.threads,
            "num_stages": self.num_stages,
            "target_arch": self.target_arch,
            "preset": self.preset,
        }


def resolve_tilelang_linear_schedule(
    x: torch.Tensor,
    *,
    out_features: int | None = None,
    activation: str | None = None,
    has_bias: bool = False,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    threads: int | None = None,
    num_stages: int | None = None,
    target_arch: str | None = None,
) -> TileLangLinearSchedule:
    """Resolve evidence-backed defaults while preserving explicit overrides."""

    resolved_target_arch = target_arch
    if resolved_target_arch is None and x.is_cuda:
        major, minor = torch.cuda.get_device_capability(x.device)
        resolved_target_arch = f"sm_{major}{minor}"

    preset = "default"
    default_block_m = 64
    default_block_n = 64
    default_block_k = 64
    default_threads = 128
    default_num_stages = 2

    # Query timing cache first
    dtype_str = (
        "bfloat16"
        if x.dtype == torch.bfloat16
        else ("float16" if x.dtype == torch.float16 else str(x.dtype).replace("torch.", ""))
    )
    m_val = int(x.shape[0]) if x.ndim >= 1 else 1
    k_val = int(x.shape[1]) if x.ndim >= 2 else (int(x.shape[0]) if x.ndim == 1 else 1)
    n_val = int(out_features) if out_features is not None else 0
    key = TimingCacheKey(
        op_type="linear",
        arch=resolved_target_arch or "cuda",
        dtype=dtype_str,
        shape=(m_val, n_val, k_val),
        extra=f"bias={bool(has_bias)},act={activation or 'none'}",
    )
    cache = get_tilelang_timing_cache()
    cached_entry = cache.lookup(key)

    if cached_entry is not None:
        preset = cached_entry.preset_name
        sched = cached_entry.schedule
        default_block_m = int(sched.get("block_m", 64))
        default_block_n = int(sched.get("block_n", 64))
        default_block_k = int(sched.get("block_k", 64))
        default_threads = int(sched.get("threads", 128))
        default_num_stages = int(sched.get("num_stages", 2))
    elif (
        x.ndim == 2
        and int(x.shape[0]) <= 4
        and int(x.shape[1]) == 4096
        and out_features == 4096
        and activation is None
        and x.dtype == torch.float16
        and resolved_target_arch == "sm_89"
    ):
        preset = "sm89_fp16_decode_m_le_4_n4096"
        default_block_m = 16
        default_block_n = 64
        default_block_k = 32
    elif (
        x.ndim >= 1
        and int(x.shape[0]) <= 4
        and x.dtype == torch.bfloat16
        and resolved_target_arch == "sm_89"
    ):
        preset = "sm89_bf16_decode_m_le_4"
        default_block_m = 16
        default_block_n = 64
        default_block_k = 32

    return TileLangLinearSchedule(
        block_m=default_block_m if block_m is None else int(block_m),
        block_n=default_block_n if block_n is None else int(block_n),
        block_k=default_block_k if block_k is None else int(block_k),
        threads=default_threads if threads is None else int(threads),
        num_stages=default_num_stages if num_stages is None else int(num_stages),
        target_arch=resolved_target_arch,
        preset=preset,
    )


def dense_linear_epilogue_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference dense Linear with optional bias and activation epilogue."""

    runtime_weight = weight.to(dtype=x.dtype, device=x.device)
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        runtime_weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


def dense_linear_epilogue_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only dense Linear path using TileLang FP16/BF16 GEMM."""

    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_or_bf16_tensors(*tensors)
    mismatch = target_arch_mismatch(target_arch, x)
    if mismatch is not None:
        raise XQTBackendError(
            f"TileLang Linear target architecture is not executable: {mismatch}"
        )
    if x.ndim != 2 or weight.ndim != 2:
        raise XQTBackendError("dense Linear TileLang path expects 2D x and weight")
    if x.shape[1] != weight.shape[1]:
        raise XQTBackendError("dense Linear TileLang path requires x.shape[1] == weight.shape[1]")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise XQTBackendError("dense Linear TileLang path expects bias shaped [out_features]")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    schedule = resolve_tilelang_linear_schedule(
        x,
        out_features=int(weight.shape[0]),
        activation=activation,
        has_bias=bias is not None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )
    if min(schedule.block_m, schedule.block_n, schedule.block_k) <= 0:
        raise XQTBackendError("dense Linear TileLang block sizes must be positive")
    if any(
        block % 16 != 0
        for block in (schedule.block_m, schedule.block_n, schedule.block_k)
    ):
        raise XQTBackendError(
            "dense Linear TileLang block_m, block_n, and block_k must be multiples of 16"
        )
    if x.shape[1] % schedule.block_k != 0:
        raise XQTBackendError("dense Linear TileLang path requires in_features to be a multiple of block_k")
    require_tilelang()
    kernel = build_tilelang_gemm_kernel(
        m=int(x.shape[0]),
        n=int(weight.shape[0]),
        k=int(x.shape[1]),
        input_dtype=("bfloat16" if x.dtype == torch.bfloat16 else "float16"),
        block_m=schedule.block_m,
        block_n=schedule.block_n,
        block_k=schedule.block_k,
        threads=schedule.threads,
        num_stages=schedule.num_stages,
        target_arch=schedule.target_arch,
        has_bias=bias is not None,
        activation=activation,
    )
    if bias is not None:
        return kernel(x, weight, bias.to(dtype=x.dtype, device=x.device))
    return kernel(x, weight)


def half_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference half Linear path used by the TileLang backend."""

    return dense_gemm_reference(
        x,
        weight.to(dtype=x.dtype, device=x.device),
        None if bias is None else bias.to(dtype=x.dtype, device=x.device),
        transpose_b=True,
    )


def half_linear_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only standalone FP16/BF16 Linear path backed by TileLang GEMM."""

    return dense_linear_epilogue_tilelang(
        x,
        weight,
        bias,
        activation=None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )

TILELANG_LINEAR_KERNEL_METADATA = {
    "dense_linear_epilogue": {
        "kernel_name": "dense_linear_epilogue",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.linear + epilogue",
        "usage": "Static-weight dense Linear path for Ada/Hopper FP16/BF16 GEMM fastpaths after one-time dequant.",
        "supported_dtypes": ["float16", "bfloat16"],
        "bfloat16_block_k_multiple": 16,
        "weight_encoding": "dense_fp16_or_bf16",
        "unpack_stage": "one_time_eager_dequant_cache",
        "fusion_status": "tilelang_dense_16bit_gemm_epilogue",
        "epilogue_stage": "tilelang_fused_bias_activation",
        "schedule_presets": {
            "default": "bm64_bn64_bk64_t128_s2",
            "sm89_fp16_decode_m_le_4_n4096": "bm16_bn64_bk32_t128_s2",
            "sm89_bf16_decode_m_le_4": "bm16_bn64_bk32_t128_s2",
        },
    },
    "linear": {
        "kernel_name": "half_linear",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.linear",
        "usage": "Standalone FP16/BF16 Linear path for direct TileLang operator benchmarking.",
        "supported_dtypes": ["float16", "bfloat16"],
        "bfloat16_block_k_multiple": 16,
        "weight_encoding": "dense_fp16_or_bf16",
        "fusion_status": "tilelang_dense_16bit_gemm",
        "epilogue_stage": None,
        "schedule_presets": {
            "default": "bm64_bn64_bk64_t128_s2",
            "sm89_fp16_decode_m_le_4_n4096": "bm16_bn64_bk32_t128_s2",
            "sm89_bf16_decode_m_le_4": "bm16_bn64_bk32_t128_s2",
        },
    },
}

__all__ = [
    "TILELANG_LINEAR_KERNEL_METADATA",
    "TileLangLinearSchedule",
    "dense_linear_epilogue_reference",
    "dense_linear_epilogue_tilelang",
    "half_linear_reference",
    "half_linear_tilelang",
    "resolve_tilelang_linear_schedule",
]
