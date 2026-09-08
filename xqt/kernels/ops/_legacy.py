"""Auto-inventory of legacy registries into xqt.kernels.registry.

Phase 2 bridge: every entry in the legacy registries gets a KernelSpec pointing
to a real callable, so old surfaces are inventoried and remotely comparable.
"""

from __future__ import annotations

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import FormatSignature, KernelBackend, KernelSpec

_BACKEND_MAP = {
    "torch": KernelBackend.TORCH,
    "triton": KernelBackend.TRITON,
    "tilelang": KernelBackend.TILELANG,
    "cutile": KernelBackend.CUTILE,
    "cutlass": KernelBackend.CUTLASS,
    "cute_dsl": KernelBackend.CUTE_DSL,
    "custom_cuda": KernelBackend.CUSTOM_CUDA,
}

_GROUP_TARGET = {
    "triton": "xqt.kernels.ops._impl.engines.triton:run_triton_kernel",
    "tilelang": "xqt.kernels.ops._impl.engines.tilelang:run_tilelang_kernel",
    "cutile": "xqt.kernels.ops._impl.engines.cutile:run_cutile_kernel",
    "cutlass": "xqt.kernels.ops._impl.engines.cutlass:run_cutlass_kernel",
    "cute_dsl": "xqt.kernels.ops._impl.engines.cute_dsl:run_cute_dsl_kernel",
}


def register_legacy_gemm() -> None:
    try:
        from xqt.kernels.ops.gemm.registry import default_registry
    except Exception:
        return
    registry = default_registry()
    for entry in registry.entries():
        backend = _BACKEND_MAP.get(entry.backend)
        if backend is None:
            continue
        try:
            register_kernel(
                KernelSpec(
                    op=f"gemm.{entry.name}",
                    backend=backend,
                    target="xqt.kernels.ops.gemm.dispatch:dispatch_gemm",
                    format_signature=FormatSignature(
                        description=f"legacy gemm {entry.name} ({entry.maturity})"
                    ),
                )
            )
        except ValueError:
            continue


_LEGACY_PATTERNS: dict[str, tuple[str, ...]] = {
    "triton": (
        "attention",
        "bias_gelu",
        "swiglu",
        "geglu",
        "rmsnorm",
        "rmsnorm_channel_first",
        "rmsnorm_residual",
        "rope",
        "gemm_fp16",
        "gemm_bf16",
        "gemm_int8",
        "gemm_fp8",
        "gemm_int4_dequant",
        "gemm_nvfp4_packed_dequant",
        "gemm_mxfp",
    ),
    "tilelang": (
        "attention",
        "conv",
        "conv3d_1x1x1",
        "dequant_gemm_epilogue",
        "dense_linear_epilogue",
        "int8_mma",
        "int8_linear",
        "int8_linear_static_activation",
        "linear",
        "linear_marlin",
        "norm",
        "fp4_packed_dequant_gemm_epilogue",
        "mxfp4_packed_dequant_gemm_epilogue",
        "nvfp4_packed_dequant_gemm_epilogue",
    ),
    "cutile": ("bias_silu",),
    "cutlass": ("gemm_epilogue",),
    "cute_dsl": ("gemm_epilogue", "grouped_gemm"),
}


def register_legacy_operator_patterns() -> None:
    for bk, target in _GROUP_TARGET.items():
        backend = KernelBackend(bk)
        patterns = _LEGACY_PATTERNS.get(bk, ())
        for pattern in patterns:
            try:
                register_kernel(
                    KernelSpec(
                        op=f"{bk}.{pattern}",
                        backend=backend,
                        target=target,
                        format_signature=FormatSignature(
                            description=f"legacy operator pattern {pattern}"
                        ),
                    )
                )
            except ValueError:
                continue


register_legacy_gemm()
register_legacy_operator_patterns()
