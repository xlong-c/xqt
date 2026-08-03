"""CuTe DSL kernel references and guarded entry points."""

from .gemm import (
    CUTE_DSL_KERNEL_METADATA,
    gemm_epilogue_cute_dsl,
    gemm_epilogue_reference,
)

__all__ = [
    "CUTE_DSL_KERNEL_METADATA",
    "gemm_epilogue_cute_dsl",
    "gemm_epilogue_reference",
]
