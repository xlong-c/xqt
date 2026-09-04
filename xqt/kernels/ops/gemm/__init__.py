"""gemm kernels."""
from __future__ import annotations

from typing import Any

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend, KernelSpec
from xqt.kernels.ops._legacy_api import load_legacy

_CUDA = frozenset({CapabilityRequirement.CUDA})


def _gemm_reference(
    a: Any,
    b: Any,
    bias: Any | None = None,
) -> Any:
    import torch

    out = torch.matmul(a, b)
    if bias is not None:
        out = out + bias
    return out


register_kernel(
    KernelSpec(
        op="gemm.bmm_fp8",
        backend=KernelBackend.TRITON,
        target="xqt.kernels.ops._impl.triton.gemm:gemm_fp8_triton",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="fp8 scaled matmul"),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.bmm_fp8",
        backend=KernelBackend.FLASHINFER,
        target="flashinfer:bmm_fp8",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="bmm fp8 flashinfer"),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.gemm_fp16",
        backend=KernelBackend.TORCH,
        target="xqt.kernels.ops.gemm:_gemm_reference",
        format_signature=FormatSignature(supported_dtypes=("float16",), description="fp16 matmul reference"),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.gemm_fp16",
        backend=KernelBackend.TRITON,
        target="xqt.kernels.ops._impl.triton.gemm:gemm_fp16_triton",
        capabilities=_CUDA,
        format_signature=FormatSignature(supported_dtypes=("float16",)),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.gemm_fp16",
        backend=KernelBackend.TILELANG,
        target="xqt.kernels.ops._impl.tilelang.linear:half_linear_tilelang",
        capabilities=_CUDA,
        format_signature=FormatSignature(supported_dtypes=("float16", "bfloat16")),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.gemm_bf16",
        backend=KernelBackend.TRITON,
        target="xqt.kernels.ops._impl.triton.gemm:gemm_bf16_triton",
        capabilities=_CUDA,
        format_signature=FormatSignature(supported_dtypes=("bfloat16",)),
    )
)

_CONTRACT_MODULES = (
    "contracts",
    "benchmark",
    "dispatch",
    "grouped_dispatch",
    "fp8",
    "layout",
    "lowbit",
    "reference",
    "p4",
    "preflight",
    "tuning_cache",
    "quantize",
    "registry",
)

__all__ = [
    "gemm_bf16_triton",
    "gemm_bmm_fp8_triton",
    "gemm_fp16_tilelang",
    "gemm_fp16_torch",
    "gemm_fp16_triton",
]


def __getattr__(name: str) -> Any:
    import importlib

    for module_name in _CONTRACT_MODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        exported = getattr(module, "__all__", ())
        if name in exported:
            value = getattr(module, name)
            globals()[name] = value
            return value
    try:
        backends = importlib.import_module("xqt.kernels.ops._impl.gemm_backends")
    except Exception:
        backends = None
    if backends is not None and name in getattr(backends, "__all__", ()):
        value = getattr(backends, name)
        globals()[name] = value
        return value
    return load_legacy("gemm", name)


def __dir__() -> list[str]:
    names = set(__all__)
    import importlib

    for module_name in _CONTRACT_MODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        names.update(item for item in getattr(module, "__all__", ()) if not str(item).startswith("_"))
    try:
        backends = importlib.import_module("xqt.kernels.ops._impl.gemm_backends")
        names.update(item for item in getattr(backends, "__all__", ()) if not str(item).startswith("_"))
    except Exception:
        pass
    return sorted(names)


def gemm_bmm_fp8_triton(*args: Any, **kwargs: Any) -> Any:
    from xqt.kernels.selector import get_kernel

    return get_kernel("gemm.bmm_fp8", KernelBackend.TRITON)(*args, **kwargs)


def gemm_fp16_triton(*args: Any, **kwargs: Any) -> Any:
    from xqt.kernels.selector import get_kernel

    return get_kernel("gemm.gemm_fp16", KernelBackend.TRITON)(*args, **kwargs)


def gemm_bf16_triton(*args: Any, **kwargs: Any) -> Any:
    from xqt.kernels.selector import get_kernel

    return get_kernel("gemm.gemm_bf16", KernelBackend.TRITON)(*args, **kwargs)


def gemm_fp16_tilelang(*args: Any, **kwargs: Any) -> Any:
    from xqt.kernels.selector import get_kernel

    return get_kernel("gemm.gemm_fp16", KernelBackend.TILELANG)(*args, **kwargs)


def gemm_fp16_torch(*args: Any, **kwargs: Any) -> Any:
    from xqt.kernels.selector import get_kernel

    return get_kernel("gemm.gemm_fp16", KernelBackend.TORCH)(*args, **kwargs)
