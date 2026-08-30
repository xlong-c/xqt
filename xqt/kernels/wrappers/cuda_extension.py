"""Optional custom CUDA extension scaffolding for XQT operator optimization.

This module intentionally does not compile or import an nvcc-built extension during
normal package import. It registers a tiny torch.library custom op with a Python
reference implementation and a FakeTensor implementation so compile/export analysis
can exercise the dispatch path before a real CUDA extension exists.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


CUSTOM_CUDA_BUILD_ENV = "XQT_BUILD_CUSTOM_CUDA"
CUSTOM_CUDA_EXTENSION_NAME = "xqt_operator_opt_cuda"
CUSTOM_CUDA_OP_NAMESPACE = "xqt_operator_opt"


@dataclass(frozen=True)
class CustomCudaExtensionCapability:
    """Preflight-friendly status for the optional custom CUDA extension."""

    extension_name: str
    build_env_var: str
    build_requested: bool
    compiled: bool
    available: bool
    requires_cuda: bool
    cuda_available: bool
    registered_ops: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "extension_name": self.extension_name,
            "build_env_var": self.build_env_var,
            "build_requested": self.build_requested,
            "compiled": self.compiled,
            "available": self.available,
            "requires_cuda": self.requires_cuda,
            "cuda_available": self.cuda_available,
            "registered_ops": list(self.registered_ops),
            "notes": list(self.notes),
        }


@torch.library.custom_op(
    f"{CUSTOM_CUDA_OP_NAMESPACE}::bias_gelu",
    mutates_args=(),
)
def fused_bias_gelu_custom_cuda(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Reference implementation for the future custom CUDA bias + GELU op."""

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if bias.ndim != 1:
        raise ValueError("bias must be one-dimensional")
    if bias.numel() != x.shape[-1]:
        raise ValueError("bias must match the last dimension of x")
    return F.gelu(x + bias)


@fused_bias_gelu_custom_cuda.register_fake
def _fused_bias_gelu_custom_cuda_fake(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    torch._check(x.dim() >= 1, lambda: "x must have at least one dimension")
    torch._check(bias.dim() == 1, lambda: "bias must be one-dimensional")
    torch._check(
        bias.shape[0] == x.shape[-1],
        lambda: "bias must match the last dimension of x",
    )
    return torch.empty_like(x)


CUSTOM_CUDA_OPS: Mapping[str, Any] = {
    "bias_gelu": fused_bias_gelu_custom_cuda,
}


def custom_cuda_build_requested(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the optional custom CUDA build switch is enabled."""

    values = env if env is not None else os.environ
    return str(values.get(CUSTOM_CUDA_BUILD_ENV, "")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def custom_cuda_extension_compiled() -> bool:
    """Return whether the optional nvcc-built extension module is importable."""

    return importlib.util.find_spec(CUSTOM_CUDA_EXTENSION_NAME) is not None


def describe_custom_cuda_extension_capability(
    env: Mapping[str, str] | None = None,
) -> CustomCudaExtensionCapability:
    """Describe custom CUDA extension readiness without compiling anything."""

    compiled = custom_cuda_extension_compiled()
    cuda_available = torch.cuda.is_available()
    build_requested = custom_cuda_build_requested(env)
    notes = [
        "torch.library custom op and FakeTensor implementation are registered.",
        "No nvcc extension is compiled during the base package import.",
    ]
    if build_requested and not compiled:
        notes.append(
            f"{CUSTOM_CUDA_BUILD_ENV} is enabled, but {CUSTOM_CUDA_EXTENSION_NAME} is not importable."
        )
    return CustomCudaExtensionCapability(
        extension_name=CUSTOM_CUDA_EXTENSION_NAME,
        build_env_var=CUSTOM_CUDA_BUILD_ENV,
        build_requested=build_requested,
        compiled=compiled,
        available=compiled and cuda_available,
        requires_cuda=True,
        cuda_available=cuda_available,
        registered_ops=tuple(sorted(CUSTOM_CUDA_OPS)),
        notes=tuple(notes),
    )


def run_custom_cuda_opcheck(
    op_name: str = "bias_gelu",
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    raise_exception: bool = True,
) -> dict[str, str]:
    """Run torch.library.opcheck for a registered custom CUDA placeholder op."""

    if op_name not in CUSTOM_CUDA_OPS:
        allowed = ", ".join(sorted(CUSTOM_CUDA_OPS))
        raise ValueError(f"Unsupported custom CUDA op: {op_name}. Known: {allowed}")
    if args is None:
        args = (torch.randn(2, 4), torch.randn(4))
    result = torch.library.opcheck(
        CUSTOM_CUDA_OPS[op_name],
        args,
        kwargs or {},
        raise_exception=raise_exception,
    )
    return {str(key): str(value) for key, value in result.items()}


__all__ = [
    "CUSTOM_CUDA_BUILD_ENV",
    "CUSTOM_CUDA_EXTENSION_NAME",
    "CUSTOM_CUDA_OP_NAMESPACE",
    "CUSTOM_CUDA_OPS",
    "CustomCudaExtensionCapability",
    "custom_cuda_build_requested",
    "custom_cuda_extension_compiled",
    "describe_custom_cuda_extension_capability",
    "fused_bias_gelu_custom_cuda",
    "run_custom_cuda_opcheck",
]
