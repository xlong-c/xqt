"""Backend capability matrix for XQT operator optimization."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch


def _package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


@dataclass(frozen=True)
class OperatorOptimizationBackendCapability:
    """Static and environment-derived capability description for one backend."""

    backend: str
    status: str
    runtime: str
    exportable: bool
    requires_cuda: bool = False
    available: bool = False
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "runtime": self.runtime,
            "exportable": self.exportable,
            "requires_cuda": self.requires_cuda,
            "available": self.available,
            "notes": list(self.notes),
            "limitations": list(self.limitations),
        }


_BASE_CAPABILITIES: dict[str, OperatorOptimizationBackendCapability] = {
    "torch_compile": OperatorOptimizationBackendCapability(
        backend="torch_compile",
        status="available",
        runtime="pytorch",
        exportable=False,
        notes=(
            "Uses torch.compile over the current PyTorch runtime module.",
            "Whole-model and component-level compile are supported in the built-in pass.",
        ),
        limitations=(
            "Dynamic Python control flow or graph breaks can reduce optimization effectiveness.",
        ),
    ),
    "deployment_backend": OperatorOptimizationBackendCapability(
        backend="deployment_backend",
        status="planned",
        runtime="deployment_backend",
        exportable=True,
        notes=(
            "Represents TensorRT, OpenVINO, or ONNX Runtime deployment fusion rather than PyTorch custom kernels.",
        ),
        limitations=("Built-in executor records capability only and does not rewrite the runtime module.",),
    ),
    "triton": OperatorOptimizationBackendCapability(
        backend="triton",
        status="planned",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=("Reserved for CUDA-only Triton fused kernels.",),
        limitations=("Built-in executor does not yet ship Triton kernels.",),
    ),
    "tilelang": OperatorOptimizationBackendCapability(
        backend="tilelang",
        status="available",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=(
            "Built-in executor ships a minimal attention target with reference fallback metadata.",
            "A minimal CUDA TileLang attention kernel path is available when tilelang and CUDA are present.",
        ),
        limitations=(
            "Current built-in execution is limited to the attention pattern.",
            "Current CUDA execution is limited to float16 attention with dropout_p=0 and seq_kv >= seq_q.",
        ),
    ),
    "cutile": OperatorOptimizationBackendCapability(
        backend="cutile",
        status="planned",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=("Reserved for CUDA-only nvvc CuTile Python DSL kernels.",),
        limitations=(
            "Built-in executor records metadata and reference fallback only.",
            "CuTile package availability and target architecture must be checked per environment.",
        ),
    ),
    "cutlass": OperatorOptimizationBackendCapability(
        backend="cutlass",
        status="planned",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=("Reserved for CUDA-only CUTLASS Python/CuTe DSL kernels.",),
        limitations=(
            "Built-in executor records metadata and reference fallback only.",
            "CUTLASS Python DSL support is version and architecture sensitive.",
        ),
    ),
    "custom_cuda": OperatorOptimizationBackendCapability(
        backend="custom_cuda",
        status="planned",
        runtime="pytorch",
        exportable=False,
        requires_cuda=True,
        notes=("Reserved for optional custom CUDA extensions.",),
        limitations=("Built-in executor does not yet build or load custom CUDA extensions.",),
    ),
}


def describe_operator_backend_capability(
    backend: str,
    *,
    torch_compile_available: Optional[bool] = None,
) -> OperatorOptimizationBackendCapability:
    """Return a capability description for an operator optimization backend."""

    try:
        base = _BASE_CAPABILITIES[backend]
    except KeyError as exc:
        allowed = ", ".join(sorted(_BASE_CAPABILITIES))
        raise ValueError(
            f"Unsupported operator optimization backend: {backend}. Known: {allowed}"
        ) from exc

    available = False
    status = base.status
    notes = list(base.notes)
    if backend == "torch_compile":
        available = (
            bool(torch_compile_available)
            if torch_compile_available is not None
            else hasattr(torch, "compile")
        )
        if not available:
            status = "unavailable"
            notes.append("torch.compile is not available in the current PyTorch build.")
    elif backend == "triton":
        available = _package_available("triton")
        try:
            from .backends.triton import list_triton_kernel_specs

            notes.append(
                "Registered patterns: "
                + ", ".join(sorted(list_triton_kernel_specs()))
            )
        except Exception:
            pass
    elif backend == "tilelang":
        available = True
        if not _package_available("tilelang"):
            notes.append("tilelang package is not importable; built-in execution is limited to reference fallback.")
        try:
            from .backends.tilelang import list_tilelang_kernel_specs

            notes.append(
                "Registered patterns: "
                + ", ".join(sorted(list_tilelang_kernel_specs()))
            )
        except Exception:
            pass
    elif backend == "cutile":
        available = _package_available("cutile")
        try:
            from .backends.cutile import list_cutile_kernel_specs

            notes.append(
                "Registered patterns: "
                + ", ".join(sorted(list_cutile_kernel_specs()))
            )
        except Exception:
            pass
    elif backend == "cutlass":
        available = _package_available("cutlass")
        try:
            from .backends.cutlass import list_cutlass_kernel_specs

            notes.append(
                "Registered patterns: "
                + ", ".join(sorted(list_cutlass_kernel_specs()))
            )
        except Exception:
            pass
    elif backend == "custom_cuda":
        try:
            from .cuda_extension import describe_custom_cuda_extension_capability

            extension = describe_custom_cuda_extension_capability()
            available = extension.available
            notes.extend(extension.notes)
            notes.append(
                "Registered custom ops: "
                + ", ".join(extension.registered_ops)
            )
            if not extension.compiled:
                notes.append("Optional nvcc extension module is not compiled.")
        except Exception:
            available = False
    elif backend == "deployment_backend":
        available = True

    return replace(base, status=status, available=available, notes=tuple(notes))


def list_operator_backend_capabilities() -> dict[str, dict[str, Any]]:
    """Return the operator optimization backend matrix as plain dictionaries."""

    return {
        name: describe_operator_backend_capability(name).to_dict()
        for name in sorted(_BASE_CAPABILITIES)
    }


__all__ = [
    "OperatorOptimizationBackendCapability",
    "describe_operator_backend_capability",
    "list_operator_backend_capabilities",
]
