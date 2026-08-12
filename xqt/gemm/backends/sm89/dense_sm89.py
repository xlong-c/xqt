"""Lazy ctypes binding for the SM89 dense CUTLASS artifact."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ..contracts import GemmSpec
from ..preflight import artifact_manifest_path, artifact_ready_for_execution
from ..registry import GemmKernelRegistration, GemmKernelRegistry


def _load_library(artifact: str | Path) -> ctypes.CDLL:
    path = Path(artifact)
    if not path.is_file():
        raise XQTBackendError(f"SM89 dense artifact not found: {path}")
    library = ctypes.CDLL(str(path))
    for name in ("dense_sm89_fp16_run", "dense_sm89_bf16_run"):
        if not hasattr(library, name):
            continue
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
    return library


def dense_sm89_artifact_available(artifact: str | Path) -> bool:
    try:
        library = _load_library(artifact)
    except (OSError, XQTBackendError):
        return False
    return hasattr(library, "dense_sm89_fp16_run") and hasattr(library, "dense_sm89_bf16_run")


def dense_sm89_executor(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    spec: GemmSpec,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    artifact: str | Path,
    **_: object,
) -> torch.Tensor:
    """Run native dense GEMM with beta source for bias/residual."""

    if spec.quant.weight_dtype != spec.quant.activation_dtype:
        raise XQTBackendError("dense SM89 executor requires matching activation/weight dtype")
    if spec.quant.weight_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("dense SM89 executor supports fp16 or bf16 only")
    if spec.quant.output_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("dense SM89 executor supports fp16 or bf16 output only")
    if spec.epilogue.activation != "none":
        raise XQTBackendError("dense SM89 artifact has no fused activation epilogue")
    if not activation.is_cuda or not weight.is_cuda:
        raise XQTBackendError("dense SM89 executor requires CUDA tensors")
    if activation.ndim != 2 or weight.ndim != 2:
        raise XQTBackendError("dense SM89 executor expects activation [M,K] and weight [N,K]")
    if tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("activation shape does not match GemmProblem")
    if tuple(weight.shape) != (spec.problem.n, spec.problem.k):
        raise XQTBackendError("weight shape does not match GemmProblem")
    expected_dtype = torch.float16 if spec.quant.weight_dtype == "fp16" else torch.bfloat16
    if activation.dtype != expected_dtype or weight.dtype != expected_dtype:
        raise XQTBackendError("dense SM89 tensors do not match declared dtype")
    if spec.problem.m < 8:
        raise XQTBackendError("dense SM89 M<8 requires the small-M fallback")
    if residual is not None and tuple(residual.shape) != (spec.problem.m, spec.problem.n):
        raise XQTBackendError("dense SM89 residual must have output shape")
    if bias is not None and int(bias.numel()) != spec.problem.n:
        raise XQTBackendError("dense SM89 bias must have N elements")
    if spec.epilogue.has_bias and bias is None:
        raise XQTBackendError("dense SM89 epilogue declares bias but bias is missing")
    if spec.epilogue.has_residual and residual is None:
        raise XQTBackendError("dense SM89 epilogue declares residual but residual is missing")
    library = _load_library(artifact)
    symbol = "dense_sm89_fp16_run" if expected_dtype == torch.float16 else "dense_sm89_bf16_run"
    if not hasattr(library, symbol):
        raise XQTBackendError(f"SM89 dense artifact lacks {symbol}")
    alignment = 8
    padded_m = ((spec.problem.m + alignment - 1) // alignment) * alignment
    padded_n = ((spec.problem.n + alignment - 1) // alignment) * alignment
    padded_k = ((spec.problem.k + alignment - 1) // alignment) * alignment
    activation_padded = activation
    if (padded_m, padded_k) != tuple(activation.shape):
        activation_padded = F.pad(
            activation,
            (0, padded_k - spec.problem.k, 0, padded_m - spec.problem.m),
        )
    weight_padded = weight
    if (padded_n, padded_k) != tuple(weight.shape):
        weight_padded = F.pad(
            weight,
            (0, padded_k - spec.problem.k, 0, padded_n - spec.problem.n),
        )
    c_source = torch.zeros(
        (padded_m, padded_n), device=activation.device, dtype=expected_dtype
    )
    if bias is not None:
        c_source[:, : spec.problem.n].add_(
            bias.to(device=activation.device, dtype=expected_dtype).reshape(1, -1)
        )
    if residual is not None:
        c_source[: spec.problem.m, : spec.problem.n].add_(
            residual.to(device=activation.device, dtype=expected_dtype)
        )
    output = torch.empty_like(c_source)
    error = getattr(library, symbol)(
        activation_padded.contiguous().data_ptr(),
        weight_padded.contiguous().data_ptr(),
        c_source.data_ptr(),
        output.data_ptr(),
        padded_m,
        padded_n,
        padded_k,
    )
    if error != 0:
        raise XQTBackendError(f"SM89 dense CUTLASS GEMM failed with CUDA error {error}")
    return output[: spec.problem.m, : spec.problem.n]


def install_sm89_dense_executors(
    registry: GemmKernelRegistry,
    *,
    artifact: str | Path,
    manifest: str | Path | None = None,
) -> bool:
    """Promote dense entries after loadability and numeric correctness gates."""

    if not dense_sm89_artifact_available(artifact):
        return False
    manifest_path = Path(manifest) if manifest is not None else artifact_manifest_path(artifact)
    if not artifact_ready_for_execution(
        manifest_path,
        kernel_name="sm89_dense_cutlass",
        target_arch="sm_89",
    ):
        return False
    for name in ("sm89_dense_fp16_cutlass", "sm89_dense_bf16_cutlass"):
        entry = registry.get(name)
        registry.replace(
            GemmKernelRegistration(
                name=entry.name,
                backend=entry.backend,
                maturity="executable",
                capability=entry.capability,
                kernel_family=entry.kernel_family,
                layout=entry.layout,
                tile_shape=entry.tile_shape,
                warp_count=entry.warp_count,
                stage_count=entry.stage_count,
                alignment=entry.alignment,
                priority=entry.priority,
                implementation="cutlass_sm89_dense_artifact",
                executor=lambda *args, **kwargs: dense_sm89_executor(
                    *args, artifact=artifact, **kwargs
                ),
            )
        )
    return True


__all__ = [
    "dense_sm89_artifact_available",
    "dense_sm89_executor",
    "install_sm89_dense_executors",
]
