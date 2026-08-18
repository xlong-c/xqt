"""SM89 INT8 executor adapter for the self-owned GEMM registry.

The CUDA source remains a separately built artifact while it is being migrated
from the historical ``operator_opt`` location.  This adapter owns the logical
``[N,K]`` contract and makes the artifact status explicit; it never reports a
missing or shape-incompatible shared object as native execution.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.gemm.common.contracts import GemmSpec, PackedWeight
from xqt.gemm.common.preflight import artifact_manifest_path, artifact_ready_for_execution
from xqt.gemm.common.registry import GemmKernelRegistration, GemmKernelRegistry


_DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "operator_opt"
    / "kernels"
    / "cute"
    / "build"
    / "int8mma_sm89.so"
)


def _load_legacy_binding(artifact: Path) -> tuple[Any, Any]:
    """Load old ctypes setup lazily while the source migration is in progress."""

    from xqt.operator_opt.kernels.cute import int8mma_binding as binding

    if not artifact.is_file():
        raise XQTBackendError(f"SM89 INT8 artifact not found: {artifact}")
    library = binding._load_lib(str(artifact))
    return binding, library


def sm89_artifact_available(artifact: str | Path | None = None) -> bool:
    """Return whether the requested shared object exists and exports metadata."""

    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    if not path.is_file():
        return False
    try:
        _, library = _load_legacy_binding(path)
        return hasattr(library, "int8mma_run_cutlass_64x128_prepacked_b")
    except (OSError, XQTBackendError, AttributeError):
        return False


def _as_weight_tensor(weight: torch.Tensor | PackedWeight) -> torch.Tensor:
    qweight = weight.qweight if isinstance(weight, PackedWeight) else weight
    if not isinstance(qweight, torch.Tensor):
        raise XQTBackendError("SM89 W8A8 weight must be a torch.Tensor or PackedWeight")
    return qweight


def prepack_sm89_int8_weight(
    weight: torch.Tensor,
    *,
    scales: torch.Tensor | None,
    zero_points: torch.Tensor | None = None,
    pack_version: str = "sm89-int8-nk-v1",
) -> PackedWeight:
    """Prepack canonical INT8 ``[N,K]`` once for the SM89 fused path."""

    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.dtype != torch.int8:
        raise XQTBackendError("SM89 prepack expects int8 canonical weight [N,K]")
    if zero_points is not None:
        raise XQTBackendError("SM89 symmetric INT8 prepack does not support zero points")
    from xqt.operator_opt.kernels.cute import int8mma_binding as binding

    canonical = weight.contiguous()
    packed = binding.prepack_qweight_t_for_ptx_sm89(canonical.transpose(0, 1).contiguous())
    from xqt.gemm.common.contracts import PackedWeightMetadata

    metadata = PackedWeightMetadata(
        logical_shape=(int(weight.shape[0]), int(weight.shape[1])),
        storage_layout="sm89_int8_nk_v1",
        pack_version=pack_version,
        weight_dtype="int8",
        padded_k=int(weight.shape[1]),
    )
    return PackedWeight(
        qweight=packed,
        scales=scales,
        zero_points=None,
        metadata=metadata,
        canonical_qweight=canonical,
    )


def sm89_w8a8_executor(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    weight_zero_points: torch.Tensor | None = None,
    activation_zero_points: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    artifact: str | Path | None = None,
) -> torch.Tensor:
    """Execute SM89 CUTLASS INT8 MMA for static scalar activation scale.

    The historical kernel's fused epilogue accepts one activation scale and a
    per-channel weight scale.  Per-token dynamic scales use an INT32 cuBLASLt
    path with an explicit scale epilogue.  Symmetric zero points, residual
    fusion and activation epilogues remain outside this native contract.
    """

    if spec.quant.weight_dtype != "int8" or spec.quant.activation_dtype != "int8":
        raise XQTBackendError("SM89 W8A8 executor requires int8 x int8")
    if spec.quant.activation_granularity not in {"per_tensor", "per_token"}:
        raise XQTBackendError("SM89 W8A8 artifact accepts per_tensor or per_token activation scale")
    if spec.quant.activation_scale_source not in {"activation_static", "activation_dynamic"}:
        raise XQTBackendError("SM89 W8A8 requires static or dynamic activation scale")
    if weight_zero_points is not None or activation_zero_points is not None:
        raise XQTBackendError("SM89 symmetric INT8 executor does not support zero points")
    if residual is not None:
        raise XQTBackendError("SM89 W8A8 artifact has no residual epilogue")
    if spec.epilogue.activation != "none":
        raise XQTBackendError("SM89 W8A8 artifact has no fused activation epilogue")
    if spec.quant.output_dtype not in {"fp16", "bf16"} or spec.epilogue.output_dtype not in {
        "fp16",
        "bf16",
    }:
        raise XQTBackendError("SM89 W8A8 artifact outputs fp16 or bf16 only")
    if not activation.is_cuda or activation.dtype != torch.int8 or activation.ndim != 2:
        raise XQTBackendError("SM89 W8A8 activation must be CUDA int8 [M,K]")
    qweight = _as_weight_tensor(weight)
    if not qweight.is_cuda or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise XQTBackendError("SM89 W8A8 weight must be CUDA int8 [N,K]")
    if tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("SM89 W8A8 activation shape does not match GemmProblem")
    if weight_scales is None or int(weight_scales.numel()) != spec.problem.n:
        raise XQTBackendError("SM89 W8A8 requires N per-channel weight scales")
    expected_scale_elements = 1 if spec.quant.activation_granularity == "per_tensor" else spec.problem.m
    if activation_scales is None or int(activation_scales.numel()) != expected_scale_elements:
        raise XQTBackendError(
            "SM89 W8A8 activation scale must have "
            f"{expected_scale_elements} element(s) for {spec.quant.activation_granularity}"
        )
    if bias is not None and int(bias.numel()) != spec.problem.n:
        raise XQTBackendError("SM89 W8A8 bias must have N elements")
    if spec.epilogue.has_bias and bias is None:
        raise XQTBackendError("SM89 W8A8 epilogue declares bias but no bias was supplied")
    if not hasattr(torch.cuda, "get_device_capability"):
        raise XQTBackendError("torch CUDA capability query is unavailable")
    major, minor = torch.cuda.get_device_capability(activation.device)
    if (major, minor) != (8, 9):
        raise XQTBackendError(f"SM89 W8A8 executor received sm_{major}{minor}")
    artifact_path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    binding, library = _load_legacy_binding(artifact_path)
    if not hasattr(library, "int8mma_run_cutlass_64x128_prepacked_b"):
        raise XQTBackendError("SM89 artifact lacks CUTLASS 64x128 fused symbol")
    canonical_qweight = (
        weight.canonical_qweight
        if isinstance(weight, PackedWeight) and weight.canonical_qweight is not None
        else qweight
    )
    logical_shape = (spec.problem.n, spec.problem.k)
    if (
        not isinstance(canonical_qweight, torch.Tensor)
        or tuple(canonical_qweight.shape) != logical_shape
    ):
        raise XQTBackendError("SM89 W8A8 weight must use canonical [N,K] shape")
    if canonical_qweight.device != activation.device:
        raise XQTBackendError("SM89 canonical weight and activation must share a CUDA device")
    scale = activation_scales.detach().to(device=activation.device, dtype=torch.float32)
    wscale = weight_scales.detach().to(device=activation.device, dtype=torch.float32).reshape(-1)
    bias_value = (
        torch.zeros(spec.problem.n, device=activation.device, dtype=torch.float32)
        if bias is None
        else bias.detach().to(device=activation.device, dtype=torch.float32).reshape(-1)
    )
    if spec.problem.m == 1:
        if spec.quant.output_dtype != "fp16" or spec.epilogue.output_dtype != "fp16":
            raise XQTBackendError("SM89 M=1 GEMV currently supports fp16 output only")
        if scale.numel() != 1:
            raise XQTBackendError("SM89 M=1 GEMV requires one activation scale")
        logical_qweight_t = canonical_qweight.transpose(0, 1).contiguous()
        prepacked = (
            qweight
            if isinstance(weight, PackedWeight)
            and tuple(qweight.shape) == logical_shape
            and weight.metadata.storage_layout == "sm89_int8_nk_v1"
            else binding.prepack_qweight_t_for_ptx_sm89(logical_qweight_t)
        )
        output = torch.empty(
            (1, spec.problem.n), device=activation.device, dtype=torch.float16
        )
        if not hasattr(library, "int8_gemv_m1_run_prepacked_b"):
            raise XQTBackendError("SM89 artifact lacks M=1 DP4A GEMV symbol")
        error = library.int8_gemv_m1_run_prepacked_b(
            activation.contiguous().data_ptr(),
            prepacked.data_ptr(),
            output.data_ptr(),
            scale.reshape(1).contiguous().data_ptr(),
            wscale.contiguous().data_ptr(),
            spec.problem.n,
            spec.problem.k,
        )
        if error != 0:
            raise XQTBackendError(f"SM89 M=1 GEMV failed with CUDA error {error}")
        return output + bias_value.to(dtype=torch.float16).reshape(1, -1)
    if spec.problem.m < 32:
        raise XQTBackendError("SM89 W8A8 M=2..31 requires the small-M fallback")
    padded_m = ((spec.problem.m + 15) // 16) * 16
    padded_n = ((spec.problem.n + 15) // 16) * 16
    padded_k = ((spec.problem.k + 31) // 32) * 32
    activation_exec = activation
    if (padded_m, padded_k) != tuple(activation.shape):
        activation_exec = F.pad(
            activation,
            (0, padded_k - spec.problem.k, 0, padded_m - spec.problem.m),
        )
    qweight_exec = canonical_qweight
    if (padded_n, padded_k) != tuple(canonical_qweight.shape):
        qweight_exec = F.pad(
            canonical_qweight,
            (0, padded_k - spec.problem.k, 0, padded_n - spec.problem.n),
        )
    qweight_t = qweight_exec.transpose(0, 1).contiguous()
    if wscale.numel() != padded_n:
        padded_wscale = torch.ones(padded_n, device=activation.device, dtype=torch.float32)
        padded_wscale[: spec.problem.n].copy_(wscale)
        wscale = padded_wscale
    if spec.quant.activation_granularity == "per_token" and scale.numel() != padded_m:
        padded_scale = torch.ones(padded_m, device=activation.device, dtype=torch.float32)
        padded_scale[: spec.problem.m].copy_(scale.reshape(-1))
        scale = padded_scale
    if bias_value.numel() != padded_n:
        padded_bias = torch.zeros(padded_n, device=activation.device, dtype=torch.float32)
        padded_bias[: spec.problem.n].copy_(bias_value)
        bias_value = padded_bias
    if (
        spec.quant.activation_granularity == "per_token"
        or scale.numel() != 1
        or spec.quant.output_dtype == "bf16"
    ):
        if not hasattr(library, "int8mma_run_cublaslt_i32"):
            raise XQTBackendError("SM89 artifact lacks cuBLASLt INT32 symbol for per-token scales")
        if scale.numel() not in {1, spec.problem.m}:
            if scale.numel() != padded_m:
                raise XQTBackendError("per-token activation scale must have M elements")
        workspace = binding._cublaslt_workspace(activation.device)
        stream = torch.cuda.current_stream(activation.device).cuda_stream
        accum = torch.empty(
            (padded_m, padded_n), device=activation.device, dtype=torch.int32
        )
        error = library.int8mma_run_cublaslt_i32(
            activation_exec.contiguous().data_ptr(),
            qweight_t.data_ptr(),
            accum.data_ptr(),
            padded_m,
            padded_n,
            padded_k,
            workspace.data_ptr(),
            workspace.numel(),
            stream,
        )
        if error != 0:
            raise XQTBackendError(f"SM89 cuBLASLt INT8 GEMM failed with status {error}")
        scale_a = scale.reshape(1, 1) if scale.numel() == 1 else scale.reshape(-1, 1)
        output = accum.to(torch.float32) * scale_a * wscale.reshape(1, -1)
        output = output + bias_value.reshape(1, -1)
        return output[: spec.problem.m, : spec.problem.n].to(
            dtype=torch.float16 if spec.quant.output_dtype == "fp16" else torch.bfloat16
        )
    prepacked = (
        qweight
        if isinstance(weight, PackedWeight)
        and tuple(qweight.shape) == (padded_n, padded_k)
        and weight.metadata.storage_layout == "sm89_int8_nk_v1"
        else binding.prepack_qweight_t_for_ptx_sm89(qweight_t)
    )
    scale = scale.reshape(1)
    scale_bias = torch.stack((scale[0] * wscale, bias_value), dim=1).contiguous()
    output = torch.empty(
        (padded_m, padded_n), device=activation.device, dtype=torch.float16
    )
    error = library.int8mma_run_cutlass_64x128_prepacked_b(
        activation_exec.contiguous().data_ptr(),
        prepacked.data_ptr(),
        output.data_ptr(),
        scale_bias.data_ptr(),
        padded_m,
        padded_n,
        padded_k,
    )
    if error != 0:
        raise XQTBackendError(f"SM89 CUTLASS W8A8 GEMM failed with CUDA error {error}")
    return output[: spec.problem.m, : spec.problem.n]


def install_sm89_w8a8_executor(
    registry: GemmKernelRegistry,
    *,
    artifact: str | Path | None = None,
    manifest: str | Path | None = None,
) -> bool:
    """Promote the entry only after loadability and numeric correctness gates."""

    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    if not sm89_artifact_available(path):
        return False
    manifest_path = Path(manifest) if manifest is not None else artifact_manifest_path(path)
    if not artifact_ready_for_execution(
        manifest_path,
        kernel_name="sm89_int8_mma_cutlass",
        target_arch="sm_89",
    ):
        return False
    entry = registry.get("sm89_int8_mma_cutlass")
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
            implementation="cutlass_sm89_artifact",
            executor=lambda *args, **kwargs: sm89_w8a8_executor(
                *args, artifact=path, **kwargs
            ),
        )
    )
    return True


__all__ = [
    "install_sm89_w8a8_executor",
    "sm89_artifact_available",
    "sm89_w8a8_executor",
    "prepack_sm89_int8_weight",
]
