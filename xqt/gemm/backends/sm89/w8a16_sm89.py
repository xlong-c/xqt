"""SM89 W8A16 weight-only adapter.

The SM89 artifact exposes an INT8xINT8 MMA kernel.  W8A16 therefore uses an
explicit dynamic INT8 activation quantization step before entering that kernel.
The composition is reported as a native W8A16 candidate only after the shared
artifact manifest has passed its correctness gate; unsupported small-M and
groupwise contracts remain transparent reference fallbacks.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from xqt.core.errors import XQTBackendError

from xqt.gemm.common.contracts import GemmSpec, PackedWeight, QuantSpec
from xqt.gemm.common.preflight import artifact_manifest_path, artifact_ready_for_execution
from xqt.gemm.common.quantize import quantize_int8_activation
from xqt.gemm.common.registry import GemmKernelRegistration, GemmKernelRegistry
from .sm89 import prepack_sm89_int8_weight, sm89_artifact_available, sm89_w8a8_executor


_DEFAULT_ARTIFACT = Path.home() / ".cache/xqt/gemm/sm89/w8a16_sm89.so"


def sm89_w8a16_artifact_available(artifact: str | Path | None = None) -> bool:
    """Return whether the W8A16 shared object exports the INT8 MMA ABI."""

    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    return sm89_artifact_available(path)


def _canonical_weight(weight: PackedWeight, *, spec: GemmSpec) -> torch.Tensor:
    """Validate and return the logical canonical ``[N,K]`` INT8 weight."""

    if weight.metadata.weight_dtype != "int8":
        raise XQTBackendError("SM89 W8A16 requires INT8 PackedWeight metadata")
    if weight.metadata.storage_layout != "xqt_int8_nk_v1":
        raise XQTBackendError(
            "SM89 W8A16 requires storage_layout='xqt_int8_nk_v1', "
            f"got {weight.metadata.storage_layout!r}"
        )
    if tuple(weight.metadata.logical_shape) != (spec.problem.n, spec.problem.k):
        raise XQTBackendError("SM89 W8A16 weight logical shape disagrees with GemmProblem")
    canonical = weight.canonical_qweight
    if canonical is None:
        canonical = weight.qweight
    if not isinstance(canonical, torch.Tensor):
        raise XQTBackendError("SM89 W8A16 canonical weight must be a torch.Tensor")
    if canonical.ndim != 2 or canonical.dtype != torch.int8:
        raise XQTBackendError("SM89 W8A16 canonical weight must be int8 [N,K]")
    if tuple(canonical.shape) != (spec.problem.n, spec.problem.k):
        raise XQTBackendError("SM89 W8A16 canonical weight must use logical [N,K] shape")
    if not canonical.is_cuda:
        raise XQTBackendError("SM89 W8A16 canonical weight must be CUDA")
    if weight.zero_points is not None:
        raise XQTBackendError("SM89 W8A16 symmetric weight-only path rejects zero points")
    if weight.scales is None:
        raise XQTBackendError("SM89 W8A16 requires per-channel weight scales")
    scales = weight.scales
    if not isinstance(scales, torch.Tensor) or not scales.is_cuda:
        raise XQTBackendError("SM89 W8A16 weight scales must be CUDA tensors")
    if scales.dtype != torch.float32 or int(scales.numel()) != spec.problem.n:
        raise XQTBackendError("SM89 W8A16 per-channel scales must be CUDA float32 [N]")
    return canonical


def _int8_spec(spec: GemmSpec, *, activation_granularity: str) -> GemmSpec:
    """Build the internal W8A8 contract used after activation quantization."""

    quant = QuantSpec(
        weight_dtype="int8",
        activation_dtype="int8",
        compute_dtype="fp32",
        accum_dtype="int32",
        output_dtype=spec.quant.output_dtype,
        weight_granularity="per_channel",
        activation_granularity=activation_granularity,
        weight_scale_source="weight_offline",
        activation_scale_source="activation_dynamic",
        storage_layout="sm89_int8_nk_v1",
        pack_version=spec.quant.pack_version,
    )
    return replace(spec, quant=quant)


def sm89_w8a16_executor(
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
    """Run the gated SM89 W8A16 composition for supported shapes."""

    if spec.quant.weight_dtype != "int8":
        raise XQTBackendError("SM89 W8A16 requires weight_dtype='int8'")
    if spec.quant.activation_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("SM89 W8A16 requires fp16 or bf16 activation")
    if spec.quant.weight_granularity != "per_channel":
        raise XQTBackendError("SM89 W8A16 native path supports per-channel scales only")
    if spec.quant.activation_granularity != "per_tensor":
        raise XQTBackendError("SM89 W8A16 activation granularity is fixed at per_tensor")
    if spec.quant.weight_zero_point or weight_zero_points is not None:
        raise XQTBackendError("SM89 W8A16 native path is symmetric and has no zero point")
    if activation_zero_points is not None or residual is not None:
        raise XQTBackendError("SM89 W8A16 has no activation zero point or residual epilogue")
    if spec.epilogue.activation != "none":
        raise XQTBackendError("SM89 W8A16 supports bias-only epilogue")
    if spec.quant.output_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("SM89 W8A16 output must be fp16 or bf16")
    if not isinstance(weight, PackedWeight):
        raise XQTBackendError("SM89 W8A16 requires a canonical PackedWeight")
    canonical = _canonical_weight(weight, spec=spec)
    if weight_scales is not None and not torch.equal(weight_scales, weight.scales):
        raise XQTBackendError("external weight_scales disagree with PackedWeight.scales")
    if not isinstance(activation, torch.Tensor) or not activation.is_cuda:
        raise XQTBackendError("SM89 W8A16 activation must be CUDA")
    expected_dtype = torch.float16 if spec.quant.activation_dtype == "fp16" else torch.bfloat16
    if activation.dtype != expected_dtype or activation.ndim != 2:
        raise XQTBackendError("SM89 W8A16 activation dtype/shape is invalid")
    if tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("SM89 W8A16 activation shape disagrees with GemmProblem")
    if bias is not None and int(bias.numel()) != spec.problem.n:
        raise XQTBackendError("SM89 W8A16 bias must have N elements")
    if spec.epilogue.has_bias and bias is None:
        raise XQTBackendError("SM89 W8A16 epilogue declares bias but no bias was supplied")
    if spec.problem.m == 0:
        return torch.empty(
            (0, spec.problem.n), device=activation.device, dtype=expected_dtype
        )
    # The legacy prepacked GEMV kernel reads each weight row with aligned
    # 32-bit loads.  Odd row strides (K not divisible by four) can therefore
    # produce a CUDA misaligned-address error; keep those decode shapes on the
    # transparent reference path until a byte-safe GEMV variant exists.
    if spec.problem.m == 1 and spec.problem.k % 4 != 0:
        raise XQTBackendError(
            "SM89 W8A16 M=1 GEMV requires K divisible by four; use reference fallback"
        )
    if 1 < spec.problem.m < 32:
        raise XQTBackendError("SM89 W8A16 M=2..31 uses the reference small-M fallback")

    artifact_path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    quantized = quantize_int8_activation(
        activation,
        granularity="per_tensor" if spec.problem.m == 1 else "per_token",
        source="activation_dynamic",
    )
    internal_spec = _int8_spec(
        spec,
        activation_granularity=quantized.granularity,
    )
    # Prepack once per dispatch. Stateful callers should retain this packed
    # payload; the stateless dispatcher intentionally does not cache tensors.
    packed_int8 = prepack_sm89_int8_weight(
        canonical,
        scales=weight.scales,
        pack_version=weight.metadata.pack_version,
    )
    return sm89_w8a8_executor(
        quantized.values,
        packed_int8,
        spec=internal_spec,
        weight_scales=weight.scales,
        activation_scales=quantized.scales,
        bias=bias,
        artifact=artifact_path,
    )


def install_sm89_w8a16_executor(
    registry: GemmKernelRegistry,
    *,
    artifact: str | Path | None = None,
    manifest: str | Path | None = None,
) -> bool:
    """Promote W8A16 only after artifact and correctness gates pass."""

    path = Path(artifact) if artifact is not None else _DEFAULT_ARTIFACT
    if not sm89_w8a16_artifact_available(path):
        return False
    manifest_path = Path(manifest) if manifest is not None else artifact_manifest_path(path)
    if not artifact_ready_for_execution(
        manifest_path,
        kernel_name="sm89_w8a16_cutlass",
        target_arch="sm_89",
    ):
        return False
    entry = registry.get("sm89_w8a16_cutlass")
    registry.replace(
        replace(
            entry,
            maturity="executable",
            implementation="cutlass_sm89_w8a16_dynamic_int8",
            executor=lambda *args, **kwargs: sm89_w8a16_executor(
                *args,
                artifact=path,
                **kwargs,
            ),
        )
    )
    return True


__all__ = [
    "install_sm89_w8a16_executor",
    "sm89_w8a16_artifact_available",
    "sm89_w8a16_executor",
]
