"""SM89 AWQ W4A16 decode backend over canonical XQT packed weights."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from xqt.core.errors import XQTBackendError

from ..contracts import GemmSpec, PackedWeight, PackedWeightMetadata
from ..layout import validate_w4a16_packed_weight


def prepack_sm89_awq_w4a16_decode(weight: PackedWeight) -> PackedWeight:
    """Prepack asymmetric canonical W4 storage for the SM89 ``M<=8`` GEMV."""

    if not isinstance(weight, PackedWeight):
        raise XQTBackendError("SM89 AWQ decode prepack requires a PackedWeight")
    metadata = weight.metadata
    if (
        metadata.weight_dtype != "int4"
        or metadata.storage_layout != "xqt_int4_nk_v1"
        or metadata.nibble_signed
        or metadata.group_size != 64
        or weight.zero_points is None
    ):
        raise XQTBackendError(
            "SM89 AWQ decode prepack requires asymmetric canonical INT4, "
            "group_size=64 and explicit zero points"
        )
    n, k = (int(value) for value in metadata.logical_shape)
    if int(metadata.padded_k) != k:
        raise XQTBackendError("SM89 AWQ decode prepack does not accept logical K padding")
    if n % 8 != 0 or k % 64 != 0:
        raise XQTBackendError("SM89 AWQ decode prepack requires N % 8 == 0 and K % 64 == 0")
    qweight = weight.qweight
    if not isinstance(qweight, torch.Tensor):
        raise XQTBackendError("SM89 AWQ decode prepack requires a tensor qweight")
    if qweight.dtype != torch.uint8 or tuple(qweight.shape) != (n, k // 2):
        raise XQTBackendError("canonical AWQ qweight must be uint8 [N,K/2]")
    if not qweight.is_cuda:
        raise XQTBackendError("SM89 AWQ decode prepack requires CUDA-resident weights")

    from xqt.operator_opt.kernels.cuda.awq_w4a16_sm89 import (
        pack_awq_w4a16_interleaved,
    )

    packed = pack_awq_w4a16_interleaved(qweight.contiguous())
    execution_metadata = PackedWeightMetadata(
        logical_shape=(n, k),
        storage_layout="sm89_awq_w4a16_interleaved_v1",
        pack_version="xqt-sm89-awq-w4a16-v1",
        weight_dtype="int4",
        padded_k=k,
        group_size=64,
        packed_bits=4,
        nibble_order="low_high",
        nibble_signed=False,
    )
    return PackedWeight(
        qweight=packed,
        scales=weight.scales,
        zero_points=weight.zero_points,
        metadata=execution_metadata,
        canonical_qweight=qweight,
    )


def _validate_prepacked(
    weight: PackedWeight,
    *,
    spec: GemmSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(weight, PackedWeight):
        raise XQTBackendError("SM89 AWQ decode requires a PackedWeight")
    metadata = weight.metadata
    if metadata.storage_layout != "sm89_awq_w4a16_interleaved_v1":
        raise XQTBackendError(
            "SM89 AWQ decode requires prepack_sm89_awq_w4a16_decode()"
        )
    canonical = weight.canonical_qweight
    if not isinstance(canonical, torch.Tensor):
        raise XQTBackendError("prepacked AWQ storage must retain canonical_qweight")
    canonical_weight = replace(
        weight,
        qweight=canonical,
        metadata=replace(
            metadata,
            storage_layout="xqt_int4_nk_v1",
            pack_version="xqt-w4a16-awq-v1",
        ),
        canonical_qweight=None,
    )
    validate_w4a16_packed_weight(
        canonical_weight,
        spec=spec.quant,
        logical_shape=(spec.problem.n, spec.problem.k),
    )
    qweight = weight.qweight
    scales = weight.scales
    zero_points = weight.zero_points
    if not isinstance(qweight, torch.Tensor) or qweight.dtype != torch.int32:
        raise XQTBackendError("SM89 AWQ execution qweight must be int32")
    if tuple(qweight.shape) != (spec.problem.n // 4, spec.problem.k // 2):
        raise XQTBackendError("SM89 AWQ execution qweight shape mismatch")
    if not isinstance(scales, torch.Tensor) or not isinstance(zero_points, torch.Tensor):
        raise XQTBackendError("SM89 AWQ decode requires scale and zero-point tensors")
    return qweight, scales, zero_points


def prepare_sm89_awq_w4a16_decode_parameters(
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare immutable execution tensors outside the hot forward path."""

    if dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("SM89 AWQ prepared parameters require FP16 or BF16")
    qweight, scales, zero_points = _validate_prepacked(weight, spec=spec)
    scale_group_output = scales.to(dtype=dtype).transpose(0, 1).contiguous()
    scaled_zeros = (
        -zero_points.to(torch.float32).transpose(0, 1)
        * scale_group_output.to(torch.float32)
    ).to(dtype).contiguous()
    return qweight.contiguous(), scale_group_output, scaled_zeros


def sm89_awq_w4a16_decode_executor(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    weight_zero_points: torch.Tensor | None = None,
    activation_zero_points: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    prepared_parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Execute asymmetric W4A16 projection for ``1 <= M <= 8`` on SM89."""

    if spec.quant.activation_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("SM89 AWQ decode requires FP16 or BF16 activations")
    if spec.quant.output_dtype != spec.quant.activation_dtype:
        raise XQTBackendError("SM89 AWQ decode requires matching input/output dtype")
    if spec.quant.group_size != 64 or spec.quant.symmetric or not spec.quant.weight_zero_point:
        raise XQTBackendError("SM89 AWQ decode requires asymmetric group_size=64 weights")
    if not (1 <= spec.problem.m <= 8):
        raise XQTBackendError("SM89 AWQ decode requires 1 <= M <= 8")
    if spec.problem.n % 8 != 0 or spec.problem.k % 64 != 0:
        raise XQTBackendError("SM89 AWQ decode requires N % 8 == 0 and K % 64 == 0")
    if spec.epilogue.activation != "none" or residual is not None:
        raise XQTBackendError("SM89 AWQ decode supports only optional bias epilogue")
    if activation_scales is not None or activation_zero_points is not None:
        raise XQTBackendError("SM89 AWQ decode does not accept activation quantization")
    expected_dtype = (
        torch.float16 if spec.quant.activation_dtype == "fp16" else torch.bfloat16
    )
    if (
        activation.ndim != 2
        or tuple(activation.shape) != (spec.problem.m, spec.problem.k)
        or activation.dtype != expected_dtype
        or not activation.is_cuda
    ):
        raise XQTBackendError("SM89 AWQ activation shape, dtype or device mismatch")
    if torch.cuda.get_device_capability(activation.device) != (8, 9):
        major, minor = torch.cuda.get_device_capability(activation.device)
        raise XQTBackendError(f"SM89 AWQ decode received sm_{major}{minor}")

    qweight, scales, zero_points = _validate_prepacked(weight, spec=spec)
    if weight_scales is not None and not torch.equal(weight_scales, scales):
        raise XQTBackendError("external weight_scales disagree with PackedWeight.scales")
    if weight_zero_points is not None and not torch.equal(
        weight_zero_points, zero_points
    ):
        raise XQTBackendError(
            "external weight_zero_points disagree with PackedWeight.zero_points"
        )
    for name, tensor in (
        ("qweight", qweight),
        ("scales", scales),
        ("zero_points", zero_points),
    ):
        if not tensor.is_cuda or tensor.device != activation.device:
            raise XQTBackendError(f"SM89 AWQ {name} must share the activation device")
    if tuple(scales.shape) != (spec.problem.n, spec.problem.k // 64):
        raise XQTBackendError("SM89 AWQ scales must have canonical shape [N,K/64]")
    if tuple(zero_points.shape) != tuple(scales.shape):
        raise XQTBackendError("SM89 AWQ zero points must match scale shape")
    if prepared_parameters is None:
        qweight_exec, scale_group_output, scaled_zeros = (
            prepare_sm89_awq_w4a16_decode_parameters(
                weight,
                spec=spec,
                dtype=expected_dtype,
            )
        )
    else:
        qweight_exec, scale_group_output, scaled_zeros = prepared_parameters
        if qweight_exec.data_ptr() != qweight.data_ptr():
            raise XQTBackendError("prepared AWQ qweight disagrees with PackedWeight")
        expected_group_output = (spec.problem.k // 64, spec.problem.n)
        for name, tensor in (
            ("prepared scales", scale_group_output),
            ("prepared scaled_zeros", scaled_zeros),
        ):
            if (
                tuple(tensor.shape) != expected_group_output
                or tensor.dtype != expected_dtype
                or tensor.device != activation.device
                or not tensor.is_contiguous()
            ):
                raise XQTBackendError(f"{name} do not match the active AWQ contract")

    if bias is not None:
        if tuple(bias.shape) not in {(spec.problem.n,), (1, spec.problem.n)}:
            raise XQTBackendError("SM89 AWQ bias must have shape [N] or [1,N]")
        from xqt.operator_opt.kernels.cuda.awq_w4a16_sm89 import (
            awq_w4a16_decode_bias,
        )

        result = awq_w4a16_decode_bias(
            activation.contiguous(),
            qweight_exec,
            scale_group_output,
            scaled_zeros,
            bias.reshape(-1),
            output=output,
        )
    elif spec.epilogue.has_bias:
        raise XQTBackendError("SM89 AWQ epilogue declares bias but none was supplied")
    else:
        from xqt.operator_opt.kernels.cuda.awq_w4a16_sm89 import awq_w4a16_decode

        result = awq_w4a16_decode(
            activation.contiguous(),
            qweight_exec,
            scale_group_output,
            scaled_zeros,
            output=output,
        )
    return result


def sm89_awq_w4a16_metadata() -> dict[str, Any]:
    """Describe the executable target and its explicit scope."""

    return {
        "implementation": "native_sm89_awq_w4a16_interleaved_gemv",
        "target_arch": "sm_89",
        "rows": [1, 8],
        "group_size": 64,
        "weight_layout": "sm89_awq_w4a16_interleaved_v1",
        "inference_only": True,
    }


__all__ = [
    "prepare_sm89_awq_w4a16_decode_parameters",
    "prepack_sm89_awq_w4a16_decode",
    "sm89_awq_w4a16_decode_executor",
    "sm89_awq_w4a16_metadata",
]
