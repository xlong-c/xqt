"""Deterministic Torch reference implementations for every P0 GEMM contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .contracts import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    GroupedGemmProblem,
    PackedWeight,
    QuantSpec,
)
from .fp8 import decode_fp8_storage, dequantize_fp8, fp8_format_spec, quantize_fp8
from .layout import (
    unpack_int2,
    unpack_int3,
    unpack_int4,
    validate_logical_shapes,
    validate_sparse2_4_mask,
    validate_w4a16_packed_weight,
)
from .quantize import dequantize_int8_activation, quantize_int8_activation


_FP4_E2M1_CODEBOOK = torch.tensor(
    (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
    dtype=torch.float32,
)


_FLOAT_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _as_float32(value: torch.Tensor) -> torch.Tensor:
    """Convert tensor values to the stable accumulator dtype."""

    return value.to(dtype=torch.float32)


def _weight_scale_matrix(
    scale: torch.Tensor,
    *,
    n: int,
    k: int,
    granularity: str,
    group_size: int | None,
    padded_k: int,
    field_name: str,
) -> torch.Tensor:
    """Expand a canonical weight scale to ``[N, padded_K]``."""

    value = _as_float32(scale)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(1, 1).expand(n, padded_k)
    if value.ndim == 1 and value.numel() == n:
        return value.reshape(n, 1).expand(n, padded_k)
    if value.ndim == 2 and tuple(value.shape) == (n, 1):
        return value.expand(n, padded_k)
    if value.ndim == 2 and tuple(value.shape) == (n, padded_k):
        return value
    if value.ndim == 2 and granularity in {"groupwise", "blockwise"}:
        if group_size is None or int(group_size) <= 0:
            raise ValueError(f"{field_name} groupwise scale needs group_size")
        groups = (padded_k + int(group_size) - 1) // int(group_size)
        if tuple(value.shape) != (n, groups):
            raise ValueError(
                f"{field_name} expected shape {(n, groups)}, got {tuple(value.shape)}"
            )
        return value.repeat_interleave(int(group_size), dim=1)[:, :padded_k]
    raise ValueError(
        f"{field_name} shape {tuple(value.shape)} is incompatible with N={n}, K={k}, "
        f"granularity={granularity!r}"
    )


def _activation_scale_matrix(
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    granularity: str,
    field_name: str,
) -> torch.Tensor:
    """Expand activation scales to ``[M, K]``."""

    value = _as_float32(scale)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(1, 1).expand(m, k)
    if value.ndim == 1 and value.numel() == m:
        return value.reshape(m, 1).expand(m, k)
    if value.ndim == 2 and tuple(value.shape) == (m, 1):
        return value.expand(m, k)
    if value.ndim == 1 and value.numel() == k:
        return value.reshape(1, k).expand(m, k)
    if value.ndim == 2 and tuple(value.shape) == (1, k):
        return value.expand(m, k)
    if value.ndim == 2 and tuple(value.shape) == (m, k):
        return value
    raise ValueError(
        f"{field_name} shape {tuple(value.shape)} is incompatible with M={m}, K={k}, "
        f"granularity={granularity!r}"
    )


def _decode_weight_values(
    qweight: torch.Tensor,
    *,
    spec: QuantSpec,
    logical_k: int,
    padded_k: int,
    packed_bits: int | None,
    nibble_signed: bool,
) -> torch.Tensor:
    if spec.weight_dtype == "int4" and packed_bits == 4:
        values = unpack_int4(
            qweight.to(torch.uint8), logical_k=padded_k, signed=nibble_signed
        )
        return _as_float32(values)
    if spec.weight_dtype == "int4":
        if qweight.ndim != 2 or int(qweight.shape[1]) != padded_k:
            raise ValueError("unpacked INT4 reference weight must have padded logical K columns")
        return _as_float32(qweight)
    if spec.weight_dtype == "int2":
        if packed_bits != 2:
            raise ValueError("INT2 reference weight requires packed_bits=2 metadata")
        return _as_float32(unpack_int2(qweight.to(torch.uint8), logical_k=padded_k))
    if spec.weight_dtype == "int3":
        if packed_bits != 3:
            raise ValueError("INT3 reference weight requires packed_bits=3 metadata")
        return _as_float32(unpack_int3(qweight.to(torch.uint8), logical_k=padded_k))
    if spec.weight_dtype in {"fp4", "mxfp4", "nvfp4"}:
        if qweight.ndim != 2 or qweight.dtype != torch.uint8:
            raise ValueError("packed FP4 reference weight must be uint8 rank-2")
        expected_columns = (int(padded_k) + 1) // 2
        if int(qweight.shape[1]) != expected_columns:
            raise ValueError(
                "packed FP4 reference weight must have "
                f"{expected_columns} columns, got {qweight.shape[1]}"
            )
        low = qweight & 0x0F
        high = (qweight >> 4) & 0x0F
        codes = torch.stack((low, high), dim=-1).reshape(qweight.shape[0], -1)
        codebook = _FP4_E2M1_CODEBOOK.to(device=qweight.device)
        return codebook[codes.long()][:, :padded_k]
    if spec.weight_dtype in {"fp8_e4m3", "fp8_e5m2"}:
        if qweight.ndim != 2 or int(qweight.shape[1]) != padded_k:
            raise ValueError("FP8 reference weight must have padded logical K columns")
        return decode_fp8_storage(qweight, format_name=spec.weight_dtype)
    if qweight.ndim != 2 or int(qweight.shape[1]) != padded_k:
        raise ValueError("quantized reference weight must have padded logical K columns")
    return _as_float32(qweight)


def dequantize_weight_reference(
    weight: torch.Tensor | PackedWeight,
    *,
    spec: QuantSpec,
    scales: torch.Tensor | None = None,
    zero_points: torch.Tensor | None = None,
    logical_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Decode a canonical packed/int tensor into a logical ``[N,K]`` float tensor."""

    metadata = weight.metadata if isinstance(weight, PackedWeight) else None
    qweight = (
        weight.canonical_qweight
        if isinstance(weight, PackedWeight) and weight.canonical_qweight is not None
        else (weight.qweight if isinstance(weight, PackedWeight) else weight)
    )
    if not isinstance(qweight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor or PackedWeight")
    if scales is None and isinstance(weight, PackedWeight):
        scales = weight.scales
    if zero_points is None and isinstance(weight, PackedWeight):
        zero_points = weight.zero_points
    if logical_shape is None and metadata is not None:
        logical_shape = metadata.logical_shape
    if logical_shape is None:
        logical_shape = (int(qweight.shape[0]), int(qweight.shape[1]))
    n, logical_k = int(logical_shape[0]), int(logical_shape[1])
    if int(qweight.shape[0]) != n:
        raise ValueError(f"weight N mismatch: expected {n}, got {qweight.shape[0]}")
    packed_bits = None if metadata is None else metadata.packed_bits
    nibble_signed = spec.symmetric if metadata is None else metadata.nibble_signed
    padded_k = logical_k if metadata is None else int(metadata.padded_k)
    if (
        packed_bits is None
        and spec.weight_dtype == "int4"
        and qweight.dtype == torch.uint8
        and int(qweight.shape[1]) == (logical_k + 1) // 2
    ):
        packed_bits = 4
        padded_k = logical_k
    sparse_mask = weight.sparse_mask if isinstance(weight, PackedWeight) else None
    if spec.weight_dtype == "codebook":
        if not isinstance(weight, PackedWeight) or weight.codebook is None:
            raise ValueError("codebook reference requires a PackedWeight with a codebook")
        codebook = weight.codebook.to(device=qweight.device)
        vector_size = int(codebook.shape[1])
        if vector_size <= 0 or padded_k % vector_size != 0:
            raise ValueError("codebook vector_size must evenly divide padded_k")
        if int(qweight.shape[1]) != padded_k // vector_size:
            raise ValueError(
                "codebook indices must have padded_k // vector_size columns, "
                f"got {int(qweight.shape[1])}"
            )
        indices = qweight.to(dtype=torch.long)
        if bool((indices >= codebook.shape[0]).any()):
            raise ValueError("codebook indices out of range")
        values = codebook[indices].reshape(n, padded_k)
    else:
        values = _decode_weight_values(
            qweight,
            spec=spec,
            logical_k=logical_k,
            padded_k=padded_k,
            packed_bits=packed_bits,
            nibble_signed=nibble_signed,
        )
    if spec.weight_dtype in {"fp16", "bf16", "fp32"}:
        if tuple(qweight.shape) != (n, logical_k):
            raise ValueError("dense reference weight must have logical [N,K] shape")
        decoded = _as_float32(qweight)
        if sparse_mask is not None:
            validate_sparse2_4_mask(sparse_mask, logical_shape=(n, logical_k))
            decoded = decoded * sparse_mask.to(device=decoded.device, dtype=decoded.dtype)
        return decoded
    if scales is None:
        raise ValueError("quantized weight reference requires scales")
    if spec.weight_zero_point and zero_points is None:
        raise ValueError("QuantSpec.weight_zero_point=True requires weight_zero_points")
    expanded_scale = _weight_scale_matrix(
        scales,
        n=n,
        k=logical_k,
        granularity=spec.weight_granularity,
        group_size=spec.group_size,
        padded_k=padded_k,
        field_name="weight scales",
    )
    expanded_zero: torch.Tensor | None = None
    if zero_points is not None:
        expanded_zero = _weight_scale_matrix(
            zero_points,
            n=n,
            k=logical_k,
            granularity=spec.weight_granularity,
            group_size=spec.group_size,
            padded_k=padded_k,
            field_name="weight zero_points",
        )
    decoded = values
    if expanded_zero is not None:
        decoded = decoded - expanded_zero
    if spec.weight_dtype == "nvfp4":
        if not isinstance(weight, PackedWeight) or weight.global_scale is None:
            raise ValueError("NVFP4 PackedWeight requires global_scale")
        global_scale = weight.global_scale.to(device=decoded.device, dtype=torch.float32).reshape(())
        expanded_scale = expanded_scale / global_scale
    decoded = (decoded * expanded_scale)[:, :logical_k]
    if sparse_mask is not None:
        validate_sparse2_4_mask(sparse_mask, logical_shape=(n, logical_k))
        decoded = decoded * sparse_mask.to(device=decoded.device, dtype=decoded.dtype)
    return decoded


def _quantize_activation_reference(
    activation: torch.Tensor,
    *,
    spec: QuantSpec,
    scales: torch.Tensor | None,
    zero_points: torch.Tensor | None,
) -> torch.Tensor:
    """Quantize/dequantize activation in reference code to model runtime error."""

    m, k = int(activation.shape[0]), int(activation.shape[1])
    values = _as_float32(activation)
    if spec.activation_dtype == "int8":
        if spec.activation_zero_point and zero_points is None:
            raise ValueError(
                "QuantSpec.activation_zero_point=True requires activation_zero_points"
            )
        if activation.dtype in {torch.int8, torch.uint8}:
            if scales is None:
                if spec.activation_scale_source != "activation_dynamic":
                    raise ValueError("static INT8 activation requires activation_scales")
                scales = values.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            expanded_scale = _activation_scale_matrix(
                scales,
                m=m,
                k=k,
                granularity=spec.activation_granularity,
                field_name="activation scales",
            )
            if zero_points is None:
                return values * expanded_scale
            expanded_zero = _activation_scale_matrix(
                zero_points,
                m=m,
                k=k,
                granularity=spec.activation_granularity,
                field_name="activation zero_points",
            )
            return (values - expanded_zero) * expanded_scale
        result = quantize_int8_activation(
            activation,
            granularity=spec.activation_granularity,
            source=spec.activation_scale_source,
            scale=scales,
            zero_point=zero_points,
        )
        return dequantize_int8_activation(result)
    if spec.activation_dtype in {"fp8_e4m3", "fp8_e5m2"}:
        format_spec = fp8_format_spec(spec.activation_dtype)
        block_k = None
        if spec.activation_granularity == "blockwise":
            if spec.group_size is None:
                raise ValueError("blockwise FP8 activation requires QuantSpec.group_size as block_k")
            block_k = int(spec.group_size)
        if activation.dtype == format_spec.torch_dtype or activation.dtype == torch.uint8:
            if scales is None:
                raise ValueError("encoded FP8 activation requires an explicit scale artifact")
            return dequantize_fp8(
                activation,
                format_name=spec.activation_dtype,
                scale=scales,
                granularity=spec.activation_granularity,
                role="activation",
                block_k=block_k,
            )
        quantized = quantize_fp8(
            activation,
            format_name=spec.activation_dtype,
            granularity=spec.activation_granularity,
            role="activation",
            source=spec.activation_scale_source,
            scale=scales,
            block_k=block_k,
        )
        return quantized.dequantize()
    return values


def _output_dtype(name: str) -> torch.dtype:
    try:
        return _FLOAT_DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"reference output dtype must be fp32/fp16/bf16, got {name!r}") from exc


def apply_epilogue_reference(
    accumulator: torch.Tensor,
    *,
    epilogue: EpilogueSpec,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply bias, residual, activation and output cast in contract order."""

    output = accumulator
    if epilogue.has_bias and bias is None:
        raise ValueError("epilogue declares bias but bias tensor is missing")
    if epilogue.has_residual and residual is None:
        raise ValueError("epilogue declares residual but residual tensor is missing")
    if bias is not None:
        if bias.ndim not in {1, 2}:
            raise ValueError("bias must have shape [N] or [M,N]")
        output = output + _as_float32(bias)
    if residual is not None:
        if tuple(residual.shape) != tuple(output.shape):
            raise ValueError("residual must have the same shape as GEMM output")
        output = output + _as_float32(residual)
    if epilogue.activation == "relu":
        output = F.relu(output)
    elif epilogue.activation == "gelu":
        output = F.gelu(output)
    elif epilogue.activation == "silu":
        output = F.silu(output)
    return output.to(dtype=_output_dtype(epilogue.output_dtype))


def dense_gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    transpose_b: bool = True,
) -> torch.Tensor:
    """Run a dense a @ b or a @ b.T reference GEMM with a small epilogue.

    This is the xqt.gemm-owned facade for legacy dense GEMM kernel references
    that do not yet carry a full GemmSpec. New contract-aware code should
    prefer reference_gemm.
    """

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("dense_gemm_reference expects rank-2 GEMM inputs")
    rhs = b.transpose(0, 1) if transpose_b else b
    if int(a.shape[1]) != int(rhs.shape[0]):
        raise ValueError(
            f"inner dimensions must match: {int(a.shape[1])} vs {int(rhs.shape[0])}"
        )

    output = torch.matmul(a, rhs)
    if bias is not None:
        output = output + bias

    if activation is None or activation == "none":
        return output
    if activation == "relu":
        return F.relu(output)
    if activation == "gelu":
        return F.gelu(output)
    if activation == "silu":
        return F.silu(output)
    raise ValueError(f"unsupported activation: {activation}")


def _resolve_spec(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    spec: GemmSpec | QuantSpec,
    *,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> tuple[GemmProblem, QuantSpec, EpilogueSpec]:
    if isinstance(spec, GemmSpec):
        logical_weight = weight.qweight if isinstance(weight, PackedWeight) else weight
        if not isinstance(logical_weight, torch.Tensor):
            raise TypeError("weight must be tensor-like")
        problem = spec.problem
        if int(activation.shape[0]) != problem.m or int(activation.shape[1]) != problem.k:
            raise ValueError("activation shape does not match GemmSpec.problem")
        if isinstance(weight, PackedWeight):
            n = weight.metadata.logical_shape[0]
        else:
            n = problem.n
        if n != problem.n:
            raise ValueError("weight shape does not match GemmSpec.problem.n")
        return problem, spec.quant, spec.epilogue
    if not isinstance(spec, QuantSpec):
        raise TypeError("spec must be GemmSpec or QuantSpec")
    qweight = weight.qweight if isinstance(weight, PackedWeight) else weight
    if not isinstance(qweight, torch.Tensor) or qweight.ndim != 2:
        raise TypeError("weight must be rank-2 tensor or PackedWeight")
    problem = GemmProblem(m=int(activation.shape[0]), n=int(qweight.shape[0]), k=int(activation.shape[1]))
    epilogue = EpilogueSpec(
        output_dtype=spec.output_dtype,
        has_bias=bias is not None,
        has_residual=residual is not None,
    )
    return problem, spec, epilogue


def reference_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    spec: GemmSpec | QuantSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    weight_zero_points: torch.Tensor | None = None,
    activation_zero_points: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``A @ W.T`` with FP32 accumulation and a declared quantization contract."""

    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise ValueError("activation must be a rank-2 tensor [M,K]")
    problem, quant, epilogue = _resolve_spec(
        activation,
        weight,
        spec,
        bias=bias,
        residual=residual,
    )
    logical_weight = weight.qweight if isinstance(weight, PackedWeight) else weight
    if not isinstance(logical_weight, torch.Tensor):
        raise TypeError("weight must be a tensor or PackedWeight")
    if isinstance(weight, PackedWeight):
        logical_weight_shape = weight.metadata.logical_shape
    else:
        raw_shape = tuple(int(item) for item in logical_weight.shape)
        if (
            quant.weight_dtype == "int4"
            and logical_weight.dtype == torch.uint8
            and raw_shape[1] == (problem.k + 1) // 2
        ):
            logical_weight_shape = (problem.n, problem.k)
        else:
            logical_weight_shape = raw_shape
    validate_logical_shapes(
        activation,
        torch.empty(logical_weight_shape, device=activation.device),
        m=problem.m,
        n=problem.n,
        k=problem.k,
    )
    decoded_weight = dequantize_weight_reference(
        weight,
        spec=quant,
        scales=weight_scales,
        zero_points=weight_zero_points,
        logical_shape=logical_weight_shape,
    ).to(device=activation.device)
    decoded_activation = _quantize_activation_reference(
        activation,
        spec=quant,
        scales=activation_scales,
        zero_points=activation_zero_points,
    )
    accumulator = _as_float32(decoded_activation) @ _as_float32(decoded_weight).transpose(0, 1)
    return apply_epilogue_reference(
        accumulator,
        epilogue=epilogue,
        bias=bias,
        residual=residual,
    )


def reference_w4a16_gemm(
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
) -> torch.Tensor:
    """Reference contract for canonical packed W4A16 weights.

    The implementation intentionally shares the numerically stable reference
    path, but validates the packed ABI first. This makes it a useful oracle for
    future fused INT4 kernels and prevents a malformed external pack from
    being mistaken for a supported native layout.
    """

    validate_w4a16_packed_weight(
        weight,
        spec=spec.quant,
        logical_shape=(spec.problem.n, spec.problem.k),
    )
    if weight_scales is not None and not torch.equal(weight_scales, weight.scales):
        raise ValueError("external weight_scales disagree with PackedWeight.scales")
    if weight_zero_points is not None and not torch.equal(
        weight_zero_points, weight.zero_points
    ):
        raise ValueError("external weight_zero_points disagree with PackedWeight.zero_points")
    return reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=weight.scales,
        activation_scales=activation_scales,
        weight_zero_points=weight.zero_points,
        activation_zero_points=activation_zero_points,
        bias=bias,
        residual=residual,
    )


def reference_grouped_gemm(
    grouped_problem: GroupedGemmProblem,
    activations: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor | PackedWeight],
    *,
    quant_specs: Sequence[QuantSpec] | QuantSpec,
    weight_scales: Sequence[torch.Tensor | None] | None = None,
    activation_scales: Sequence[torch.Tensor | None] | None = None,
    weight_zero_points: Sequence[torch.Tensor | None] | None = None,
    activation_zero_points: Sequence[torch.Tensor | None] | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reference grouped GEMM; one output tensor is returned for each group."""

    count = grouped_problem.group_count
    if len(activations) != count or len(weights) != count:
        raise ValueError("activations and weights must match grouped_problem.group_count")
    if isinstance(quant_specs, QuantSpec):
        specs = (quant_specs,) * count
    else:
        if len(quant_specs) != count:
            raise ValueError("quant_specs must match grouped_problem.group_count")
        specs = tuple(quant_specs)
    if weight_scales is None:
        weight_scales = (None,) * count
    if activation_scales is None:
        activation_scales = (None,) * count
    if weight_zero_points is None:
        weight_zero_points = (None,) * count
    if activation_zero_points is None:
        activation_zero_points = (None,) * count
    if bias is None:
        bias = (None,) * count
    grouped_values = {
        "weight_scales": weight_scales,
        "activation_scales": activation_scales,
        "weight_zero_points": weight_zero_points,
        "activation_zero_points": activation_zero_points,
        "bias": bias,
    }
    for name, values in grouped_values.items():
        if len(values) != count:
            raise ValueError(
                f"{name} must match grouped_problem.group_count"
            )
    outputs: list[torch.Tensor] = []
    for index, problem in enumerate(grouped_problem.problems):
        outputs.append(
            reference_gemm(
                activations[index],
                weights[index],
                spec=GemmSpec(
                    problem=problem,
                    quant=specs[index],
                    epilogue=EpilogueSpec(
                        has_bias=bias[index] is not None,
                        output_dtype=specs[index].output_dtype,
                    ),
                ),
                weight_scales=weight_scales[index],
                activation_scales=activation_scales[index],
                weight_zero_points=weight_zero_points[index],
                activation_zero_points=activation_zero_points[index],
                bias=bias[index],
            )
        )
    return tuple(outputs)


def _split_grouped_activation_artifact(
    value: torch.Tensor | Sequence[torch.Tensor | None] | None,
    *,
    grouped_problem: GroupedGemmProblem,
    quant_specs: tuple[QuantSpec, ...],
    field_name: str,
) -> tuple[torch.Tensor | None, ...]:
    """Split one packed activation artifact into explicit per-group values."""

    count = grouped_problem.group_count
    if value is None:
        return (None,) * count
    if not isinstance(value, torch.Tensor):
        values = tuple(value)
        if len(values) != count:
            raise ValueError(
                f"{field_name} must match grouped_problem.group_count"
            )
        return values

    granularities = {
        quant.activation_granularity for quant in quant_specs
    }
    if len(granularities) != 1:
        raise ValueError(
            f"packed {field_name} requires one shared activation granularity"
        )
    granularity = granularities.pop()
    if granularity == "per_tensor":
        if int(value.numel()) == 1:
            return tuple(value.reshape(()) for _ in range(count))
        if value.ndim in {1, 2} and int(value.numel()) == count:
            return tuple(value.reshape(count)[index] for index in range(count))
        raise ValueError(
            f"packed per_tensor {field_name} must be scalar or [group_count]"
        )
    if granularity in {"per_token", "blockwise"}:
        if value.ndim == 0 or int(value.shape[0]) != grouped_problem.total_m:
            raise ValueError(
                f"packed {granularity} {field_name} must lead with total_M"
            )
        offsets = _grouped_offsets(grouped_problem)
        return tuple(
            value[offsets[index] : offsets[index + 1]]
            for index in range(count)
        )
    if granularity in {"none", "per_channel"}:
        return tuple(value for _ in range(count))
    raise ValueError(
        f"packed {field_name} does not support activation granularity "
        f"{granularity!r}; pass an explicit per-group sequence"
    )


def _grouped_offsets(
    grouped_problem: GroupedGemmProblem,
) -> tuple[int, ...]:
    """Return explicit cumulative offsets for a grouped problem."""

    if grouped_problem.m_offsets is not None:
        return grouped_problem.m_offsets
    offsets = [0]
    for problem in grouped_problem.problems:
        offsets.append(offsets[-1] + problem.m)
    return tuple(offsets)


def reference_packed_grouped_gemm(
    grouped_problem: GroupedGemmProblem,
    activation: torch.Tensor,
    weights: Sequence[torch.Tensor | PackedWeight],
    *,
    quant_specs: Sequence[QuantSpec] | QuantSpec,
    weight_scales: Sequence[torch.Tensor | None] | None = None,
    activation_scales: torch.Tensor
    | Sequence[torch.Tensor | None]
    | None = None,
    weight_zero_points: Sequence[torch.Tensor | None] | None = None,
    activation_zero_points: torch.Tensor
    | Sequence[torch.Tensor | None]
    | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Run the explicit reference fallback for one packed routed-token input."""

    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise ValueError("grouped activation must be a rank-2 tensor [total_M,K]")
    if tuple(activation.shape) != (
        grouped_problem.total_m,
        grouped_problem.k,
    ):
        raise ValueError(
            "grouped activation shape must match "
            f"[{grouped_problem.total_m},{grouped_problem.k}]"
        )
    count = grouped_problem.group_count
    if len(weights) != count:
        raise ValueError("weights must match grouped_problem.group_count")
    if isinstance(quant_specs, QuantSpec):
        specs = (quant_specs,) * count
    else:
        if len(quant_specs) != count:
            raise ValueError(
                "quant_specs must match grouped_problem.group_count"
            )
        specs = tuple(quant_specs)
    offsets = _grouped_offsets(grouped_problem)
    activations = tuple(
        activation[offsets[index] : offsets[index + 1]]
        for index in range(count)
    )
    grouped_activation_scales = _split_grouped_activation_artifact(
        activation_scales,
        grouped_problem=grouped_problem,
        quant_specs=specs,
        field_name="activation_scales",
    )
    grouped_activation_zero_points = _split_grouped_activation_artifact(
        activation_zero_points,
        grouped_problem=grouped_problem,
        quant_specs=specs,
        field_name="activation_zero_points",
    )
    outputs = reference_grouped_gemm(
        grouped_problem,
        activations,
        weights,
        quant_specs=specs,
        weight_scales=weight_scales,
        activation_scales=grouped_activation_scales,
        weight_zero_points=weight_zero_points,
        activation_zero_points=grouped_activation_zero_points,
        bias=bias,
    )
    packed_output = torch.cat(outputs, dim=0)
    if grouped_problem.output_rows is None:
        return packed_output
    output = torch.empty_like(packed_output)
    destination = torch.tensor(
        grouped_problem.output_rows,
        dtype=torch.int64,
        device=packed_output.device,
    )
    output.index_copy_(0, destination, packed_output)
    return output


def reference_w8a16_gemm(
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
) -> torch.Tensor:
    """Reference contract for canonical packed W8A16 weights (per-channel/groupwise)."""
    validate_logical_shapes(
        activation,
        torch.empty((weight.metadata.logical_shape[0], weight.metadata.logical_shape[1]), device=activation.device),
        m=spec.problem.m,
        n=spec.problem.n,
        k=spec.problem.k,
    )
    if weight_scales is not None and not torch.equal(weight_scales, weight.scales):
        raise ValueError("external weight_scales disagree with PackedWeight.scales")
    if weight_zero_points is not None and not torch.equal(weight_zero_points, weight.zero_points):
        raise ValueError("external weight_zero_points disagree with PackedWeight.zero_points")
    return reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=weight.scales,
        activation_scales=activation_scales,
        weight_zero_points=weight.zero_points,
        activation_zero_points=activation_zero_points,
        bias=bias,
        residual=residual,
    )


__all__ = [
    "apply_epilogue_reference",
    "dense_gemm_reference",
    "dequantize_weight_reference",
    "reference_dense_gemm",
    "reference_gemm",
    "reference_grouped_gemm",
    "reference_packed_grouped_gemm",
    "reference_w4a16_gemm",
    "reference_w8a16_gemm",
]
