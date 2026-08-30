"""P5 low-bit and structured-sparsity reference contracts.

These modules own the canonical value/scale/storage ABI for W3A16, W2A16,
channel-wise 2:4 sparse GEMM, and vector-codebook/TurboQuant GEMM.  They are
CPU-safe references: native promotion must come from a separate backend with
its own artifact, correctness, SASS, and benchmark evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Any

import torch
import torch.nn.functional as F

from .contracts import EpilogueSpec, GemmProblem, GemmSpec, PackedWeight, QuantSpec
from .layout import (
    build_packed_weight,
    pack_int2_signed,
    pack_int3_signed,
    validate_sparse2_4_mask,
)
from .reference import reference_gemm


LOWBIT_ACTIVATION_DTYPES: tuple[str, ...] = ("fp16", "bf16")
LOWBIT_OUTPUT_DTYPES: tuple[str, ...] = ("fp16", "bf16", "fp32")


def _positive_group_size(group_size: int, *, name: str) -> int:
    if isinstance(group_size, bool) or int(group_size) != group_size or int(group_size) <= 0:
        raise ValueError(f"{name} must be a positive int")
    return int(group_size)


def _padded_lowbit_k(k: int, group_size: int, pack_unit: int) -> int:
    padded = ((int(k) + int(group_size) - 1) // int(group_size)) * int(group_size)
    unit = lcm(int(group_size), int(pack_unit))
    return ((padded + unit - 1) // unit) * unit


def _canonical_lowbit_scales(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    n: int,
    groups: int,
) -> torch.Tensor:
    if value is None:
        return torch.ones((n, groups), dtype=torch.float32, device=device)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32, device=device)
    value = value.to(device=device, dtype=torch.float32)
    if value.ndim == 1 and int(value.numel()) == n:
        value = value.reshape(n, 1)
    if tuple(value.shape) == (n, 1):
        value = value.expand(n, groups)
    if tuple(value.shape) != (n, groups):
        raise ValueError(
            "low-bit scales must be [N], [N,1], or [N,groups], "
            f"got {tuple(value.shape)} for groups={groups}"
        )
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError("low-bit scales must be finite and positive")
    return value.contiguous()


@dataclass(frozen=True, slots=True)
class W3A16Contract:
    """Signed 3-bit weight plus FP16/BF16 activation contract."""

    activation_dtype: str = "fp16"
    weight_group_size: int = 16
    output_dtype: str = "fp16"
    storage_layout: str = "xqt_int3_nk_v1"
    pack_version: str = "xqt-w3a16-v1"

    def __post_init__(self) -> None:
        if self.activation_dtype not in LOWBIT_ACTIVATION_DTYPES:
            raise ValueError(f"W3A16 activation_dtype must be one of {LOWBIT_ACTIVATION_DTYPES}")
        _positive_group_size(self.weight_group_size, name="W3A16 weight_group_size")
        if self.output_dtype not in LOWBIT_OUTPUT_DTYPES:
            raise ValueError(f"W3A16 output_dtype must be one of {LOWBIT_OUTPUT_DTYPES}")
        if not self.storage_layout.strip() or not self.pack_version.strip():
            raise ValueError("W3A16 storage_layout and pack_version must be non-empty")

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype="int3",
            activation_dtype=self.activation_dtype,
            compute_dtype="fp32",
            accum_dtype="fp32",
            output_dtype=self.output_dtype,
            weight_granularity="groupwise",
            activation_granularity="per_tensor",
            group_size=int(self.weight_group_size),
            symmetric=True,
            weight_scale_source="weight_offline",
            storage_layout=self.storage_layout,
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_dtype": "int3",
            "activation_dtype": self.activation_dtype,
            "weight_granularity": "groupwise",
            "weight_group_size": int(self.weight_group_size),
            "output_dtype": self.output_dtype,
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
        }


@dataclass(frozen=True, slots=True)
class W2A16Contract:
    """Signed 2-bit weight plus FP16/BF16 activation contract."""

    activation_dtype: str = "fp16"
    weight_group_size: int = 32
    output_dtype: str = "fp16"
    storage_layout: str = "xqt_int2_nk_v1"
    pack_version: str = "xqt-w2a16-v1"

    def __post_init__(self) -> None:
        if self.activation_dtype not in LOWBIT_ACTIVATION_DTYPES:
            raise ValueError(f"W2A16 activation_dtype must be one of {LOWBIT_ACTIVATION_DTYPES}")
        _positive_group_size(self.weight_group_size, name="W2A16 weight_group_size")
        if self.output_dtype not in LOWBIT_OUTPUT_DTYPES:
            raise ValueError(f"W2A16 output_dtype must be one of {LOWBIT_OUTPUT_DTYPES}")
        if not self.storage_layout.strip() or not self.pack_version.strip():
            raise ValueError("W2A16 storage_layout and pack_version must be non-empty")

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype="int2",
            activation_dtype=self.activation_dtype,
            compute_dtype="fp32",
            accum_dtype="fp32",
            output_dtype=self.output_dtype,
            weight_granularity="groupwise",
            activation_granularity="per_tensor",
            group_size=int(self.weight_group_size),
            symmetric=True,
            weight_scale_source="weight_offline",
            storage_layout=self.storage_layout,
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_dtype": "int2",
            "activation_dtype": self.activation_dtype,
            "weight_granularity": "groupwise",
            "weight_group_size": int(self.weight_group_size),
            "output_dtype": self.output_dtype,
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
        }


def pack_w3a16_weight(
    weight: torch.Tensor,
    *,
    contract: W3A16Contract | None = None,
    scales: torch.Tensor | None = None,
) -> PackedWeight:
    """Pack signed 3-bit groupwise weights into canonical INT3 storage."""

    return _pack_lowbit_weight(weight, contract or W3A16Contract(), scales=scales)


def pack_w2a16_weight(
    weight: torch.Tensor,
    *,
    contract: W2A16Contract | None = None,
    scales: torch.Tensor | None = None,
) -> PackedWeight:
    """Pack signed 2-bit groupwise weights into canonical INT2 storage."""

    return _pack_lowbit_weight(weight, contract or W2A16Contract(), scales=scales)


def _pack_lowbit_weight(
    weight: torch.Tensor,
    contract: W3A16Contract | W2A16Contract,
    *,
    scales: torch.Tensor | None,
) -> PackedWeight:
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("low-bit weight must be rank-2 [N,K]")
    n, k = (int(weight.shape[0]), int(weight.shape[1]))
    spec = contract.quant_spec()
    group_size = int(contract.weight_group_size)
    pack_unit = 8 if spec.weight_dtype == "int3" else 4
    if spec.weight_dtype == "int3":
        min_code, max_code, value_range = -4, 3, 3
    else:
        min_code, max_code, value_range = -2, 1, 1
    padded_k = _padded_lowbit_k(k, group_size, pack_unit)
    groups = padded_k // group_size
    if weight.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        codes = weight.to(torch.int16)
        if bool(((codes < min_code) | (codes > max_code)).any()):
            raise ValueError(f"{spec.weight_dtype} integer codes must be in [{min_code}, {max_code}]")
        resolved_scales = _canonical_lowbit_scales(
            scales, device=weight.device, n=n, groups=groups
        )
    else:
        values = weight.to(dtype=torch.float32)
        padded = F.pad(values, (0, padded_k - k))
        grouped = padded.reshape(n, groups, group_size)
        derived = grouped.abs().amax(dim=-1).clamp_min(1e-8) / float(value_range)
        resolved_scales = _canonical_lowbit_scales(
            scales if scales is not None else derived,
            device=weight.device,
            n=n,
            groups=groups,
        )
        codes = torch.round(grouped / resolved_scales.unsqueeze(-1)).clamp(
            min_code, max_code
        ).reshape(n, padded_k).to(torch.int8)
    packed = (
        pack_int3_signed(codes.to(torch.int8))
        if spec.weight_dtype == "int3"
        else pack_int2_signed(codes.to(torch.int8))
    )
    return build_packed_weight(
        packed,
        logical_shape=(n, k),
        spec=spec,
        scales=resolved_scales,
        padded_k=padded_k,
        storage_layout=contract.storage_layout,
        pack_version=contract.pack_version,
    )


def reference_w3a16_gemm(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    contract: W3A16Contract,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the W3A16 dequantized reference with the declared contract."""

    spec = GemmSpec(
        problem=GemmProblem(
            m=int(activation.shape[0]),
            n=weight.metadata.logical_shape[0],
            k=int(activation.shape[1]),
        ),
        quant=contract.quant_spec(),
        epilogue=EpilogueSpec(
            output_dtype=contract.output_dtype,
            has_bias=bias is not None,
            has_residual=residual is not None,
        ),
    )
    return reference_gemm(activation, weight, spec=spec, bias=bias, residual=residual)


def reference_w2a16_gemm(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    contract: W2A16Contract,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the W2A16 dequantized reference with the declared contract."""

    spec = GemmSpec(
        problem=GemmProblem(
            m=int(activation.shape[0]),
            n=weight.metadata.logical_shape[0],
            k=int(activation.shape[1]),
        ),
        quant=contract.quant_spec(),
        epilogue=EpilogueSpec(
            output_dtype=contract.output_dtype,
            has_bias=bias is not None,
            has_residual=residual is not None,
        ),
    )
    return reference_gemm(activation, weight, spec=spec, bias=bias, residual=residual)


@dataclass(frozen=True, slots=True)
class Sparse2_4Contract:
    """Channel-wise 2:4 sparsity over dense FP16/BF16 or quantized INT8 weights."""

    weight_dtype: str = "int8"
    activation_dtype: str = "fp16"
    weight_group_size: int = 8
    output_dtype: str = "fp16"
    storage_layout: str = "xqt_sparse2_4_v1"
    pack_version: str = "xqt-sparse2-4-v1"

    def __post_init__(self) -> None:
        if self.weight_dtype not in {"fp16", "bf16", "int8"}:
            raise ValueError("Sparse2_4 weight_dtype must be fp16, bf16, or int8")
        if self.activation_dtype not in LOWBIT_ACTIVATION_DTYPES:
            raise ValueError("Sparse2_4 activation_dtype must be fp16 or bf16")
        _positive_group_size(self.weight_group_size, name="Sparse2_4 weight_group_size")
        if self.output_dtype not in LOWBIT_OUTPUT_DTYPES:
            raise ValueError("Sparse2_4 output_dtype must be fp16, bf16, or fp32")
        if not self.storage_layout.strip() or not self.pack_version.strip():
            raise ValueError("Sparse2_4 storage_layout and pack_version must be non-empty")

    def quant_spec(self) -> QuantSpec:
        if self.weight_dtype == "int8":
            return QuantSpec(
                weight_dtype="int8",
                activation_dtype=self.activation_dtype,
                compute_dtype="fp32",
                accum_dtype="fp32",
                output_dtype=self.output_dtype,
                weight_granularity="groupwise",
                activation_granularity="per_tensor",
                group_size=int(self.weight_group_size),
                symmetric=True,
                weight_scale_source="weight_offline",
                storage_layout=self.storage_layout,
                pack_version=self.pack_version,
            )
        return QuantSpec(
            weight_dtype=self.weight_dtype,
            activation_dtype=self.activation_dtype,
            compute_dtype="fp32",
            accum_dtype="fp32",
            output_dtype=self.output_dtype,
            weight_granularity="per_tensor",
            activation_granularity="per_tensor",
            storage_layout=self.storage_layout,
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "weight_group_size": int(self.weight_group_size),
            "output_dtype": self.output_dtype,
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
        }


def pack_sparse2_4_weight(
    weight: torch.Tensor,
    mask: torch.Tensor,
    *,
    contract: Sparse2_4Contract | None = None,
    scales: torch.Tensor | None = None,
) -> PackedWeight:
    """Materialize a 2:4 sparse dense or INT8 packed weight with a validated mask."""

    resolved = contract or Sparse2_4Contract()
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("sparse 2:4 weight must be rank-2 [N,K]")
    n, k = (int(weight.shape[0]), int(weight.shape[1]))
    validate_sparse2_4_mask(mask, logical_shape=(n, k))
    spec = resolved.quant_spec()
    if resolved.weight_dtype == "int8":
        values = weight.to(dtype=torch.float32)
        groups = k // int(resolved.weight_group_size)
        grouped = values.reshape(n, groups, int(resolved.weight_group_size))
        resolved_scales = _canonical_lowbit_scales(
            scales if scales is not None else grouped.abs().amax(dim=-1).clamp_min(1e-8) / 127.0,
            device=weight.device,
            n=n,
            groups=groups,
        )
        codes = torch.round(grouped / resolved_scales.unsqueeze(-1)).clamp(
            -127, 127
        ).reshape(n, k).to(torch.int8)
    else:
        expected = {"fp16": torch.float16, "bf16": torch.bfloat16}[resolved.weight_dtype]
        if weight.dtype != expected:
            raise TypeError(f"sparse 2:4 {resolved.weight_dtype} weight must use {expected}")
        codes = weight
        resolved_scales = None
    return build_packed_weight(
        codes,
        logical_shape=(n, k),
        spec=spec,
        scales=resolved_scales,
        padded_k=k,
        storage_layout=resolved.storage_layout,
        pack_version=resolved.pack_version,
        sparse_mask=mask,
    )


def reference_sparse2_4_gemm(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    contract: Sparse2_4Contract,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the 2:4 sparse reference: dense matmul with masked-out weights."""

    spec = GemmSpec(
        problem=GemmProblem(
            m=int(activation.shape[0]),
            n=weight.metadata.logical_shape[0],
            k=int(activation.shape[1]),
        ),
        quant=contract.quant_spec(),
        epilogue=EpilogueSpec(
            output_dtype=contract.output_dtype,
            has_bias=bias is not None,
            has_residual=residual is not None,
        ),
    )
    return reference_gemm(activation, weight, spec=spec, bias=bias, residual=residual)


@dataclass(frozen=True, slots=True)
class VectorCodebookContract:
    """TurboQuant-style codebook lookup weight contract."""

    num_vectors: int = 256
    vector_size: int = 4
    vectors_per_group: int = 8
    activation_dtype: str = "fp16"
    output_dtype: str = "fp16"
    storage_layout: str = "xqt_codebook_v1"
    pack_version: str = "xqt-codebook-v1"

    def __post_init__(self) -> None:
        _positive_group_size(self.num_vectors, name="codebook num_vectors")
        _positive_group_size(self.vector_size, name="codebook vector_size")
        _positive_group_size(self.vectors_per_group, name="codebook vectors_per_group")
        if self.activation_dtype not in LOWBIT_ACTIVATION_DTYPES:
            raise ValueError("codebook activation_dtype must be fp16 or bf16")
        if self.output_dtype not in LOWBIT_OUTPUT_DTYPES:
            raise ValueError("codebook output_dtype must be fp16, bf16, or fp32")
        if not self.storage_layout.strip() or not self.pack_version.strip():
            raise ValueError("codebook storage_layout and pack_version must be non-empty")

    @property
    def group_size(self) -> int:
        return int(self.vector_size) * int(self.vectors_per_group)

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype="codebook",
            activation_dtype=self.activation_dtype,
            compute_dtype="fp32",
            accum_dtype="fp32",
            output_dtype=self.output_dtype,
            weight_granularity="groupwise",
            activation_granularity="per_tensor",
            group_size=self.group_size,
            symmetric=True,
            weight_scale_source="weight_offline",
            storage_layout=self.storage_layout,
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_dtype": "codebook",
            "num_vectors": int(self.num_vectors),
            "vector_size": int(self.vector_size),
            "vectors_per_group": int(self.vectors_per_group),
            "group_size": self.group_size,
            "activation_dtype": self.activation_dtype,
            "output_dtype": self.output_dtype,
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
        }


def build_vector_codebook_weight(
    codebook: torch.Tensor,
    indices: torch.Tensor,
    scales: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    contract: VectorCodebookContract | None = None,
) -> PackedWeight:
    """Build a codebook PackedWeight from explicit codebook, indices, and scales."""

    resolved = contract or VectorCodebookContract(
        num_vectors=int(codebook.shape[0]),
        vector_size=int(codebook.shape[1]),
    )
    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if not isinstance(codebook, torch.Tensor) or codebook.ndim != 2:
        raise ValueError("codebook must be rank-2 [num_vectors, vector_size]")
    if tuple(codebook.shape) != (int(resolved.num_vectors), int(resolved.vector_size)):
        raise ValueError("codebook shape disagrees with VectorCodebookContract")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
        raise ValueError("codebook indices must be rank-2 [N, K/vector_size]")
    if k % int(resolved.vector_size) != 0:
        raise ValueError("codebook logical K must be divisible by vector_size")
    if tuple(indices.shape) != (n, k // int(resolved.vector_size)):
        raise ValueError(
            f"codebook indices must have shape {(n, k // int(resolved.vector_size))}, "
            f"got {tuple(indices.shape)}"
        )
    if indices.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("codebook indices must use an integer dtype")
    if int(resolved.num_vectors) > 256:
        raise ValueError("codebook uint8 index storage supports at most 256 vectors")
    indices_values = indices.to(dtype=torch.int64)
    if bool(((indices_values < 0) | (indices_values >= int(resolved.num_vectors))).any()):
        raise ValueError("codebook indices out of range")
    indices = indices.to(dtype=torch.uint8)
    groups = k // resolved.group_size
    resolved_scales = _canonical_lowbit_scales(
        scales, device=indices.device, n=n, groups=groups
    )
    return build_packed_weight(
        indices,
        logical_shape=(n, k),
        spec=resolved.quant_spec(),
        scales=resolved_scales,
        padded_k=k,
        storage_layout=resolved.storage_layout,
        pack_version=resolved.pack_version,
        codebook=codebook,
    )


def quantize_vector_codebook_reference(
    values: torch.Tensor,
    *,
    contract: VectorCodebookContract,
) -> PackedWeight:
    """Deterministic nearest-codebook reference quantization.

    The codebook is seeded from the first distinct K-blocks of the input, each
    block is assigned to its nearest codebook vector, and group scales keep the
    reconstructed weights bounded by the original per-group absmax.
    """

    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError("codebook quantization expects a rank-2 [N,K] tensor")
    n, k = (int(values.shape[0]), int(values.shape[1]))
    vector_size = int(contract.vector_size)
    if k % vector_size != 0:
        raise ValueError("codebook logical K must be divisible by vector_size")
    if int(contract.num_vectors) > 256:
        raise ValueError("codebook uint8 index storage supports at most 256 vectors")
    blocks = values.to(dtype=torch.float32).reshape(n, k // vector_size, vector_size)
    flat = blocks.reshape(-1, vector_size)
    rounded = torch.round(flat * 1024) / 1024
    unique = rounded.unique(dim=0)
    if unique.shape[0] < int(contract.num_vectors):
        repeats = (int(contract.num_vectors) + unique.shape[0] - 1) // unique.shape[0]
        unique = unique.repeat(repeats, 1)
    codebook = unique[: int(contract.num_vectors)].contiguous()
    distances = torch.cdist(flat, codebook)
    indices = distances.argmin(dim=1).to(torch.uint8).reshape(n, k // vector_size)
    reconstructed = codebook[indices.to(dtype=torch.long).reshape(-1)].reshape(
        n, k // vector_size, vector_size
    )
    groups = k // contract.group_size
    vectors_per_group = int(contract.vectors_per_group)
    grouped_original = blocks.reshape(n, groups, vectors_per_group, vector_size)
    grouped_recon = reconstructed.reshape(n, groups, vectors_per_group, vector_size)
    scales = (
        grouped_original.abs().amax(dim=(-2, -1))
        / grouped_recon.abs().amax(dim=(-2, -1)).clamp_min(1e-8)
    ).clamp_min(1e-8)
    return build_vector_codebook_weight(
        codebook,
        indices,
        scales,
        logical_shape=(n, k),
        contract=contract,
    )


def reference_vector_codebook_gemm(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    contract: VectorCodebookContract,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the codebook dequantized reference GEMM."""

    spec = GemmSpec(
        problem=GemmProblem(
            m=int(activation.shape[0]),
            n=weight.metadata.logical_shape[0],
            k=int(activation.shape[1]),
        ),
        quant=contract.quant_spec(),
        epilogue=EpilogueSpec(
            output_dtype=contract.output_dtype,
            has_bias=bias is not None,
            has_residual=residual is not None,
        ),
    )
    return reference_gemm(activation, weight, spec=spec, bias=bias, residual=residual)


__all__ = [
    "LOWBIT_ACTIVATION_DTYPES",
    "LOWBIT_OUTPUT_DTYPES",
    "Sparse2_4Contract",
    "VectorCodebookContract",
    "W2A16Contract",
    "W3A16Contract",
    "build_vector_codebook_weight",
    "pack_sparse2_4_weight",
    "pack_w2a16_weight",
    "pack_w3a16_weight",
    "quantize_vector_codebook_reference",
    "reference_sparse2_4_gemm",
    "reference_vector_codebook_gemm",
    "reference_w2a16_gemm",
    "reference_w3a16_gemm",
]
