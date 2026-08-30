"""Explicit FP8 storage and scale contracts for XQT GEMM.

The module keeps FP8 encoding separate from GEMM execution.  A canonical FP8
tensor is one byte per element in row-major order plus an explicit FP32 scale
tensor.  Native CUDA backends may consume the byte buffer directly, while the
reference path decodes the same bytes before accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_FP8_DTYPES: dict[str, torch.dtype] = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
_FP8_MAX_FINITE: dict[str, float] = {
    "fp8_e4m3": 448.0,
    "fp8_e5m2": 57344.0,
}
_FP8_EXPONENT_BITS: dict[str, int] = {"fp8_e4m3": 4, "fp8_e5m2": 5}
_FP8_MANTISSA_BITS: dict[str, int] = {"fp8_e4m3": 3, "fp8_e5m2": 2}
_FP8_STORAGE_LAYOUT = "xqt_fp8_rowmajor_v1"
FP8_BLOCK_K_VALUES = (32, 64, 128)


def validate_fp8_block_k(block_k: int) -> int:
    """Return ``block_k`` when it belongs to the legal FP8 blockwise set."""

    value = int(block_k)
    if value not in FP8_BLOCK_K_VALUES:
        raise ValueError(
            f"FP8 blockwise block_k must be one of {FP8_BLOCK_K_VALUES}, got {block_k!r}"
        )
    return value


def fp8_block_count(cols: int, block_k: int) -> int:
    """Return ceil(cols / block_k), the scale block axis of one blockwise row."""

    if int(cols) <= 0:
        raise ValueError(f"FP8 blockwise cols must be positive, got {cols!r}")
    return (int(cols) + validate_fp8_block_k(block_k) - 1) // validate_fp8_block_k(block_k)


@dataclass(frozen=True, slots=True)
class FP8FormatSpec:
    """Wire-level facts for one supported FP8 encoding."""

    name: str
    torch_dtype: torch.dtype
    max_finite: float
    exponent_bits: int
    mantissa_bits: int
    nan_encoding: str
    infinity_encoding: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "torch_dtype": str(self.torch_dtype),
            "max_finite": self.max_finite,
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
            "nan_encoding": self.nan_encoding,
            "infinity_encoding": self.infinity_encoding,
        }


_FORMAT_SPECS: dict[str, FP8FormatSpec] = {
    "fp8_e4m3": FP8FormatSpec(
        name="fp8_e4m3",
        torch_dtype=torch.float8_e4m3fn,
        max_finite=448.0,
        exponent_bits=4,
        mantissa_bits=3,
        nan_encoding="canonical_nan",
        infinity_encoding="not_representable",
    ),
    "fp8_e5m2": FP8FormatSpec(
        name="fp8_e5m2",
        torch_dtype=torch.float8_e5m2,
        max_finite=57344.0,
        exponent_bits=5,
        mantissa_bits=2,
        nan_encoding="ieee_nan",
        infinity_encoding="ieee_inf",
    ),
}


def fp8_format_spec(format_name: str) -> FP8FormatSpec:
    """Return the immutable format contract for ``fp8_e4m3`` or ``fp8_e5m2``."""

    try:
        return _FORMAT_SPECS[str(format_name)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported FP8 format {format_name!r}; expected fp8_e4m3 or fp8_e5m2"
        ) from exc


def _validate_role_granularity(role: str, granularity: str) -> None:
    if role == "weight" and granularity not in {"per_tensor", "per_channel", "blockwise"}:
        raise ValueError(
            "FP8 weight contract currently supports per_tensor, per_channel or blockwise scales"
        )
    if role == "activation" and granularity not in {"per_tensor", "per_token", "blockwise"}:
        raise ValueError(
            "FP8 activation contract currently supports per_tensor, per_token or blockwise scales"
        )
    if role not in {"weight", "activation"}:
        raise ValueError("FP8 scale role must be 'weight' or 'activation'")


def _canonical_scale(
    scale: torch.Tensor,
    *,
    rows: int,
    granularity: str,
    role: str,
    device: torch.device,
    name: str,
    cols: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """Normalize only explicitly supported scale shapes.

    ``blockwise`` scales are strictly ``[rows, ceil(cols/block_k)]``; partial
    trailing blocks keep their own scale slot, never a padded extra column.
    """

    _validate_role_granularity(role, granularity)
    value = torch.as_tensor(scale, dtype=torch.float32, device=device)
    if granularity == "blockwise":
        if cols is None or block_k is None:
            raise ValueError(f"{name} blockwise scale requires cols and block_k")
        blocks = fp8_block_count(int(cols), int(block_k))
        if tuple(value.shape) != (rows, blocks):
            raise ValueError(
                f"{name} blockwise scale must be [{rows},{blocks}] "
                f"(ceil(cols={cols}/block_k={block_k})), got {tuple(value.shape)}"
            )
        canonical = value
    elif granularity == "per_tensor":
        if value.ndim == 0 or tuple(value.shape) == (1, 1):
            canonical = value.reshape(1, 1)
        else:
            raise ValueError(f"{name} per_tensor scale must be scalar or [1,1], got {tuple(value.shape)}")
    else:
        if value.ndim == 1 and int(value.numel()) == rows:
            canonical = value.reshape(rows, 1)
        elif tuple(value.shape) == (rows, 1):
            canonical = value
        else:
            raise ValueError(
                f"{name} {granularity} scale must be [{rows},1] or [{rows}], "
                f"got {tuple(value.shape)}"
            )
    if not bool(torch.isfinite(canonical).all()) or bool((canonical <= 0).any()):
        raise ValueError(f"{name} scales must be finite and positive")
    return canonical.contiguous()


def _expanded_scale(
    scale: torch.Tensor,
    *,
    rows: int,
    cols: int,
    granularity: str,
    role: str,
    device: torch.device,
    name: str,
    block_k: int | None = None,
) -> torch.Tensor:
    canonical = _canonical_scale(
        scale,
        rows=rows,
        granularity=granularity,
        role=role,
        device=device,
        name=name,
        cols=cols if granularity == "blockwise" else None,
        block_k=block_k,
    )
    if granularity == "blockwise":
        if block_k is None:
            raise ValueError(f"{name} blockwise scale expansion requires block_k")
        return canonical.repeat_interleave(int(block_k), dim=1)[:, :cols]
    return canonical.expand(rows, cols)


def calibrate_fp8_scale(
    tensor: torch.Tensor,
    *,
    format_name: str,
    granularity: str,
    role: str,
    eps: float = 1e-8,
    block_k: int | None = None,
) -> torch.Tensor:
    """Compute a positive FP32 scale artifact from an FP32/FP16 tensor.

    ``blockwise`` calibration returns one scale per ``[row, K block]`` with
    shape ``[rows, ceil(cols/block_k)]``; a partial trailing block is
    calibrated on its own columns only.
    """

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError("FP8 calibration tensor must have shape [rows, cols]")
    if float(eps) <= 0.0:
        raise ValueError("FP8 calibration eps must be positive")
    format_spec = fp8_format_spec(format_name)
    rows, cols = (int(tensor.shape[0]), int(tensor.shape[1]))
    _validate_role_granularity(role, granularity)
    values = tensor.detach().to(torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("FP8 scale calibration requires finite input values")
    if granularity == "blockwise":
        if block_k is None:
            raise ValueError("FP8 blockwise calibration requires block_k")
        blocks = fp8_block_count(cols, int(block_k))
        padded = torch.nn.functional.pad(values, (0, blocks * int(block_k) - cols))
        amax = padded.view(rows, blocks, int(block_k)).abs().amax(dim=2)
    elif granularity == "per_tensor":
        amax = values.abs().amax().reshape(1, 1)
    else:
        amax = values.abs().amax(dim=1, keepdim=True)
    scale = amax.clamp_min(float(eps)) / format_spec.max_finite
    return _canonical_scale(
        scale,
        rows=rows,
        granularity=granularity,
        role=role,
        device=tensor.device,
        name="calibrated FP8",
        cols=cols if granularity == "blockwise" else None,
        block_k=block_k,
    )


def decode_fp8_storage(storage: torch.Tensor, *, format_name: str) -> torch.Tensor:
    """Decode canonical uint8 bytes or a matching torch FP8 tensor to FP32."""

    format_spec = fp8_format_spec(format_name)
    if not isinstance(storage, torch.Tensor):
        raise TypeError("FP8 storage must be a torch.Tensor")
    if storage.dtype == torch.uint8:
        return storage.contiguous().view(format_spec.torch_dtype).to(torch.float32)
    if storage.dtype == format_spec.torch_dtype:
        return storage.to(torch.float32)
    raise TypeError(
        f"FP8 storage for {format_name} must be uint8 or {format_spec.torch_dtype}, "
        f"got {storage.dtype}"
    )


@dataclass(frozen=True, slots=True)
class FP8QuantizedTensor:
    """Canonical FP8 byte payload plus its explicit scale artifact.

    ``block_k`` is set only for ``blockwise`` granularity and fixes the
    ``[rows, ceil(cols/block_k)]`` scale layout; other granularities keep
    ``block_k=None``.
    """

    storage: torch.Tensor
    scale: torch.Tensor
    format_name: str
    granularity: str
    role: str
    source: str
    logical_shape: tuple[int, int]
    saturation_count: int
    nan_count: int
    inf_count: int
    block_k: int | None = None

    def __post_init__(self) -> None:
        format_spec = fp8_format_spec(self.format_name)
        if self.storage.dtype != torch.uint8:
            raise TypeError("FP8QuantizedTensor.storage must be canonical uint8 bytes")
        if self.storage.ndim != 2:
            raise ValueError("FP8QuantizedTensor.storage must be rank-2")
        if tuple(int(item) for item in self.storage.shape) != tuple(self.logical_shape):
            raise ValueError("FP8QuantizedTensor.logical_shape must match storage shape")
        if self.granularity == "blockwise":
            if self.block_k is None:
                raise ValueError("blockwise FP8QuantizedTensor requires block_k")
            validate_fp8_block_k(self.block_k)
        elif self.block_k is not None:
            raise ValueError("block_k is only valid for blockwise FP8QuantizedTensor")
        _canonical_scale(
            self.scale,
            rows=int(self.logical_shape[0]),
            granularity=self.granularity,
            role=self.role,
            device=self.storage.device,
            name="FP8QuantizedTensor",
            cols=int(self.logical_shape[1]) if self.granularity == "blockwise" else None,
            block_k=self.block_k,
        )
        if self.source not in {
            "weight_offline",
            "weight_load_time",
            "activation_static",
            "activation_dynamic",
        }:
            raise ValueError(f"unsupported FP8 scale source: {self.source!r}")
        for name in ("saturation_count", "nan_count", "inf_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        del format_spec

    @property
    def dtype(self) -> torch.dtype:
        """Return the logical torch FP8 dtype represented by the bytes."""

        return fp8_format_spec(self.format_name).torch_dtype

    @property
    def saturation_ratio(self) -> float:
        return float(self.saturation_count) / float(self.storage.numel())

    def as_float8(self) -> torch.Tensor:
        """View the canonical bytes as a torch FP8 tensor without copying."""

        return self.storage.contiguous().view(self.dtype)

    def dequantize(self) -> torch.Tensor:
        return dequantize_fp8(
            self.storage,
            format_name=self.format_name,
            scale=self.scale,
            granularity=self.granularity,
            role=self.role,
            block_k=self.block_k,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format_name,
            "storage_dtype": "uint8",
            "storage_layout": _FP8_STORAGE_LAYOUT,
            "logical_shape": list(self.logical_shape),
            "scale_shape": list(self.scale.shape),
            "granularity": self.granularity,
            "block_k": self.block_k,
            "role": self.role,
            "source": self.source,
            "saturation_count": int(self.saturation_count),
            "saturation_ratio": self.saturation_ratio,
            "nan_count": int(self.nan_count),
            "inf_count": int(self.inf_count),
        }


def quantize_fp8(
    tensor: torch.Tensor,
    *,
    format_name: str,
    granularity: str,
    role: str,
    source: str,
    scale: torch.Tensor | None = None,
    nonfinite_policy: str = "saturate",
    eps: float = 1e-8,
    block_k: int | None = None,
) -> FP8QuantizedTensor:
    """Quantize a rank-2 tensor with explicit static or dynamic scale semantics.

    ``nonfinite_policy='saturate'`` maps NaN to zero and +/-Inf to the nearest
    finite endpoint before encoding.  ``'error'`` rejects any non-finite input.
    E4M3FN never receives an out-of-range value, avoiding backend-dependent NaN
    conversion for finite overflow.  ``blockwise`` quantization requires
    ``block_k`` and produces one scale per ``[row, K block]``.
    """

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError("FP8 quantization tensor must have shape [rows, cols]")
    if nonfinite_policy not in {"saturate", "error"}:
        raise ValueError("FP8 nonfinite_policy must be 'saturate' or 'error'")
    if float(eps) <= 0.0:
        raise ValueError("FP8 quantization eps must be positive")
    format_spec = fp8_format_spec(format_name)
    _validate_role_granularity(role, granularity)
    if granularity == "blockwise":
        if block_k is None:
            raise ValueError("FP8 blockwise quantization requires block_k")
        validate_fp8_block_k(block_k)
    elif block_k is not None:
        raise ValueError("block_k is only valid for blockwise FP8 quantization")
    expected_sources = (
        {"weight_offline", "weight_load_time"}
        if role == "weight"
        else {"activation_static", "activation_dynamic"}
    )
    if source not in expected_sources:
        raise ValueError(
            f"FP8 {role} source must be one of {sorted(expected_sources)}, got {source!r}"
        )
    rows, cols = (int(tensor.shape[0]), int(tensor.shape[1]))
    values = tensor.detach().to(torch.float32)
    nan_count = int(torch.isnan(values).sum().item())
    inf_count = int(torch.isinf(values).sum().item())
    if nonfinite_policy == "error" and (nan_count or inf_count):
        raise ValueError(
            f"FP8 quantization received non-finite values: nan={nan_count}, inf={inf_count}"
        )
    if source.endswith("dynamic"):
        sanitized = values.nan_to_num(
            0.0, posinf=format_spec.max_finite, neginf=format_spec.max_finite
        )
        if granularity == "blockwise":
            blocks = fp8_block_count(cols, int(block_k))
            padded = torch.nn.functional.pad(sanitized, (0, blocks * int(block_k) - cols))
            amax = padded.view(rows, blocks, int(block_k)).abs().amax(dim=2)
        elif granularity == "per_tensor":
            amax = sanitized.abs().amax().reshape(1, 1)
        else:
            amax = sanitized.abs().amax(dim=1, keepdim=True)
        canonical_scale = _canonical_scale(
            amax.clamp_min(float(eps)) / format_spec.max_finite,
            rows=rows,
            granularity=granularity,
            role=role,
            device=tensor.device,
            name="dynamic FP8",
            cols=cols if granularity == "blockwise" else None,
            block_k=block_k,
        )
    else:
        if scale is None:
            raise ValueError(f"{source} FP8 quantization requires an explicit scale artifact")
        canonical_scale = _canonical_scale(
            scale,
            rows=rows,
            granularity=granularity,
            role=role,
            device=tensor.device,
            name="static FP8",
            cols=cols if granularity == "blockwise" else None,
            block_k=block_k,
        )
    expanded = _expanded_scale(
        canonical_scale,
        rows=rows,
        cols=cols,
        granularity=granularity,
        role=role,
        device=tensor.device,
        name="FP8",
        block_k=block_k,
    )
    normalized = values / expanded
    finite = torch.isfinite(normalized)
    overflow = (finite & (normalized.abs() > format_spec.max_finite)) | torch.isinf(normalized)
    if nonfinite_policy == "saturate":
        normalized = torch.nan_to_num(
            normalized,
            nan=0.0,
            posinf=format_spec.max_finite,
            neginf=-format_spec.max_finite,
        )
    normalized = normalized.clamp(-format_spec.max_finite, format_spec.max_finite)
    encoded = normalized.to(format_spec.torch_dtype)
    storage = encoded.contiguous().view(torch.uint8)
    return FP8QuantizedTensor(
        storage=storage,
        scale=canonical_scale,
        format_name=format_name,
        granularity=granularity,
        role=role,
        source=source,
        logical_shape=(rows, cols),
        saturation_count=int(overflow.sum().item()),
        nan_count=nan_count,
        inf_count=inf_count,
        block_k=block_k,
    )


def dequantize_fp8(
    storage: torch.Tensor,
    *,
    format_name: str,
    scale: torch.Tensor,
    granularity: str,
    role: str,
    block_k: int | None = None,
) -> torch.Tensor:
    """Decode FP8 bytes and apply an explicit tensorwise/rowwise/blockwise scale."""

    if storage.ndim != 2:
        raise ValueError("FP8 storage must have shape [rows, cols]")
    rows, cols = (int(storage.shape[0]), int(storage.shape[1]))
    decoded = decode_fp8_storage(storage, format_name=format_name)
    expanded = _expanded_scale(
        scale,
        rows=rows,
        cols=cols,
        granularity=granularity,
        role=role,
        device=storage.device,
        name="FP8",
        block_k=block_k,
    )
    return decoded * expanded


__all__ = [
    "FP8FormatSpec",
    "FP8QuantizedTensor",
    "FP8_BLOCK_K_VALUES",
    "calibrate_fp8_scale",
    "decode_fp8_storage",
    "dequantize_fp8",
    "fp8_block_count",
    "fp8_format_spec",
    "quantize_fp8",
    "validate_fp8_block_k",
]
