"""Canonical logical layout and small, reversible packing helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .contracts import PackedWeight, PackedWeightMetadata, QuantSpec


def validate_logical_shapes(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
) -> tuple[int, int, int]:
    """Validate ``A[M,K]`` and logical ``W[N,K]`` without transposing silently."""

    if not isinstance(activation, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("activation and weight must be torch.Tensor")
    if activation.ndim != 2:
        raise ValueError(f"activation must have shape [M,K], got {tuple(activation.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    inferred_m, inferred_k = (int(activation.shape[0]), int(activation.shape[1]))
    inferred_n, weight_k = (int(weight.shape[0]), int(weight.shape[1]))
    if inferred_k != weight_k:
        raise ValueError(
            f"logical K mismatch: activation has {inferred_k}, weight has {weight_k}"
        )
    expected = (m, n, k)
    actual = (inferred_m, inferred_n, inferred_k)
    for field_name, requested, observed in zip(("m", "n", "k"), expected, actual):
        if requested is not None and int(requested) != observed:
            raise ValueError(f"{field_name} mismatch: expected {requested}, got {observed}")
    return actual


def expected_weight_scale_shape(
    logical_shape: tuple[int, int],
    *,
    granularity: str,
    group_size: int | None = None,
) -> tuple[int, ...]:
    """Return canonical scale shape for a ``[N,K]`` weight."""

    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if granularity == "per_tensor":
        return (1, 1)
    if granularity == "per_channel":
        return (n, 1)
    if granularity in {"groupwise", "blockwise"}:
        if group_size is None or int(group_size) <= 0:
            raise ValueError("groupwise/blockwise scales require group_size > 0")
        return (n, (k + int(group_size) - 1) // int(group_size))
    raise ValueError(f"unsupported weight scale granularity: {granularity!r}")


def expected_activation_scale_shape(
    activation_shape: tuple[int, int], *, granularity: str
) -> tuple[int, ...]:
    """Return canonical scale shape for an ``A[M,K]`` activation."""

    m, k = (int(activation_shape[0]), int(activation_shape[1]))
    if granularity == "per_tensor":
        return (1, 1)
    if granularity == "per_token":
        return (m, 1)
    if granularity == "per_channel":
        return (1, k)
    if granularity in {"groupwise", "blockwise"}:
        raise ValueError("activation groupwise/blockwise shape needs an explicit group_size")
    raise ValueError(f"unsupported activation scale granularity: {granularity!r}")


def pack_int4_signed(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values using the canonical low-nibble-first order."""

    if values.ndim != 2:
        raise ValueError("INT4 values must be rank-2 [N,K]")
    if values.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("INT4 values must use an integer dtype")
    values_i = values.to(torch.int16)
    if bool(((values_i < -8) | (values_i > 7)).any()):
        raise ValueError("signed INT4 values must be in [-8, 7]")
    if values_i.shape[1] % 2:
        values_i = F.pad(values_i, (0, 1), value=0)
    encoded = torch.where(values_i < 0, values_i + 16, values_i).to(torch.uint8)
    return (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).contiguous()


def pack_int4_unsigned(values: torch.Tensor) -> torch.Tensor:
    """Pack unsigned INT4 codes using the canonical low-nibble-first order."""

    if values.ndim != 2:
        raise ValueError("INT4 values must be rank-2 [N,K]")
    if values.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("INT4 values must use an integer dtype")
    values_i = values.to(torch.int16)
    if bool(((values_i < 0) | (values_i > 15)).any()):
        raise ValueError("unsigned INT4 values must be in [0, 15]")
    if values_i.shape[1] % 2:
        values_i = F.pad(values_i, (0, 1), value=0)
    encoded = values_i.to(torch.uint8)
    return (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).contiguous()


def pack_int2_signed(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT2 values in ``[-2, 1]`` with four low-bit-first codes per byte."""

    if values.ndim != 2:
        raise ValueError("INT2 values must be rank-2 [N,K]")
    if values.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("INT2 values must use an integer dtype")
    values_i = values.to(torch.int16)
    if bool(((values_i < -2) | (values_i > 1)).any()):
        raise ValueError("signed INT2 values must be in [-2, 1]")
    if values_i.shape[1] % 4:
        values_i = F.pad(values_i, (0, 4 - values_i.shape[1] % 4), value=0)
    codes = (values_i + 2).to(torch.uint8)
    groups = codes.reshape(codes.shape[0], -1, 4)
    packed = (
        groups[:, :, 0]
        | (groups[:, :, 1] << 2)
        | (groups[:, :, 2] << 4)
        | (groups[:, :, 3] << 6)
    )
    return packed.contiguous()


def unpack_int2(packed: torch.Tensor, *, logical_k: int) -> torch.Tensor:
    """Decode canonical INT2 storage back to signed INT8 ``[N, logical_k]`` codes."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise TypeError("INT2 storage must be uint8 rank-2")
    if int(logical_k) <= 0 or int(packed.shape[1]) != (int(logical_k) + 3) // 4:
        raise ValueError("INT2 storage columns must match ceil(logical_k / 4)")
    codes = torch.stack(
        (
            packed & 0x03,
            (packed >> 2) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 6) & 0x03,
        ),
        dim=-1,
    ).reshape(packed.shape[0], -1)[:, : int(logical_k)]
    signed = (codes.to(torch.int16) - 2).to(torch.int8)
    return signed.contiguous()


def pack_int3_signed(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT3 values in ``[-4, 3]`` with eight codes per three bytes."""

    if values.ndim != 2:
        raise ValueError("INT3 values must be rank-2 [N,K]")
    if values.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("INT3 values must use an integer dtype")
    values_i = values.to(torch.int16)
    if bool(((values_i < -4) | (values_i > 3)).any()):
        raise ValueError("signed INT3 values must be in [-4, 3]")
    if values_i.shape[1] % 8:
        values_i = F.pad(values_i, (0, 8 - values_i.shape[1] % 8), value=0)
    codes = (values_i + 4).to(torch.uint8)
    groups = codes.reshape(codes.shape[0], -1, 8)
    byte0 = (
        groups[:, :, 0]
        | (groups[:, :, 1] << 3)
        | ((groups[:, :, 2] & 0x03) << 6)
    )
    byte1 = (
        ((groups[:, :, 2] >> 2) & 0x01)
        | (groups[:, :, 3] << 1)
        | (groups[:, :, 4] << 4)
        | ((groups[:, :, 5] & 0x01) << 7)
    )
    byte2 = (
        ((groups[:, :, 5] >> 1) & 0x03)
        | (groups[:, :, 6] << 2)
        | (groups[:, :, 7] << 5)
    )
    packed = torch.stack((byte0, byte1, byte2), dim=-1).reshape(codes.shape[0], -1)
    return packed.contiguous()


def unpack_int3(packed: torch.Tensor, *, logical_k: int) -> torch.Tensor:
    """Decode canonical INT3 storage back to signed INT8 ``[N, logical_k]`` codes."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise TypeError("INT3 storage must be uint8 rank-2")
    groups_per_row = (int(logical_k) + 7) // 8
    if int(logical_k) <= 0 or int(packed.shape[1]) != groups_per_row * 3:
        raise ValueError("INT3 storage columns must match ceil(logical_k / 8) * 3")
    groups = packed.reshape(packed.shape[0], groups_per_row, 3)
    byte0, byte1, byte2 = groups[:, :, 0], groups[:, :, 1], groups[:, :, 2]
    codes = torch.stack(
        (
            byte0 & 0x07,
            (byte0 >> 3) & 0x07,
            ((byte0 >> 6) & 0x03) | ((byte1 & 0x01) << 2),
            (byte1 >> 1) & 0x07,
            (byte1 >> 4) & 0x07,
            ((byte1 >> 7) & 0x01) | ((byte2 & 0x03) << 1),
            (byte2 >> 2) & 0x07,
            (byte2 >> 5) & 0x07,
        ),
        dim=-1,
    ).reshape(packed.shape[0], -1)[:, : int(logical_k)]
    signed = (codes.to(torch.int16) - 4).to(torch.int8)
    return signed.contiguous()


def _unpack_int32_codes(
    packed: torch.Tensor,
    *,
    reverse_order: bool,
) -> torch.Tensor:
    """Decode eight 4-bit lanes from each GPTQ/AWQ int32 word."""

    if packed.ndim != 2 or packed.dtype not in {
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("int32 packed weights must be a rank-2 integer tensor")
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int64)
    values = packed.to(torch.int64)
    unpacked = ((values.unsqueeze(-1) >> shifts) & 0x0F).to(torch.uint8)
    if reverse_order:
        order = torch.tensor(
            (0, 4, 1, 5, 2, 6, 3, 7),
            device=packed.device,
            dtype=torch.long,
        )
        unpacked = unpacked[..., order]
    return unpacked


def _source_int4_codes(
    qweight: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    method: str,
) -> torch.Tensor:
    """Decode GPTQ/AWQ source storage into unsigned codes ``[N,K]``."""

    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if qweight.ndim != 2:
        raise ValueError("source INT4 qweight must be rank-2")
    if qweight.dtype == torch.uint8 and qweight.shape[0] == n:
        if int(qweight.shape[1]) < (k + 1) // 2:
            raise ValueError("canonical INT4 qweight has too few packed K columns")
        return unpack_int4(qweight, logical_k=k, signed=False).to(torch.uint8)
    if qweight.dtype not in {torch.int16, torch.int32, torch.int64}:
        raise TypeError("GPTQ/AWQ source qweight must be an int32-like tensor")
    if method == "gptq":
        if tuple(qweight.shape) == ((k + 7) // 8, n):
            unpacked = _unpack_int32_codes(qweight, reverse_order=False)
            return unpacked.permute(1, 0, 2).reshape(n, -1)[:, :k].contiguous()
        if tuple(qweight.shape) == (k, (n + 7) // 8):
            unpacked = _unpack_int32_codes(qweight, reverse_order=False)
            return unpacked.reshape(k, -1)[:, :n].transpose(0, 1).contiguous()
    elif method == "awq":
        if tuple(qweight.shape) == (k, (n + 7) // 8):
            unpacked = _unpack_int32_codes(qweight, reverse_order=True)
            return unpacked.reshape(k, -1)[:, :n].transpose(0, 1).contiguous()
        if tuple(qweight.shape) == ((k + 7) // 8, n):
            unpacked = _unpack_int32_codes(qweight, reverse_order=True)
            return unpacked.permute(1, 0, 2).reshape(n, -1)[:, :k].contiguous()
    raise ValueError(
        f"unsupported {method} INT4 qweight shape {tuple(qweight.shape)} "
        f"for logical [N,K]=[{n},{k}]"
    )


def _canonical_group_parameter(
    value: torch.Tensor,
    *,
    n: int,
    groups: int,
    name: str,
) -> torch.Tensor:
    """Normalize group scales/zero points to float32 ``[N,G]``."""

    tensor = value.detach().to(torch.float32)
    if tensor.ndim == 3 and tuple(tensor.shape[-1:]) == (1,):
        tensor = tensor[..., 0]
    if tensor.ndim == 1:
        if tensor.numel() == n:
            tensor = tensor.reshape(n, 1).expand(n, groups)
        elif tensor.numel() == groups:
            tensor = tensor.reshape(1, groups).expand(n, groups)
        else:
            raise ValueError(f"{name} shape {tuple(value.shape)} is not [N,G] or [G,N]")
    elif tensor.ndim == 2:
        if tuple(tensor.shape) == (n, groups):
            pass
        elif tuple(tensor.shape) == (groups, n):
            tensor = tensor.transpose(0, 1)
        elif tuple(tensor.shape) == (n, 1):
            tensor = tensor.expand(n, groups)
        elif tuple(tensor.shape) == (1, groups):
            tensor = tensor.expand(n, groups)
        else:
            raise ValueError(f"{name} shape {tuple(value.shape)} is not [N,G] or [G,N]")
    else:
        raise ValueError(f"{name} shape {tuple(value.shape)} is not [N,G] or [G,N]")
    return tensor.contiguous()


def _canonical_group_zero_points(
    value: torch.Tensor | None,
    *,
    n: int,
    groups: int,
    method: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.dtype in {torch.int16, torch.int32, torch.int64}:
        if method == "gptq" and tuple(value.shape) == (groups, (n + 7) // 8):
            unpacked = _unpack_int32_codes(value, reverse_order=False)
            return unpacked.reshape(groups, -1)[:, :n].transpose(0, 1).to(torch.float32).contiguous()
        if method == "awq" and tuple(value.shape) == (groups, (n + 7) // 8):
            unpacked = _unpack_int32_codes(value, reverse_order=True)
            return unpacked.reshape(groups, -1)[:, :n].transpose(0, 1).to(torch.float32).contiguous()
    return _canonical_group_parameter(value, n=n, groups=groups, name="zero_points")


def _validate_sequential_g_idx(
    g_idx: torch.Tensor | None,
    *,
    logical_k: int,
    group_size: int,
) -> None:
    if g_idx is None:
        return
    if g_idx.ndim != 1 or int(g_idx.numel()) != logical_k:
        raise ValueError("g_idx must be a rank-1 tensor with K entries")
    expected = torch.arange(logical_k, device=g_idx.device, dtype=torch.long) // group_size
    observed = g_idx.to(device=g_idx.device, dtype=torch.long)
    if not torch.equal(observed, expected):
        raise ValueError(
            "non-sequential g_idx/act-order is not supported by canonical repack; "
            "materialize the input permutation before packing"
        )


def _build_canonical_w4(
    codes: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    group_size: int,
    signed: bool,
    symmetric: bool,
    pack_version: str,
    storage_layout: str,
) -> PackedWeight:
    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if group_size not in {32, 64, 128}:
        raise ValueError("W4A16 canonical repack supports group_size 32, 64, or 128")
    if tuple(codes.shape) != (n, k):
        raise ValueError(f"decoded INT4 codes must have shape {(n, k)}, got {tuple(codes.shape)}")
    padded_k = ((k + group_size - 1) // group_size) * group_size
    if padded_k != k:
        codes = F.pad(codes, (0, padded_k - k), value=0)
    groups = padded_k // group_size
    canonical_scales = _canonical_group_parameter(scales, n=n, groups=groups, name="scales")
    canonical_zero_points = _canonical_group_zero_points(
        zero_points, n=n, groups=groups, method="gptq" if signed else "awq"
    )
    if signed:
        signed_codes = codes.to(torch.int16) - 8
        packed = pack_int4_signed(signed_codes.to(torch.int8))
    else:
        packed = pack_int4_unsigned(codes.to(torch.uint8))
    spec = QuantSpec(
        weight_dtype="int4",
        activation_dtype="fp16",
        output_dtype="fp16",
        weight_granularity="groupwise",
        group_size=group_size,
        symmetric=symmetric,
        weight_zero_point=canonical_zero_points is not None,
        weight_scale_source="weight_offline",
    )
    return build_packed_weight(
        packed,
        logical_shape=(n, k),
        spec=spec,
        scales=canonical_scales,
        zero_points=canonical_zero_points,
        padded_k=padded_k,
        storage_layout=storage_layout,
        pack_version=pack_version,
    )


def _build_canonical_w8(
    qweight: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    scales: torch.Tensor,
    zero_points: torch.Tensor | None,
    group_size: int,
    symmetric: bool,
    pack_version: str,
    storage_layout: str,
) -> PackedWeight:
    """Build canonical signed INT8 storage for per-channel/groupwise scales."""

    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if qweight.ndim != 2 or tuple(qweight.shape) != (n, k):
        raise ValueError(f"INT8 qweight must have shape {(n, k)}, got {tuple(qweight.shape)}")
    if qweight.dtype != torch.int8:
        raise TypeError("INT8 qweight must use torch.int8")
    if zero_points is not None:
        raise ValueError("symmetric INT8 canonical storage does not support zero_points")

    scale_shape = tuple(int(item) for item in scales.shape)
    if scale_shape in {(n,), (n, 1)}:
        granularity = "per_channel"
        canonical_scales = scales.detach().to(torch.float32).reshape(n, 1).contiguous()
        padded_k = k
        spec_group_size: int | None = None
    elif scale_shape in {(), (1, 1)}:
        granularity = "per_tensor"
        canonical_scales = scales.detach().to(torch.float32).reshape(1, 1).contiguous()
        padded_k = k
        spec_group_size = None
    else:
        if int(group_size) <= 0:
            raise ValueError("groupwise INT8 canonical repack requires group_size > 0")
        padded_k = ((k + int(group_size) - 1) // int(group_size)) * int(group_size)
        groups = padded_k // int(group_size)
        granularity = "groupwise"
        canonical_scales = _canonical_group_parameter(
            scales,
            n=n,
            groups=groups,
            name="scales",
        )
        spec_group_size = int(group_size)

    padded_weight = qweight
    if padded_k != k:
        padded_weight = F.pad(qweight, (0, padded_k - k), value=0)
    spec = QuantSpec(
        weight_dtype="int8",
        activation_dtype="int8",
        output_dtype="fp16",
        weight_granularity=granularity,
        group_size=spec_group_size,
        symmetric=symmetric,
        weight_scale_source="weight_offline",
        activation_scale_source="activation_dynamic",
    )
    return build_packed_weight(
        padded_weight,
        logical_shape=(n, k),
        spec=spec,
        scales=canonical_scales,
        zero_points=None,
        padded_k=padded_k,
        storage_layout=storage_layout,
        pack_version=pack_version,
    )


def repack_gptq_int4(
    qweight: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    scales: torch.Tensor,
    zero_points: torch.Tensor | None = None,
    group_size: int = 128,
    g_idx: torch.Tensor | None = None,
    pack_version: str = "xqt-w4a16-gptq-v1",
) -> PackedWeight:
    """Convert GPTQ int32 words to signed canonical XQT W4 storage."""

    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if zero_points is not None:
        raise ValueError("GPTQ signed canonical INT4 does not support zero_points")
    _validate_sequential_g_idx(g_idx, logical_k=k, group_size=group_size)
    codes = _source_int4_codes(qweight, logical_shape=(n, k), method="gptq")
    return _build_canonical_w4(
        codes,
        logical_shape=(n, k),
        scales=scales,
        zero_points=zero_points,
        group_size=group_size,
        signed=True,
        symmetric=True,
        pack_version=pack_version,
        storage_layout="xqt_int4_nk_v1",
    )


def repack_awq_int4(
    qweight: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    scales: torch.Tensor,
    zero_points: torch.Tensor | None = None,
    group_size: int = 128,
    g_idx: torch.Tensor | None = None,
    pack_version: str = "xqt-w4a16-awq-v1",
) -> PackedWeight:
    """Convert AWQ int32 words to unsigned canonical XQT W4 storage."""

    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    _validate_sequential_g_idx(g_idx, logical_k=k, group_size=group_size)
    codes = _source_int4_codes(qweight, logical_shape=(n, k), method="awq")
    return _build_canonical_w4(
        codes,
        logical_shape=(n, k),
        scales=scales,
        zero_points=zero_points,
        group_size=group_size,
        signed=False,
        symmetric=False,
        pack_version=pack_version,
        storage_layout="xqt_int4_nk_v1",
    )


def repack_w8a8_int8(
    qweight: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    scales: torch.Tensor,
    zero_points: torch.Tensor | None = None,
    group_size: int = 128,
    g_idx: torch.Tensor | None = None,
    pack_version: str = "xqt-w8a8-int8-v1",
) -> PackedWeight:
    """Convert int8 weights to canonical XQT W8 storage (per-channel/groupwise)."""
    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if zero_points is not None:
        raise ValueError("W8A8 does not support zero_points")
    _validate_sequential_g_idx(g_idx, logical_k=k, group_size=group_size)
    return _build_canonical_w8(
        qweight,
        logical_shape=(n, k),
        scales=scales,
        zero_points=zero_points,
        group_size=group_size,
        symmetric=True,
        pack_version=pack_version,
        storage_layout="xqt_int8_nk_v1",
    )


def unpack_int4_signed(packed: torch.Tensor, *, logical_k: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4_signed`, preserving logical K."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise TypeError("packed INT4 weights must be uint8 rank-2")
    if int(logical_k) <= 0 or packed.shape[1] < (int(logical_k) + 1) // 2:
        raise ValueError("logical_k is incompatible with packed INT4 shape")
    return unpack_int4(packed, logical_k=logical_k, signed=True)


def unpack_int4(
    packed: torch.Tensor, *, logical_k: int, signed: bool
) -> torch.Tensor:
    """Decode low-nibble-first INT4 as signed values or unsigned codes."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise TypeError("packed INT4 weights must be uint8 rank-2")
    if int(logical_k) <= 0 or packed.shape[1] < (int(logical_k) + 1) // 2:
        raise ValueError("logical_k is incompatible with packed INT4 shape")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)[:, :logical_k]
    if signed:
        return torch.where(values >= 8, values.to(torch.int16) - 16, values.to(torch.int16)).to(torch.int8)
    return values.to(torch.uint8)


def _validate_scale_tensor(
    value: torch.Tensor | None,
    *,
    logical_shape: tuple[int, int],
    granularity: str,
    group_size: int | None,
    name: str,
) -> None:
    if value is None:
        raise ValueError(f"{name} is required for quantized packed weights")
    n, k = logical_shape
    expected = expected_weight_scale_shape(
        logical_shape,
        granularity=granularity,
        group_size=group_size,
    )
    actual = tuple(int(item) for item in value.shape)
    accepted = {expected}
    if expected == (1, 1):
        accepted.add(())
    if expected == (n, 1):
        accepted.add((n,))
    if actual not in accepted:
        raise ValueError(f"{name} shape {actual} is incompatible; expected one of {sorted(accepted)}")


def validate_sparse2_4_mask(
    mask: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
) -> torch.Tensor:
    """Validate a channel-wise 2:4 mask: exactly two nonzeros per K quad."""

    n, k = (int(logical_shape[0]), int(logical_shape[1]))
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
        raise TypeError("sparse 2:4 mask must be a rank-2 torch.bool tensor")
    if tuple(mask.shape) != (n, k):
        raise ValueError(f"sparse 2:4 mask must have shape {(n, k)}, got {tuple(mask.shape)}")
    if k % 4 != 0:
        raise ValueError("sparse 2:4 logical K must be divisible by four")
    quads = mask.reshape(n, k // 4, 4)
    counts = quads.to(dtype=torch.int64).sum(dim=-1)
    if bool((counts != 2).any()):
        raise ValueError("sparse 2:4 mask must keep exactly two values per K quad")
    return mask.contiguous()


def build_packed_weight(
    qweight: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
    spec: QuantSpec,
    scales: torch.Tensor | None,
    zero_points: torch.Tensor | None = None,
    padded_k: int | None = None,
    storage_layout: str | None = None,
    pack_version: str | None = None,
    local_shape: tuple[int, int] | None = None,
    shard_axis: int | None = None,
    global_scale: torch.Tensor | None = None,
    sparse_mask: torch.Tensor | None = None,
    codebook: torch.Tensor | None = None,
) -> PackedWeight:
    """Create a versioned packed-weight object without changing tensor values."""

    n, k = int(logical_shape[0]), int(logical_shape[1])
    if qweight.ndim != 2 or int(qweight.shape[0]) != n:
        raise ValueError("qweight must have shape [N, packed-or-logical-K]")
    if padded_k is None:
        padded_k = k
    if spec.weight_dtype in {"int4", "fp4", "mxfp4", "nvfp4"}:
        if spec.weight_dtype in {"fp4", "mxfp4", "nvfp4"} and qweight.dtype != torch.uint8:
            raise TypeError("FP4-family qweight must use torch.uint8 packed storage")
        expected_columns = (int(padded_k) + 1) // 2
        if int(qweight.shape[1]) != expected_columns:
            raise ValueError(
                f"packed 4-bit qweight must have {expected_columns} columns, got {qweight.shape[1]}"
            )
        packed_bits: int | None = 4
        nibble_order: str | None = "low_high"
    elif spec.weight_dtype == "int2":
        if qweight.dtype != torch.uint8:
            raise TypeError("INT2 qweight must use torch.uint8 packed storage")
        expected_columns = (int(padded_k) + 3) // 4
        if int(qweight.shape[1]) != expected_columns:
            raise ValueError(
                f"packed INT2 qweight must have {expected_columns} columns, got {qweight.shape[1]}"
            )
        packed_bits = 2
        nibble_order = None
    elif spec.weight_dtype == "int3":
        if qweight.dtype != torch.uint8:
            raise TypeError("INT3 qweight must use torch.uint8 packed storage")
        expected_columns = ((int(padded_k) + 7) // 8) * 3
        if int(qweight.shape[1]) != expected_columns:
            raise ValueError(
                f"packed INT3 qweight must have {expected_columns} columns, got {qweight.shape[1]}"
            )
        packed_bits = 3
        nibble_order = None
    elif spec.weight_dtype == "codebook":
        if qweight.dtype != torch.uint8:
            raise TypeError("codebook indices must use torch.uint8")
        if codebook is None or codebook.ndim != 2:
            raise ValueError("codebook packed weights require a rank-2 codebook tensor")
        if codebook.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
            raise TypeError("codebook must use a floating dtype")
        vector_size = int(codebook.shape[1])
        if vector_size <= 0 or int(padded_k) % vector_size != 0:
            raise ValueError("codebook vector_size must evenly divide padded_k")
        if spec.group_size is None or int(spec.group_size) % vector_size != 0:
            raise ValueError("codebook group_size must be a multiple of vector_size")
        expected_columns = int(padded_k) // vector_size
        if int(qweight.shape[1]) != expected_columns:
            raise ValueError(
                f"codebook indices must have {expected_columns} columns, got {qweight.shape[1]}"
            )
        if bool((qweight.to(torch.int64) >= codebook.shape[0]).any()):
            raise ValueError("codebook indices out of range")
        packed_bits = None
        nibble_order = None
    else:
        if int(qweight.shape[1]) != int(padded_k):
            raise ValueError("non-INT4 qweight must use padded logical K columns")
        packed_bits = None
        nibble_order = None
    if spec.weight_dtype in {
        "int4",
        "int8",
        "int3",
        "int2",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp4",
        "mxfp4",
        "nvfp4",
        "codebook",
    }:
        _validate_scale_tensor(
            scales,
            logical_shape=(n, int(padded_k)),
            granularity=spec.weight_granularity,
            group_size=spec.group_size,
            name="scales",
        )
        if zero_points is not None:
            _validate_scale_tensor(
                zero_points,
                logical_shape=(n, int(padded_k)),
                granularity=spec.weight_granularity,
                group_size=spec.group_size,
                name="zero_points",
            )
    if sparse_mask is not None:
        validate_sparse2_4_mask(sparse_mask, logical_shape=(n, k))
        if int(padded_k) != k:
            raise ValueError("2:4 sparse weights require padded_k == logical K")
    metadata = PackedWeightMetadata(
        logical_shape=(n, k),
        storage_layout=storage_layout or spec.storage_layout,
        pack_version=pack_version or spec.pack_version,
        weight_dtype=spec.weight_dtype,
        padded_k=int(padded_k),
        group_size=spec.group_size,
        packed_bits=packed_bits,
        nibble_order=nibble_order,
        # FP4 family nibbles are sign-bit codebook indices, not two's
        # complement INT4 values.  Keep that distinction in the ABI.
        nibble_signed=(
            False
            if spec.weight_dtype in {"fp4", "mxfp4", "nvfp4"}
            else spec.symmetric
        ),
        local_shape=local_shape,
        shard_axis=shard_axis,
        global_scale=global_scale,
    )
    if spec.weight_dtype == "nvfp4":
        if global_scale is None:
            raise ValueError("NVFP4 packed weights require a scalar global_scale")
        if not isinstance(global_scale, torch.Tensor) or global_scale.numel() != 1:
            raise ValueError("NVFP4 global_scale must be a scalar tensor")
        if not bool(torch.isfinite(global_scale).all()) or bool((global_scale <= 0).any()):
            raise ValueError("NVFP4 global_scale must be finite and positive")
        global_scale = global_scale.detach().to(dtype=torch.float32).reshape(())
    elif global_scale is not None:
        raise ValueError("global_scale is only valid for NVFP4 packed weights")
    return PackedWeight(
        qweight=qweight.contiguous(),
        scales=scales,
        zero_points=zero_points,
        metadata=metadata,
        global_scale=global_scale,
        sparse_mask=sparse_mask,
        codebook=codebook,
    )


def validate_w4a16_packed_weight(
    weight: PackedWeight,
    *,
    spec: QuantSpec,
    logical_shape: tuple[int, int] | None = None,
) -> None:
    """Validate the canonical packed W4A16 ABI before backend dispatch.

    This check is deliberately stricter than the generic reference decoder. A
    W4 backend must know whether nibbles are signed, which group size was used,
    and whether a zero point is present; silently treating an AWQ buffer as a
    GPTQ buffer would produce plausible but wrong output.
    """

    if not isinstance(weight, PackedWeight):
        raise TypeError("W4A16 validation requires a PackedWeight")
    if spec.weight_dtype != "int4":
        raise ValueError("W4A16 validation requires QuantSpec.weight_dtype='int4'")
    metadata = weight.metadata
    if metadata.weight_dtype != "int4" or metadata.packed_bits != 4:
        raise ValueError("W4A16 PackedWeight must contain packed 4-bit storage")
    if metadata.storage_layout != "xqt_int4_nk_v1":
        raise ValueError(
            "W4A16 backend requires storage_layout='xqt_int4_nk_v1', "
            f"got {metadata.storage_layout!r}"
        )
    if logical_shape is None:
        raise ValueError("W4A16 validation requires logical_shape=(N,K)")
    expected_shape = logical_shape
    expected_shape = (int(expected_shape[0]), int(expected_shape[1]))
    if tuple(metadata.logical_shape) != expected_shape:
        raise ValueError(
            "W4A16 logical shape mismatch: "
            f"metadata={metadata.logical_shape}, expected={expected_shape}"
        )
    n, logical_k = expected_shape
    if metadata.padded_k < logical_k:
        raise ValueError("W4A16 padded_k cannot be smaller than logical K")
    if metadata.group_size not in {32, 64, 128}:
        raise ValueError("W4A16 canonical storage requires group_size 32, 64, or 128")
    if spec.group_size != metadata.group_size:
        raise ValueError(
            "W4A16 group_size mismatch: "
            f"spec={spec.group_size}, metadata={metadata.group_size}"
        )
    if metadata.nibble_order != "low_high":
        raise ValueError("W4A16 canonical storage requires low_high nibble order")
    if metadata.nibble_signed != spec.symmetric:
        raise ValueError(
            "W4A16 signedness mismatch: "
            f"metadata.nibble_signed={metadata.nibble_signed}, spec.symmetric={spec.symmetric}"
        )
    if spec.symmetric and spec.weight_zero_point:
        raise ValueError("symmetric W4A16 cannot declare a weight zero point")
    if not spec.symmetric and not spec.weight_zero_point:
        raise ValueError("asymmetric W4A16 requires weight_zero_point=True")
    if not isinstance(weight.qweight, torch.Tensor):
        raise TypeError("W4A16 qweight must be a torch.Tensor")
    expected_columns = (int(metadata.padded_k) + 1) // 2
    if weight.qweight.dtype != torch.uint8 or tuple(weight.qweight.shape) != (n, expected_columns):
        raise ValueError(
            "W4A16 qweight must be uint8 with shape "
            f"{(n, expected_columns)}, got dtype={weight.qweight.dtype}, "
            f"shape={tuple(weight.qweight.shape)}"
        )
    _validate_scale_tensor(
        weight.scales,
        logical_shape=(n, int(metadata.padded_k)),
        granularity=spec.weight_granularity,
        group_size=spec.group_size,
        name="W4A16 scales",
    )
    if spec.weight_zero_point:
        _validate_scale_tensor(
            weight.zero_points,
            logical_shape=(n, int(metadata.padded_k)),
            granularity=spec.weight_granularity,
            group_size=spec.group_size,
            name="W4A16 zero_points",
        )
    elif weight.zero_points is not None:
        raise ValueError("symmetric W4A16 must not carry zero_points")


def canonical_weight_view(weight: torch.Tensor) -> torch.Tensor:
    """Return a contiguous logical ``[N,K]`` view, rejecting transposed input."""

    if weight.ndim != 2:
        raise ValueError("canonical weight must be rank-2 [N,K]")
    return weight.contiguous()


def reference_weight_view(weight: torch.Tensor) -> torch.Tensor:
    """Reference backend view; it is exactly the canonical ``[N,K]`` layout."""

    return canonical_weight_view(weight)


def cutlass_weight_view(weight: torch.Tensor) -> torch.Tensor:
    """Logical CUTLASS view for a dense operand, ``[K,N]`` and contiguous."""

    return canonical_weight_view(weight).transpose(0, 1).contiguous()


__all__ = [
    "build_packed_weight",
    "canonical_weight_view",
    "cutlass_weight_view",
    "expected_activation_scale_shape",
    "expected_weight_scale_shape",
    "pack_int2_signed",
    "pack_int3_signed",
    "pack_int4_signed",
    "pack_int4_unsigned",
    "repack_awq_int4",
    "repack_gptq_int4",
    "reference_weight_view",
    "unpack_int2",
    "unpack_int3",
    "unpack_int4",
    "unpack_int4_signed",
    "validate_logical_shapes",
    "validate_sparse2_4_mask",
    "validate_w4a16_packed_weight",
]
