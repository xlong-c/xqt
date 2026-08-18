"""HF GPTQ/AWQ int32 pack/unpack helpers (vLLM layout bits)."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from xqt.contracts.packing_int4 import _unpack_int4
from xqt.core.errors import XQTArtifactError


def awq_reverse_pack_order(bits: int) -> list[int]:
    """Return AWQ nibble reverse order inside each int32 word (vLLM order)."""

    if bits != 4:
        raise XQTArtifactError(
            f"AWQ reverse pack order only defined for 4-bit, got {bits}"
        )
    return [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_int32_packed(
    packed: torch.Tensor,
    *,
    bits: int,
    reverse_order: bool,
) -> torch.Tensor:
    """Unpack int32-packed values to per-lane unsigned codes."""

    if bits not in {4, 8}:
        raise XQTArtifactError(f"unsupported pack bits: {bits}")
    mask = (1 << bits) - 1
    shifts = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int32)
    values = packed.to(torch.int32)
    unpacked = (values.unsqueeze(-1) >> shifts) & mask
    if reverse_order and bits == 4:
        order = torch.tensor(
            awq_reverse_pack_order(bits),
            device=packed.device,
            dtype=torch.long,
        )
        unpacked = unpacked[..., order]
    return unpacked


def gptq_qweight_to_signed_matrix(
    qweight: torch.Tensor,
    *,
    bits: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Convert GPTQ (or XQT-native) qweight to signed ``[out, in]`` codes."""

    pack_factor = 32 // bits
    if qweight.ndim != 2:
        raise XQTArtifactError(f"qweight must be 2D, got shape {tuple(qweight.shape)}")

    if qweight.dtype in {torch.uint8, torch.int8} and qweight.shape[0] == out_features:
        if bits == 4 and qweight.shape[1] * 2 >= in_features:
            return _unpack_int4(qweight, in_features)[:, :in_features].to(torch.float32)
        if bits == 8 and qweight.shape[1] >= in_features:
            return qweight.to(torch.float32)[:, :in_features]
        raise XQTArtifactError(
            f"unrecognized XQT-like qweight shape {tuple(qweight.shape)} "
            f"for out={out_features} in={in_features} bits={bits}"
        )

    if qweight.shape[0] * pack_factor == in_features and qweight.shape[1] == out_features:
        unpacked = unpack_int32_packed(qweight, bits=bits, reverse_order=False)
        codes = unpacked.reshape(-1, out_features)[:in_features, :]
        codes = codes.transpose(0, 1).contiguous()
    elif qweight.shape[0] == in_features and qweight.shape[1] * pack_factor == out_features:
        unpacked = unpack_int32_packed(qweight, bits=bits, reverse_order=False)
        codes = unpacked.reshape(in_features, -1)[:, :out_features]
        codes = codes.transpose(0, 1).contiguous()
    else:
        raise XQTArtifactError(
            f"unsupported GPTQ qweight shape {tuple(qweight.shape)} "
            f"for out={out_features} in={in_features} bits={bits}"
        )

    if bits == 4:
        return codes.to(torch.float32) - 8.0
    return codes.to(torch.float32) - 128.0


def awq_qweight_to_codes_matrix(
    qweight: torch.Tensor,
    *,
    bits: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Convert AWQ int32 qweight to raw ``[out, in]`` codes."""

    pack_factor = 32 // bits
    if qweight.dtype in {torch.uint8, torch.int8} and qweight.shape[0] == out_features:
        return gptq_qweight_to_signed_matrix(
            qweight,
            bits=bits,
            in_features=in_features,
            out_features=out_features,
        )
    if qweight.ndim != 2:
        raise XQTArtifactError(f"AWQ qweight must be 2D, got {tuple(qweight.shape)}")

    if qweight.shape[0] == in_features and qweight.shape[1] * pack_factor == out_features:
        unpacked = unpack_int32_packed(qweight, bits=bits, reverse_order=True)
        codes = unpacked.reshape(in_features, -1)[:, :out_features]
        return codes.transpose(0, 1).contiguous().to(torch.float32)
    if qweight.shape[0] * pack_factor == in_features and qweight.shape[1] == out_features:
        unpacked = unpack_int32_packed(qweight, bits=bits, reverse_order=True)
        codes = unpacked.reshape(-1, out_features)[:in_features, :]
        return codes.transpose(0, 1).contiguous().to(torch.float32)
    raise XQTArtifactError(
        f"unsupported AWQ qweight shape {tuple(qweight.shape)} "
        f"for out={out_features} in={in_features} bits={bits}"
    )


def normalize_scales(
    scales: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    """Return scale as ``[out, groups, 1]`` float32."""

    groups = max(1, (in_features + group_size - 1) // group_size)
    tensor = scales.detach().to(torch.float32)
    if tensor.ndim == 3 and tensor.shape[0] == out_features and tensor.shape[-1] == 1:
        return tensor
    if tensor.ndim == 2:
        if tensor.shape[0] == groups and tensor.shape[1] == out_features:
            tensor = tensor.transpose(0, 1).contiguous()
        if tensor.shape[0] == out_features and tensor.shape[1] == groups:
            return tensor.unsqueeze(-1)
        if tensor.shape[0] == out_features and tensor.shape[1] == 1:
            return tensor.unsqueeze(-1).expand(out_features, groups, 1).contiguous()
    if tensor.ndim == 1 and tensor.numel() == out_features:
        return (
            tensor.reshape(out_features, 1, 1)
            .expand(out_features, groups, 1)
            .contiguous()
        )
    raise XQTArtifactError(
        f"unsupported scales shape {tuple(scales.shape)} for out={out_features} "
        f"groups={groups}"
    )


def apply_zero_points(
    codes: torch.Tensor,
    qzeros: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    method: str,
) -> torch.Tensor:
    """Apply packed qzeros when present (AWQ asymmetric)."""

    if qzeros is None:
        if method == "awq" and bits == 4:
            return codes - 8.0
        return codes

    out_features, in_features = codes.shape
    groups = max(1, (in_features + group_size - 1) // group_size)
    pack_factor = 32 // bits
    zeros = qzeros.detach()
    try:
        if zeros.dtype in {torch.int32, torch.int64, torch.int16} and zeros.ndim == 2:
            if zeros.shape[0] == groups and zeros.shape[1] * pack_factor >= out_features:
                unpacked = unpack_int32_packed(
                    zeros, bits=bits, reverse_order=(method == "awq")
                )
                zp = unpacked.reshape(groups, -1)[:, :out_features].transpose(0, 1)
            elif zeros.shape[0] == out_features:
                zp = zeros[:, :groups]
            else:
                unpacked = unpack_int32_packed(
                    zeros, bits=bits, reverse_order=(method == "awq")
                )
                flat = unpacked.reshape(-1)
                zp = flat[: out_features * groups].reshape(out_features, groups)
        else:
            zp = zeros.to(torch.float32)
            if zp.ndim == 1:
                zp = zp.reshape(out_features, 1).expand(out_features, groups)
            elif zp.shape[0] == groups and zp.shape[1] == out_features:
                zp = zp.transpose(0, 1)
        zp_f = zp.to(torch.float32)[:, :groups]
        expanded = zp_f.unsqueeze(-1).expand(out_features, groups, group_size)
        expanded = expanded.reshape(out_features, groups * group_size)[:, :in_features]
        return codes - expanded
    except (RuntimeError, XQTArtifactError):
        if bits == 4:
            return codes - 8.0
        return codes


def group_qweight_keys(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, str]]:
    """Group GPTQ/AWQ-style keys by module prefix."""

    suffixes = (
        "qweight",
        "qzeros",
        "scales",
        "g_idx",
        "bias",
        "weight_scale",
        "weight_packed",
        "weight",
    )
    groups: dict[str, dict[str, str]] = {}
    for key in state:
        for suffix in suffixes:
            token = f".{suffix}"
            if key.endswith(token) or key == suffix:
                prefix = key[: -len(token)] if key.endswith(token) else ""
                groups.setdefault(prefix, {})[suffix] = key
                break
    return groups


def replace_submodule(
    root: nn.Module,
    qualified_name: str,
    new_module: nn.Module,
) -> None:
    """Replace a dotted submodule on ``root``."""

    if not qualified_name:
        raise XQTArtifactError(
            "cannot replace the root module in place via empty name"
        )
    parts = qualified_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


__all__ = [
    "apply_zero_points",
    "awq_qweight_to_codes_matrix",
    "awq_reverse_pack_order",
    "gptq_qweight_to_signed_matrix",
    "group_qweight_keys",
    "normalize_scales",
    "replace_submodule",
    "unpack_int32_packed",
]
