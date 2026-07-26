"""INT8 sm_89 B-matrix prepack: math [K,N] row-major -> packed [N,K] for contiguous-K MMA loads."""

from __future__ import annotations

from typing import Any

import torch

from xqt.core.errors import XQTBackendError

from .base import PrepackLayoutId, PrepackResult, PrepackSpec, register_prepack_spec

INT8_SM89_B_NK = "int8:sm_89:b_nk"


def prepack_int8_b_nk(weight_kn: torch.Tensor, **_kwargs: Any) -> PrepackResult:
    if weight_kn.ndim != 2:
        raise XQTBackendError("int8:sm_89:b_nk expects 2D weight [K, N]")
    if weight_kn.dtype != torch.int8:
        raise XQTBackendError("int8:sm_89:b_nk expects torch.int8")
    k, n = int(weight_kn.shape[0]), int(weight_kn.shape[1])
    if k <= 0 or n <= 0:
        raise XQTBackendError("int8:sm_89:b_nk dimensions must be positive")
    packed = weight_kn.transpose(0, 1).contiguous()
    layout = PrepackLayoutId("int8", "sm_89", "b_nk")
    return PrepackResult(
        packed=packed,
        layout_id=layout,
        math_shape=(k, n),
        packed_shape=(n, k),
        metadata={
            "mma": "m16n8k32.s8",
            "math_layout": "K,N_row_major",
            "packed_layout": "N,K_row_major",
            "fragment_b": "contiguous_K_at_fixed_N",
            "kernel": "ptx_sm89_or_cuda_sm89",
        },
    )


def unpack_int8_b_nk(
    packed: torch.Tensor,
    *,
    math_shape: tuple[int, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> torch.Tensor:
    if packed.ndim != 2:
        raise XQTBackendError("int8:sm_89:b_nk packed buffer must be 2D [N, K]")
    if packed.dtype != torch.int8:
        raise XQTBackendError("int8:sm_89:b_nk packed buffer must be torch.int8")
    n, k = int(packed.shape[0]), int(packed.shape[1])
    if math_shape is not None:
        if tuple(int(x) for x in math_shape) != (k, n):
            raise XQTBackendError(
                f"math_shape {math_shape} does not match packed [N,K]=[{n},{k}]"
            )
    return packed.transpose(0, 1).contiguous()


def _register() -> None:
    register_prepack_spec(
        PrepackSpec(
            layout_id=PrepackLayoutId("int8", "sm_89", "b_nk"),
            description=(
                "INT8 B prepack for Ada sm_89: store weight as [N,K] so MMA B fragment "
                "loads consecutive K for sm_89 INT8 MMA (ptx_sm89 / cuda_sm89)"
            ),
            status="implemented",
            math_shape_order="K,N",
            packed_shape_order="N,K",
            pack=prepack_int8_b_nk,
            unpack=unpack_int8_b_nk,
            notes="Used by ptx_sm89 and cuda_sm89. Offline transpose of qweight_t.",
            metadata_defaults={"mma": "m16n8k32.s8"},
        )
    )


_register()
