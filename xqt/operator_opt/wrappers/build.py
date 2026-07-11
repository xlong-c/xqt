"""TileLang candidate model builder and shared helper."""

from __future__ import annotations

import copy
from typing import Any, Callable

import torch
from torch import nn

from xqt.core.errors import XQTBackendError

from ..types import OperatorOptimizationTargetPlan
from .attention import _TileLangAttentionWrapper
from .conv import _TileLangConvWrapper
from .conv3d import _TileLangConv3dWrapper
from .dequant_gemm import _build_tilelang_dequant_candidate
from .linear import _TileLangLinearWrapper
from .norm import _TileLangNormWrapper
from .xqt_attention import _TileLangXqtAttentionWrapper


def _wrap_direct_or_member(
    target_model: nn.Module,
    *,
    module_type: type[nn.Module],
    member_name: str,
    make_wrapper: Callable[[nn.Module], nn.Module],
    error_message: str,
    nested_member: bool = False,
) -> nn.Module:
    """Wrap a direct target or the first supported member without changing topology."""

    if isinstance(target_model, module_type):
        return make_wrapper(target_model)
    member = getattr(target_model, member_name, None)
    if isinstance(member, module_type):
        copied = copy.deepcopy(target_model)
        setattr(copied, member_name, make_wrapper(member))
        return copied
    for child_name, child in target_model.named_children():
        if nested_member:
            nested = getattr(child, member_name, None)
            if not isinstance(nested, module_type):
                continue
            copied = copy.deepcopy(target_model)
            copied_child = copied.get_submodule(child_name)
            setattr(copied_child, member_name, make_wrapper(getattr(copied_child, member_name)))
            return copied
        if isinstance(child, module_type):
            copied = copy.deepcopy(target_model)
            setattr(copied, child_name, make_wrapper(child))
            return copied
    raise XQTBackendError(error_message)


def build_tilelang_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Build a TileLang candidate for one supported target pattern."""

    patterns = target.patterns or ["attention"]
    settings = dict(target.tilelang)
    settings["preferred_patterns"] = list(patterns)
    if patterns == ["attention"]:
        from xqt import nn as xqt_nn

        if isinstance(target_model, xqt_nn.Attention):
            return _TileLangXqtAttentionWrapper(
                target_model,
                fallback=target.fallback,
                settings=settings,
            )
        for child_name, child in target_model.named_children():
            if isinstance(child, xqt_nn.Attention):
                wrapped = copy.deepcopy(target_model)
                setattr(
                    wrapped,
                    child_name,
                    _TileLangXqtAttentionWrapper(
                        getattr(wrapped, child_name),
                        fallback=target.fallback,
                        settings=settings,
                    ),
                )
                return wrapped
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.MultiheadAttention,
            member_name="attention",
            make_wrapper=lambda module: _TileLangAttentionWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message=(
                "TileLang attention target requires xqt.nn.Attention, "
                "nn.MultiheadAttention, or a module with an attention submodule"
            ),
            nested_member=True,
        )
    if patterns == ["conv"]:
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.Conv2d,
            member_name="conv",
            make_wrapper=lambda module: _TileLangConvWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang conv target requires nn.Conv2d or a module with a Conv2d child",
        )
    if patterns == ["conv3d_1x1x1"]:
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.Conv3d,
            member_name="conv",
            make_wrapper=lambda module: _TileLangConv3dWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang conv3d_1x1x1 target requires nn.Conv3d or a module with a Conv3d child",
        )
    if patterns in (["linear"], ["linear_marlin"]):
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.Linear,
            member_name="linear",
            make_wrapper=lambda module: _TileLangLinearWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang linear target requires nn.Linear or a module with a Linear child",
        )
    if patterns == ["norm"]:
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.LayerNorm,
            member_name="norm",
            make_wrapper=lambda module: _TileLangNormWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang norm target requires nn.LayerNorm or a module with a LayerNorm child",
        )
    if patterns in (
        ["dequant_gemm_epilogue"],
        ["fp4_packed_dequant_gemm_epilogue"],
        ["nvfp4_packed_dequant_gemm_epilogue"],
    ):
        return _build_tilelang_dequant_candidate(target_model, target, settings)
    raise XQTBackendError(
        "built-in TileLang executor currently supports attention, conv, conv3d_1x1x1, linear, norm, and dequant_gemm_epilogue patterns"
    )
