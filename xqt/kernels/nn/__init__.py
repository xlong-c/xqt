"""Semantic facades, convert API and smoke fixtures for XQT kernels."""

from __future__ import annotations

from xqt.kernels.precision import (
    FeedForwardPrecisionPolicy,
    FusionIntent,
    ModuleContract,
    OperatorContract,
    OperatorKind,
    PrecisionPolicy,
    TensorStorageSpec,
)

from .attention import Attention
from .block import TransformerBlock
from .feedforward import FeedForward, FeedForwardFusionConfig
from .linear import Conv2d, LayerNorm, Linear
from .norm import RMSNorm

__all__ = [
    "Attention",
    "Conv2d",
    "FeedForward",
    "FeedForwardFusionConfig",
    "FeedForwardPrecisionPolicy",
    "FusionIntent",
    "LayerNorm",
    "Linear",
    "ModuleContract",
    "OperatorContract",
    "OperatorKind",
    "PrecisionPolicy",
    "RMSNorm",
    "TensorStorageSpec",
    "TransformerBlock",
]
