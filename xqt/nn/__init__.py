"""Small operator facades for conversion-oriented XQT APIs."""

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
    "LayerNorm",
    "Linear",
    "RMSNorm",
    "TransformerBlock",
]
