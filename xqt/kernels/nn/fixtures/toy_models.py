"""Small operator optimization toy models used by XQT smoke recipes."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ToyDequantGemmBlock(nn.Module):
    """Minimal dequantize + GEMM + epilogue block for TileLang workflows."""

    def __init__(
        self,
        input_dim: int = 32,
        output_dim: int = 64,
        *,
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        self.activation = activation
        self.qweight = nn.Parameter(torch.randn(output_dim, input_dim, dtype=torch.float32))
        self.scale = nn.Parameter(torch.rand(output_dim, dtype=torch.float32) + 0.01)
        self.bias = nn.Parameter(torch.randn(output_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_scale = self.scale.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        weight = self.qweight.to(dtype=x.dtype, device=x.device) * weight_scale
        output = x.matmul(weight.t())
        output = output + self.bias.to(dtype=output.dtype, device=output.device)
        if self.activation is None:
            return output
        if self.activation == "gelu":
            return F.gelu(output)
        if self.activation == "silu":
            return F.silu(output)
        if self.activation == "relu":
            return F.relu(output)
        raise ValueError(f"unsupported activation: {self.activation}")


class ToyFP4MLP(nn.Module):
    """Small MLP with a named Linear target for FP4 -> TileLang workflows."""

    def __init__(
        self,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = self.norm(hidden)
        return self.fc2(hidden)


class ToyTransformerClassifier(nn.Module):
    """Tiny Transformer-like classifier with a named encoder target."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 8,
        num_heads: int = 2,
        dim_feedforward: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        encoded = self.encoder(hidden)
        return self.head(encoded.mean(dim=1))


class ToySwiGLUMLP(nn.Module):
    """Small SwiGLU MLP block used by the Triton MLP recipe."""

    def __init__(
        self,
        hidden_dim: int = 8,
        intermediate_dim: int = 16,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ToyLLMMLPClassifier(nn.Module):
    """Tiny decoder-MLP classifier with a named MLP target."""

    def __init__(
        self,
        hidden_dim: int = 8,
        intermediate_dim: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = ToySwiGLUMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(x))


class ToyAttentionBlock(nn.Module):
    """Single-input attention wrapper suitable for component-level targeting."""

    def __init__(
        self,
        hidden_dim: int = 8,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(x, x, x, need_weights=False)
        return output


class ToyAttentionClassifier(nn.Module):
    """Tiny attention classifier with a named attention_block target."""

    def __init__(
        self,
        hidden_dim: int = 8,
        num_heads: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.attention_block = ToyAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attention_block(x).mean(dim=1))


class ToyConvBlock(nn.Module):
    """Small Conv2d block with a named conv target for operator-family routing tests."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 8,
        out_channels: int = 4,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.conv(x)))


class ToyConv3dBlock(nn.Module):
    """Small Conv3d block with a named conv target for operator-family routing tests."""

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.proj = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.conv(x)))


class ToyLinearBlock(nn.Module):
    """Small Linear block with a named linear target for direct half operator routing tests."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.linear(x)))


class ToyNormBlock(nn.Module):
    """Small LayerNorm block with a named norm target for direct half operator routing tests."""

    def __init__(
        self,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


def build_toy_dequant_gemm_block(
    input_dim: int = 32,
    output_dim: int = 64,
    *,
    activation: str | None = "silu",
) -> ToyDequantGemmBlock:
    """Build a small dequant GEMM block for TileLang recipes and tests."""

    return ToyDequantGemmBlock(
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
    )


def build_toy_fp4_mlp(
    hidden_dim: int = 64,
) -> ToyFP4MLP:
    """Build a small MLP for FP4 quantization and TileLang operator recipes."""

    return ToyFP4MLP(hidden_dim=hidden_dim)


def build_toy_transformer_classifier(
    input_dim: int = 8,
    hidden_dim: int = 8,
    num_heads: int = 2,
    dim_feedforward: int = 16,
    num_classes: int = 2,
) -> ToyTransformerClassifier:
    """Build a small Transformer classifier for component compile recipes."""

    return ToyTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
    )


def build_toy_llm_mlp_classifier(
    hidden_dim: int = 8,
    intermediate_dim: int = 16,
    num_classes: int = 2,
) -> ToyLLMMLPClassifier:
    """Build a small SwiGLU classifier for Triton MLP recipes."""

    return ToyLLMMLPClassifier(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_classes=num_classes,
    )


def build_toy_attention_classifier(
    hidden_dim: int = 8,
    num_heads: int = 2,
    num_classes: int = 2,
) -> ToyAttentionClassifier:
    """Build a small attention classifier for TileLang attention recipes."""

    return ToyAttentionClassifier(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_classes=num_classes,
    )


def build_toy_conv_block(
    in_channels: int = 3,
    hidden_channels: int = 8,
    out_channels: int = 4,
) -> ToyConvBlock:
    """Build a small Conv2d block for operator-family routing tests."""

    return ToyConvBlock(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )


def build_toy_conv3d_block(
    in_channels: int = 8,
    hidden_channels: int = 64,
    out_channels: int = 32,
) -> ToyConv3dBlock:
    """Build a small Conv3d block for direct TileLang Conv3d routing tests."""

    return ToyConv3dBlock(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )


def build_toy_linear_block(
    input_dim: int = 64,
    hidden_dim: int = 64,
    output_dim: int = 32,
) -> ToyLinearBlock:
    """Build a small Linear block for direct TileLang half Linear tests."""

    return ToyLinearBlock(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )


def build_toy_norm_block(
    hidden_dim: int = 64,
) -> ToyNormBlock:
    """Build a small LayerNorm block for direct TileLang half norm tests."""

    return ToyNormBlock(hidden_dim=hidden_dim)


class HeteroQuantToyModel(nn.Module):
    """Tiny chain model with named components for quantization tests."""

    def __init__(
        self,
        in_features: int = 4,
        hidden_features: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(in_features, hidden_features)
        self.projector = nn.Linear(hidden_features, hidden_features)
        self.decoder = nn.Linear(hidden_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = torch.relu(self.vision_encoder(inputs))
        projected = torch.relu(self.projector(encoded))
        return self.decoder(projected)


def build_hetero_quant_toy_model(
    in_features: int = 4,
    hidden_features: int = 4,
    num_classes: int = 2,
) -> HeteroQuantToyModel:
    return HeteroQuantToyModel(
        in_features=in_features,
        hidden_features=hidden_features,
        num_classes=num_classes,
    )


class StructuredPruningToyCNN(nn.Module):
    """Small Conv-BN-ReLU CNN used by structured-pruning tests."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: int = 8,
        out_channels: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.head(self.flatten(self.pool(features)))


__all__ = [
    "HeteroQuantToyModel",
    "StructuredPruningToyCNN",
    "ToyDequantGemmBlock",
    "ToyFP4MLP",
    "ToyAttentionBlock",
    "ToyAttentionClassifier",
    "ToyConvBlock",
    "ToyConv3dBlock",
    "ToyLinearBlock",
    "ToyLLMMLPClassifier",
    "ToyNormBlock",
    "ToySwiGLUMLP",
    "ToyTransformerClassifier",
    "build_toy_attention_classifier",
    "build_hetero_quant_toy_model",
    "build_toy_conv_block",
    "build_toy_conv3d_block",
    "build_toy_dequant_gemm_block",
    "build_toy_fp4_mlp",
    "build_toy_linear_block",
    "build_toy_llm_mlp_classifier",
    "build_toy_norm_block",
    "build_toy_transformer_classifier",
]
