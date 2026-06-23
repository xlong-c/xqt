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


__all__ = [
    "ToyDequantGemmBlock",
    "ToyFP4MLP",
    "ToyAttentionBlock",
    "ToyAttentionClassifier",
    "ToyLLMMLPClassifier",
    "ToySwiGLUMLP",
    "ToyTransformerClassifier",
    "build_toy_attention_classifier",
    "build_toy_dequant_gemm_block",
    "build_toy_fp4_mlp",
    "build_toy_llm_mlp_classifier",
    "build_toy_transformer_classifier",
]
