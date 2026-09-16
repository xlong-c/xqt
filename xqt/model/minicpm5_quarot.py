"""QuaRot-style offline rotation for MiniCPM5's Llama architecture.

The transform follows the official QuaRot Llama weight-space convention:
RMSNorm scales are fused into adjacent projections, a randomized Hadamard
rotation is applied to the residual coordinate system, and inverse rotations
are absorbed into successor projections. The first implementation deliberately
keeps the attention/MLP online Hadamard sites disabled so a vanilla HF forward
remains functionally equivalent. Online KV/value rotations are a separate
runtime contract and must not be silently claimed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from xqt.compression.quant.quantizers.convrot_4bit import build_regular_hadamard_matrix


@dataclass(frozen=True)
class QuaRotTransformReport:
    """Metadata for one offline QuaRot coordinate transform."""

    hidden_size: int
    intermediate_size: int
    head_dim: int
    seed: int
    norm_fused: bool
    embedding_mean_baked: bool
    online_hadamard_enabled: bool
    transformed_linear_count: int
    max_weight_abs_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "quarot_hidden_coordinate_rotation",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "head_dim": self.head_dim,
            "seed": self.seed,
            "norm_fused": self.norm_fused,
            "embedding_mean_baked": self.embedding_mean_baked,
            "online_hadamard_enabled": self.online_hadamard_enabled,
            "transformed_linear_count": self.transformed_linear_count,
            "max_weight_abs_delta": self.max_weight_abs_delta,
            "status": "vanilla_hf_forward_compatible",
        }


def _hadamard_power_of_two(order: int) -> torch.Tensor:
    """Build a normalized Sylvester Hadamard matrix."""

    if order < 1 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {order}")
    matrix = torch.ones((1, 1), dtype=torch.float64)
    size = 1
    while size < order:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
        size *= 2
    return matrix / math.sqrt(float(order))


def build_quarot_rotation(
    hidden_size: int,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Build QuaRot's randomized Hadamard rotation ``H @ diag(signs)``."""

    if hidden_size < 1 or hidden_size & (hidden_size - 1):
        raise ValueError("MiniCPM5 QuaRot hidden rotation requires power-of-two hidden_size")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    try:
        matrix = build_regular_hadamard_matrix(hidden_size).to(torch.float64)
        matrix = matrix / math.sqrt(float(hidden_size))
    except ValueError:
        matrix = _hadamard_power_of_two(hidden_size)
    signs = torch.randint(0, 2, (hidden_size,), generator=generator, dtype=torch.float64)
    signs = signs.mul(2.0).sub(1.0)
    return matrix * signs.unsqueeze(0)


def _hadamard_12() -> torch.Tensor:
    """Return the order-12 Hadamard factor used by QuaRot for 6144 features."""

    values = [
        [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
        [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1],
        [1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1],
        [1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
        [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
        [1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1],
        [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1],
    ]
    matrix = torch.tensor(values, dtype=torch.float64)
    if not torch.allclose(
        matrix @ matrix.T,
        12.0 * torch.eye(12, dtype=torch.float64),
    ):
        raise RuntimeError("QuaRot order-12 Hadamard factor is not orthogonal")
    return matrix / math.sqrt(12.0)


def _partial_hadamard(dim: int) -> torch.Tensor:
    """Build the QuaRot partial Hadamard operator for a supported feature width."""

    if dim == 6144:
        small = _hadamard_12()
        large = _hadamard_power_of_two(512)
        operator = torch.kron(small, large)
        return operator
    if dim < 1 or dim & (dim - 1):
        raise ValueError(f"unsupported QuaRot Hadamard feature width: {dim}")
    return _hadamard_power_of_two(dim)


def _copy_weight(linear: nn.Linear, weight: torch.Tensor) -> float:
    """Replace a Linear weight and return the maximum absolute change."""

    before = linear.weight.detach().to(torch.float64)
    linear.weight.data.copy_(weight.to(device=linear.weight.device, dtype=linear.weight.dtype))
    return float((before - linear.weight.detach().to(torch.float64)).abs().max().item())


def _fuse_norm(linear_layers: tuple[nn.Linear, ...], norm: nn.Module) -> None:
    """Absorb one RMSNorm scale into all adjacent Linear input columns."""

    scale = norm.weight.detach().to(torch.float64)
    for linear in linear_layers:
        weight = linear.weight.detach().to(torch.float64)
        if weight.shape[1] != scale.numel():
            raise ValueError("RMSNorm and adjacent Linear dimensions do not match")
        linear.weight.data.copy_((weight * scale.unsqueeze(0)).to(linear.weight.dtype))
    norm.weight.data.fill_(1.0)


def _rotate_input(linear: nn.Linear, rotation: torch.Tensor) -> float:
    weight = linear.weight.detach().to(torch.float64)
    return _copy_weight(linear, weight @ rotation.to(weight.device))


def _rotate_output(linear: nn.Linear, rotation: torch.Tensor) -> float:
    weight = linear.weight.detach().to(torch.float64)
    rotated = rotation.to(weight.device).T @ weight
    delta = _copy_weight(linear, rotated)
    if linear.bias is not None:
        bias = linear.bias.detach().to(torch.float64)
        linear.bias.data.copy_((rotation.to(bias.device).T @ bias).to(linear.bias.dtype))
    return delta


def _rotate_per_head_output(linear: nn.Linear, head_dim: int) -> float:
    """Apply the exact per-head Hadamard to a projection's output columns."""

    if linear.out_features % head_dim:
        raise ValueError("projection output width must be divisible by head_dim")
    rotation = _hadamard_power_of_two(head_dim).to(linear.weight.device)
    weight = linear.weight.detach().to(torch.float64)
    reshaped = weight.T.reshape(linear.in_features, -1, head_dim)
    rotated = torch.matmul(reshaped, rotation).reshape(linear.in_features, linear.out_features).T
    return _copy_weight(linear, rotated)


def _rotate_input_blocks(linear: nn.Linear, operator: torch.Tensor) -> float:
    """Apply a block/partial Hadamard to a Linear input dimension."""

    if operator.shape != (linear.in_features, linear.in_features):
        raise ValueError("partial Hadamard shape does not match Linear input width")
    weight = linear.weight.detach().to(torch.float64)
    return _copy_weight(linear, weight @ operator.to(weight.device))


def apply_quarot_minicpm5(
    model: nn.Module,
    *,
    seed: int = 42,
    fuse_norms: bool = True,
    rotate_ov_projections: bool = False,
    rotate_mlp_down_projection: bool = False,
) -> QuaRotTransformReport:
    """Apply QuaRot's offline Llama coordinate transform to MiniCPM5.

    The default mode is compatible with an unmodified Hugging Face forward.
    Setting ``rotate_ov_projections`` or ``rotate_mlp_down_projection`` requires
    matching online Hadamard runtime nodes and therefore raises for now.
    """

    if rotate_ov_projections or rotate_mlp_down_projection:
        raise NotImplementedError(
            "online QuaRot Hadamard runtime nodes are not yet materialized for HF forward"
        )
    config = getattr(model, "config", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if config is None or layers is None:
        raise TypeError("model must be a Hugging Face LlamaForCausalLM-like module")
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_heads = int(config.num_attention_heads)
    if hidden_size % num_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    head_dim = hidden_size // num_heads
    rotation = build_quarot_rotation(hidden_size, seed=seed)
    max_delta = 0.0
    linear_count = 0

    if fuse_norms:
        for layer in layers:
            _fuse_norm(
                (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
                layer.input_layernorm,
            )
            _fuse_norm(
                (layer.mlp.gate_proj, layer.mlp.up_proj),
                layer.post_attention_layernorm,
            )
        _fuse_norm((model.lm_head,), model.model.norm)

    embed = model.model.embed_tokens
    embed_weight = embed.weight.detach().to(torch.float64)
    embed.weight.data.copy_((embed_weight @ rotation.to(embed_weight.device)).to(embed.weight.dtype))

    max_delta = max(max_delta, _rotate_input(model.lm_head, rotation))
    linear_count += 1
    for layer in layers:
        for linear in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
        ):
            max_delta = max(max_delta, _rotate_input(linear, rotation))
            linear_count += 1
        max_delta = max(max_delta, _rotate_output(layer.self_attn.o_proj, rotation))
        max_delta = max(max_delta, _rotate_output(layer.mlp.down_proj, rotation))
        linear_count += 2

    setattr(model, "_xqt_quarot_report", QuaRotTransformReport(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        seed=int(seed),
        norm_fused=bool(fuse_norms),
        embedding_mean_baked=False,
        online_hadamard_enabled=False,
        transformed_linear_count=linear_count,
        max_weight_abs_delta=max_delta,
    ))
    return model._xqt_quarot_report


__all__ = [
    "QuaRotTransformReport",
    "apply_quarot_minicpm5",
    "build_quarot_rotation",
]
