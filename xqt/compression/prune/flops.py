"""Shape-based FLOPs estimation for pruned model reports."""

from __future__ import annotations

from typing import Any

from torch import nn


def estimate_module_flops(module: nn.Module) -> float | None:
    """Return a shape-based FLOPs estimate or ``None`` for unknown modules.

    Estimates are per output position (per token for Linear, per spatial
    position for Conv2d) and intentionally separate from latency benchmark.
    """

    if isinstance(module, nn.Linear):
        return 2.0 * float(module.in_features) * float(module.out_features)
    if isinstance(module, nn.Conv2d):
        kernel_volume = float(
            int(module.kernel_size[0])
            * int(module.kernel_size[1])
        )
        in_per_group = float(module.in_channels) / max(int(module.groups), 1)
        return (
            2.0
            * in_per_group
            * kernel_volume
            * float(module.out_channels)
        )
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return 2.0 * float(module.num_features)
    if isinstance(module, nn.LayerNorm):
        normalized = module.normalized_shape
        if isinstance(normalized, int):
            return 2.0 * float(normalized)
        return 2.0 * float(int(normalized[-1]))

    embed_dim = getattr(module, "embed_dim", None)
    inner_dim = getattr(module, "inner_dim", None)
    num_heads = getattr(module, "num_heads", None)
    head_dim = getattr(module, "head_dim", None)
    if isinstance(embed_dim, int) and isinstance(inner_dim, int):
        return 8.0 * float(embed_dim) * float(inner_dim)
    if isinstance(embed_dim, int) and isinstance(num_heads, int) and isinstance(
        head_dim, int
    ):
        inner = float(num_heads) * float(head_dim)
        return 8.0 * float(embed_dim) * inner
    return None


def estimate_model_flops(model: nn.Module) -> dict[str, Any]:
    """Estimate per-module FLOPs and the model total."""

    total = 0.0
    per_module: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        flops = estimate_module_flops(module)
        if flops is None or flops <= 0.0:
            continue
        per_module.append(
            {
                "module_name": name,
                "module_type": type(module).__name__,
                "flops": flops,
            }
        )
        total += flops
    return {
        "flops": total,
        "estimate_kind": "shape_based_per_output_position",
        "per_module": per_module,
    }


__all__ = ["estimate_model_flops", "estimate_module_flops"]
