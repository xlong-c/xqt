"""SVDQuant — low-rank branch + quantized residual for 4-bit inference.

Reference:
  SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models
  (Li et al., NVIDIA/MIT, 2024) — https://arxiv.org/abs/2411.05007

Architecture per Linear layer:

  Input x ──┬── down_proj(x) ──┬── up_proj(residual) ── y_lora ──┐
            │                  │                                      ├── y + bias
            └── act_quantize → x_q ── W4A4 MMA + dequant ── y_main ─┘

Where down_proj and act_quantize share the same input x (candidate for kernel fusion),
and up_proj and 4-bit MMA share the same output accumulator (candidate for kernel fusion).

Phase 1 (this file): pure-Python reference implementation — correctness baseline.
Phase 2-3 (future): CuTe DSL kernels for TRUE INT4 MMA + SVDQuant fusion kernels.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from xqt.analysis.svd_analysis import (
    SVDQuantAnalysis,
    decompose_weight_svd,
)

from .fp4_backend import (
    _pack_int4,
    _unpack_int4,
)
from .policy import QuantizationPolicy, should_quantize_module
from .strategy import normalize_quant_strategy


# ── SVDQuant result dataclass ────────────────────────────────────────────


@dataclass
class SVDQuantResult:
    """Result returned by the SVDQuant backend."""

    model: nn.Module
    backend: str = "svdquant"
    strategy: str = "svd_fp4"
    quantized_modules: list[str] = field(default_factory=list)
    svd_analysis: Optional[SVDQuantAnalysis] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Low-rank branch ──────────────────────────────────────────────────────


class LowRankBranch(nn.Module):
    """Two-layer low-rank branch that absorbs weight outliers.

    W_lr = L2 @ L1, where:
      L1: (r, in_features) — down-projection
      L2: (out_features, r) — up-projection

    Forward: y = L2(L1(x)) = x @ L1.T @ L2.T
    """

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        rank, in_features = down_weight.shape
        out_features, rank2 = up_weight.shape
        if rank != rank2:
            raise ValueError(
                f"Rank mismatch: down_proj rank={rank}, up_proj rank={rank2}"
            )
        self.down_proj = nn.Linear(in_features, rank, bias=False)
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(rank, out_features, bias=False)
        self.up_proj.weight.data = up_weight.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank correction: L2(L1(x))."""
        return self.up_proj(self.down_proj(x))


# ── Packed INT4 residual weight helpers ──────────────────────────────────


def _quantize_residual_int4(
    weight_res: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize residual weight to signed INT4 with per-group scales.

    Returns (packed_weight, scale, padded_in_features).
    """
    out_features, in_features = weight_res.shape
    normalized_group_size = max(1, min(group_size, in_features))
    padded_in_features = (
        (in_features + normalized_group_size - 1) // normalized_group_size
    ) * normalized_group_size

    # Pad to group_size boundary
    weight_f32 = weight_res.detach().to(torch.float32)
    if padded_in_features != in_features:
        weight_f32 = F.pad(weight_f32, (0, padded_in_features - in_features))

    # Per-group absmax quantization to [-8, 7]
    grouped = weight_f32.reshape(out_features, -1, normalized_group_size)
    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    scale = torch.where(
        max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs)
    )  # shape: (out_features, num_groups, 1)
    quantized = torch.clamp(
        torch.round(grouped / (scale + 1e-12)), min=-8, max=7
    ).to(torch.int8)
    scale = scale.squeeze(-1)  # (out_features, num_groups)

    packed = _pack_int4(quantized.reshape(out_features, padded_in_features))
    return packed, scale.to(torch.float32), padded_in_features


def _dequantize_residual_int4(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    input_features: int,
    group_size: int,
    padded_input_features: int,
) -> torch.Tensor:
    """Dequantize packed INT4 residual weight back to float."""
    codes = _unpack_int4(packed_weight, padded_input_features)
    grouped = codes.reshape(codes.shape[0], -1, group_size)  # (out, num_groups, gs)
    scale_expanded = scale.unsqueeze(-1)  # (out, num_groups, 1)
    dequantized = grouped * scale_expanded
    return dequantized.reshape(codes.shape[0], padded_input_features)[
        :, :input_features
    ]


# ── Combined SVDQuant Linear module ──────────────────────────────────────


class SVDQuantLinear(nn.Module):
    """SVDQuant Linear: low-rank FP16 branch + quantized INT4/FP4 residual.

    Shapes:
      - down_proj.weight: (r, in_features)       ← L1
      - up_proj.weight:   (out_features, r)       ← L2
      - packed_residual:  (out_features, padded_in // 2) — uint8 packed INT4
      - residual_scale:   (out_features, num_groups)
      - bias (optional):  (out_features,)

    Forward (reference path):
      x_q    = act_quantize(x)                    # simulated: pass-through in reference
      y_main = dequant_gemm(x, packed_residual)   # dequant + fp16 matmul
      y_lora = up_proj(down_proj(x))              # low-rank correction
      return y_main + y_lora + bias
    """

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        packed_residual: torch.Tensor,
        residual_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        quant_dtype: str = "int4",
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.quant_dtype = str(quant_dtype)

        # Low-rank branch
        self.down_proj = nn.Linear(input_features, down_weight.shape[0], bias=False)
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(up_weight.shape[0], output_features, bias=False)
        self.up_proj.weight.data = up_weight.detach().clone()

        # Quantized residual
        self.register_buffer("packed_residual", packed_residual.to(torch.uint8))
        self.register_buffer("residual_scale", residual_scale.to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rank: int,
        group_size: int = 128,
        quant_dtype: str = "int4",
    ) -> "SVDQuantLinear":
        """Build an SVDQuantLinear from a regular nn.Linear via SVD decomposition."""
        weight = module.weight.detach()
        weight_2d = weight.reshape(module.out_features, module.in_features)
        if weight_2d.shape[0] != module.out_features:
            weight_2d = weight_2d.reshape(module.out_features, -1)

        # 1. SVD decomposition
        decomp = decompose_weight_svd(weight_2d, rank=rank)

        # 2. Extract low-rank components
        L1, L2 = decomp.low_rank_components()  # L1: (r, in), L2: (out, r)

        # 3. Compute residual and quantize
        weight_res = decomp.residual_weight(weight_2d).to(torch.float32)
        if quant_dtype == "int4":
            packed_residual, residual_scale, padded_in = _quantize_residual_int4(
                weight_res, group_size=group_size
            )
        else:
            raise ValueError(
                f"Unsupported quant_dtype '{quant_dtype}' for SVDQuant. "
                f"Supported: int4"
            )

        bias = None if module.bias is None else module.bias.detach().to(torch.float32)

        return cls(
            down_weight=L1,
            up_weight=L2,
            packed_residual=packed_residual,
            residual_scale=residual_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=group_size,
            padded_input_features=padded_in,
            quant_dtype=quant_dtype,
        )

    def dequantize_residual(self) -> torch.Tensor:
        """Dequantize the packed residual weight for reference computation."""
        if self.quant_dtype == "int4":
            packed = self.packed_residual
            scale = self.residual_scale
            if not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor):
                raise RuntimeError(
                    "packed_residual and residual_scale must be tensors"
                )
            return _dequantize_residual_int4(
                packed,
                scale,
                self.input_features,
                self.group_size,
                self.padded_input_features,
            )
        raise RuntimeError(f"Unsupported quant_dtype: {self.quant_dtype}")

    def low_rank_weight(self) -> torch.Tensor:
        """Reconstruct the low-rank branch weight W_lr = L2 @ L1."""
        return self.up_proj.weight.data @ self.down_proj.weight.data

    def full_weight_dequant(self) -> torch.Tensor:
        """Reconstruct the full dequantized weight: W_lr + W_res_deq."""
        return self.low_rank_weight() + self.dequantize_residual()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference forward pass.

        Fusion opportunities (Phase 2-3):
          - FUSE_DOWN: down_proj(x) + act_quantize(x) share input → single kernel
          - FUSE_UP: up_proj(h) + dequant_gemm_epilogue share accumulator → single kernel
        """
        device = x.device
        dtype = x.dtype

        # Main path: dequant residual → fp16 GEMM ([FUSE_UP candidate: epilogue fusion])
        weight_deq = self.dequantize_residual().to(device=device, dtype=dtype)
        y_main = F.linear(x, weight_deq)

        # Low-rank correction ([FUSE_DOWN candidate: input-sharing with act_quantize])
        h = self.down_proj(x)  # (batch, r)
        y_lora = self.up_proj(h)  # (batch, out)  [FUSE_UP candidate: add to accum]

        y = y_main + y_lora
        if self.bias is not None:
            y = y + self.bias.to(device=device, dtype=dtype)
        return y


# ── Submodule replacement ─────────────────────────────────────────────────


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


# ── Policy from mapping ──────────────────────────────────────────────────


def _policy_from_mapping(policy: Mapping[str, Any]) -> QuantizationPolicy:
    kwargs: dict[str, Any] = {}
    for key, value in policy.items():
        if key == "dtype":
            kwargs["dtype"] = str(value)
        elif key == "scheme":
            kwargs["scheme"] = str(value)
        elif key in {
            "include_module_types",
            "exclude_module_types",
            "include_name_patterns",
            "exclude_name_patterns",
            "include_module_names",
            "exclude_module_names",
        }:
            kwargs[key] = tuple(str(item) for item in value)
        elif key == "min_parameters":
            kwargs[key] = int(value)
    return QuantizationPolicy(**kwargs)


# ── Main quantization entry point ────────────────────────────────────────


def quantize_with_svd(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    rank: int = 32,
    group_size: int = 128,
    quant_dtype: str = "int4",
    inplace: bool = True,
    collect_analysis: bool = True,
) -> SVDQuantResult:
    """Apply SVDQuant to all qualifying Linear layers in a model.

    For each Linear layer:
      1. SVD decompose weight → low-rank branch (L1, L2) + residual
      2. Quantize residual to INT4/FP4 with per-group scales
      3. Replace module with SVDQuantLinear

    Args:
        model: PyTorch model to quantize.
        policy: Module selection policy (which layers to quantize).
        strategy: Canonical strategy name ("svd_fp4" or "svd_int4").
        rank: Low-rank branch rank (r). Typical: 16–64.
        group_size: Per-group quantization granularity.
        quant_dtype: Residual quantization dtype ("int4").
        inplace: If True, modify model in-place.
        collect_analysis: If True, collect SVD metrics for all layers.

    Returns:
        SVDQuantResult with the modified model and analysis.
    """
    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    configured_rank = int(policy_mapping.get("rank", rank))
    configured_group_size = int(policy_mapping.get("group_size", group_size) or group_size)
    configured_quant_dtype = str(policy_mapping.get("quant_dtype", quant_dtype))

    selected_strategy = (
        normalize_quant_strategy(strategy, policy_mapping)
        or f"svd_{configured_quant_dtype}"
    )

    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []
    svd_analysis = SVDQuantAnalysis() if collect_analysis else None

    for name, module in list(target_model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not should_quantize_module(name, module, quant_policy):
            continue

        svd_module = SVDQuantLinear.from_linear(
            module,
            rank=configured_rank,
            group_size=configured_group_size,
            quant_dtype=configured_quant_dtype,
        )

        _replace_submodule(target_model, name, svd_module)
        quantized_modules.append(name)

        if svd_analysis is not None:
            weight = module.weight.detach()
            try:
                decomp = decompose_weight_svd(
                    weight.reshape(module.out_features, module.in_features),
                    rank=configured_rank,
                )
                svd_analysis.decompositions[name] = decomp
            except ValueError:
                # Layer too small for SVD at this rank — skip analysis
                pass

    return SVDQuantResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        svd_analysis=svd_analysis,
        metadata={
            "implementation": "svdquant_reference",
            "rank": configured_rank,
            "group_size": configured_group_size,
            "quant_dtype": configured_quant_dtype,
            "low_rank_branch_dtype": "fp16",
            "fusion_status": "none",
            "fusion_note": (
                "Reference path uses separate down_proj, dequant+GEMM, and up_proj. "
                "CuTe DSL kernel fusion (FUSE_DOWN + FUSE_UP) pending Phase 2-3."
            ),
            "policy": {
                "dtype": quant_policy.dtype,
                "scheme": quant_policy.scheme,
                "include_module_types": list(quant_policy.include_module_types),
                "exclude_module_types": list(quant_policy.exclude_module_types),
                "include_name_patterns": list(quant_policy.include_name_patterns),
                "exclude_name_patterns": list(quant_policy.exclude_name_patterns),
                "include_module_names": list(quant_policy.include_module_names),
                "exclude_module_names": list(quant_policy.exclude_module_names),
                "min_parameters": quant_policy.min_parameters,
                "rank": configured_rank,
                "group_size": configured_group_size,
                "quant_dtype": configured_quant_dtype,
            },
            "svd_analysis": svd_analysis.to_dict() if svd_analysis is not None else None,
        },
    )


__all__ = [
    "LowRankBranch",
    "SVDQuantLinear",
    "SVDQuantResult",
    "quantize_with_svd",
]
