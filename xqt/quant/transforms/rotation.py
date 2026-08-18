"""Rotation absorb transform: bake groupwise R into predecessor Linear weights.

ConvRot / QuaRot-style residual convention for sequential Linear pairs:

- Successor weight is left unchanged here (quantizer still rotates offline).
- Predecessor weight absorbs R so its output already lives in the rotated
  space: ``W_pred' = R^T @ W_pred`` on the output-feature groups.
- Successor modules are marked ``input_already_rotated=True`` when they expose
  the attribute, so online Hadamard can be skipped.

Structures that cannot absorb keep online rotation (declared via
``required_kernels``) as the honest fallback.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.quant.quantizers.convrot_4bit import (
    _apply_groupwise_rotation,
    _normalize_rot_size,
    _normalized_regular_hadamard,
)

from .base import TransformPlan, TransformReport


def _linear_children(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    ]


def _set_input_already_rotated(module: nn.Module, value: bool) -> bool:
    if hasattr(module, "input_already_rotated"):
        setattr(module, "input_already_rotated", bool(value))
        return True
    module.register_buffer(
        "_xqt_input_already_rotated",
        torch.tensor(bool(value), dtype=torch.bool),
        persistent=False,
    )
    return True


def _pair_rot_ok(pred: nn.Linear, succ: nn.Linear, rot_size: int) -> int | None:
    if int(pred.out_features) != int(succ.in_features):
        return None
    rot = _normalize_rot_size(rot_size, int(succ.in_features))
    if rot < 4:
        return None
    if int(succ.in_features) % rot != 0 or int(pred.out_features) % rot != 0:
        return None
    return rot


def _collect_absorb_pairs(
    model: nn.Module,
    *,
    rot_size: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (absorbable sequential pairs, residual/branch online pairs).

    Sequential: adjacent Linear names in named_modules order with matching dims.
    Residual/branch: Linear under a module that also has a non-Linear residual
    path (e.g. ``branch`` + ``residual``), or explicit ``_xqt_residual_pair``.
    """

    linears = _linear_children(model)
    absorb: list[dict[str, str]] = []
    online: list[dict[str, str]] = []
    for index in range(len(linears) - 1):
        pred_name, pred = linears[index]
        succ_name, succ = linears[index + 1]
        if _pair_rot_ok(pred, succ, rot_size) is None:
            continue
        absorb.append(
            {
                "predecessor": pred_name,
                "successor": succ_name,
                "kind": "sequential",
            }
        )

    for name, module in model.named_modules():
        if not name:
            continue
        children = list(module.named_children())
        child_map = dict(children)
        if "branch" in child_map and "residual" in child_map:
            branch = child_map["branch"]
            residual = child_map["residual"]
            if isinstance(branch, nn.Linear) and isinstance(residual, nn.Linear):
                branch_name = f"{name}.branch" if name else "branch"
                residual_name = f"{name}.residual" if name else "residual"
                if _pair_rot_ok(branch, residual, rot_size) is not None:
                    online.append(
                        {
                            "predecessor": branch_name,
                            "successor": residual_name,
                            "kind": "residual_branch",
                        }
                    )
        explicit = getattr(module, "_xqt_residual_pair", None)
        if isinstance(explicit, (list, tuple)) and len(explicit) == 2:
            pred_n, succ_n = str(explicit[0]), str(explicit[1])
            try:
                pred = model.get_submodule(pred_n)
                succ = model.get_submodule(succ_n)
            except AttributeError:
                continue
            if isinstance(pred, nn.Linear) and isinstance(succ, nn.Linear):
                if _pair_rot_ok(pred, succ, rot_size) is not None:
                    online.append(
                        {
                            "predecessor": pred_n,
                            "successor": succ_n,
                            "kind": "explicit_residual",
                        }
                    )
    return absorb, online


def preflight_hadamard_kernel() -> dict[str, Any]:
    """Honest capability note for online ``hadamard_groupwise`` (U7 / V3).

    When tilelang/triton declare the capability, status is ``engine_registered``.
    Otherwise residual online path stays reference groupwise matmul.
    """

    try:
        from xqt.contracts.engine_resolve import engines_providing
    except ImportError:
        engines_providing = None  # type: ignore[assignment]

    providers: list[str] = []
    if callable(engines_providing):
        providers = list(engines_providing("hadamard_groupwise"))
    if providers:
        return {
            "kernel": "hadamard_groupwise",
            "status": "engine_registered",
            "providers": providers,
            "performance_note": (
                "engines declare hadamard_groupwise; fused prologue still "
                "shape/device gated at call site"
            ),
        }
    return {
        "kernel": "hadamard_groupwise",
        "status": "reference_only",
        "providers": [],
        "performance_note": (
            "no fused hadamard_groupwise engine registered; "
            "online residual path uses reference groupwise matmul"
        ),
    }


def _absorb_one_pair(
    pred: nn.Linear,
    succ: nn.Linear,
    *,
    rot: int,
) -> None:
    rotation = _normalized_regular_hadamard(rot, device=pred.weight.device)
    weight = pred.weight.detach().to(torch.float32)
    out_features, in_features = weight.shape
    reshaped = weight.reshape(out_features // rot, rot, in_features)
    rotated = torch.einsum(
        "ij,bjk->bik",
        rotation.T.to(dtype=weight.dtype, device=weight.device),
        reshaped,
    )
    pred.weight.data.copy_(
        rotated.reshape(out_features, in_features).to(dtype=pred.weight.dtype)
    )
    if pred.bias is not None:
        bias = pred.bias.detach().to(torch.float32)
        bias_rot = _apply_groupwise_rotation(
            bias.unsqueeze(0),
            rot_size=rot,
            rotation_matrix=rotation,
        ).reshape(-1)
        pred.bias.data.copy_(bias_rot.to(dtype=pred.bias.dtype))
    _set_input_already_rotated(succ, True)


class RotationAbsorbTransform:
    """Absorb groupwise rotation into predecessor Linear weights when safe."""

    name: str = "rotation_absorb"
    required_kernels: tuple[str, ...] = ("hadamard_groupwise",)

    def __init__(self, *, rot_size: int = 32) -> None:
        self.rot_size = int(rot_size)

    def match(self, model: nn.Module) -> TransformPlan | None:
        absorb, online = _collect_absorb_pairs(model, rot_size=self.rot_size)
        if not absorb and not online:
            return None
        targets = tuple(
            f"{p['predecessor']}->{p['successor']}" for p in absorb + online
        )
        return TransformPlan(
            transform_name=self.name,
            targets=targets,
            absorbed_ops=(
                ("groupwise_rotation_into_predecessor_weight",) if absorb else ()
            ),
            online_ops=(
                ("hadamard_groupwise_residual_or_branch",) if online else ()
            ),
            metadata={
                "pairs": absorb,
                "online_pairs": online,
                "rot_size": self.rot_size,
                "hadamard_preflight": preflight_hadamard_kernel(),
            },
        )

    def apply(self, model: nn.Module, plan: TransformPlan) -> TransformReport:
        modules = dict(model.named_modules())
        pairs = plan.metadata.get("pairs", [])
        online_pairs = plan.metadata.get("online_pairs", [])
        absorbed: list[str] = []
        online: list[str] = []
        notes: list[str] = []
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            pred_name = str(pair.get("predecessor", ""))
            succ_name = str(pair.get("successor", ""))
            pred = modules.get(pred_name)
            succ = modules.get(succ_name)
            if not isinstance(pred, nn.Linear) or not isinstance(succ, nn.Linear):
                online.append(f"{pred_name}->{succ_name}")
                notes.append(f"skip_non_linear:{pred_name}->{succ_name}")
                continue
            rot = _pair_rot_ok(pred, succ, self.rot_size)
            if rot is None:
                online.append(f"{pred_name}->{succ_name}")
                notes.append(f"skip_indivisible:{pred_name}->{succ_name}")
                continue
            _absorb_one_pair(pred, succ, rot=rot)
            absorbed.append(f"{pred_name}->{succ_name}")

        for pair in online_pairs:
            if not isinstance(pair, dict):
                continue
            pred_name = str(pair.get("predecessor", ""))
            succ_name = str(pair.get("successor", ""))
            kind = str(pair.get("kind", "residual"))
            online.append(f"{pred_name}->{succ_name}")
            notes.append(f"online_{kind}:{pred_name}->{succ_name}")

        preflight = preflight_hadamard_kernel()
        if online:
            notes.append(f"hadamard_preflight:{preflight['status']}")
            if preflight.get("performance_note"):
                notes.append(str(preflight["performance_note"]))

        applied = bool(absorbed) or bool(online)
        return TransformReport(
            transform_name=self.name,
            applied=applied,
            absorbed_ops=tuple(absorbed),
            online_ops=tuple(online) if online else (),
            required_kernels=() if not online else self.required_kernels,
            targets=plan.targets,
            notes=tuple(notes),
            metadata={
                "rot_size": self.rot_size,
                "absorbed_count": len(absorbed),
                "online_fallback_count": len(online),
                "hadamard_preflight": preflight,
            },
        )


__all__ = [
    "RotationAbsorbTransform",
    "preflight_hadamard_kernel",
]
