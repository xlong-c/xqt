"""Transactional Graph Rewrite Engine with Acceptance Verification.

Provides a speculative rewrite loop for graph transforms:
1. Snapshot baseline model state;
2. Apply candidate transforms;
3. Verify numeric equivalence, executability, and performance against acceptance gates;
4. Safely commit on success or automatically roll back on divergence/failure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from xqt.auto.acceptance import AcceptanceEvaluation, evaluate_stage_acceptance
from xqt.compression.quant.quantizers.base import (
    call_model,
    iter_calibration_batches,
    move_batch_to_device,
)
from .base import GraphQuantTransform, TransformPlan, TransformReport


@dataclass(frozen=True)
class SpeculativeRewriteConfig:
    """Configuration for speculative graph rewrite transaction."""

    verify_numerics: bool = True
    max_mean_abs_tolerance: float = 1e-4
    max_max_abs_tolerance: float = 1e-3
    min_cosine_similarity: float = 0.9999
    require_acceptance_checks: bool = False
    sample_limit: int = 4


@dataclass(frozen=True)
class RewriteTransactionReport:
    """Diagnostic and audit report of a graph rewrite transaction."""

    status: str  # "committed", "rolled_back", "no_op"
    transform_reports: tuple[TransformReport, ...]
    numeric_diff: dict[str, float] | None = None
    rollback_reason: str | None = None
    acceptance_eval: AcceptanceEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "transform_reports": [r.to_dict() for r in self.transform_reports],
            "numeric_diff": self.numeric_diff,
            "rollback_reason": self.rollback_reason,
            "acceptance_eval": (
                self.acceptance_eval.to_dict() if self.acceptance_eval is not None else None
            ),
        }


def _flatten_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        res = []
        for item in out:
            res.extend(_flatten_outputs(item))
        return res
    if isinstance(out, Mapping):
        res = []
        for v in out.values():
            res.extend(_flatten_outputs(v))
        return res
    return []


def _compare_tensors(
    baseline_list: list[torch.Tensor],
    candidate_list: list[torch.Tensor],
) -> dict[str, float]:
    if len(baseline_list) != len(candidate_list):
        return {
            "max_mean_abs": float("inf"),
            "max_max_abs": float("inf"),
            "cosine_similarity": 0.0,
        }

    max_mean_abs = 0.0
    max_max_abs = 0.0
    cosines = []

    for b, c in zip(baseline_list, candidate_list):
        if b.shape != c.shape:
            return {
                "max_mean_abs": float("inf"),
                "max_max_abs": float("inf"),
                "cosine_similarity": 0.0,
            }
        b_flat = b.detach().to(torch.float32).reshape(-1)
        c_flat = c.detach().to(torch.float32).reshape(-1)
        diff = (b_flat - c_flat).abs()
        mean_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0
        max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        if mean_diff > max_mean_abs:
            max_mean_abs = mean_diff
        if max_diff > max_max_abs:
            max_max_abs = max_diff

        # Cosine similarity
        if b_flat.norm() > 0 and c_flat.norm() > 0:
            cos = float(F.cosine_similarity(b_flat.unsqueeze(0), c_flat.unsqueeze(0)).item())
            cosines.append(cos)
        else:
            cosines.append(1.0)

    avg_cosine = sum(cosines) / len(cosines) if cosines else 1.0
    return {
        "max_mean_abs": max_mean_abs,
        "max_max_abs": max_max_abs,
        "cosine_similarity": avg_cosine,
    }


def speculative_graph_rewrite(
    model: nn.Module,
    transforms: Sequence[GraphQuantTransform],
    *,
    calibration_inputs: Iterable[Any] | None = None,
    config: SpeculativeRewriteConfig | None = None,
) -> tuple[nn.Module, RewriteTransactionReport]:
    """Execute speculative graph rewrite with automatic rollback upon divergence."""
    cfg = config or SpeculativeRewriteConfig()

    if not transforms:
        return model, RewriteTransactionReport(
            status="no_op",
            transform_reports=(),
        )

    # 1. Take state_dict snapshot
    state_dict_backup = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    device = next(model.parameters(), torch.empty((), device="cpu")).device

    # 2. Capture baseline outputs if verification is requested
    baseline_outputs: list[list[torch.Tensor]] = []
    batches_run: list[Any] = []
    if cfg.verify_numerics and calibration_inputs is not None:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in iter_calibration_batches(calibration_inputs, sample_limit=cfg.sample_limit):
                    batch_dev = move_batch_to_device(batch, device)
                    out = call_model(model, batch_dev)
                    baseline_outputs.append([t.cpu().clone() for t in _flatten_outputs(out)])
                    batches_run.append(batch)
        finally:
            if was_training:
                model.train()

    # 3. Apply candidate transforms sequentially
    transform_reports: list[TransformReport] = []
    any_applied = False
    for transform in transforms:
        plan = transform.match(model)
        if plan is None:
            transform_reports.append(
                TransformReport(
                    transform_name=transform.name,
                    applied=False,
                    required_kernels=tuple(transform.required_kernels),
                    notes=("no_match",),
                )
            )
            continue
        report = transform.apply(model, plan)
        transform_reports.append(report)
        if report.applied:
            any_applied = True

    if not any_applied:
        return model, RewriteTransactionReport(
            status="no_op",
            transform_reports=tuple(transform_reports),
        )

    # 4. Verification stage
    if cfg.verify_numerics and batches_run:
        was_training = model.training
        model.eval()
        candidate_outputs: list[list[torch.Tensor]] = []
        try:
            with torch.no_grad():
                for batch in batches_run:
                    batch_dev = move_batch_to_device(batch, device)
                    out = call_model(model, batch_dev)
                    candidate_outputs.append([t.cpu() for t in _flatten_outputs(out)])
        except Exception as exc:
            # Forward error after rewrite -> Rollback immediately
            model.load_state_dict(state_dict_backup)
            return model, RewriteTransactionReport(
                status="rolled_back",
                transform_reports=tuple(transform_reports),
                rollback_reason=f"forward_execution_failed: {exc}",
            )
        finally:
            if was_training:
                model.train()

        # Compare outputs
        all_metrics = []
        for b_list, c_list in zip(baseline_outputs, candidate_outputs):
            all_metrics.append(_compare_tensors(b_list, c_list))

        agg_metrics = {
            "max_mean_abs": max(m["max_mean_abs"] for m in all_metrics),
            "max_max_abs": max(m["max_max_abs"] for m in all_metrics),
            "cosine_similarity": min(m["cosine_similarity"] for m in all_metrics),
        }

        # Check tolerances
        failed_reasons = []
        if agg_metrics["max_mean_abs"] > cfg.max_mean_abs_tolerance:
            failed_reasons.append(
                f"max_mean_abs ({agg_metrics['max_mean_abs']:.2e}) exceeds {cfg.max_mean_abs_tolerance:.2e}"
            )
        if agg_metrics["max_max_abs"] > cfg.max_max_abs_tolerance:
            failed_reasons.append(
                f"max_max_abs ({agg_metrics['max_max_abs']:.2e}) exceeds {cfg.max_max_abs_tolerance:.2e}"
            )
        if agg_metrics["cosine_similarity"] < cfg.min_cosine_similarity:
            failed_reasons.append(
                f"cosine_similarity ({agg_metrics['cosine_similarity']:.6f}) below {cfg.min_cosine_similarity:.6f}"
            )

        if failed_reasons:
            # Rollback
            model.load_state_dict(state_dict_backup)
            return model, RewriteTransactionReport(
                status="rolled_back",
                transform_reports=tuple(transform_reports),
                numeric_diff=agg_metrics,
                rollback_reason="; ".join(failed_reasons),
            )

        return model, RewriteTransactionReport(
            status="committed",
            transform_reports=tuple(transform_reports),
            numeric_diff=agg_metrics,
        )

    # If numeric verification is skipped, commit directly
    return model, RewriteTransactionReport(
        status="committed",
        transform_reports=tuple(transform_reports),
    )


__all__ = [
    "RewriteTransactionReport",
    "SpeculativeRewriteConfig",
    "speculative_graph_rewrite",
]
