"""Detection postprocess and decoded diff helpers for XQT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from dataclasses import dataclass as _dataclass
from typing import Any as _Any


@_dataclass
class DetectionPrediction:
    """One image worth of decoded detections (xqt-local, no xdl dependency)."""

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def to_dict(self) -> dict[str, _Any]:
        return {
            "boxes": self.boxes.detach().cpu().tolist(),
            "scores": self.scores.detach().cpu().tolist(),
            "labels": self.labels.detach().cpu().tolist(),
        }


@dataclass
class DecodedDetectionDiff:
    """Comparison between two decoded detection result sets."""

    box_mae: float
    score_mae: float
    label_match_rate: float
    prediction_count_reference: int
    prediction_count_candidate: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "box_mae": self.box_mae,
            "score_mae": self.score_mae,
            "label_match_rate": self.label_match_rate,
            "prediction_count_reference": self.prediction_count_reference,
            "prediction_count_candidate": self.prediction_count_candidate,
        }


def compare_decoded_detections(
    reference: Sequence[DetectionPrediction],
    candidate: Sequence[DetectionPrediction],
) -> DecodedDetectionDiff:
    """Compare two decoded detection result sets without ground-truth targets."""

    image_count = min(len(reference), len(candidate))
    if image_count == 0:
        return DecodedDetectionDiff(
            box_mae=0.0,
            score_mae=0.0,
            label_match_rate=1.0,
            prediction_count_reference=sum(item.boxes.shape[0] for item in reference),
            prediction_count_candidate=sum(item.boxes.shape[0] for item in candidate),
        )

    box_diffs: list[torch.Tensor] = []
    score_diffs: list[torch.Tensor] = []
    label_matches: list[torch.Tensor] = []
    for ref_item, cand_item in zip(
        reference[:image_count],
        candidate[:image_count],
        strict=False,
    ):
        compare_count = min(ref_item.boxes.shape[0], cand_item.boxes.shape[0])
        if compare_count == 0:
            continue
        box_diffs.append(
            (ref_item.boxes[:compare_count] - cand_item.boxes[:compare_count])
            .abs()
            .reshape(-1)
        )
        score_diffs.append(
            (ref_item.scores[:compare_count] - cand_item.scores[:compare_count])
            .abs()
            .reshape(-1)
        )
        label_matches.append(
            (ref_item.labels[:compare_count] == cand_item.labels[:compare_count]).float()
        )

    return DecodedDetectionDiff(
        box_mae=float(torch.cat(box_diffs).mean().item()) if box_diffs else 0.0,
        score_mae=float(torch.cat(score_diffs).mean().item()) if score_diffs else 0.0,
        label_match_rate=(
            float(torch.cat(label_matches).mean().item()) if label_matches else 1.0
        ),
        prediction_count_reference=sum(item.boxes.shape[0] for item in reference),
        prediction_count_candidate=sum(item.boxes.shape[0] for item in candidate),
    )


__all__ = [
    "DecodedDetectionDiff",
    "compare_decoded_detections",
]
