"""HF text classification data-role builders for XQT."""

from __future__ import annotations

from typing import Any, Mapping

from xqt.distill.hf_text import (
    HFTextClassificationDataSpec,
    build_hf_text_classification_data,
)


def build_hf_text_classification_loader(
    split_name: str,
    *,
    model_params: Mapping[str, Any] | None = None,
    split_params: Mapping[str, Any] | None = None,
    batch_size: int = 1,
    sample_limit: int | None = None,
):
    """Build a role-specific HF text classification dataloader."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    merged = dict(model_params or {})
    merged.update(dict(split_params or {}))

    train_sample_limit = sample_limit if split_name == "train" else None
    validation_sample_limit = sample_limit if split_name != "train" else None
    spec = HFTextClassificationDataSpec(
        dataset_name=str(merged.get("dataset_name", "")),
        dataset_config_name=merged.get("dataset_config_name"),
        text_column=str(merged.get("text_column", "text")),
        label_column=str(merged.get("label_column", "label")),
        max_length=int(merged.get("max_length", 128)),
        train_split=str(merged.get("train_split", "train")),
        validation_split=str(merged.get("validation_split", "validation")),
        tokenizer_name_or_path=merged.get("tokenizer_name_or_path"),
        teacher_name_or_path=merged.get("teacher_name_or_path"),
        student_name_or_path=merged.get("student_name_or_path"),
        train_sample_limit=train_sample_limit,
        validation_sample_limit=validation_sample_limit,
    )
    bundle = build_hf_text_classification_data(
        spec,
        train_batch_size=batch_size,
        validation_batch_size=batch_size,
    )
    if split_name == "train":
        return bundle.train_loader
    if bundle.validation_loader is None:
        raise ValueError(f"{split_name} split requires a validation loader")
    return bundle.validation_loader


__all__ = ["build_hf_text_classification_loader"]
