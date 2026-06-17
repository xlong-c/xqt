"""HuggingFace text classification helpers for XQT recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from xqt.core.errors import XQTBackendError

from .training import DistillationTrainReport, train_logit_distillation


@dataclass
class HFTextClassificationSpec:
    """Settings for a HuggingFace sequence classification recipe."""

    teacher_name_or_path: str
    student_name_or_path: str
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 128
    num_labels: Optional[int] = None
    train_split: str = "train"
    validation_split: str = "validation"
    batch_size: int = 8
    sample_limit: Optional[int] = None
    tokenizer_name_or_path: Optional[str] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HFTextClassificationBundle:
    """Built HF text classification objects."""

    teacher: nn.Module
    student: nn.Module
    tokenizer: Any
    train_loader: DataLoader[Any]
    validation_loader: Optional[DataLoader[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def _import_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
        )
    except ImportError as exc:
        raise XQTBackendError(
            "transformers is required for HF text classification recipes"
        ) from exc
    return AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


def _import_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise XQTBackendError("datasets is required for HF text classification recipes") from exc
    return load_dataset


def _maybe_select(dataset: Any, sample_limit: Optional[int]) -> Any:
    if sample_limit is None:
        return dataset
    return dataset.select(range(min(int(sample_limit), len(dataset))))


def _remove_columns(dataset: Any, keep_columns: Sequence[str]) -> list[str]:
    return [column for column in dataset.column_names if column not in set(keep_columns)]


def build_hf_text_classification_bundle(
    spec: HFTextClassificationSpec,
) -> HFTextClassificationBundle:
    """Build teacher, student, tokenizer, and dataloaders for text KD/prune recipes."""

    (
        auto_model_for_sequence_classification,
        auto_tokenizer,
        data_collator_with_padding,
    ) = _import_transformers()
    load_dataset = _import_datasets()

    tokenizer_name = spec.tokenizer_name_or_path or spec.teacher_name_or_path
    tokenizer = auto_tokenizer.from_pretrained(tokenizer_name)
    model_kwargs = dict(spec.model_kwargs)
    if spec.num_labels is not None:
        model_kwargs["num_labels"] = spec.num_labels

    teacher = auto_model_for_sequence_classification.from_pretrained(
        spec.teacher_name_or_path,
        **model_kwargs,
    )
    student = auto_model_for_sequence_classification.from_pretrained(
        spec.student_name_or_path,
        **model_kwargs,
    )

    raw = load_dataset(spec.dataset_name, spec.dataset_config_name)
    train_dataset = _maybe_select(raw[spec.train_split], spec.sample_limit)
    validation_dataset = raw.get(spec.validation_split)
    if validation_dataset is not None:
        validation_dataset = _maybe_select(validation_dataset, spec.sample_limit)

    def tokenize(batch: Mapping[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            batch[spec.text_column],
            truncation=True,
            max_length=spec.max_length,
        )
        encoded["labels"] = batch[spec.label_column]
        return encoded

    keep = ["input_ids", "attention_mask", "labels"]
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=_remove_columns(train_dataset, keep),
    )
    if validation_dataset is not None:
        validation_dataset = validation_dataset.map(
            tokenize,
            batched=True,
            remove_columns=_remove_columns(validation_dataset, keep),
        )

    collator = data_collator_with_padding(tokenizer=tokenizer)
    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=spec.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader: Optional[DataLoader[Any]] = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    return HFTextClassificationBundle(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        metadata={
            "teacher_name_or_path": spec.teacher_name_or_path,
            "student_name_or_path": spec.student_name_or_path,
            "dataset_name": spec.dataset_name,
            "train_split": spec.train_split,
            "validation_split": spec.validation_split,
        },
    )


def build_hf_text_classification_bundle_from_params(
    params: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> HFTextClassificationBundle:
    """Build an HF text classification bundle from YAML params."""

    merged = dict(params or {})
    merged.update(kwargs)
    return build_hf_text_classification_bundle(HFTextClassificationSpec(**merged))


def train_hf_text_classification_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str | torch.device = "cpu",
    max_steps: Optional[int] = None,
) -> DistillationTrainReport:
    """Train HF sequence classification models with logit KD."""

    return train_logit_distillation(
        student,
        teacher,
        dataloader,
        optimizer,
        temperature=temperature,
        alpha=alpha,
        device=device,
        max_steps=max_steps,
    )


__all__ = [
    "HFTextClassificationBundle",
    "HFTextClassificationSpec",
    "build_hf_text_classification_bundle",
    "build_hf_text_classification_bundle_from_params",
    "train_hf_text_classification_distillation",
]
