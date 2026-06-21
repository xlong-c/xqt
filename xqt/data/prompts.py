"""Prompt data helpers for diffusion-facing XQT recipes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from xqt.diffusion_distill.spec import PromptRecord


@dataclass
class PromptBatch:
    """Normalized in-memory prompt entry."""

    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    guidance_scale: float | None = None
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    condition_image: str | None = None
    condition_mask: str | None = None
    reference_image: str | None = None
    latent_cache_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_record(self) -> PromptRecord:
        """Convert the normalized batch entry to a diffusion prompt record."""

        metadata = dict(self.metadata)
        optional_fields = {
            "guidance_scale": self.guidance_scale,
            "steps": self.steps,
            "width": self.width,
            "height": self.height,
            "condition_image": self.condition_image,
            "condition_mask": self.condition_mask,
            "reference_image": self.reference_image,
            "latent_cache_key": self.latent_cache_key,
        }
        for key, value in optional_fields.items():
            if value is not None:
                metadata.setdefault(key, value)
        return PromptRecord(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            guidance_scale=self.guidance_scale,
            steps=self.steps,
            width=self.width,
            height=self.height,
            condition_image=self.condition_image,
            condition_mask=self.condition_mask,
            reference_image=self.reference_image,
            latent_cache_key=self.latent_cache_key,
            metadata=metadata,
        )


def _mapping_to_prompt_batch(item: Mapping[str, Any]) -> PromptBatch:
    condition = item.get("condition")
    condition_mapping = condition if isinstance(condition, Mapping) else {}

    def _pick_str(*keys: str) -> str | None:
        for key in keys:
            value = item.get(key)
            if value is None and condition_mapping:
                value = condition_mapping.get(key)
            if value is not None:
                return str(value)
        return None

    prompt = item.get("prompt")
    if prompt is None:
        raise ValueError("prompt mapping requires a 'prompt' field")

    reserved_keys = {
        "prompt",
        "negative_prompt",
        "seed",
        "guidance_scale",
        "steps",
        "width",
        "height",
        "condition",
        "condition_image",
        "condition_mask",
        "reference_image",
        "latent_cache_key",
        "image",
        "mask",
    }
    metadata = {
        str(key): value
        for key, value in item.items()
        if key not in reserved_keys
    }
    return PromptBatch(
        prompt=str(prompt),
        negative_prompt=(
            None
            if item.get("negative_prompt") is None
            else str(item.get("negative_prompt"))
        ),
        seed=None if item.get("seed") is None else int(item.get("seed")),
        guidance_scale=(
            None
            if item.get("guidance_scale") is None
            else float(item.get("guidance_scale"))
        ),
        steps=None if item.get("steps") is None else int(item.get("steps")),
        width=None if item.get("width") is None else int(item.get("width")),
        height=None if item.get("height") is None else int(item.get("height")),
        condition_image=_pick_str("condition_image", "image"),
        condition_mask=_pick_str("condition_mask", "mask"),
        reference_image=_pick_str("reference_image"),
        latent_cache_key=_pick_str("latent_cache_key"),
        metadata=metadata,
    )


def build_prompt_list(
    prompts: Sequence[str | Mapping[str, Any] | PromptBatch | PromptRecord],
    *,
    sample_limit: int | None = None,
) -> list[PromptBatch]:
    """Build an in-memory prompt list from strings, mappings, or prompt objects."""

    built: list[PromptBatch] = []
    for item in prompts:
        if isinstance(item, PromptBatch):
            built.append(item)
        elif isinstance(item, PromptRecord):
            built.append(
                PromptBatch(
                    prompt=item.prompt,
                    negative_prompt=item.negative_prompt,
                    seed=item.seed,
                    guidance_scale=item.guidance_scale,
                    steps=item.steps,
                    width=item.width,
                    height=item.height,
                    condition_image=item.condition_image,
                    condition_mask=item.condition_mask,
                    reference_image=item.reference_image,
                    latent_cache_key=item.latent_cache_key,
                    metadata=dict(item.metadata),
                )
            )
        elif isinstance(item, str):
            built.append(PromptBatch(prompt=item))
        elif isinstance(item, Mapping):
            built.append(_mapping_to_prompt_batch(item))
        else:
            raise TypeError(
                "prompt items must be strings, mappings, PromptBatch, or PromptRecord"
            )
        if sample_limit is not None and len(built) >= sample_limit:
            break
    return built


def _load_prompt_payload(path: Path) -> Iterable[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, Mapping):
        prompts = payload.get("prompts")
        if isinstance(prompts, Sequence):
            return prompts
        raise ValueError("prompt JSON object must contain a 'prompts' array")
    if isinstance(payload, Sequence):
        return payload
    raise ValueError("prompt file must contain a JSON array or object with 'prompts'")


def build_prompt_list_from_file(
    path: str | Path,
    *,
    sample_limit: int | None = None,
) -> list[PromptBatch]:
    """Build prompts from a JSON or JSONL file."""

    resolved_path = Path(path).expanduser()
    return build_prompt_list(_load_prompt_payload(resolved_path), sample_limit=sample_limit)


def prompt_summary(prompts: Sequence[PromptBatch]) -> dict[str, Any]:
    """Summarize prompt role data for metrics and manifests."""

    prompt_list = list(prompts)
    return {
        "count": len(prompt_list),
        "has_seed": any(item.seed is not None for item in prompt_list),
        "has_negative_prompt": any(item.negative_prompt is not None for item in prompt_list),
        "has_condition_image": any(item.condition_image is not None for item in prompt_list),
        "has_condition_mask": any(item.condition_mask is not None for item in prompt_list),
        "has_reference_image": any(item.reference_image is not None for item in prompt_list),
        "has_latent_cache_key": any(
            item.latent_cache_key is not None for item in prompt_list
        ),
        "sample_prompts": [item.prompt for item in prompt_list[:3]],
    }


__all__ = [
    "PromptBatch",
    "build_prompt_list",
    "build_prompt_list_from_file",
    "prompt_summary",
]
