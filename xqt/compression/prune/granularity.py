"""Canonical prune granularity vocabulary and alias normalization.

The roadmap requires finer structured-pruning granularity names
(``channel``, ``filter``, ``mlp_neuron``, ``attention_head``, ``block``,
``token``) with one obvious canonical spelling per concept.  This module is
the vocabulary fact source: aliases normalize to a canonical name, and every
granularity declares whether XQT rewrites structure or only records metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PruneGranularitySpec:
    """One canonical prune granularity entry."""

    canonical: str
    rewrites_structure: bool
    exportable: bool
    runtime_support: str
    module_types: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_granularity": self.canonical,
            "rewrites_structure": self.rewrites_structure,
            "exportable": self.exportable,
            "runtime_support": self.runtime_support,
            "module_types": list(self.module_types),
            "model_families": list(self.model_families),
            "aliases": list(self.aliases),
            "notes": list(self.notes),
        }


_GRANULARITY_SPECS: tuple[PruneGranularitySpec, ...] = (
    PruneGranularitySpec(
        "channel",
        True,
        True,
        "pytorch_rewrite",
        ("Conv2d", "BatchNorm2d"),
        ("ConvNet", "Detection", "MBConv"),
        (),
        ("Rewrites Conv/BN channel dimensions; latency must be benchmarked per backend.",),
    ),
    PruneGranularitySpec(
        "filter",
        True,
        True,
        "pytorch_rewrite",
        ("Conv2d",),
        ("ConvNet", "Detection"),
        (),
        ("Filter pruning is represented as output-channel rewrite.",),
    ),
    PruneGranularitySpec(
        "mlp_neuron",
        True,
        True,
        "pytorch_rewrite",
        ("Linear",),
        ("Transformer", "ViT", "LLM"),
        (),
        ("Pairs producer/consumer MLP Linear modules; gated MLP keeps gate/up/down aligned.",),
    ),
    PruneGranularitySpec(
        "head",
        True,
        True,
        "pytorch_rewrite",
        ("Linear", "Attention"),
        ("Transformer", "ViT", "LLM"),
        ("attention_head",),
        ("Attention head pruning rewrites q/k/v/out projections when adapters can prove topology.",),
    ),
    PruneGranularitySpec(
        "block",
        True,
        True,
        "pytorch_rewrite",
        ("Sequential", "ModuleList"),
        ("ConvNet", "Transformer"),
        (),
        ("Container/block pruning requires homogeneous or adapter-supported child topology.",),
    ),
    PruneGranularitySpec(
        "stage",
        True,
        True,
        "pytorch_rewrite",
        ("Sequential", "ModuleList"),
        ("ConvNet",),
        (),
        ("CNN stage pruning drops shape-compatible stage children.",),
    ),
    PruneGranularitySpec(
        "token",
        False,
        False,
        "metadata_only",
        ("metadata",),
        ("Transformer", "Multimodal"),
        (),
        (
            "Token pruning is tracked as a model-family plan item; "
            "no generic XQT runtime rewrite yet.",
        ),
    ),
    PruneGranularitySpec(
        "expert",
        True,
        True,
        "pytorch_rewrite",
        ("ModuleList", "Linear"),
        ("MoE",),
        (),
        ("Expert pruning rewrites expert list and router output where adapter proves layout.",),
    ),
    PruneGranularitySpec(
        "hidden_width",
        True,
        True,
        "pytorch_rewrite",
        ("Linear", "Conv2d", "LayerNorm"),
        ("ViT", "Transformer"),
        (),
        ("Width pruning preserves model-wide alignment constraints such as head groups.",),
    ),
    PruneGranularitySpec(
        "embedding_width",
        True,
        True,
        "pytorch_rewrite",
        ("Conv2d", "Linear", "LayerNorm"),
        ("ViT",),
        (),
        ("Embedding-width pruning rewrites patch projection, normalization, and head shape.",),
    ),
    PruneGranularitySpec(
        "nm",
        False,
        True,
        "cuda_sparse_candidate",
        ("Linear", "Conv2d"),
        ("generic",),
        (),
        (
            "2:4 pattern may accelerate on NVIDIA sparse kernels; "
            "XQT reports compliance separately from speedup.",
        ),
    ),
    PruneGranularitySpec(
        "block_sparse",
        False,
        False,
        "metadata_only",
        ("Linear", "Conv2d"),
        ("generic",),
        (),
        ("Block-sparse masks are reported; no generic sparse runtime backend is wired.",),
    ),
)

_CANONICAL_SPECS: dict[str, PruneGranularitySpec] = {
    spec.canonical: spec for spec in _GRANULARITY_SPECS
}
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: spec.canonical for spec in _GRANULARITY_SPECS for alias in spec.aliases
}


def normalize_prune_granularity(granularity: str) -> str:
    """Normalize an alias to its canonical granularity name."""

    raw = str(granularity).strip().lower()
    if raw in _CANONICAL_SPECS:
        return raw
    canonical = _ALIAS_TO_CANONICAL.get(raw)
    if canonical is not None:
        return canonical
    allowed = ", ".join(_CANONICAL_SPECS)
    raise ValueError(f"Unsupported prune granularity '{granularity}'. Allowed: {allowed}")


def describe_prune_granularity(granularity: str) -> dict[str, Any]:
    """Return the canonical spec dict for a granularity name or alias."""

    canonical = normalize_prune_granularity(granularity)
    return _CANONICAL_SPECS[canonical].to_dict()


def supported_prune_granularities() -> list[dict[str, Any]]:
    """Return all canonical granularity specs."""

    return [spec.to_dict() for spec in _GRANULARITY_SPECS]


def rewrite_supported_granularities() -> tuple[str, ...]:
    """Return granularities with a real structural rewrite in XQT."""

    return tuple(
        spec.canonical for spec in _GRANULARITY_SPECS if spec.rewrites_structure
    )


__all__ = [
    "PruneGranularitySpec",
    "describe_prune_granularity",
    "normalize_prune_granularity",
    "rewrite_supported_granularities",
    "supported_prune_granularities",
]
