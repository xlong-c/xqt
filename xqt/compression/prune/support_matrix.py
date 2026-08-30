from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PruneGranularitySupport:
    granularity: str
    method: str
    module_types: tuple[str, ...]
    model_families: tuple[str, ...]
    rewrites_structure: bool
    exportable: bool
    speedup_verified: bool
    runtime_support: str
    notes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "method": self.method,
            "module_types": list(self.module_types),
            "model_families": list(self.model_families),
            "rewrites_structure": self.rewrites_structure,
            "exportable": self.exportable,
            "speedup_verified": self.speedup_verified,
            "runtime_support": self.runtime_support,
            "notes": list(self.notes),
            "aliases": list(self.aliases),
        }


_STRUCTURED_SUPPORT_ROWS: tuple[PruneGranularitySupport, ...] = (
    PruneGranularitySupport("channel", "structured", ("Conv2d", "BatchNorm2d"), ("ConvNet", "Detection", "MBConv"), True, True, False, "pytorch_rewrite", ("Rewrites Conv/BN channel dimensions; latency must be benchmarked per backend.",)),
    PruneGranularitySupport("filter", "structured", ("Conv2d",), ("ConvNet", "Detection"), True, True, False, "pytorch_rewrite", ("Filter pruning is represented as output-channel rewrite.",)),
    PruneGranularitySupport("mlp_neuron", "structured", ("Linear",), ("Transformer", "ViT", "LLM"), True, True, False, "pytorch_rewrite", ("Pairs producer/consumer MLP Linear modules; gated MLP keeps gate/up/down aligned.",)),
    PruneGranularitySupport("head", "structured", ("Linear", "Attention"), ("Transformer", "ViT", "LLM"), True, True, False, "pytorch_rewrite", ("Attention head pruning rewrites q/k/v/out projections when adapters can prove topology.",), ("attention_head",)),
    PruneGranularitySupport("block", "structured", ("Sequential", "ModuleList"), ("ConvNet", "Transformer"), True, True, False, "pytorch_rewrite", ("Container/block pruning requires homogeneous or adapter-supported child topology.",)),
    PruneGranularitySupport("token", "structured", ("metadata",), ("Transformer", "Multimodal"), False, False, False, "metadata_only", ("Token pruning is tracked as a model-family plan item; no generic XQT runtime rewrite yet.",)),
    PruneGranularitySupport("expert", "structured", ("ModuleList", "Linear"), ("MoE",), True, True, False, "pytorch_rewrite", ("Expert pruning rewrites expert list and router output where adapter proves layout.",)),
    PruneGranularitySupport("hidden_width", "structured", ("Linear", "Conv2d", "LayerNorm"), ("ViT", "Transformer"), True, True, False, "pytorch_rewrite", ("Width pruning preserves model-wide alignment constraints such as head groups.",)),
    PruneGranularitySupport("embedding_width", "structured", ("Conv2d", "Linear", "LayerNorm"), ("ViT",), True, True, False, "pytorch_rewrite", ("Embedding-width pruning rewrites patch projection, normalization, and head shape.",)),
    PruneGranularitySupport("nm", "nm_structured", ("Linear", "Conv2d"), ("generic",), False, True, False, "cuda_sparse_candidate", ("2:4 pattern may accelerate on NVIDIA sparse kernels; XQT reports compliance separately from speedup.",)),
    PruneGranularitySupport("block_sparse", "block_sparse", ("Linear", "Conv2d"), ("generic",), False, False, False, "metadata_only", ("Block-sparse masks are reported; no generic sparse runtime backend is wired.",)),
)


def structured_prune_support_matrix(*, include_aliases: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _STRUCTURED_SUPPORT_ROWS:
        payload = row.to_dict()
        rows.append(payload)
        if include_aliases:
            for alias in row.aliases:
                alias_payload = dict(payload)
                alias_payload["granularity"] = alias
                alias_payload["canonical_granularity"] = row.granularity
                rows.append(alias_payload)
    return rows


__all__ = ["PruneGranularitySupport", "structured_prune_support_matrix"]
