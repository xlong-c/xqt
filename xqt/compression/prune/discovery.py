"""Structured-pruning target discovery and plan validation."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from xqt.core.base import XQTConfigError
from xqt.contracts import ModelStructureContract
from xqt.contracts.model_structure import (
    is_module_path_within,
    resolve_structure_role,
    structure_contract_keep_high_precision_paths,
)

from .candidates import (
    _CandidateDiscoveryResult,
    _StructuredPruningAdapter,
    _StructuredCandidate,
    collect_candidates,
)
from .candidates_attention import collect_head_candidates
from .candidates_container import (
    collect_block_candidates,
    collect_cnn_stage_candidates,
)
from .candidates_conv import collect_concat_branch_candidates, collect_conv_candidates
from .candidates_mbconv import collect_mbconv_candidates
from .candidates_mlp import collect_mlp_candidates
from .candidates_moe import collect_expert_candidates
from .candidates_residual import collect_residual_conv_candidates
from .candidates_vit import (
    collect_vit_embedding_width_candidates,
    collect_vit_hidden_width_candidates,
)
from .granularity import (
    describe_prune_granularity,
    normalize_prune_granularity,
    rewrite_supported_granularities,
)
from .report import StructuredPruningAction
from .rewrite import attention_variant, infer_attention_role, validate_conv2d_keep_indices

SUPPORTED_GRANULARITIES = rewrite_supported_granularities()
SUPPORTED_IMPORTANCE_METRICS = ("l1", "l2", "bn_gamma", "usage")
SUPPORTED_SCOPES = ("global", "per_layer")

_PASSTHROUGH_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.SiLU,
    nn.Identity,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.Flatten,
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool2d,
)


def get_module(root: nn.Module, name: str) -> nn.Module:
    """Resolve a module name emitted by structured-pruning discovery."""

    if name in {"", "<root>"}:
        return root
    module = root
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _collect_conv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_conv_candidates(
        model,
        importance_metric=importance_metric,
        get_module=get_module,
        passthrough_types=_PASSTHROUGH_TYPES,
    )


def _collect_concat_branch_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_concat_branch_candidates(
        model,
        importance_metric=importance_metric,
        get_module=get_module,
    )


def _collect_mbconv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_mbconv_candidates(
        model,
        importance_metric=importance_metric,
        get_module=get_module,
    )


def _collect_residual_conv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_residual_conv_candidates(
        model,
        importance_metric=importance_metric,
        get_module=get_module,
    )


def _collect_mlp_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_mlp_candidates(model, importance_metric=importance_metric)


def _collect_head_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_head_candidates(
        model,
        importance_metric=importance_metric,
        infer_attention_role=infer_attention_role,
        attention_variant=attention_variant,
    )


def _collect_vit_hidden_width_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_vit_hidden_width_candidates(
        model,
        importance_metric=importance_metric,
        get_module=get_module,
    )


def _collect_vit_embedding_width_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_vit_embedding_width_candidates(
        model,
        importance_metric=importance_metric,
        get_module=get_module,
    )


def _collect_expert_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_expert_candidates(model, importance_metric=importance_metric)


def _collect_cnn_stage_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_cnn_stage_candidates(model, importance_metric=importance_metric)


def _collect_block_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    return collect_block_candidates(model, importance_metric=importance_metric)


_STRUCTURED_ADAPTERS: tuple[_StructuredPruningAdapter, ...] = (
    _StructuredPruningAdapter(
        name="cnn_chain_adapter",
        granularity="channel",
        structure_family="cnn",
        collect=_collect_conv_candidates,
    ),
    _StructuredPruningAdapter(
        name="mbconv_adapter",
        granularity="channel",
        structure_family="cnn_mbconv",
        collect=_collect_mbconv_candidates,
    ),
    _StructuredPruningAdapter(
        name="concat_branch_adapter",
        granularity="channel",
        structure_family="cnn_branch",
        collect=_collect_concat_branch_candidates,
    ),
    _StructuredPruningAdapter(
        name="residual_cnn_adapter",
        granularity="channel",
        structure_family="cnn_residual",
        collect=_collect_residual_conv_candidates,
    ),
    _StructuredPruningAdapter(
        name="cnn_chain_adapter",
        granularity="filter",
        structure_family="cnn",
        collect=_collect_conv_candidates,
    ),
    _StructuredPruningAdapter(
        name="mlp_pair_adapter",
        granularity="mlp_neuron",
        structure_family="transformer",
        collect=_collect_mlp_candidates,
    ),
    _StructuredPruningAdapter(
        name="attention_adapter",
        granularity="head",
        structure_family="transformer",
        collect=_collect_head_candidates,
    ),
    _StructuredPruningAdapter(
        name="container_adapter",
        granularity="block",
        structure_family="container",
        collect=_collect_block_candidates,
    ),
    _StructuredPruningAdapter(
        name="cnn_stage_adapter",
        granularity="stage",
        structure_family="cnn_stage",
        collect=_collect_cnn_stage_candidates,
    ),
    _StructuredPruningAdapter(
        name="vit_hidden_width_adapter",
        granularity="hidden_width",
        structure_family="transformer_width",
        collect=_collect_vit_hidden_width_candidates,
    ),
    _StructuredPruningAdapter(
        name="vit_embedding_width_adapter",
        granularity="embedding_width",
        structure_family="transformer_width",
        collect=_collect_vit_embedding_width_candidates,
    ),
    _StructuredPruningAdapter(
        name="moe_expert_adapter",
        granularity="expert",
        structure_family="moe_expert",
        collect=_collect_expert_candidates,
    ),
)


def _candidate_touches_protected(
    candidate: _StructuredCandidate,
    protected: tuple[str, ...],
) -> bool:
    """Return whether one candidate's rewrite scope intersects protected paths.

    双向判定: 候选模块位于保护路径下 (候选本身受保护), 或保护模块位于候选
    模块之下 (容器级候选的结构改写会波及受保护子模块, 如删 expert 必然改
    router), 都视为相交并拦截.
    """

    return any(
        is_module_path_within(candidate.module_name, path)
        or is_module_path_within(path, candidate.module_name)
        for path in protected
    )


def _apply_structure_contract(
    result: _CandidateDiscoveryResult,
    contract: ModelStructureContract,
) -> None:
    """Filter and annotate discovery results against a structure contract.

    契约是模型结构的声明式事实源: 与 ``keep_high_precision`` 组件路径相交
    的候选不产出, 记入 blocked/protected; 其余候选标注 ``structure_role``,
    未声明路径保留并标 ``undeclared``, 不静默丢弃.
    """

    protected = structure_contract_keep_high_precision_paths(contract)
    kept: list[_StructuredCandidate] = []
    for candidate in result.candidates:
        if _candidate_touches_protected(candidate, protected):
            result.blocked_modules.append(candidate.module_name)
            result.protected_modules.append(candidate.module_name)
            continue
        candidate.metadata["structure_role"] = (
            resolve_structure_role(contract, candidate.module_name) or "undeclared"
        )
        kept.append(candidate)
    result.candidates = kept


def collect_structured_candidates(
    model: nn.Module,
    *,
    granularity: str,
    importance_metric: str,
    structure_contract: ModelStructureContract | None = None,
) -> _CandidateDiscoveryResult:
    """Collect candidates through the supported model-family adapters.

    ``structure_contract`` 可选: 提供时按契约保护 ``keep_high_precision``
    组件并为候选标注 ``structure_role``; 缺省保持纯启发式发现.
    """

    canonical = normalize_prune_granularity(granularity)
    result = collect_candidates(
        model,
        granularity=canonical,
        importance_metric=importance_metric,
        adapters=_STRUCTURED_ADAPTERS,
        supported_granularities=SUPPORTED_GRANULARITIES,
    )
    if structure_contract is not None:
        from xqt.contracts.model_structure import is_structure_contract_valid_for_model

        if not is_structure_contract_valid_for_model(structure_contract, model):
            raise XQTConfigError(
                "Structure contract is invalid or expired for the given model"
            )
        _apply_structure_contract(result, structure_contract)
    return result


def validate_action_keep_indices(
    model: nn.Module,
    action: StructuredPruningAction,
) -> None:
    """Validate one selected action against the current model topology."""

    if action.action_type == "residual_stage_channels":
        consumers = action.metadata.get("consumers")
        if consumers is None:
            raise ValueError("residual_stage_channels requires metadata.consumers")
        if not isinstance(consumers, list):
            raise ValueError("metadata.consumers must be a list")
        for item in consumers:
            if not isinstance(item, Mapping):
                raise ValueError("metadata.consumers entries must be mappings")
            consumer_name = item.get("name")
            consumer_type = item.get("type")
            if consumer_type != "Conv2d" or not isinstance(consumer_name, str):
                continue
            consumer = get_module(model, consumer_name)
            if not isinstance(consumer, nn.Conv2d):
                raise TypeError(f"{consumer_name} is not a Conv2d module")
            validate_conv2d_keep_indices(consumer, action.keep_indices, axis="in")
        return
    if action.action_type == "attention_heads":
        attention = get_module(model, action.module_name)
        num_heads = getattr(attention, "num_heads", None)
        num_kv_heads = getattr(attention, "num_kv_heads", num_heads)
        if not isinstance(num_heads, int) or not isinstance(num_kv_heads, int):
            raise TypeError(f"{action.module_name} is not a supported attention module")
        variant = str(action.metadata.get("attention_variant", "mha"))
        if variant in {"gqa", "mqa"}:
            if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"{action.module_name} has invalid num_heads/num_kv_heads configuration"
                )
            if action.keep_indices[0] < 0 or action.keep_indices[-1] >= num_kv_heads:
                raise ValueError(
                    f"selection.keep_indices for '{action.module_name}' must reference kv heads"
                )
        return
    if action.action_type == "drop_stages":
        container = get_module(model, action.module_name)
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            raise TypeError(
                f"{action.module_name} is not a ModuleList or Sequential stage container"
            )
        if not bool(action.metadata.get("shape_compatible", False)):
            raise ValueError(
                f"CNN stage pruning candidate '{action.module_name}' is not shape compatible"
            )
        return
    if action.action_type == "vit_hidden_width":
        num_heads = int(action.metadata.get("num_heads", 0))
        if num_heads <= 0:
            raise ValueError("vit_hidden_width requires metadata.num_heads")
        if len(action.keep_indices) % num_heads != 0:
            raise ValueError("vit_hidden_width keep count must be divisible by num_heads")
        return
    if action.action_type == "drop_experts":
        router = get_module(model, str(action.metadata["router_name"]))
        experts = get_module(model, str(action.metadata["experts_name"]))
        if not isinstance(router, nn.Linear):
            raise TypeError("drop_experts requires a Linear router")
        if not isinstance(experts, (nn.ModuleList, nn.Sequential)):
            raise TypeError("drop_experts requires an expert container")
        if router.out_features != len(list(experts.children())):
            raise ValueError("router.out_features must match expert count")
        return
    if action.action_type != "conv_channel_group":
        return
    producer = get_module(model, action.module_name)
    if not isinstance(producer, nn.Conv2d):
        raise TypeError(f"{action.module_name} is not a Conv2d module")
    validate_conv2d_keep_indices(producer, action.keep_indices, axis="out")
    if action.consumer_name is None or action.consumer_type != "Conv2d":
        return
    consumer = get_module(model, action.consumer_name)
    if not isinstance(consumer, nn.Conv2d):
        raise TypeError(f"{action.consumer_name} is not a Conv2d module")
    validate_conv2d_keep_indices(consumer, action.keep_indices, axis="in")


__all__ = [
    "SUPPORTED_GRANULARITIES",
    "SUPPORTED_IMPORTANCE_METRICS",
    "SUPPORTED_SCOPES",
    "collect_structured_candidates",
    "get_module",
    "validate_action_keep_indices",
]
