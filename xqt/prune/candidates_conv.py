"""Internal Conv2d-based candidate discovery helpers for structured pruning."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


def _named_leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    leaves: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not any(module.children()):
            leaves.append((name, module))
    return leaves


def _is_chain_like_module_tree(model: nn.Module) -> bool:
    for name, module in model.named_modules():
        if not name:
            continue
        if any(module.children()) and not isinstance(module, nn.Sequential):
            return False
    return True


def _channel_scores(
    module: nn.Conv2d,
    normalization: Optional[nn.modules.batchnorm._BatchNorm],
    metric: str,
) -> torch.Tensor:
    if metric == "bn_gamma":
        if normalization is None or normalization.weight is None:
            raise ValueError("importance.metric=bn_gamma requires a following BatchNorm")
        return normalization.weight.detach().abs().to(dtype=torch.float32, device="cpu")

    weight = module.weight.detach().to(dtype=torch.float32, device="cpu")
    flattened = weight.reshape(weight.shape[0], -1)
    if metric == "l1":
        return flattened.abs().sum(dim=1)
    if metric == "l2":
        return torch.linalg.vector_norm(flattened, dim=1)
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _find_consumer(
    leaves: list[tuple[str, nn.Module]],
    start_index: int,
    producer: nn.Conv2d,
    *,
    passthrough_types: Sequence[type[nn.Module]],
) -> tuple[Optional[str], Optional[str], Optional[str], int]:
    normalization_name: Optional[str] = None
    current_index = start_index + 1

    if current_index < len(leaves):
        maybe_norm_name, maybe_norm_module = leaves[current_index]
        if isinstance(maybe_norm_module, nn.modules.batchnorm._BatchNorm):
            if maybe_norm_module.num_features != producer.out_channels:
                raise ValueError(
                    f"BatchNorm '{maybe_norm_name}' does not match producer "
                    f"'{leaves[start_index][0]}' out_channels"
                )
            normalization_name = maybe_norm_name
            current_index += 1

    while current_index < len(leaves):
        consumer_name, consumer_module = leaves[current_index]
        if isinstance(consumer_module, nn.Conv2d):
            if consumer_module.in_channels != producer.out_channels:
                raise ValueError(
                    f"Consumer '{consumer_name}' in_channels do not match "
                    f"producer '{leaves[start_index][0]}' out_channels"
                )
            return consumer_name, "Conv2d", normalization_name, 1
        if isinstance(consumer_module, nn.Linear):
            if consumer_module.in_features % producer.out_channels != 0:
                raise ValueError(
                    f"Linear consumer '{consumer_name}' in_features must be divisible by "
                    f"producer '{leaves[start_index][0]}' out_channels"
                )
            block_size = consumer_module.in_features // producer.out_channels
            return consumer_name, "Linear", normalization_name, block_size
        if not isinstance(consumer_module, tuple(passthrough_types)):
            raise ValueError(
                f"Unsupported module '{consumer_name}' of type "
                f"{type(consumer_module).__name__} between structured pruning candidates"
            )
        current_index += 1

    return None, None, normalization_name, 1


def _is_depthwise_conv2d(module: nn.Conv2d) -> bool:
    return module.groups == module.in_channels == module.out_channels


def _conv2d_group_alignment_constraint(
    module: nn.Conv2d,
    *,
    module_name: str,
    axis: str,
) -> Optional[dict[str, Any]]:
    if axis not in {"in", "out"}:
        raise ValueError("axis must be 'in' or 'out'")
    if module.groups == 1 or _is_depthwise_conv2d(module):
        return None
    total_channels = module.in_channels if axis == "in" else module.out_channels
    if total_channels % module.groups != 0:
        raise ValueError(
            f"Grouped Conv2d '{module_name}' must have {axis}_channels divisible by groups"
        )
    return {
        "module_name": module_name,
        "axis": axis,
        "groups": int(module.groups),
        "total_channels": int(total_channels),
        "channels_per_group": int(total_channels // module.groups),
    }


def _conv_channel_group_constraints(
    model: nn.Module,
    *,
    producer_name: str,
    producer: nn.Conv2d,
    consumer_name: Optional[str],
    consumer_type: Optional[str],
    get_module: Callable[[nn.Module, str], nn.Module],
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    producer_constraint = _conv2d_group_alignment_constraint(
        producer,
        module_name=producer_name,
        axis="out",
    )
    if producer_constraint is not None:
        constraints.append(producer_constraint)
    if consumer_name is not None and consumer_type == "Conv2d":
        consumer = get_module(model, consumer_name)
        if not isinstance(consumer, nn.Conv2d):
            raise TypeError(f"{consumer_name} is not a Conv2d module")
        consumer_constraint = _conv2d_group_alignment_constraint(
            consumer,
            module_name=consumer_name,
            axis="in",
        )
        if consumer_constraint is not None:
            constraints.append(consumer_constraint)
    return constraints


def _concat_branch_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    prefix = "" if module_name in {"", "<root>"} else f"{module_name}."
    branch_specs: list[dict[str, Any]] = []
    for branch_index, branch_attr in enumerate(("branch1", "branch2")):
        branch_module = getattr(module, branch_attr, None)
        if not isinstance(branch_module, nn.Sequential):
            return None
        children = list(branch_module.children())
        if not children or not isinstance(children[0], nn.Conv2d):
            return None
        conv = children[0]
        if conv.groups != 1:
            return None
        bn_name: Optional[str] = None
        if len(children) >= 2:
            maybe_bn = children[1]
            if isinstance(maybe_bn, nn.modules.batchnorm._BatchNorm):
                if maybe_bn.num_features != conv.out_channels:
                    return None
                bn_name = f"{module_name}.{branch_attr}.1"
        branch_specs.append(
            {
                "branch_attr": branch_attr,
                "branch_index": branch_index,
                "conv_name": f"{prefix}{branch_attr}.0",
                "bn_name": (
                    f"{prefix}{branch_attr}.1" if bn_name is not None else None
                ),
                "out_channels": int(conv.out_channels),
            }
        )

    consumer = getattr(module, "fuse", None)
    if not isinstance(consumer, nn.Conv2d) or consumer.groups != 1:
        return None
    total_out_channels = sum(int(spec["out_channels"]) for spec in branch_specs)
    if consumer.in_channels != total_out_channels:
        return None
    return {
        "module_name": module_name,
        "branch_specs": branch_specs,
        "consumer_name": f"{prefix}fuse",
        "consumer_type": "Conv2d",
        "total_out_channels": total_out_channels,
    }


def collect_conv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
    get_module: Callable[[nn.Module, str], nn.Module],
    passthrough_types: Sequence[type[nn.Module]],
) -> _CandidateDiscoveryResult:
    """Collect chain-like Conv2d channel pruning candidates."""

    if not _is_chain_like_module_tree(model):
        return _CandidateDiscoveryResult()
    leaves = _named_leaf_modules(model)
    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    blocked_producers: set[str] = set()
    for module_name, module in model.named_modules():
        descriptor_module_name = module_name or "<root>"
        descriptor = _concat_branch_descriptor(descriptor_module_name, module)
        if descriptor is None:
            continue
        for branch_spec in descriptor["branch_specs"]:
            if isinstance(branch_spec, Mapping):
                conv_name = branch_spec.get("conv_name")
                if isinstance(conv_name, str):
                    blocked_producers.add(conv_name)

    for index, (module_name, module) in enumerate(leaves):
        if not isinstance(module, nn.Conv2d):
            continue
        if module_name in blocked_producers:
            continue
        try:
            consumer_name, consumer_type, normalization_name, feature_block_size = _find_consumer(
                leaves,
                index,
                module,
                passthrough_types=passthrough_types,
            )
        except ValueError:
            continue
        if consumer_name is None or consumer_type is None:
            continue
        normalization = None
        if normalization_name is not None:
            normalization = get_module(model, normalization_name)
            if not isinstance(normalization, nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{normalization_name} is not a BatchNorm module")
        scores = _channel_scores(module, normalization, importance_metric)
        group_constraints = _conv_channel_group_constraints(
            model,
            producer_name=module_name,
            producer=module,
            consumer_name=consumer_name,
            consumer_type=consumer_type,
            get_module=get_module,
        )
        candidates.append(
            _StructuredCandidate(
                adapter="cnn_chain_adapter",
                structure_family="cnn",
                action_type="conv_channel_group",
                module_name=module_name,
                module_type="Conv2d",
                granularity="channel",
                dependency_group=module_name,
                consumer_name=consumer_name,
                consumer_type=consumer_type,
                normalization_name=normalization_name,
                feature_block_size=feature_block_size,
                scores=scores,
                metadata={
                    "normalization_name": normalization_name,
                    "consumer_name": consumer_name,
                    "consumer_type": consumer_type,
                    "feature_block_size": feature_block_size,
                    "group_alignment_constraints": [dict(item) for item in group_constraints],
                },
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[consumer_name],
            merge=None,
            shape_constraints={
                "consumer_type": consumer_type,
                "feature_block_size": feature_block_size,
                "normalization_name": normalization_name,
                "group_alignment_constraints": [dict(item) for item in group_constraints],
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def collect_concat_branch_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> _CandidateDiscoveryResult:
    """Collect concat-branch Conv2d channel pruning candidates."""

    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        descriptor_module_name = module_name or "<root>"
        descriptor = _concat_branch_descriptor(descriptor_module_name, module)
        if descriptor is None:
            continue
        branch_specs = descriptor["branch_specs"]
        consumer_name = str(descriptor["consumer_name"])
        for branch_spec in branch_specs:
            conv_name = str(branch_spec["conv_name"])
            conv = get_module(model, conv_name)
            if not isinstance(conv, nn.Conv2d):
                raise TypeError(f"{conv_name} is not a Conv2d module")
            bn_name = branch_spec.get("bn_name")
            normalization = None
            if isinstance(bn_name, str):
                normalization = get_module(model, bn_name)
                if not isinstance(normalization, nn.modules.batchnorm._BatchNorm):
                    raise TypeError(f"{bn_name} is not a BatchNorm module")
            scores = _channel_scores(conv, normalization, importance_metric)
            metadata = {
                "container_name": module_name,
                "branch_index": int(branch_spec["branch_index"]),
                "branch_attr": str(branch_spec["branch_attr"]),
                "branch_specs": [dict(item) for item in branch_specs],
                "merge": "concat",
                "consumer_name": consumer_name,
                "consumer_type": "Conv2d",
            }
            candidates.append(
                _StructuredCandidate(
                    adapter="concat_branch_adapter",
                    structure_family="cnn_branch",
                    action_type="concat_branch_channels",
                    module_name=conv_name,
                    module_type="Conv2d",
                    granularity="channel",
                    dependency_group=f"{module_name}:{branch_spec['branch_attr']}",
                    consumer_name=consumer_name,
                    consumer_type="Conv2d",
                    normalization_name=bn_name if isinstance(bn_name, str) else None,
                    feature_block_size=1,
                    scores=scores,
                    metadata=metadata,
                )
            )
            graph.add_group(
                name=f"{module_name}:{branch_spec['branch_attr']}",
                producer=conv_name,
                consumers=[consumer_name],
                merge="concat",
                shape_constraints={
                    "branch_index": int(branch_spec["branch_index"]),
                    "branch_attr": str(branch_spec["branch_attr"]),
                    "branch_specs": [dict(item) for item in branch_specs],
                    "consumer_name": consumer_name,
                },
            )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)
