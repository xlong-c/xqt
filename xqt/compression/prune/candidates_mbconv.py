"""Internal MBConv candidate discovery helpers for structured pruning."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


def _mbconv_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    expand = getattr(module, "expand_conv", None)
    expand_bn = getattr(module, "expand_bn", None)
    depthwise = getattr(module, "depthwise_conv", None)
    depthwise_bn = getattr(module, "depthwise_bn", None)
    project = getattr(module, "project_conv", None)
    project_bn = getattr(module, "project_bn", None)
    if not (
        isinstance(expand, nn.Conv2d)
        and isinstance(expand_bn, nn.modules.batchnorm._BatchNorm)
        and isinstance(depthwise, nn.Conv2d)
        and isinstance(depthwise_bn, nn.modules.batchnorm._BatchNorm)
        and isinstance(project, nn.Conv2d)
        and isinstance(project_bn, nn.modules.batchnorm._BatchNorm)
    ):
        return None
    mid_channels = int(expand.out_channels)
    if (
        expand.groups != 1
        or expand_bn.num_features != mid_channels
        or depthwise.in_channels != mid_channels
        or depthwise.out_channels != mid_channels
        or depthwise.groups != mid_channels
        or depthwise_bn.num_features != mid_channels
        or project.in_channels != mid_channels
    ):
        return None
    return {
        "module_name": module_name,
        "mid_channels": mid_channels,
        "expand_conv_name": f"{module_name}.expand_conv",
        "expand_bn_name": f"{module_name}.expand_bn",
        "depthwise_conv_name": f"{module_name}.depthwise_conv",
        "depthwise_bn_name": f"{module_name}.depthwise_bn",
        "project_conv_name": f"{module_name}.project_conv",
        "project_bn_name": f"{module_name}.project_bn",
    }


def _mbconv_scores(
    descriptor: Mapping[str, Any],
    model: nn.Module,
    metric: str,
    *,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> torch.Tensor:
    expand_conv = get_module(model, str(descriptor["expand_conv_name"]))
    expand_bn = get_module(model, str(descriptor["expand_bn_name"]))
    depthwise_conv = get_module(model, str(descriptor["depthwise_conv_name"]))
    depthwise_bn = get_module(model, str(descriptor["depthwise_bn_name"]))
    project_conv = get_module(model, str(descriptor["project_conv_name"]))
    if not all(
        isinstance(module, nn.Conv2d)
        for module in (expand_conv, depthwise_conv, project_conv)
    ):
        raise TypeError("MBConv pruning requires Conv2d modules")
    if not all(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in (expand_bn, depthwise_bn)
    ):
        raise TypeError("MBConv pruning requires BatchNorm modules")
    expand_weight = expand_conv.weight.detach().to(dtype=torch.float32, device="cpu")
    depthwise_weight = depthwise_conv.weight.detach().to(dtype=torch.float32, device="cpu")
    project_weight = project_conv.weight.detach().to(dtype=torch.float32, device="cpu")
    if metric == "bn_gamma":
        return (
            expand_bn.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            + depthwise_bn.weight.detach().abs().to(dtype=torch.float32, device="cpu")
        )
    if metric == "l1":
        return (
            expand_weight.abs().sum(dim=(1, 2, 3))
            + depthwise_weight.abs().sum(dim=(1, 2, 3))
            + project_weight.abs().sum(dim=(0, 2, 3))
        )
    if metric == "l2":
        return torch.sqrt(
            torch.linalg.vector_norm(
                expand_weight.reshape(expand_conv.out_channels, -1),
                dim=1,
            ).square()
            + torch.linalg.vector_norm(
                depthwise_weight.reshape(depthwise_conv.out_channels, -1),
                dim=1,
            ).square()
            + torch.linalg.vector_norm(
                project_weight.permute(1, 0, 2, 3).reshape(project_conv.in_channels, -1),
                dim=1,
            ).square()
        )
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def collect_mbconv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> _CandidateDiscoveryResult:
    """Collect MBConv mid-channel pruning candidates."""

    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        descriptor = _mbconv_descriptor(module_name, module)
        if descriptor is None:
            continue
        scores = _mbconv_scores(
            descriptor,
            model,
            importance_metric,
            get_module=get_module,
        )
        candidates.append(
            _StructuredCandidate(
                adapter="mbconv_adapter",
                structure_family="cnn_mbconv",
                action_type="mbconv_mid_channels",
                module_name=module_name,
                module_type=type(module).__name__,
                granularity="channel",
                dependency_group=module_name,
                consumer_name=str(descriptor["project_conv_name"]),
                consumer_type="Conv2d",
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata=dict(descriptor),
            )
        )
        graph.add_group(
            name=module_name,
            producer=str(descriptor["expand_conv_name"]),
            consumers=[str(descriptor["project_conv_name"])],
            merge=None,
            shape_constraints={
                "mid_channels": int(descriptor["mid_channels"]),
                "depthwise_groups": int(descriptor["mid_channels"]),
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)
