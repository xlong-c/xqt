"""Internal container candidate discovery helpers for structured pruning."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


def _block_scores(
    block: nn.Module,
    metric: str,
) -> float:
    parameters = [
        parameter.detach().to(dtype=torch.float32, device="cpu").flatten()
        for parameter in block.parameters()
    ]
    if not parameters:
        return 0.0
    flat = torch.cat(parameters)
    if metric == "l1":
        return float(flat.abs().sum().item())
    if metric == "l2":
        return float(torch.linalg.vector_norm(flat).item())
    raise ValueError(f"Unsupported structured importance metric: {metric}")


def _is_cnn_stage_passthrough_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
            nn.Identity,
            nn.Dropout,
            nn.Dropout2d,
            nn.Dropout3d,
        ),
    )


def _conv2d_preserves_spatial_shape(module: nn.Conv2d) -> bool:
    if module.stride != (1, 1):
        return False
    if isinstance(module.padding, str):
        return module.padding == "same"
    return all(
        2 * int(module.padding[index])
        == int(module.dilation[index]) * (int(module.kernel_size[index]) - 1)
        for index in range(2)
    )


def _cnn_stage_descriptor(
    stage_name: str,
    stage: nn.Module,
) -> Optional[dict[str, Any]]:
    leaves = [
        (leaf_name, leaf_module)
        for leaf_name, leaf_module in stage.named_modules()
        if leaf_name and not any(leaf_module.children())
    ]
    if not leaves:
        return None

    convs: list[tuple[str, nn.Conv2d]] = []
    batchnorm_features: list[int] = []
    for leaf_name, leaf_module in leaves:
        if isinstance(leaf_module, nn.Conv2d):
            if not _conv2d_preserves_spatial_shape(leaf_module):
                return None
            convs.append((leaf_name, leaf_module))
            continue
        if isinstance(leaf_module, nn.modules.batchnorm._BatchNorm):
            batchnorm_features.append(int(leaf_module.num_features))
            continue
        if _is_cnn_stage_passthrough_module(leaf_module):
            continue
        return None

    if not convs:
        return None
    input_channels = int(convs[0][1].in_channels)
    output_channels = int(convs[-1][1].out_channels)
    if input_channels != output_channels:
        return None
    conv_output_channels = {int(conv.out_channels) for _, conv in convs}
    if any(num_features not in conv_output_channels for num_features in batchnorm_features):
        return None
    return {
        "stage_name": stage_name,
        "stage_type": type(stage).__name__,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "conv_count": len(convs),
        "conv_names": [f"{stage_name}.{leaf_name}" for leaf_name, _ in convs],
        "shape_compatible": True,
    }


def collect_cnn_stage_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    """Collect compatible CNN stage pruning candidates."""

    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        named_children = list(module.named_children())
        if len(named_children) <= 1:
            continue

        descriptors: list[dict[str, Any]] = []
        for child_name, child in named_children:
            descriptor = _cnn_stage_descriptor(f"{module_name}.{child_name}", child)
            if descriptor is None:
                descriptors = []
                break
            descriptors.append(descriptor)
        if not descriptors:
            continue

        input_channels = {int(descriptor["input_channels"]) for descriptor in descriptors}
        output_channels = {int(descriptor["output_channels"]) for descriptor in descriptors}
        if len(input_channels) != 1 or len(output_channels) != 1:
            continue
        if input_channels != output_channels:
            continue

        stage_types = [str(descriptor["stage_type"]) for descriptor in descriptors]
        heterogeneous_children = len(set(stage_types)) != 1
        children = [child for _, child in named_children]
        scores = torch.tensor(
            [_block_scores(child, importance_metric) for child in children],
            dtype=torch.float32,
        )
        metadata = {
            "container_type": type(module).__name__,
            "stage_type": stage_types[0] if not heterogeneous_children else "heterogeneous",
            "stage_types": list(stage_types),
            "heterogeneous_children": heterogeneous_children,
            "stage_count": len(descriptors),
            "stage_names": [str(descriptor["stage_name"]) for descriptor in descriptors],
            "input_channels": input_channels.pop(),
            "output_channels": output_channels.pop(),
            "shape_compatible": True,
            "compatibility_rule": "same input/output channels and spatial-preserving Conv2d leaves",
            "stage_descriptors": [dict(descriptor) for descriptor in descriptors],
        }
        candidates.append(
            _StructuredCandidate(
                adapter="cnn_stage_adapter",
                structure_family="cnn_stage",
                action_type="drop_stages",
                module_name=module_name,
                module_type=type(module).__name__,
                granularity="stage",
                dependency_group=module_name,
                consumer_name=None,
                consumer_type=None,
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata=metadata,
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[],
            merge=None,
            shape_constraints={
                "container_type": type(module).__name__,
                "stage_count": len(descriptors),
                "input_channels": int(metadata["input_channels"]),
                "output_channels": int(metadata["output_channels"]),
                "shape_compatible": True,
                "compatibility_rule": str(metadata["compatibility_rule"]),
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)


def _is_composite_block_module(module: nn.Module) -> bool:
    return any(True for _ in module.children())


def collect_block_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
) -> _CandidateDiscoveryResult:
    """Collect composite block container pruning candidates."""

    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        children = list(module.children())
        if len(children) <= 1:
            continue
        if not all(_is_composite_block_module(child) for child in children):
            continue
        block_types = [child.__class__.__name__ for child in children]
        heterogeneous_children = len(set(block_types)) != 1
        scores = torch.tensor(
            [_block_scores(child, importance_metric) for child in children],
            dtype=torch.float32,
        )
        candidates.append(
            _StructuredCandidate(
                adapter="container_adapter",
                structure_family="container",
                action_type="drop_blocks",
                module_name=module_name,
                module_type=type(module).__name__,
                granularity="block",
                dependency_group=module_name,
                consumer_name=None,
                consumer_type=None,
                normalization_name=None,
                feature_block_size=1,
                scores=scores,
                metadata={
                    "container_type": type(module).__name__,
                    "block_type": block_types[0] if not heterogeneous_children else "heterogeneous",
                    "block_types": list(block_types),
                    "heterogeneous_children": heterogeneous_children,
                },
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[],
            merge=None,
            shape_constraints={
                "container_type": type(module).__name__,
                "block_type": block_types[0] if not heterogeneous_children else "heterogeneous",
                "block_types": list(block_types),
                "heterogeneous_children": heterogeneous_children,
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)
