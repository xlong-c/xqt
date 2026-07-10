"""Internal residual-stage candidate discovery helpers for structured pruning."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from .candidates import _CandidateDiscoveryResult, _StructuredCandidate
from .graph import PruningDependencyGraph


def _residual_block_descriptor(
    module_name: str,
    module: nn.Module,
) -> Optional[dict[str, Any]]:
    conv1 = getattr(module, "conv1", None)
    conv2 = getattr(module, "conv2", None)
    bn1 = getattr(module, "bn1", None)
    bn2 = getattr(module, "bn2", None)
    if (
        isinstance(conv1, nn.Conv2d)
        and isinstance(conv2, nn.Conv2d)
        and isinstance(bn1, nn.modules.batchnorm._BatchNorm)
        and isinstance(bn2, nn.modules.batchnorm._BatchNorm)
        and conv1.out_channels == conv2.out_channels == bn1.num_features == bn2.num_features
    ):
        block_type = type(module).__name__
        if block_type == "BasicBlock":
            return {
                "block_name": module_name,
                "block_type": block_type,
                "out_channels": conv2.out_channels,
                "conv1_name": f"{module_name}.conv1",
                "bn1_name": f"{module_name}.bn1",
                "conv2_name": f"{module_name}.conv2",
                "bn2_name": f"{module_name}.bn2",
                "downsample_conv_name": (
                    f"{module_name}.downsample.0"
                    if isinstance(getattr(module, "downsample", None), nn.Sequential)
                    and len(getattr(module, "downsample")) >= 2
                    and isinstance(module.downsample[0], nn.Conv2d)
                    and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                    else None
                ),
                "downsample_bn_name": (
                    f"{module_name}.downsample.1"
                    if isinstance(getattr(module, "downsample", None), nn.Sequential)
                    and len(getattr(module, "downsample")) >= 2
                    and isinstance(module.downsample[0], nn.Conv2d)
                    and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                    else None
                ),
            }

    conv3 = getattr(module, "conv3", None)
    bn3 = getattr(module, "bn3", None)
    if (
        isinstance(conv1, nn.Conv2d)
        and isinstance(conv2, nn.Conv2d)
        and isinstance(conv3, nn.Conv2d)
        and isinstance(bn1, nn.modules.batchnorm._BatchNorm)
        and isinstance(bn2, nn.modules.batchnorm._BatchNorm)
        and isinstance(bn3, nn.modules.batchnorm._BatchNorm)
        and conv2.out_channels == bn2.num_features
        and conv3.out_channels == bn3.num_features
        and type(module).__name__ == "Bottleneck"
    ):
        return {
            "block_name": module_name,
            "block_type": "Bottleneck",
            "out_channels": conv2.out_channels,
            "conv1_name": f"{module_name}.conv1",
            "bn1_name": f"{module_name}.bn1",
            "conv2_name": f"{module_name}.conv2",
            "bn2_name": f"{module_name}.bn2",
            "conv3_name": f"{module_name}.conv3",
            "bn3_name": f"{module_name}.bn3",
            "project_out_channels": conv3.out_channels,
            "downsample_conv_name": (
                f"{module_name}.downsample.0"
                if isinstance(getattr(module, "downsample", None), nn.Sequential)
                and len(getattr(module, "downsample")) >= 2
                and isinstance(module.downsample[0], nn.Conv2d)
                and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                else None
            ),
            "downsample_bn_name": (
                f"{module_name}.downsample.1"
                if isinstance(getattr(module, "downsample", None), nn.Sequential)
                and len(getattr(module, "downsample")) >= 2
                and isinstance(module.downsample[0], nn.Conv2d)
                and isinstance(module.downsample[1], nn.modules.batchnorm._BatchNorm)
                else None
            ),
        }
    return None


def _find_next_feature_consumer(
    model: nn.Module,
    *,
    module_name: str,
    out_channels: int,
) -> tuple[Optional[str], Optional[str], int]:
    module_prefix = f"{module_name}."
    passed = False
    for name, module in model.named_modules():
        if name == module_name:
            passed = True
            continue
        if not passed:
            continue
        if name.startswith(module_prefix):
            continue
        if isinstance(module, nn.Conv2d) and module.in_channels == out_channels:
            return name, "Conv2d", 1
        if isinstance(module, nn.Linear):
            if module.in_features % out_channels != 0:
                continue
            return name, "Linear", module.in_features // out_channels
    return None, None, 1


def _residual_stage_descriptor(
    stage_name: str,
    stage: nn.Module,
) -> Optional[dict[str, Any]]:
    if not isinstance(stage, nn.Sequential):
        return None
    blocks = list(stage.children())
    if not blocks:
        return None
    block_type = type(blocks[0]).__name__
    if block_type not in {"BasicBlock", "Bottleneck"}:
        return None
    if any(type(block).__name__ != block_type for block in blocks):
        return None

    block_descriptors: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        descriptor = _residual_block_descriptor(f"{stage_name}.{block_index}", block)
        if descriptor is None:
            return None
        block_descriptors.append(descriptor)

    first = block_descriptors[0]
    if first.get("downsample_conv_name") is None or first.get("downsample_bn_name") is None:
        return None

    if block_type == "BasicBlock":
        stage_out_channels = int(first["out_channels"])
        for descriptor in block_descriptors:
            if int(descriptor["out_channels"]) != stage_out_channels:
                return None
    else:
        stage_out_channels = int(first["project_out_channels"])
        for descriptor in block_descriptors:
            if int(descriptor["project_out_channels"]) != stage_out_channels:
                return None

    return {
        "stage_name": stage_name,
        "block_type": block_type,
        "stage_out_channels": stage_out_channels,
        "block_count": len(block_descriptors),
        "blocks": block_descriptors,
    }


def _residual_stage_scores(
    descriptor: Mapping[str, Any],
    model: nn.Module,
    metric: str,
    *,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> torch.Tensor:
    block_type = str(descriptor["block_type"])
    stage_out_channels = int(descriptor["stage_out_channels"])
    score = torch.zeros(stage_out_channels, dtype=torch.float32)
    blocks = descriptor["blocks"]
    if not isinstance(blocks, list):
        raise TypeError("descriptor.blocks must be a list")

    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError("descriptor.blocks items must be mappings")
        if block_type == "BasicBlock":
            conv1 = get_module(model, str(block["conv1_name"]))
            conv2 = get_module(model, str(block["conv2_name"]))
            bn1 = get_module(model, str(block["bn1_name"]))
            bn2 = get_module(model, str(block["bn2_name"]))
            if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d):
                raise TypeError("BasicBlock residual pruning requires conv1/conv2")
            if not isinstance(bn1, nn.modules.batchnorm._BatchNorm) or not isinstance(
                bn2, nn.modules.batchnorm._BatchNorm
            ):
                raise TypeError("BasicBlock residual pruning requires bn1/bn2")
            conv1_weight = conv1.weight.detach().to(dtype=torch.float32, device="cpu")
            conv2_weight = conv2.weight.detach().to(dtype=torch.float32, device="cpu")
            if metric == "bn_gamma":
                score = score + bn2.weight.detach().abs().to(dtype=torch.float32, device="cpu")
                continue
            if metric == "l1":
                score = score + conv1_weight.abs().sum(dim=(1, 2, 3))
                if conv1.in_channels == stage_out_channels:
                    score = score + conv1_weight.abs().sum(dim=(0, 2, 3))
                score = score + conv2_weight.abs().sum(dim=(0, 2, 3))
                score = score + conv2_weight.abs().sum(dim=(1, 2, 3))
                score = score + bn1.weight.detach().abs().to(dtype=torch.float32, device="cpu")
                score = score + bn2.weight.detach().abs().to(dtype=torch.float32, device="cpu")
                continue
            if metric == "l2":
                conv1_out = torch.linalg.vector_norm(
                    conv1_weight.reshape(conv1.out_channels, -1),
                    dim=1,
                ).square()
                conv1_in = torch.zeros_like(conv1_out)
                if conv1.in_channels == stage_out_channels:
                    conv1_in = torch.linalg.vector_norm(
                        conv1_weight.permute(1, 0, 2, 3).reshape(stage_out_channels, -1),
                        dim=1,
                    ).square()
                score = score + torch.sqrt(
                    conv1_out
                    + conv1_in
                    + torch.linalg.vector_norm(
                        conv2_weight.permute(1, 0, 2, 3).reshape(stage_out_channels, -1),
                        dim=1,
                    ).square()
                    + torch.linalg.vector_norm(
                        conv2_weight.reshape(stage_out_channels, -1),
                        dim=1,
                    ).square()
                )
                continue
            raise ValueError(f"Unsupported structured importance metric: {metric}")

        conv1 = get_module(model, str(block["conv1_name"]))
        conv3 = get_module(model, str(block["conv3_name"]))
        bn3 = get_module(model, str(block["bn3_name"]))
        if not isinstance(conv1, nn.Conv2d) or not isinstance(conv3, nn.Conv2d):
            raise TypeError("Bottleneck residual pruning requires conv1/conv3")
        if not isinstance(bn3, nn.modules.batchnorm._BatchNorm):
            raise TypeError("Bottleneck residual pruning requires bn3")
        conv1_weight = conv1.weight.detach().to(dtype=torch.float32, device="cpu")
        conv3_weight = conv3.weight.detach().to(dtype=torch.float32, device="cpu")
        if metric == "bn_gamma":
            score = score + bn3.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            continue
        if metric == "l1":
            if conv1.in_channels == stage_out_channels:
                score = score + conv1_weight.abs().sum(dim=(0, 2, 3))
            score = score + conv3_weight.abs().sum(dim=(1, 2, 3))
            score = score + bn3.weight.detach().abs().to(dtype=torch.float32, device="cpu")
            continue
        if metric == "l2":
            conv1_in = torch.zeros_like(score)
            if conv1.in_channels == stage_out_channels:
                conv1_in = torch.linalg.vector_norm(
                    conv1_weight.permute(1, 0, 2, 3).reshape(stage_out_channels, -1),
                    dim=1,
                ).square()
            score = score + torch.sqrt(
                conv1_in
                + torch.linalg.vector_norm(
                    conv3_weight.reshape(stage_out_channels, -1),
                    dim=1,
                ).square()
            )
            continue
        raise ValueError(f"Unsupported structured importance metric: {metric}")

    if block_type == "Bottleneck":
        first = blocks[0]
        if isinstance(first, Mapping) and first.get("downsample_conv_name") is not None:
            downsample_conv = get_module(model, str(first["downsample_conv_name"]))
            if isinstance(downsample_conv, nn.Conv2d):
                downsample_weight = downsample_conv.weight.detach().to(
                    dtype=torch.float32,
                    device="cpu",
                )
                if metric == "l1":
                    score = score + downsample_weight.abs().sum(dim=(1, 2, 3))
                elif metric == "l2":
                    score = score + torch.sqrt(
                        torch.linalg.vector_norm(
                            downsample_weight.reshape(stage_out_channels, -1),
                            dim=1,
                        ).square()
                    )
    return score


def collect_residual_conv_candidates(
    model: nn.Module,
    *,
    importance_metric: str,
    get_module: Callable[[nn.Module, str], nn.Module],
) -> _CandidateDiscoveryResult:
    """Collect residual-stage channel pruning candidates."""

    candidates: list[_StructuredCandidate] = []
    graph = PruningDependencyGraph()
    stage_descriptors: list[dict[str, Any]] = []
    for stage_name, stage in model.named_modules():
        if not stage_name:
            continue
        descriptor = _residual_stage_descriptor(stage_name, stage)
        if descriptor is not None:
            stage_descriptors.append(descriptor)

    for stage_index, descriptor in enumerate(stage_descriptors):
        module_name = str(descriptor["stage_name"])
        scores = _residual_stage_scores(
            descriptor,
            model,
            importance_metric,
            get_module=get_module,
        )
        consumers: list[dict[str, Any]] = []
        if stage_index + 1 < len(stage_descriptors):
            next_stage = stage_descriptors[stage_index + 1]
            next_blocks = next_stage["blocks"]
            if not isinstance(next_blocks, list) or not next_blocks:
                raise TypeError("next_stage.blocks must be a non-empty list")
            first_block = next_blocks[0]
            if not isinstance(first_block, Mapping):
                raise TypeError("next_stage first block descriptor must be a mapping")
            consumers.append(
                {"name": str(first_block["conv1_name"]), "type": "Conv2d", "feature_block_size": 1}
            )
            if first_block.get("downsample_conv_name") is not None:
                consumers.append(
                    {
                        "name": str(first_block["downsample_conv_name"]),
                        "type": "Conv2d",
                        "feature_block_size": 1,
                    }
                )
        else:
            next_consumer_name, next_consumer_type, feature_block_size = _find_next_feature_consumer(
                model,
                module_name=module_name,
                out_channels=int(descriptor["stage_out_channels"]),
            )
            if next_consumer_name is not None and next_consumer_type is not None:
                consumers.append(
                    {
                        "name": next_consumer_name,
                        "type": next_consumer_type,
                        "feature_block_size": feature_block_size,
                    }
                )

        primary_consumer = consumers[0] if consumers else None
        metadata = dict(descriptor)
        metadata["consumers"] = consumers
        metadata["consumer_name"] = primary_consumer["name"] if primary_consumer is not None else None
        metadata["consumer_type"] = primary_consumer["type"] if primary_consumer is not None else None
        metadata["feature_block_size"] = (
            int(primary_consumer["feature_block_size"]) if primary_consumer is not None else 1
        )
        metadata["merge"] = "add"
        candidates.append(
            _StructuredCandidate(
                adapter="residual_cnn_adapter",
                structure_family="cnn_residual",
                action_type="residual_stage_channels",
                module_name=module_name,
                module_type="Sequential",
                granularity="channel",
                dependency_group=module_name,
                consumer_name=metadata["consumer_name"],
                consumer_type=metadata["consumer_type"],
                normalization_name=None,
                feature_block_size=int(metadata["feature_block_size"]),
                scores=scores,
                metadata=metadata,
            )
        )
        graph.add_group(
            name=module_name,
            producer=module_name,
            consumers=[str(item["name"]) for item in consumers],
            merge="add",
            shape_constraints={
                "block_type": str(descriptor["block_type"]),
                "stage_out_channels": int(descriptor["stage_out_channels"]),
                "block_count": int(descriptor["block_count"]),
                "consumers": [dict(item) for item in consumers],
            },
        )
    return _CandidateDiscoveryResult(candidates=candidates, dependency_graph=graph)
