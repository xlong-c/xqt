"""torch.export candidate scanning."""

from __future__ import annotations

from typing import Any

import torch
from torch import fx, nn

from ._graph_utils import (
    _FLATTEN_SUFFIXES,
    _RESHAPE_SUFFIXES,
    _ancestor_subclass_keys,
    _ancestor_support_nodes,
    _is_target,
    _matches_any_suffix,
    _normalize_target,
)
from .candidate import _build_candidate, OperatorPatternCandidate
from .finders import (
    _find_epilogue_path,
    _find_qdq_epilogue_nodes,
    _find_rmsnorm_nodes,
    _find_rmsnorm_residual_nodes,
    _find_rope_nodes,
    _find_user_path,
)


def _match_export_patterns(graph_module: fx.GraphModule) -> list[OperatorPatternCandidate]:
    candidates: list[OperatorPatternCandidate] = []
    nodes = list(graph_module.graph.nodes)
    for index, node in enumerate(nodes):
        target_name = _normalize_target(node.target)
        if node.op == "call_function" and target_name.endswith("aten.gelu.default"):
            previous = nodes[index - 1] if index > 0 else None
            if previous is not None and _normalize_target(previous.target).endswith("aten.add.Tensor"):
                candidates.append(
                    _build_candidate(
                        "bias_gelu",
                        "torch_export",
                        node,
                        [previous, node],
                        estimated_kernel_count=2,
                    )
                )
        if node.op == "call_function" and target_name.endswith("aten.mul.Tensor"):
            rmsnorm_nodes = _find_rmsnorm_nodes(node)
            if rmsnorm_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rmsnorm",
                        "torch_export",
                        node,
                        rmsnorm_nodes,
                        estimated_kernel_count=len(rmsnorm_nodes),
                    )
                )
            rmsnorm_nodes = _find_rmsnorm_residual_nodes(node)
            if rmsnorm_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rmsnorm_residual",
                        "torch_export",
                        node,
                        rmsnorm_nodes,
                        estimated_kernel_count=len(rmsnorm_nodes),
                    )
                )
            previous = nodes[index - 1] if index > 0 else None
            if previous is not None and _normalize_target(previous.target).endswith("aten.silu.default"):
                candidates.append(
                    _build_candidate(
                        "swiglu",
                        "torch_export",
                        node,
                        [previous, node],
                        estimated_kernel_count=2,
                    )
                )
        if node.op == "call_function" and _matches_any_suffix(node, _FLATTEN_SUFFIXES):
            rope_nodes = _find_rope_nodes(node)
            if rope_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rope",
                        "torch_export",
                        node,
                        rope_nodes,
                        estimated_kernel_count=len(rope_nodes),
                    )
                )
        if node.op == "call_function" and target_name.endswith("aten.scaled_dot_product_attention.default"):
            candidates.append(
                _build_candidate(
                    "attention",
                    "torch_export",
                    node,
                    [node],
                    estimated_kernel_count=1,
                )
            )
        if node.op == "call_function" and (
            target_name.endswith("aten.linear.default")
            or target_name.endswith("aten.matmul.default")
            or target_name.endswith("aten.mm.default")
        ):
            dequant_nodes = [
                arg
                for arg in node.args
                if isinstance(arg, fx.Node)
                and _is_target(arg, "aten.mul.Tensor")
                and any(
                    isinstance(source, fx.Node) and _is_target(source, "aten.to.dtype")
                    for source in arg.all_input_nodes
                )
            ]
            if dequant_nodes:
                candidates.append(
                    _build_candidate(
                        "dequant_gemm",
                        "torch_export",
                        node,
                        [*dequant_nodes, node],
                        estimated_kernel_count=3,
                    )
                )
                epilogue_nodes = _find_epilogue_path(
                    nodes,
                    node,
                    passthrough_suffixes=_RESHAPE_SUFFIXES,
                )
                if epilogue_nodes:
                    candidates.append(
                        _build_candidate(
                            "dequant_gemm_epilogue",
                            "torch_export",
                            epilogue_nodes[-1],
                            [*dequant_nodes, node, *epilogue_nodes],
                            estimated_kernel_count=3 + len(epilogue_nodes),
                        )
                    )
                qdq_nodes = _find_qdq_epilogue_nodes(nodes, node)
                if qdq_nodes is not None:
                    candidates.append(
                        _build_candidate(
                            "qdq_epilogue",
                            "torch_export",
                            qdq_nodes[-1],
                            [node, *qdq_nodes],
                            estimated_kernel_count=1 + len(qdq_nodes),
                        )
                    )
            ancestry_keys = _ancestor_subclass_keys(node)
            if "int_data" in ancestry_keys:
                scale_path = _find_user_path(
                    nodes,
                    node,
                    target_suffixes=("mul.Tensor",),
                )
                if scale_path is not None:
                    epilogue_nodes = _find_epilogue_path(
                        nodes,
                        scale_path[-1],
                        passthrough_suffixes=_RESHAPE_SUFFIXES,
                    )
                    if epilogue_nodes:
                        support_nodes = _ancestor_support_nodes(
                            nodes,
                            node,
                            {"int_data", "scale"},
                        )
                        candidates.append(
                            _build_candidate(
                                "weight_only_matmul_epilogue",
                                "torch_export",
                                epilogue_nodes[-1],
                                [*support_nodes, node, *scale_path, *epilogue_nodes],
                                estimated_kernel_count=1 + len(scale_path) + len(epilogue_nodes),
                            )
                        )
            if "qdata" in ancestry_keys and "scale" in ancestry_keys:
                epilogue_nodes = _find_epilogue_path(
                    nodes,
                    node,
                    passthrough_suffixes=_RESHAPE_SUFFIXES,
                )
                if epilogue_nodes:
                    support_nodes = _ancestor_support_nodes(
                        nodes,
                        node,
                        {"qdata", "scale"},
                    )
                    candidates.append(
                        _build_candidate(
                            "fp8_scale_cast_matmul_epilogue",
                            "torch_export",
                            epilogue_nodes[-1],
                            [*support_nodes, node, *epilogue_nodes],
                            estimated_kernel_count=1 + len(epilogue_nodes),
                        )
                    )
    return candidates


def scan_export_candidates(
    model: nn.Module,
    example_input: Any,
) -> list[OperatorPatternCandidate]:
    """Scan a torch.export graph for operator optimization candidates."""

    if not hasattr(torch, "export"):
        raise RuntimeError("torch.export is not available in the current PyTorch build")
    args = example_input if isinstance(example_input, tuple) else (example_input,)
    exported = torch.export.export(model.eval(), args)
    return _match_export_patterns(exported.module())
