"""FX symbolic-trace candidate scanning."""

from __future__ import annotations

import operator
from typing import Any

import torch
from torch import fx, nn

from ._graph_utils import _FLATTEN_SUFFIXES, _matches_any_suffix, _normalize_target
from .candidate import _build_candidate, OperatorPatternCandidate
from .finders import (
    _find_rmsnorm_nodes,
    _find_rmsnorm_residual_nodes,
    _find_rope_nodes,
)


def _match_fx_patterns(graph_module: fx.GraphModule) -> list[OperatorPatternCandidate]:
    candidates: list[OperatorPatternCandidate] = []
    nodes = list(graph_module.graph.nodes)
    for index, node in enumerate(nodes):
        target_name = _normalize_target(node.target)
        if node.op == "call_function" and target_name.endswith("gelu"):
            previous = nodes[index - 1] if index > 0 else None
            if previous is not None and previous.op == "call_function":
                previous_target = _normalize_target(previous.target)
                if previous.target is operator.add or previous_target.endswith("add"):
                    candidates.append(
                        _build_candidate(
                            "bias_gelu",
                            "fx",
                            node,
                            [previous, node],
                            estimated_kernel_count=2,
                        )
                    )
        if node.op == "call_function" and target_name.endswith("mul"):
            rmsnorm_nodes = _find_rmsnorm_nodes(node)
            if rmsnorm_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rmsnorm",
                        "fx",
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
                        "fx",
                        node,
                        rmsnorm_nodes,
                        estimated_kernel_count=len(rmsnorm_nodes),
                    )
                )
            mul_args = [arg for arg in node.args if isinstance(arg, fx.Node)]
            silu_node = next(
                (
                    arg
                    for arg in mul_args
                    if arg.op == "call_function"
                    and _normalize_target(arg.target).endswith("silu")
                ),
                None,
            )
            if silu_node is not None:
                candidates.append(
                    _build_candidate(
                        "swiglu",
                        "fx",
                        node,
                        [silu_node, node],
                        estimated_kernel_count=2,
                    )
                )
        if node.op in {"call_function", "call_method"} and _matches_any_suffix(
            node,
            _FLATTEN_SUFFIXES,
        ):
            rope_nodes = _find_rope_nodes(node)
            if rope_nodes is not None:
                candidates.append(
                    _build_candidate(
                        "rope",
                        "fx",
                        node,
                        rope_nodes,
                        estimated_kernel_count=len(rope_nodes),
                    )
                )
        if node.op == "call_function" and (
            target_name.endswith("linear") or target_name.endswith("matmul")
        ):
            candidates.append(
                _build_candidate(
                    "linear_gemm",
                    "fx",
                    node,
                    [node],
                    estimated_kernel_count=1,
                )
            )
            dequant_nodes = [
                arg
                for arg in node.args
                if isinstance(arg, fx.Node)
                and _normalize_target(arg.target).endswith("mul")
                and any(
                    isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("to")
                    for source in arg.all_input_nodes
                )
            ]
            if dequant_nodes:
                candidates.append(
                    _build_candidate(
                        "dequant_gemm",
                        "fx",
                        node,
                        [*dequant_nodes, node],
                        estimated_kernel_count=3,
                    )
                )
        if node.op == "call_module":
            module = graph_module.get_submodule(str(node.target))
            try:
                import xqt.kernels.nn as xqt_nn

                is_xqt_attention = isinstance(module, xqt_nn.Attention)
            except Exception:
                is_xqt_attention = False
            if isinstance(module, nn.Linear):
                candidates.append(
                    _build_candidate(
                        "linear_gemm",
                        "fx",
                        node,
                        [node],
                        estimated_kernel_count=1,
                    )
                )
            if isinstance(module, nn.MultiheadAttention) or is_xqt_attention:
                candidates.append(
                    _build_candidate(
                        "attention",
                        "fx",
                        node,
                        [node],
                        estimated_kernel_count=1,
                    )
                )
        if node.op == "call_function" and (
            target_name.endswith("gelu")
            or target_name.endswith("relu")
            or target_name.endswith("silu")
            or target_name.endswith("add")
        ):
            previous = nodes[index - 1] if index > 0 else None
            if previous is None:
                continue
            if not _normalize_target(previous.target).endswith("mul"):
                continue
            clamp_node = next(
                (
                    source
                    for source in previous.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("clamp")
                ),
                None,
            )
            if clamp_node is None:
                continue
            round_node = next(
                (
                    source
                    for source in clamp_node.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("round")
                ),
                None,
            )
            if round_node is None:
                continue
            div_node = next(
                (
                    source
                    for source in round_node.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("truediv")
                ),
                None,
            )
            if div_node is None:
                continue
            linear_node = next(
                (
                    source
                    for source in div_node.all_input_nodes
                    if isinstance(source, fx.Node)
                    and _normalize_target(source.target).endswith("linear")
                ),
                None,
            )
            if linear_node is None:
                continue
            candidates.append(
                _build_candidate(
                    "qdq_epilogue",
                    "fx",
                    node,
                    [linear_node, div_node, round_node, clamp_node, previous, node],
                    estimated_kernel_count=6,
                )
            )
    return candidates


def scan_fx_candidates(
    model: nn.Module,
    example_input: Any,
) -> list[OperatorPatternCandidate]:
    """Scan a symbolic FX graph for operator optimization candidates."""

    traced = fx.symbolic_trace(model.eval())
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        args = example_input if isinstance(example_input, tuple) else (example_input,)
        ShapeProp(traced).propagate(*args)
    except Exception:
        pass
    return _match_fx_patterns(traced)
