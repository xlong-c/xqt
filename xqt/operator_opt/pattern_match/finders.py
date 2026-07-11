"""Pattern finders: QDQ, epilogue, RMSNorm, RoPE."""

from __future__ import annotations

from typing import Iterable, Optional

from torch import fx

from ._graph_utils import (
    _ADD_SUFFIXES,
    _ACTIVATION_SUFFIXES,
    _FLATTEN_SUFFIXES,
    _MEAN_SUFFIXES,
    _MUL_SUFFIXES,
    _POW_SUFFIXES,
    _RSQRT_SUFFIXES,
    _SLICE_SUFFIXES,
    _STACK_SUFFIXES,
    _find_first_user_with_suffix,
    _find_node_users,
    _is_target,
    _matches_any_suffix,
)


def _find_qdq_epilogue_nodes(
    nodes: list[fx.Node],
    linear_node: fx.Node,
) -> Optional[list[fx.Node]]:
    div_node = _find_first_user_with_suffix(nodes, linear_node, "div.Tensor")
    if div_node is None:
        return None
    round_node = _find_first_user_with_suffix(nodes, div_node, "round.default")
    if round_node is None:
        return None
    clamp_node = _find_first_user_with_suffix(nodes, round_node, "clamp.default")
    if clamp_node is None:
        return None
    requant_mul = _find_first_user_with_suffix(nodes, clamp_node, "mul.Tensor")
    if requant_mul is None:
        return None
    epilogue = next(
        (
            node
            for node in nodes
            if requant_mul in node.all_input_nodes
            and (
                _is_target(node, "add.Tensor")
                or _is_target(node, "gelu.default")
                or _is_target(node, "relu.default")
                or _is_target(node, "silu.default")
            )
        ),
        None,
    )
    if epilogue is None:
        return None
    return [div_node, round_node, clamp_node, requant_mul, epilogue]


def _find_user_path(
    nodes: list[fx.Node],
    start: fx.Node,
    *,
    target_suffixes: Iterable[str],
    passthrough_suffixes: Iterable[str] = (),
) -> Optional[list[fx.Node]]:
    pending: list[tuple[fx.Node, list[fx.Node]]] = [(start, [])]
    visited: set[fx.Node] = {start}
    while pending:
        current, path = pending.pop(0)
        for user in _find_node_users(nodes, current):
            if user in visited:
                continue
            visited.add(user)
            next_path = [*path, user]
            if _matches_any_suffix(user, target_suffixes):
                return next_path
            if _matches_any_suffix(user, passthrough_suffixes):
                pending.append((user, next_path))
    return None


def _find_epilogue_path(
    nodes: list[fx.Node],
    start: fx.Node,
    *,
    passthrough_suffixes: Iterable[str] = (),
) -> list[fx.Node]:
    chain: list[fx.Node] = []
    current = start
    add_path = _find_user_path(
        nodes,
        current,
        target_suffixes=_ADD_SUFFIXES,
        passthrough_suffixes=passthrough_suffixes,
    )
    if add_path is not None:
        chain.extend(add_path)
        current = add_path[-1]
    activation_path = _find_user_path(
        nodes,
        current,
        target_suffixes=_ACTIVATION_SUFFIXES,
        passthrough_suffixes=passthrough_suffixes,
    )
    if activation_path is not None:
        chain.extend(activation_path)
    return chain


def _find_rmsnorm_residual_nodes(anchor: fx.Node) -> Optional[list[fx.Node]]:
    if not _matches_any_suffix(anchor, _MUL_SUFFIXES):
        return None
    rms_mul = next(
        (
            source
            for source in anchor.all_input_nodes
            if _matches_any_suffix(source, _MUL_SUFFIXES)
        ),
        None,
    )
    if rms_mul is None:
        return None
    merged_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if _matches_any_suffix(source, _ADD_SUFFIXES)
        ),
        None,
    )
    rsqrt_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if _matches_any_suffix(source, _RSQRT_SUFFIXES)
        ),
        None,
    )
    if merged_node is None or rsqrt_node is None:
        return None
    eps_add = next(
        (
            source
            for source in rsqrt_node.all_input_nodes
            if _matches_any_suffix(source, _ADD_SUFFIXES)
        ),
        None,
    )
    if eps_add is None:
        return None
    mean_node = next(
        (
            source
            for source in eps_add.all_input_nodes
            if _matches_any_suffix(source, _MEAN_SUFFIXES)
        ),
        None,
    )
    if mean_node is None:
        return None
    pow_node = next(
        (
            source
            for source in mean_node.all_input_nodes
            if _matches_any_suffix(source, _POW_SUFFIXES)
        ),
        None,
    )
    if pow_node is None or merged_node not in pow_node.all_input_nodes:
        return None
    return [merged_node, pow_node, mean_node, eps_add, rsqrt_node, rms_mul, anchor]


def _find_rmsnorm_nodes(anchor: fx.Node) -> Optional[list[fx.Node]]:
    if not _matches_any_suffix(anchor, _MUL_SUFFIXES):
        return None
    rms_mul = next(
        (
            source
            for source in anchor.all_input_nodes
            if _matches_any_suffix(source, _MUL_SUFFIXES)
        ),
        None,
    )
    if rms_mul is None:
        return None
    x_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if not _matches_any_suffix(source, _RSQRT_SUFFIXES)
        ),
        None,
    )
    rsqrt_node = next(
        (
            source
            for source in rms_mul.all_input_nodes
            if _matches_any_suffix(source, _RSQRT_SUFFIXES)
        ),
        None,
    )
    if x_node is None or rsqrt_node is None:
        return None
    eps_add = next(
        (
            source
            for source in rsqrt_node.all_input_nodes
            if _matches_any_suffix(source, _ADD_SUFFIXES)
        ),
        None,
    )
    if eps_add is None:
        return None
    mean_node = next(
        (
            source
            for source in eps_add.all_input_nodes
            if _matches_any_suffix(source, _MEAN_SUFFIXES)
        ),
        None,
    )
    if mean_node is None:
        return None
    pow_node = next(
        (
            source
            for source in mean_node.all_input_nodes
            if _matches_any_suffix(source, _POW_SUFFIXES)
        ),
        None,
    )
    if pow_node is None or x_node not in pow_node.all_input_nodes:
        return None
    if any(_matches_any_suffix(source, _ADD_SUFFIXES) for source in x_node.all_input_nodes):
        return None
    return [x_node, pow_node, mean_node, eps_add, rsqrt_node, rms_mul, anchor]


def _find_rope_nodes(anchor: fx.Node) -> Optional[list[fx.Node]]:
    if not _matches_any_suffix(anchor, _FLATTEN_SUFFIXES):
        return None
    stack_node = next(
        (
            source
            for source in anchor.all_input_nodes
            if _matches_any_suffix(source, _STACK_SUFFIXES)
        ),
        None,
    )
    if stack_node is None:
        return None
    rotation_nodes = list(stack_node.all_input_nodes)
    if len(rotation_nodes) != 2:
        return None
    if not any(_matches_any_suffix(node, ("sub", "sub.Tensor")) for node in rotation_nodes):
        return None
    if not any(_matches_any_suffix(node, _ADD_SUFFIXES) for node in rotation_nodes):
        return None
    mul_nodes = [
        source
        for node in rotation_nodes
        for source in node.all_input_nodes
        if _matches_any_suffix(source, _MUL_SUFFIXES)
    ]
    if len(mul_nodes) != 4:
        return None
    slice_nodes: list[fx.Node] = []
    for mul_node in mul_nodes:
        slice_nodes.extend(
            source
            for source in mul_node.all_input_nodes
            if _matches_any_suffix(source, _SLICE_SUFFIXES)
        )
    unique_slice_nodes = list(dict.fromkeys(slice_nodes))
    if len(unique_slice_nodes) != 2:
        return None
    if not all(slice_nodes.count(node) == 2 for node in unique_slice_nodes):
        return None
    return [*unique_slice_nodes, *mul_nodes, *rotation_nodes, stack_node, anchor]
