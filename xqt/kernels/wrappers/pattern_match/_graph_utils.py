"""Shared graph utilities used across pattern finders, FX, and export scanners."""

from __future__ import annotations

from typing import Iterable, Optional

from torch import fx


_RESHAPE_SUFFIXES = (
    "reshape.default",
    "view.default",
)
_ADD_SUFFIXES = (
    "add",
    "add.Tensor",
    "add_.Tensor",
)
_ACTIVATION_SUFFIXES = (
    "gelu.default",
    "relu.default",
    "silu.default",
)
_MUL_SUFFIXES = (
    "mul",
    "mul.Tensor",
)
_POW_SUFFIXES = (
    "pow",
    "pow.Tensor_Scalar",
)
_MEAN_SUFFIXES = (
    "mean",
    "mean.dim",
)
_RSQRT_SUFFIXES = (
    "rsqrt",
    "rsqrt.default",
)
_SLICE_SUFFIXES = (
    "getitem",
    "slice.Tensor",
)
_STACK_SUFFIXES = (
    "stack",
    "stack.default",
)
_FLATTEN_SUFFIXES = (
    "flatten",
    "flatten.using_ints",
)


def _normalize_target(target: object) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        module_name = getattr(target, "__module__", "")
        qualname = getattr(target, "__name__", str(target))
        return f"{module_name}.{qualname}" if module_name else qualname
    return str(target)


def _is_target(node: fx.Node, suffix: str) -> bool:
    return _normalize_target(node.target).endswith(suffix)


def _matches_any_suffix(node: fx.Node, suffixes: Iterable[str]) -> bool:
    target_name = _normalize_target(node.target)
    return any(target_name.endswith(suffix) for suffix in suffixes)


def _find_node_users(nodes: Iterable[fx.Node], value: fx.Node) -> list[fx.Node]:
    return [node for node in nodes if value in node.all_input_nodes]


def _subclass_key(node: fx.Node) -> Optional[str]:
    if not _is_target(node, "export.access_subclass_inner_tensor.default"):
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], str):
        return None
    return str(node.args[1])


def _ancestor_subclass_keys(node: fx.Node) -> set[str]:
    keys: set[str] = set()
    pending = list(node.all_input_nodes)
    visited: set[fx.Node] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        key = _subclass_key(current)
        if key is not None:
            keys.add(key)
        pending.extend(current.all_input_nodes)
    return keys


def _ancestor_support_nodes(
    nodes: Iterable[fx.Node],
    anchor: fx.Node,
    keys: set[str],
) -> list[fx.Node]:
    if not keys:
        return []
    pending = list(anchor.all_input_nodes)
    ancestors: set[fx.Node] = set()
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(current.all_input_nodes)
    return [
        node
        for node in nodes
        if node in ancestors and _subclass_key(node) in keys
    ]


def _find_first_user_with_suffix(
    nodes: Iterable[fx.Node],
    value: fx.Node,
    suffix: str,
) -> Optional[fx.Node]:
    for node in nodes:
        if value in node.all_input_nodes and _is_target(node, suffix):
            return node
    return None
