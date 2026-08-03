"""Compatibility re-export: NVFP4 bridge lives in xqt.runtime.bridges.nvfp4."""

from __future__ import annotations

from xqt.runtime.bridges.nvfp4 import (
    NVFP4LinearBridge,
    NVFP4TensorLayout,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    expand_group_scale,
    infer_nvfp4_tensor_layout,
    unpack_nvfp4e2m1,
)

__all__ = [
    "NVFP4LinearBridge",
    "NVFP4TensorLayout",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "expand_group_scale",
    "infer_nvfp4_tensor_layout",
    "unpack_nvfp4e2m1",
]
