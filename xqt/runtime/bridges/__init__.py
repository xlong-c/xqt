"""Runtime bridge modules for packed quantized weights (Infer / operator_opt).

These adapters expose packed storage to kernels. They are not quant algorithms
and must not live under xqt.quant as a hard dependency of operator_opt.
"""

from .nvfp4 import (
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
