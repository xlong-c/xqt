"""Runtime bridge modules for packed quantized weights (Infer / operator_opt).

These adapters expose packed storage to kernels. They are not quant algorithms
and must not live under xqt.quant as a hard dependency of operator_opt.
"""

from .external_materialize import create_weight_plans
from .external_weight_only import ExternalLoadReport, load_external_quantized_model
from .hf_int4_layout import process_weights_after_loading
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
    "ExternalLoadReport",
    "NVFP4LinearBridge",
    "NVFP4TensorLayout",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "create_weight_plans",
    "expand_group_scale",
    "infer_nvfp4_tensor_layout",
    "load_external_quantized_model",
    "process_weights_after_loading",
    "unpack_nvfp4e2m1",
]
