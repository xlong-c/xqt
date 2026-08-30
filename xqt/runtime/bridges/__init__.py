"""Runtime bridge modules for packed quantized weights (Infer / kernels.wrappers).

These adapters expose packed storage to kernels. They are not quant algorithms
and must not live under xqt.compression.quant as a hard dependency of wrappers.
"""

from .external_materialize import create_weight_plans
from .external_weight_only import ExternalLoadReport, load_external_quantized_model
from .hf_int4_layout import process_weights_after_loading

__all__ = [
    "ExternalLoadReport",
    "create_weight_plans",
    "load_external_quantized_model",
    "process_weights_after_loading",
]
