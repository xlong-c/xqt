"""Concrete model adapters and compatibility profiles for XQT."""

from .adapter import ModelAdapter
from .minicpm5 import (
    MINICPM5_2B_PROFILE_ID,
    MINICPM5_2B_REPO_ID,
    load_minicpm5,
    materialize_minicpm5_int8_runtime,
    materialize_minicpm5_weight_only_runtime,
    QuaRotTransformReport,
    apply_quarot_minicpm5,
    build_quarot_rotation,
    minicpm5_linear_summary,
    minicpm5_mlp_only_quantization_policy,
    minicpm5_edge_protected_quantization_policy,
    minicpm5_quantization_policy,
    quantize_minicpm5,
)
from .config import ModelProfile
from .registry import (
    model_profile_names,
    register_model_profile,
    resolve_model_adapter,
    resolve_model_profile,
)
from .resolver import profile_from_model_config

__all__ = [
    "ModelProfile",
    "ModelAdapter",
    "MINICPM5_2B_PROFILE_ID",
    "MINICPM5_2B_REPO_ID",
    "load_minicpm5",
    "materialize_minicpm5_int8_runtime",
    "materialize_minicpm5_weight_only_runtime",
    "QuaRotTransformReport",
    "apply_quarot_minicpm5",
    "build_quarot_rotation",
    "minicpm5_linear_summary",
    "minicpm5_mlp_only_quantization_policy",
    "minicpm5_edge_protected_quantization_policy",
    "minicpm5_quantization_policy",
    "quantize_minicpm5",
    "model_profile_names",
    "profile_from_model_config",
    "register_model_profile",
    "resolve_model_adapter",
    "resolve_model_profile",
]
