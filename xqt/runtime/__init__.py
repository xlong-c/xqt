"""Runtime helpers for hybrid inference over quantized model artifacts.

This package only consumes quantized modules and execution policies.
It never runs quantizers, calibration, or sensitivity analysis.
"""

from .channel import (
    SUPPORTED_CHANNEL_AXES,
    ChannelHybridSpec,
    apply_channel_hybrid_policy,
    build_channel_mask,
    channel_hybrid_linear_reference,
    collect_channel_hybrid_map,
    normalize_channel_axis,
    select_outlier_channels,
    split_linear_tensors_by_channel,
)
from .engine import HybridInferenceEngine, HybridInferenceResult
from .package import (
    LoadedModelPackage,
    MODEL_PACKAGE_ARTIFACT_TYPE,
    MODEL_PACKAGE_SCHEMA_VERSION,
    ModelPackageManifest,
    ONNXRuntimeRunner,
    create_inference_runner,
    load_model_package,
    write_model_package,
)
from .policy import (
    SUPPORTED_COMPUTE_PRECISIONS,
    apply_execution_policy,
    build_execution_policy_payload,
    collect_module_precision_map,
    normalize_compute_precision,
    precision_overrides_to_map,
    set_module_compute_precision,
)

__all__ = [
    "SUPPORTED_CHANNEL_AXES",
    "SUPPORTED_COMPUTE_PRECISIONS",
    "ChannelHybridSpec",
    "HybridInferenceEngine",
    "HybridInferenceResult",
    "LoadedModelPackage",
    "MODEL_PACKAGE_ARTIFACT_TYPE",
    "MODEL_PACKAGE_SCHEMA_VERSION",
    "ModelPackageManifest",
    "ONNXRuntimeRunner",
    "apply_channel_hybrid_policy",
    "apply_execution_policy",
    "build_channel_mask",
    "build_execution_policy_payload",
    "create_inference_runner",
    "channel_hybrid_linear_reference",
    "collect_channel_hybrid_map",
    "collect_module_precision_map",
    "load_model_package",
    "normalize_channel_axis",
    "normalize_compute_precision",
    "precision_overrides_to_map",
    "select_outlier_channels",
    "set_module_compute_precision",
    "split_linear_tensors_by_channel",
    "write_model_package",
]
