"""Runtime helpers for hybrid inference over quantized model artifacts.

This package only consumes quantized modules and execution policies.
It never runs quantizers, calibration, or sensitivity analysis.
"""

# ---------------------------------------------------------------------------
# re-exported from xqt.contracts
# ---------------------------------------------------------------------------

from xqt.contracts import (
    SUPPORTED_CHANNEL_AXES,
    SUPPORTED_COMPUTE_PRECISIONS,
    ChannelHybridSpec,
    SupportsChannelHybrid,
    SupportsComputePrecision,
    normalize_channel_axis,
    normalize_compute_precision,
)

# ---------------------------------------------------------------------------
# runtime channel helpers
# ---------------------------------------------------------------------------

from .channel import (
    apply_channel_hybrid_policy,
    channel_hybrid_linear_reference,
    collect_channel_hybrid_map,
)

# ---------------------------------------------------------------------------
# runtime policy
# ---------------------------------------------------------------------------

from .policy import (
    apply_execution_policy,
    build_execution_policy_payload,
    collect_module_precision_map,
    precision_overrides_to_map,
    set_module_compute_precision,
)

# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

from .engine import HybridInferenceEngine, HybridInferenceResult
from .engine_resolve import (
    EngineResolveResult,
    resolve_engine,
    resolve_int8_mma_engine,
)

# ---------------------------------------------------------------------------
# package
# ---------------------------------------------------------------------------

from .package import (
    MODEL_PACKAGE_ARTIFACT_TYPE,
    MODEL_PACKAGE_SCHEMA_VERSION,
    LoadedModelPackage,
    ModelPackageManifest,
    ONNXRuntimeRunner,
    create_inference_runner,
    load_model_package,
    write_model_package,
)

__all__ = [
    # re-exported from xqt.contracts
    "SUPPORTED_CHANNEL_AXES",
    "SUPPORTED_COMPUTE_PRECISIONS",
    "ChannelHybridSpec",
    "SupportsChannelHybrid",
    "SupportsComputePrecision",
    "normalize_channel_axis",
    "normalize_compute_precision",
    # runtime channel
    "apply_channel_hybrid_policy",
    "channel_hybrid_linear_reference",
    "collect_channel_hybrid_map",
    # runtime policy
    "apply_execution_policy",
    "build_execution_policy_payload",
    "collect_module_precision_map",
    "precision_overrides_to_map",
    "set_module_compute_precision",
    # engine
    "EngineResolveResult",
    "HybridInferenceEngine",
    "HybridInferenceResult",
    "resolve_engine",
    "resolve_int8_mma_engine",
    # package
    "LoadedModelPackage",
    "MODEL_PACKAGE_ARTIFACT_TYPE",
    "MODEL_PACKAGE_SCHEMA_VERSION",
    "ModelPackageManifest",
    "ONNXRuntimeRunner",
    "create_inference_runner",
    "load_model_package",
    "write_model_package",
]
