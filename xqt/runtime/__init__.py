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
from .graph_decode import CudaGraphDecodeSession, GraphDecodeResult
from .model_runner import ModelRunner, ModelRunnerReport
from .cuda_graph import (
    CUDAGraphBlockRunner,
    CUDAGraphCache,
    CUDAGraphCacheKey,
    CUDAGraphEntry,
    compute_model_param_fingerprint,
)
from .deploy_loader import (
    DeployedModelInstance,
    DeploymentExecutionReport,
    StandaloneDeployLoader,
)
from .composite_branch import (
    CompositeBranchModule,
    MaterializerFunc,
    SupportsStaticActivationCalibration,
    get_materializer,
    register_materializer,
    replace_submodule,
)
from .composite_combine import (
    AddCombine,
    CombineStrategy,
    ConcatCombine,
    SelectCombine,
    SUPPORTED_COMBINE_STRATEGIES,
    get_combine_strategy,
)
from .composite_materialize import materialize_composite_compute
from .modules.composite_add import materialize_composite_w4a4
from .composite_inference import (
    calibrate_static_activation_scales,
    fuse_composite_modules,
    materialize_svd_gelu_mlps,
    materialize_svd_for_inference,
    override_composite_execution,
)
from .modules.svd_flux_api import (
    materialize_svd_flux_attention,
    pack_diffusers_flux_rotary_emb,
    svd_flux_attention_metadata,
)
from .modules.svd_flux_transformer_api import materialize_svd_flux_transformer
from .modules import (
    AWQW4A16Linear,
    CompositeAddLinear,
    CompositeAddModule,
    CompositeAddFp8Linear,
    CompositeAddW4A4Linear,
    ConvRotExecutionView,
    ConvRotInt8ExecutionView,
    ConvRotW4A4ExecutionView,
    materialize_convrot_execution_views,
    RMSNormCompositeLinear,
    Fp8MmaLinear,
    Int8MmaLinear,
    LowRankBranch,
    SVDQuantFp8Linear,
    SVDQuantAdaLayerNormZero,
    SVDQuantAdaLayerNormZeroSingle,
    SVDQuantGeluMLP,
    SVDQuantFluxAttention,
    SVDQuantFluxRotaryEmb,
    SVDQuantFluxSingleTransformerBlock,
    SVDQuantFluxTransformerBlock,
    SVDQuantFluxTransformer2DModel,
    SVDQuantInt8MmaLinear,
    SVDQuantLinear,
    W4StorageInt8MmaLinear,
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
    create_inference_session,
    load_model_package,
    write_model_package,
)
from .inference import (
    ImageClassificationAdapter,
    InferenceAdapter,
    InferenceSession,
    TensorInferenceAdapter,
    create_inference_adapter,
    inference_adapter_names,
    register_inference_adapter,
)
from xqt.contracts.runtime_manifest import (
    RUNTIME_MANIFEST_KEY,
    RUNTIME_MANIFEST_SCHEMA_VERSION,
    RuntimeManifest,
    build_runtime_manifest,
)
from .serving_config import generate_serving_config, write_serving_config

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
    "HybridInferenceEngine",
    "HybridInferenceResult",
    "CudaGraphDecodeSession",
    "GraphDecodeResult",
    "ModelRunner",
    "ModelRunnerReport",
    "materialize_composite_compute",
    "materialize_composite_w4a4",
    "calibrate_static_activation_scales",
    "fuse_composite_modules",
    "materialize_svd_gelu_mlps",
    "materialize_svd_for_inference",
    "override_composite_execution",
    "AWQW4A16Linear",
    "CompositeAddLinear",
    "CompositeAddModule",
    "CompositeAddFp8Linear",
    "CompositeAddW4A4Linear",
    "ConvRotExecutionView",
    "ConvRotInt8ExecutionView",
    "ConvRotW4A4ExecutionView",
    "materialize_convrot_execution_views",
    "RMSNormCompositeLinear",
    "W4StorageInt8MmaLinear",
    "SVDQuantLinear",
    "SVDQuantFp8Linear",
    "SVDQuantGeluMLP",
    "SVDQuantAdaLayerNormZero",
    "SVDQuantAdaLayerNormZeroSingle",
    "SVDQuantFluxAttention",
    "SVDQuantFluxRotaryEmb",
    "SVDQuantFluxSingleTransformerBlock",
    "SVDQuantFluxTransformerBlock",
    "SVDQuantFluxTransformer2DModel",
    "SVDQuantInt8MmaLinear",
    "LowRankBranch",
    "materialize_svd_flux_attention",
    "materialize_svd_flux_transformer",
    "pack_diffusers_flux_rotary_emb",
    "svd_flux_attention_metadata",
    "Fp8MmaLinear",
    "Int8MmaLinear",
    # composite branch protocol + registry
    "AddCombine",
    "CombineStrategy",
    "CompositeBranchModule",
    "ConcatCombine",
    "SelectCombine",
    "SUPPORTED_COMBINE_STRATEGIES",
    "SupportsStaticActivationCalibration",
    "get_combine_strategy",
    "get_materializer",
    "register_materializer",
    "replace_submodule",
    # package
    "LoadedModelPackage",
    "MODEL_PACKAGE_ARTIFACT_TYPE",
    "MODEL_PACKAGE_SCHEMA_VERSION",
    "ModelPackageManifest",
    "ONNXRuntimeRunner",
    "create_inference_runner",
    "create_inference_session",
    "ImageClassificationAdapter",
    "InferenceAdapter",
    "InferenceSession",
    "TensorInferenceAdapter",
    "create_inference_adapter",
    "inference_adapter_names",
    "register_inference_adapter",
    "load_model_package",
    "write_model_package",
    "RUNTIME_MANIFEST_KEY",
    "RUNTIME_MANIFEST_SCHEMA_VERSION",
    "RuntimeManifest",
    "build_runtime_manifest",
    "generate_serving_config",
    "write_serving_config",
    "CUDAGraphBlockRunner",
    "CUDAGraphCache",
    "CUDAGraphCacheKey",
    "CUDAGraphEntry",
    "compute_model_param_fingerprint",
    "DeployedModelInstance",
    "DeploymentExecutionReport",
    "StandaloneDeployLoader",
]
