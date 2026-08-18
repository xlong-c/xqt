"""Framework-owned smoke models, family facts, and output hooks."""

from .families import (
    FamilySmokeReport,
    classify_model_family,
    component_grouping,
    family_smoke_report,
    model_family_names,
)
from .hooks import ModuleOutputCapture, capture_module_outputs, collect_module_outputs
from .smoke_detection import SmokeDetectionModule, build_smoke_detection_module
from .smoke_diffusion import (
    DiffusionSmokeReport,
    SmokeDiffusionDenoiser,
    build_smoke_diffusion_denoiser,
    diffusion_smoke_report,
)
from .smoke_llm import SmokeLLM, SmokeLLMBlock, build_smoke_llm
from .smoke_moe import (
    MoEFamilyReport,
    SmokeMoE,
    SmokeMoEBlock,
    build_smoke_moe,
    moe_family_report,
    suggest_expert_pruning,
)
from .smoke_multimodal import (
    EncoderCacheMetadata,
    MultimodalInputSignature,
    SmokeMultimodalClassifier,
    VisualTokenCompressionMetadata,
    build_smoke_multimodal_classifier,
    encoder_cache_metadata,
    multimodal_input_signature,
    visual_token_compression_metadata,
)
from .smoke_vit import SmokeViTClassifier, build_smoke_vit_classifier

__all__ = [
    "DiffusionSmokeReport",
    "EncoderCacheMetadata",
    "FamilySmokeReport",
    "MoEFamilyReport",
    "ModuleOutputCapture",
    "MultimodalInputSignature",
    "SmokeDetectionModule",
    "SmokeDiffusionDenoiser",
    "SmokeLLM",
    "SmokeLLMBlock",
    "SmokeMoE",
    "SmokeMoEBlock",
    "SmokeMultimodalClassifier",
    "SmokeViTClassifier",
    "VisualTokenCompressionMetadata",
    "build_smoke_detection_module",
    "build_smoke_diffusion_denoiser",
    "build_smoke_llm",
    "build_smoke_moe",
    "build_smoke_multimodal_classifier",
    "build_smoke_vit_classifier",
    "capture_module_outputs",
    "classify_model_family",
    "collect_module_outputs",
    "component_grouping",
    "diffusion_smoke_report",
    "encoder_cache_metadata",
    "family_smoke_report",
    "model_family_names",
    "moe_family_report",
    "multimodal_input_signature",
    "suggest_expert_pruning",
    "visual_token_compression_metadata",
]
