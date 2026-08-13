"""Model-side helpers used by XQT recipes."""

from .families import (
    FamilySmokeReport,
    classify_model_family,
    component_grouping,
    family_smoke_report,
    model_family_names,
)
from .hooks import ModuleOutputCapture, capture_module_outputs, collect_module_outputs
from .smoke_diffusion import (
    DiffusionSmokeReport,
    SmokeDiffusionDenoiser,
    build_smoke_diffusion_denoiser,
    diffusion_smoke_report,
)
from .smoke_detection import SmokeDetectionModule, build_smoke_detection_module
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

_FLUX2_KLEIN_NVFP4_EXPORTS = {
    "FLUX2_KLEIN_4B_NVFP4_FILENAME",
    "FLUX2_KLEIN_4B_NVFP4_REPO_ID",
    "FLUX2_KLEIN_4B_REPO_ID",
    "FLUX2_KLEIN_NVFP4_ENGINES",
    "Flux2KleinNVFP4EngineResult",
    "Flux2KleinNVFP4CompiledTransformerResult",
    "Flux2KleinNVFP4CudaGraphTransformerResult",
    "Flux2KleinNVFP4PairedBenchmarkResult",
    "Flux2KleinNVFP4TargetSummary",
    "FamilySmokeReport",
    "benchmark_flux2_klein_nvfp4_transformer_paired",
    "benchmark_flux2_klein_nvfp4_transformer_forward",
    "capture_flux2_klein_nvfp4_transformer_cuda_graph",
    "compile_flux2_klein_nvfp4_transformer",
    "collect_flux2_klein_nvfp4_engine_targets",
    "collect_flux2_klein_nvfp4_targets",
    "flux2_klein_nvfp4_single_file_url",
    "load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_4bit",
    "load_flux2_klein_bf16_pipeline",
    "load_flux2_klein_bf16_transformer",
    "load_flux2_klein_nvfp4_pipeline",
    "load_flux2_klein_nvfp4_transformer",
    "materialize_flux2_klein_nvfp4_engine",
    "normalize_flux2_klein_nvfp4_engine",
    "optimize_flux2_klein_nvfp4_transformer",
    "quantize_flux2_klein_bf16_pipeline_to_convrot_4bit",
    "quantize_flux2_klein_bf16_transformer_to_convrot_4bit",
    "run_flux2_klein_bf16_convrot_4bit_inference",
    "run_flux2_klein_nvfp4_inference",
    "warmup_flux2_klein_nvfp4_transformer",
}

_FLUX2_KLEIN_CONVROT_INT8_EXPORTS = {
    "benchmark_flux2_klein_convrot_int8_transformer_forward",
    "benchmark_flux2_klein_convrot_int8_transformer_paired",
    "flux2_klein_bf16_convrot_int8_default_policy",
    "load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_int8",
    "optimize_flux2_klein_convrot_int8_transformer",
    "quantize_flux2_klein_bf16_pipeline_to_convrot_int8",
    "quantize_flux2_klein_bf16_transformer_to_convrot_int8",
    "run_flux2_klein_bf16_convrot_int8_inference",
}

_WAN21_VAE_EXPORTS = {
    "WAN21_VAE_OPTIMIZATION_KINDS",
    "WAN21_VAE_REPO_ID",
    "WAN21_VAE_RUN_MODES",
    "WAN21_VAE_SUBFOLDER",
    "Wan21VAECompileResult",
    "Wan21VAECudaGraphResult",
    "Wan21VAEOptimizationSummary",
    "Wan21VAEPairedBenchmarkResult",
    "benchmark_wan21_vae_paired",
    "benchmark_wan21_vae_runner",
    "build_wan21_vae_runner",
    "capture_wan21_vae_cuda_graph",
    "collect_wan21_vae_conv3d_targets",
    "collect_wan21_vae_rmsnorm_targets",
    "compile_wan21_vae_runner",
    "load_wan21_pipeline_with_vae",
    "load_wan21_vae",
    "materialize_wan21_vae_conv3d_fastpath",
    "materialize_wan21_vae_rmsnorm_fastpath",
    "optimize_wan21_vae",
    "run_wan21_vae_inference",
    "warmup_wan21_vae_runner",
}

_HUNYUAN_OCR_EXPORTS = {
    "HUNYUAN_OCR_DFLASH_SUBFOLDER",
    "HUNYUAN_OCR_REPO_ID",
    "HUNYUAN_OCR_SVD_INT4_STRATEGY",
    "HunyuanOCRBlockOptimization",
    "HunyuanOCRInt4QuantResult",
    "HunyuanOcrTileLangCudaGraphRunner",
    "HunyuanOcrTileLangDecodeBlock",
    "HunyuanOcrTileLangDecodeSpec",
    "benchmark_hunyuan_ocr_tilelang_decode_graph",
    "load_hunyuan_ocr",
    "load_hunyuan_ocr_dflash",
    "optimize_hunyuan_ocr_svd_int4_blocks",
    "optimize_hunyuan_ocr_dflash_svd_int4_blocks",
}

_UNLIMITED_OCR_EXPORTS = {
    "UNLIMITED_OCR_CONVROT_INT8_STRATEGY",
    "UNLIMITED_OCR_REPO_ID",
    "UnlimitedOcrConvRotCalibration",
    "UnlimitedOcrConvRotInt8Result",
    "calibrate_unlimited_ocr_convrot_activation_scales",
    "force_unlimited_ocr_fp16_runtime",
    "load_unlimited_ocr",
    "quantize_unlimited_ocr_convrot_int8",
    "select_unlimited_ocr_convrot_modules",
    "unlimited_ocr_convrot_default_policy",
}


def __getattr__(name: str) -> object:
    if name in _FLUX2_KLEIN_CONVROT_INT8_EXPORTS:
        from . import flux2_klein

        return getattr(flux2_klein, name)
    if name in _FLUX2_KLEIN_NVFP4_EXPORTS:
        from . import flux2_klein_nvfp4

        return getattr(flux2_klein_nvfp4, name)
    if name in _WAN21_VAE_EXPORTS:
        from . import wan21_vae

        return getattr(wan21_vae, name)
    if name in _HUNYUAN_OCR_EXPORTS:
        if name.startswith("HunyuanOcrTileLang") or name == (
            "benchmark_hunyuan_ocr_tilelang_decode_graph"
        ):
            from . import hunyuan_ocr_tilelang

            return getattr(hunyuan_ocr_tilelang, name)
        from . import hunyuan_ocr

        return getattr(hunyuan_ocr, name)
    if name in _UNLIMITED_OCR_EXPORTS:
        from . import unlimited_ocr

        return getattr(unlimited_ocr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FLUX2_KLEIN_4B_NVFP4_FILENAME",
    "FLUX2_KLEIN_4B_NVFP4_REPO_ID",
    "FLUX2_KLEIN_4B_REPO_ID",
    "FLUX2_KLEIN_NVFP4_ENGINES",
    "Flux2KleinNVFP4EngineResult",
    "Flux2KleinNVFP4CompiledTransformerResult",
    "Flux2KleinNVFP4CudaGraphTransformerResult",
    "Flux2KleinNVFP4PairedBenchmarkResult",
    "Flux2KleinNVFP4TargetSummary",
    "HUNYUAN_OCR_REPO_ID",
    "HUNYUAN_OCR_DFLASH_SUBFOLDER",
    "HUNYUAN_OCR_SVD_INT4_STRATEGY",
    "HunyuanOCRBlockOptimization",
    "HunyuanOCRInt4QuantResult",
    "HunyuanOcrTileLangCudaGraphRunner",
    "HunyuanOcrTileLangDecodeBlock",
    "HunyuanOcrTileLangDecodeSpec",
    "UNLIMITED_OCR_CONVROT_INT8_STRATEGY",
    "UNLIMITED_OCR_REPO_ID",
    "UnlimitedOcrConvRotCalibration",
    "UnlimitedOcrConvRotInt8Result",
    "DiffusionSmokeReport",
    "EncoderCacheMetadata",
    "MoEFamilyReport",
    "MultimodalInputSignature",
    "benchmark_hunyuan_ocr_tilelang_decode_graph",
    "ModuleOutputCapture",
    "SmokeDiffusionDenoiser",
    "SmokeDetectionModule",
    "SmokeLLM",
    "SmokeLLMBlock",
    "SmokeMoE",
    "SmokeMoEBlock",
    "SmokeMultimodalClassifier",
    "SmokeViTClassifier",
    "VisualTokenCompressionMetadata",
    "benchmark_flux2_klein_nvfp4_transformer_paired",
    "benchmark_flux2_klein_nvfp4_transformer_forward",
    "build_smoke_diffusion_denoiser",
    "build_smoke_detection_module",
    "build_smoke_llm",
    "build_smoke_moe",
    "build_smoke_multimodal_classifier",
    "build_smoke_vit_classifier",
    "classify_model_family",
    "capture_flux2_klein_nvfp4_transformer_cuda_graph",
    "compile_flux2_klein_nvfp4_transformer",
    "collect_flux2_klein_nvfp4_engine_targets",
    "collect_flux2_klein_nvfp4_targets",
    "capture_module_outputs",
    "calibrate_unlimited_ocr_convrot_activation_scales",
    "force_unlimited_ocr_fp16_runtime",
    "collect_module_outputs",
    "component_grouping",
    "diffusion_smoke_report",
    "encoder_cache_metadata",
    "flux2_klein_nvfp4_single_file_url",
    "flux2_klein_bf16_convrot_int8_default_policy",
    "family_smoke_report",
    "load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_4bit",
    "load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_int8",
    "load_flux2_klein_bf16_pipeline",
    "load_hunyuan_ocr",
    "load_hunyuan_ocr_dflash",
    "load_unlimited_ocr",
    "moe_family_report",
    "multimodal_input_signature",
    "model_family_names",
    "load_flux2_klein_bf16_transformer",
    "load_flux2_klein_nvfp4_pipeline",
    "load_flux2_klein_nvfp4_transformer",
    "materialize_flux2_klein_nvfp4_engine",
    "normalize_flux2_klein_nvfp4_engine",
    "optimize_flux2_klein_convrot_int8_transformer",
    "optimize_flux2_klein_nvfp4_transformer",
    "optimize_hunyuan_ocr_svd_int4_blocks",
    "optimize_hunyuan_ocr_dflash_svd_int4_blocks",
    "quantize_flux2_klein_bf16_pipeline_to_convrot_4bit",
    "quantize_flux2_klein_bf16_pipeline_to_convrot_int8",
    "quantize_flux2_klein_bf16_transformer_to_convrot_4bit",
    "quantize_flux2_klein_bf16_transformer_to_convrot_int8",
    "quantize_unlimited_ocr_convrot_int8",
    "run_flux2_klein_bf16_convrot_4bit_inference",
    "run_flux2_klein_bf16_convrot_int8_inference",
    "run_flux2_klein_nvfp4_inference",
    "select_unlimited_ocr_convrot_modules",
    "suggest_expert_pruning",
    "unlimited_ocr_convrot_default_policy",
    "visual_token_compression_metadata",
    "warmup_flux2_klein_nvfp4_transformer",
    "WAN21_VAE_OPTIMIZATION_KINDS",
    "WAN21_VAE_REPO_ID",
    "WAN21_VAE_RUN_MODES",
    "WAN21_VAE_SUBFOLDER",
    "Wan21VAECompileResult",
    "Wan21VAECudaGraphResult",
    "Wan21VAEOptimizationSummary",
    "Wan21VAEPairedBenchmarkResult",
    "benchmark_wan21_vae_paired",
    "benchmark_wan21_vae_runner",
    "build_wan21_vae_runner",
    "capture_wan21_vae_cuda_graph",
    "collect_wan21_vae_conv3d_targets",
    "collect_wan21_vae_rmsnorm_targets",
    "compile_wan21_vae_runner",
    "load_wan21_pipeline_with_vae",
    "load_wan21_vae",
    "materialize_wan21_vae_conv3d_fastpath",
    "materialize_wan21_vae_rmsnorm_fastpath",
    "optimize_wan21_vae",
    "run_wan21_vae_inference",
    "warmup_wan21_vae_runner",
]
