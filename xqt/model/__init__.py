"""Model-side helpers used by XQT recipes."""

from .hooks import ModuleOutputCapture, capture_module_outputs, collect_module_outputs
from .smoke_detection import SmokeDetectionModule, build_smoke_detection_module

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


def __getattr__(name: str) -> object:
    if name in _FLUX2_KLEIN_NVFP4_EXPORTS:
        from . import flux2_klein_nvfp4

        return getattr(flux2_klein_nvfp4, name)
    if name in _WAN21_VAE_EXPORTS:
        from . import wan21_vae

        return getattr(wan21_vae, name)
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
    "ModuleOutputCapture",
    "SmokeDetectionModule",
    "benchmark_flux2_klein_nvfp4_transformer_paired",
    "benchmark_flux2_klein_nvfp4_transformer_forward",
    "build_smoke_detection_module",
    "capture_flux2_klein_nvfp4_transformer_cuda_graph",
    "compile_flux2_klein_nvfp4_transformer",
    "collect_flux2_klein_nvfp4_engine_targets",
    "collect_flux2_klein_nvfp4_targets",
    "capture_module_outputs",
    "collect_module_outputs",
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
