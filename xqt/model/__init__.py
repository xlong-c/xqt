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
    "load_flux2_klein_nvfp4_pipeline",
    "load_flux2_klein_nvfp4_transformer",
    "materialize_flux2_klein_nvfp4_engine",
    "normalize_flux2_klein_nvfp4_engine",
    "optimize_flux2_klein_nvfp4_transformer",
    "run_flux2_klein_nvfp4_inference",
    "warmup_flux2_klein_nvfp4_transformer",
}


def __getattr__(name: str) -> object:
    if name in _FLUX2_KLEIN_NVFP4_EXPORTS:
        from . import flux2_klein_nvfp4

        return getattr(flux2_klein_nvfp4, name)
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
    "load_flux2_klein_nvfp4_pipeline",
    "load_flux2_klein_nvfp4_transformer",
    "materialize_flux2_klein_nvfp4_engine",
    "normalize_flux2_klein_nvfp4_engine",
    "optimize_flux2_klein_nvfp4_transformer",
    "run_flux2_klein_nvfp4_inference",
    "warmup_flux2_klein_nvfp4_transformer",
]
