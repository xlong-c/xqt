"""Wan 2.1 VAE model-side high-performance inference helpers."""

from .wan21.types import (
    WAN21_VAE_OPTIMIZATION_KINDS,
    WAN21_VAE_REPO_ID,
    WAN21_VAE_RUN_MODES,
    WAN21_VAE_SUBFOLDER,
    Wan21VAECompileResult,
    Wan21VAECudaGraphResult,
    Wan21VAEOptimizationSummary,
    Wan21VAEPairedBenchmarkResult,
)
from .wan21.targets import (
    collect_wan21_vae_conv3d_targets,
    collect_wan21_vae_rmsnorm_targets,
    materialize_wan21_vae_conv3d_fastpath,
    materialize_wan21_vae_rmsnorm_fastpath,
)
from .wan21.runtime import (
    build_wan21_vae_runner,
    capture_wan21_vae_cuda_graph,
    compile_wan21_vae_runner,
    warmup_wan21_vae_runner,
)
from .wan21.optimize import (
    benchmark_wan21_vae_paired,
    benchmark_wan21_vae_runner,
    optimize_wan21_vae,
)
from .wan21.load import (
    load_wan21_pipeline_with_vae,
    load_wan21_vae,
    run_wan21_vae_inference,
)

__all__ = [
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
