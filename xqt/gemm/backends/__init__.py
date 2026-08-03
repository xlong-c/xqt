"""Optional native GEMM backends.

Backend modules are lazy and artifact-driven.  Importing ``xqt.gemm`` never
loads a shared object or compiles CUDA code.
"""

from .sm89 import (
    install_sm89_w8a8_executor,
    prepack_sm89_int8_weight,
    sm89_artifact_available,
    sm89_w8a8_executor,
)
from .sm89_build import (
    Sm89BuildConfig,
    Sm89DenseBuildConfig,
    Sm89MixedInputProbeBuildConfig,
    Sm89W4A16BuildConfig,
    Sm89W4A16FusedBuildConfig,
    Sm89Fp8ProbeBuildConfig,
    Sm89Fp8BuildConfig,
    build_sm89_artifact,
    build_sm89_dense_artifact,
    build_sm89_w4a16_dequant_artifact,
    build_sm89_w4a16_fused_artifact,
    build_sm89_fp8_probe_artifact,
    build_sm89_fp8_artifact,
    build_sm89_mixed_input_probe_artifact,
)
from .dense_sm89 import (
    dense_sm89_artifact_available,
    dense_sm89_executor,
    install_sm89_dense_executors,
)
from .w4a16_sm89 import (
    Sm89W4A16ResourceReport,
    install_sm89_w4a16_dequant_executor,
    query_sm89_w4a16_resources,
    sm89_w4a16_dequant_artifact_available,
    sm89_w4a16_dequant_executor,
)
from .w4a16_fused_sm89 import (
    install_sm89_w4a16_fused_executor,
    select_fused_split_k,
    sm89_w4a16_fused_artifact_available,
    sm89_w4a16_fused_decode_supported,
    sm89_w4a16_fused_executor,
    sm89_w4a16_fused_splitk_supported,
)
from .mixed_input_probe_sm89 import (
    run_sm89_mixed_input_probe,
    sm89_mixed_input_probe_artifact_available,
)
from .fp8_probe_sm89 import run_sm89_fp8_probe, sm89_fp8_probe_artifact_available
from .fp8_sm89 import (
    Sm89Fp8BlockwiseResourceReport,
    fp8_blockwise_split_k_partition,
    fp8_sm89_executor,
    install_sm89_fp8_executors,
    query_sm89_fp8_blockwise_resources,
    select_fp8_blockwise_split_k,
    sm89_fp8_artifact_available,
)

__all__ = [
    "install_sm89_w8a8_executor",
    "sm89_artifact_available",
    "sm89_w8a8_executor",
    "prepack_sm89_int8_weight",
    "Sm89BuildConfig",
    "Sm89DenseBuildConfig",
    "Sm89MixedInputProbeBuildConfig",
    "Sm89W4A16BuildConfig",
    "Sm89W4A16FusedBuildConfig",
    "Sm89Fp8ProbeBuildConfig",
    "Sm89Fp8BuildConfig",
    "build_sm89_artifact",
    "build_sm89_dense_artifact",
    "build_sm89_w4a16_dequant_artifact",
    "build_sm89_w4a16_fused_artifact",
    "build_sm89_fp8_probe_artifact",
    "build_sm89_fp8_artifact",
    "build_sm89_mixed_input_probe_artifact",
    "dense_sm89_artifact_available",
    "dense_sm89_executor",
    "install_sm89_dense_executors",
    "install_sm89_w4a16_dequant_executor",
    "Sm89W4A16ResourceReport",
    "query_sm89_w4a16_resources",
    "sm89_w4a16_dequant_artifact_available",
    "sm89_w4a16_dequant_executor",
    "install_sm89_w4a16_fused_executor",
    "select_fused_split_k",
    "sm89_w4a16_fused_artifact_available",
    "sm89_w4a16_fused_decode_supported",
    "sm89_w4a16_fused_executor",
    "sm89_w4a16_fused_splitk_supported",
    "run_sm89_mixed_input_probe",
    "sm89_mixed_input_probe_artifact_available",
    "run_sm89_fp8_probe",
    "sm89_fp8_probe_artifact_available",
    "fp8_sm89_executor",
    "install_sm89_fp8_executors",
    "sm89_fp8_artifact_available",
    "Sm89Fp8BlockwiseResourceReport",
    "fp8_blockwise_split_k_partition",
    "query_sm89_fp8_blockwise_resources",
    "select_fp8_blockwise_split_k",
]
