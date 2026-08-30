from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.kernels.ops.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    QuantSpec,
    Sm120BuildConfig,
    Sm120Fp8Contract,
    Sm120Nvfp4Contract,
    Sm90Fp8WgmmaBuildConfig,
    build_packed_weight,
    calibrate_fp8_scale,
    default_registry,
    dispatch_gemm,
    quantize_fp8,
    select_kernel,
)
from xqt.kernels.ops._impl.gemm_backends._sm1xx_runtime import cutlass_blockscale_shape


def test_sm90_and_sm120_build_configs_are_architecture_specific() -> None:
    sm90 = Sm90Fp8WgmmaBuildConfig()
    sm120 = Sm120BuildConfig()

    assert sm90.target_arch == "sm_90"
    assert sm90.source.name == "sm90_fp8_wgmma.cu"
    assert sm120.target_arch == "sm_120"
    assert sm120.compile_target_arch == "sm_120a"
    assert "--expt-relaxed-constexpr" in sm120.extra_flags
    assert sm120.source.name == "sm120_gemm.cu"


def test_target_manifests_declare_stream_aware_runtime_abi() -> None:
    root = Path(__file__).resolve().parents[3]
    sm90_source = (root / "xqt/kernels/ops/_impl/gemm_backends/sm90/sm90_fp8_wgmma.py").read_text()
    sm120_source = (root / "xqt/kernels/ops/_impl/gemm_backends/sm120/sm120.py").read_text()

    assert '"runtime_stream_abi": "torch_current_stream_void_p"' in sm90_source
    assert '"runtime_stream_abi": "torch_current_stream_void_p"' in sm120_source
    assert '"compile_target_arch": resolved.compile_target_arch' in sm120_source
    assert '"required_compile_flags": list(_SM120_REQUIRED_FLAGS)' in sm120_source
    assert '"dense_schedule_variants"' in sm90_source
    assert '"fp8_groupwise_pingpong_probe_present": True' in sm90_source
    assert '"fp8_groupwise_schedule_variants"' in sm90_source
    assert '"nvfp4_k256_probe_present": True' in sm120_source
    assert '"nvfp4_schedule_variants": ["cooperative", "pingpong"]' in sm120_source


def test_sm120_fp8_contract_keeps_xqt_blockwise_abi_for_groupwise_probe() -> None:
    contract = Sm120Fp8Contract(scale_granularity="groupwise")
    quant = contract.quant_spec()

    assert contract.to_dict()["scale_granularity"] == "groupwise"
    assert contract.to_dict()["xqt_scale_mode"] == "w:blockwise/a:blockwise"
    assert quant.scale_mode == "w:blockwise/a:blockwise"
    assert quant.group_size == 128


def test_sm120_nvfp4_contract_exposes_unverified_runtime_probe() -> None:
    contract = Sm120Nvfp4Contract()

    assert contract.quant_spec().weight_dtype == "nvfp4"
    assert contract.quant_spec().activation_dtype == "fp16"
    assert contract.quant_spec().activation_granularity == "per_tensor"
    assert contract.to_dict()["scale_dtype"] == "float_ue4m3"
    assert contract.to_dict()["tile_shapes"] == [[128, 128, 128], [128, 128, 256]]
    assert contract.to_dict()["maturity"] == "metadata_only"


def test_sm90_dense_and_sm120_fp8_candidates_are_architecture_isolated() -> None:
    registry = default_registry()
    sm90_spec = GemmSpec(
        problem=GemmProblem(m=128, n=128, k=128, sm=90, device="cuda:0"),
        quant=QuantSpec(
            weight_dtype="fp16",
            activation_dtype="fp16",
            output_dtype="fp16",
        ),
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    sm120_spec = GemmSpec(
        problem=GemmProblem(m=128, n=128, k=128, sm=120, device="cuda:0"),
        quant=QuantSpec(
            weight_dtype="fp8_e4m3",
            activation_dtype="fp8_e4m3",
            output_dtype="fp32",
            weight_granularity="blockwise",
            activation_granularity="blockwise",
            group_size=128,
            weight_scale_source="weight_offline",
            activation_scale_source="activation_static",
            storage_layout="xqt_fp8_rowmajor_v1",
            pack_version="xqt-sm120-fp8-blockwise-v1",
        ),
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )

    sm90_names = [entry.name for entry in select_kernel(sm90_spec, registry=registry)]
    sm120_names = [entry.name for entry in select_kernel(sm120_spec, registry=registry)]

    assert sm90_names[0] == "sm90_dense_fp16_wgmma"
    assert "sm120_fp8_e4m3_tcgen05" not in sm90_names
    assert sm120_names[0] == "sm120_fp8_e4m3_tcgen05"
    assert "sm90_dense_fp16_wgmma" not in sm120_names


def test_sm120_fp8_metadata_candidate_falls_back_to_reference() -> None:
    torch.manual_seed(20260813)
    m, n, k = 3, 5, 37
    quant = QuantSpec(
        weight_dtype="fp8_e4m3",
        activation_dtype="fp8_e4m3",
        output_dtype="fp32",
        weight_granularity="blockwise",
        activation_granularity="blockwise",
        group_size=128,
        weight_scale_source="weight_offline",
        activation_scale_source="activation_dynamic",
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-sm120-fp8-blockwise-v1",
    )
    weight = torch.randn(n, k, dtype=torch.float32)
    encoded = quantize_fp8(
        weight,
        format_name="fp8_e4m3",
        granularity="blockwise",
        role="weight",
        source="weight_offline",
        scale=calibrate_fp8_scale(
            weight,
            format_name="fp8_e4m3",
            granularity="blockwise",
            role="weight",
            block_k=128,
        ),
        block_k=128,
    )
    padded = torch.nn.functional.pad(encoded.storage, (0, 128 - k))
    packed = build_packed_weight(
        padded,
        logical_shape=(n, k),
        spec=quant,
        scales=encoded.scale,
        padded_k=128,
        storage_layout="xqt_fp8_rowmajor_v1",
        pack_version="xqt-sm120-fp8-blockwise-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, sm=120, device="cuda:0"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp32"),
    )

    result = dispatch_gemm(torch.randn(m, k), packed, spec=spec)

    assert result.report.selected_kernel == "fp8_blockwise_reference"
    assert result.report.native is False
    assert "sm120_fp8_e4m3_tcgen05" in result.report.fallback_chain
    assert result.output.shape == (m, n)


def test_sm120_nvfp4_metadata_candidate_is_architecture_isolated() -> None:
    registry = default_registry()
    spec = GemmSpec(
        problem=GemmProblem(m=128, n=128, k=128, sm=120, device="cuda:0"),
        quant=QuantSpec(
            weight_dtype="nvfp4",
            activation_dtype="fp16",
            output_dtype="bf16",
            weight_granularity="groupwise",
            activation_granularity="per_tensor",
            group_size=16,
            weight_scale_source="weight_offline",
            activation_scale_source="none",
            storage_layout="xqt_fp4_nk_v1",
            pack_version="xqt-sm120-nvfp4-v1",
        ),
        epilogue=EpilogueSpec(output_dtype="bf16"),
    )

    names = [entry.name for entry in select_kernel(spec, registry=registry)]

    assert names[0] == "sm120_nvfp4_tcgen05"
    assert "nvfp4_reference" in names
    assert "sm100_nvfp4_cutlass" not in names


def test_arch_sources_keep_runtime_probe_abi_explicit() -> None:
    root = Path(__file__).resolve().parents[3]
    sm90_source = (root / "xqt/kernels/jit/csrc/gemm/sm90_fp8_wgmma.cu").read_text()
    sm120_source = (root / "xqt/kernels/jit/csrc/gemm/sm120_gemm.cu").read_text()

    assert "Gemm::get_workspace_size(arguments)" in sm90_source
    assert "gemm.can_implement(arguments)" in sm90_source
    assert "gemm.initialize(arguments, workspace.get(), stream)" in sm90_source
    assert "gemm.run(stream)" in sm90_source
    assert "cudaStream_t stream" in sm90_source
    assert "xqt_sm90_fp8_e4m3_fp16_wgmma_run" in sm90_source
    assert "xqt_sm90_dense_cooperative_128_fp16_wgmma_run" in sm90_source
    assert "xqt_sm90_dense_pingpong_128_bf16_wgmma_run" in sm90_source
    assert "xqt_sm90_dense_cooperative_256_bf16_wgmma_run" in sm90_source
    assert "xqt_sm90_dense_cooperative_128_cluster2x2_bf16_wgmma_run" in sm90_source
    assert "xqt_sm90_dense_pingpong_64_cluster2x2_bf16_wgmma_run" in sm90_source
    assert "xqt_sm90_fp8_e4m3_groupwise_pingpong_bf16_run" in sm90_source
    assert "xqt_sm90_fp8_e4m3_groupwise_cooperative_256_bf16_run" in sm90_source
    assert "xqt_sm120_fp8_e4m3_blockwise_bf16_run" in sm120_source
    assert "xqt_sm120_fp8_e4m3_blockwise_pingpong_bf16_run" in sm120_source
    assert "xqt_sm120_fp8_e4m3_groupwise_bf16_run" in sm120_source
    assert "xqt_sm120_fp8_e4m3_groupwise_pingpong_bf16_run" in sm120_source
    assert "xqt_sm120_nvfp4_bf16_run" in sm120_source
    assert "xqt_sm120_nvfp4_k256_bf16_run" in sm120_source
    assert "xqt_sm120_nvfp4_pingpong_bf16_run" in sm120_source
    assert "xqt_sm120_nvfp4_k256_pingpong_bf16_run" in sm120_source
    assert "float_ue4m3_t" in sm120_source
    assert "tile_atom_to_shape_SFA" in sm120_source
    assert "gemm.initialize(arguments, workspace.get(), stream)" in sm120_source
    assert "gemm.run(stream)" in sm120_source


def test_cutlass_blockscale_shapes_are_explicitly_layout_specific() -> None:
    assert cutlass_blockscale_shape(256, 4096) == (2, 32)
    assert cutlass_blockscale_shape(256, 4096, row_block=64) == (4, 32)
    assert cutlass_blockscale_shape(256, 4096, row_block=1) == (256, 32)
    assert cutlass_blockscale_shape(257, 4097) == (3, 33)


def test_cutlass_blockscale_storage_is_column_major_over_block_grid() -> None:
    from xqt.kernels.ops._impl.gemm_backends._sm1xx_runtime import flatten_cutlass_blockscale_grid

    grid = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert flatten_cutlass_blockscale_grid(grid).tolist() == [
        1.0,
        4.0,
        2.0,
        5.0,
        3.0,
        6.0,
    ]
    assert flatten_cutlass_blockscale_grid(grid, major="k").tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]


def test_cutlass_nvfp4_scale_layout_matches_sm120_sfvec16_offsets() -> None:
    from xqt.kernels.ops._impl.gemm_backends._sm1xx_runtime import (
        cutlass_nvfp4_scale_shape,
        cutlass_nvfp4_scale_storage_offset,
        cutlass_nvfp4_scale_storage_size,
    )

    assert cutlass_nvfp4_scale_shape(129, 257) == (129, 17)
    assert cutlass_nvfp4_scale_storage_size(256, 256) == 4096
    assert cutlass_nvfp4_scale_storage_offset(
        0,
        0,
        padded_rows=256,
        padded_cols=256,
    ) == 0
    assert cutlass_nvfp4_scale_storage_offset(
        16,
        0,
        padded_rows=256,
        padded_cols=256,
    ) == 256
    assert cutlass_nvfp4_scale_storage_offset(
        0,
        8,
        padded_rows=256,
        padded_cols=256,
    ) == 1024
    assert cutlass_nvfp4_scale_storage_offset(
        128,
        0,
        padded_rows=256,
        padded_cols=256,
    ) == 2048


def test_sm1xx_runtime_rejects_canonical_per_row_scale_shape() -> None:
    from xqt.kernels.ops._impl.gemm_backends._sm1xx_runtime import prepare_cutlass_blockscales
    from xqt.core.errors import XQTBackendError

    with pytest.raises(XQTBackendError, match="canonical XQT per-row scales"):
        prepare_cutlass_blockscales(
            torch.ones((256, 32), dtype=torch.float32),
            rows=256,
            cols=4096,
            name="scale_a",
        )
