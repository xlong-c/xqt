"""T12: C4 process storage can feed tilelang/triton packed dequant arg APIs."""

from __future__ import annotations

import torch
from torch import nn

from xqt.contracts.layout_kernel_report import layout_report_from_module_shapes
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.runtime.bridges.hf_int4_layout import process_weights_after_loading
from xqt.runtime.bridges.hf_int4_pack import awq_reverse_pack_order


def _pack_gptq_int32(codes_out_in: torch.Tensor, *, bits: int = 4) -> torch.Tensor:
    pack = 32 // bits
    out_features, in_features = codes_out_in.shape
    unsigned = codes_out_in.to(torch.int32)
    mat = unsigned.T.contiguous()
    reshaped = mat.reshape(in_features // pack, pack, out_features)
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    return (reshaped << shifts.view(1, -1, 1)).sum(dim=1).to(torch.int32)


def test_process_product_exposes_tilelang_and_triton_packed_args() -> None:
    bits = 4
    group_size = 32
    in_features = 64
    out_features = 32
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    qweight = _pack_gptq_int32(codes, bits=bits)
    scales = torch.rand(in_features // group_size, out_features) * 0.05 + 0.01
    module = process_weights_after_loading(
        method="gptq",
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        scales=scales,
        qzeros=None,
        bias=None,
        g_idx=None,
    )
    assert isinstance(module, AWQGPTQWeightOnlyLinear)
    device = torch.device("cpu")
    dtype = torch.float32
    packed_t = module.tilelang_packed_dequant_gemm_args(dtype=dtype, device=device)
    packed_r = module.triton_packed_dequant_gemm_args(dtype=dtype, device=device)
    assert packed_t[0].ndim == 2
    assert packed_r[0].shape == packed_t[0].shape
    report = layout_report_from_module_shapes(
        bits=bits,
        group_size=group_size,
        symmetric=True,
        zero_point=False,
        desc_act=False,
        g_idx_applied=False,
        out_features=out_features,
        in_features=in_features,
        padded_in_features=module.padded_input_features,
        storage_layout="xqt_awq_gptq_int4_v1",
        selected_kernel="tilelang_packed_dequant_gemm",
        fallback_reason="cpu_reference_dequant_fp16",
        scale_time="weight_offline",
    ).to_dict()
    assert report["selected_kernel"] == "tilelang_packed_dequant_gemm"
    assert report["fallback_reason"] == "cpu_reference_dequant_fp16"
    # Forward still works via reference apply
    y = module(torch.randn(2, in_features))
    assert y.shape == (2, out_features)
