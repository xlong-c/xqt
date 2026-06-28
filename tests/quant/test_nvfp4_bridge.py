from __future__ import annotations

import torch

from xqt.quant import (
    NVFP4LinearBridge,
    bridge_module_to_nvfp4_linear,
    expand_group_scale,
    infer_nvfp4_tensor_layout,
    unpack_nvfp4e2m1,
)


class _FakeCompressedNVFP4Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 4
        self.out_features = 2
        self.register_buffer(
            "qweight",
            torch.tensor([[0x10, 0x32], [0x54, 0x76]], dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.tensor(
                [
                    [[2.0], [4.0]],
                    [[1.0], [3.0]],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "weight_global_scale",
            torch.tensor([0.5], dtype=torch.float32),
        )
        self.register_buffer("bias", torch.tensor([0.25, -0.5], dtype=torch.float32))


class _FakeCompressedNVFP4LinearWithFlatScale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 4
        self.out_features = 2
        self.register_buffer(
            "weight_packed",
            torch.tensor([[0x10, 0x32], [0x54, 0x76]], dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.tensor(
                [
                    [2.0, 4.0],
                    [1.0, 3.0],
                ],
                dtype=torch.float32,
            ).to(torch.float8_e4m3fn),
        )
        self.register_buffer("weight_global_scale", torch.tensor([0.5], dtype=torch.float32))


def test_unpack_nvfp4e2m1_matches_enumerated_codebook() -> None:
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)

    unpacked = unpack_nvfp4e2m1(packed, input_features=16)

    expected = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]],
        dtype=torch.float32,
    )
    assert torch.equal(unpacked, expected)


def test_expand_group_scale_expands_per_group_tensor() -> None:
    scale = torch.tensor([[[2.0], [4.0]]], dtype=torch.float32)

    expanded = expand_group_scale(scale, group_size=2, input_features=4)

    assert torch.equal(expanded, torch.tensor([[2.0, 2.0, 4.0, 4.0]], dtype=torch.float32))


def test_expand_group_scale_accepts_compressed_tensors_flat_scale() -> None:
    scale = torch.tensor([[2.0, 4.0]], dtype=torch.float32).to(torch.float8_e4m3fn)

    expanded = expand_group_scale(scale, group_size=2, input_features=4)

    assert torch.equal(
        expanded.float(),
        torch.tensor([[2.0, 2.0, 4.0, 4.0]], dtype=torch.float32),
    )


def test_infer_nvfp4_tensor_layout_from_fake_module() -> None:
    module = _FakeCompressedNVFP4Linear()

    layout = infer_nvfp4_tensor_layout(module)

    assert layout is not None
    assert layout.packed_weight_name == "qweight"
    assert layout.weight_scale_name == "weight_scale"
    assert layout.weight_global_scale_name == "weight_global_scale"
    assert layout.group_size == 2
    assert layout.input_features == 4
    assert layout.output_features == 2


def test_infer_nvfp4_tensor_layout_from_compressed_tensors_flat_scale() -> None:
    module = _FakeCompressedNVFP4LinearWithFlatScale()

    layout = infer_nvfp4_tensor_layout(module)

    assert layout is not None
    assert layout.packed_weight_name == "weight_packed"
    assert layout.weight_scale_name == "weight_scale"
    assert layout.weight_global_scale_name == "weight_global_scale"
    assert layout.group_size == 2
    assert layout.input_features == 4
    assert layout.output_features == 2


def test_bridge_module_to_nvfp4_linear_exposes_tilelang_args() -> None:
    module = _FakeCompressedNVFP4Linear()

    bridge = bridge_module_to_nvfp4_linear(module)

    assert isinstance(bridge, NVFP4LinearBridge)
    packed_weight, scale, bias, activation, input_features, group_size, weight_global_scale = (
        bridge.tilelang_packed_nvfp4_dequant_gemm_args(
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
    )
    assert activation is None
    assert input_features == 4
    assert group_size == 2
    assert packed_weight.dtype == torch.uint8
    assert scale.dtype == torch.float16
    assert bias is not None and bias.dtype == torch.float16
    assert weight_global_scale is not None and weight_global_scale.dtype == torch.float16


def test_bridge_module_to_nvfp4_linear_normalizes_flat_scale_for_tilelang() -> None:
    module = _FakeCompressedNVFP4LinearWithFlatScale()

    bridge = bridge_module_to_nvfp4_linear(module)

    assert isinstance(bridge, NVFP4LinearBridge)
    _, scale, _, _, _, group_size, _ = bridge.tilelang_packed_nvfp4_dequant_gemm_args(
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert group_size == 2
    assert scale.shape == (2, 2, 1)
    assert scale.dtype == torch.float16


def test_nvfp4_linear_bridge_forward_dequantizes_weight() -> None:
    module = _FakeCompressedNVFP4Linear()
    bridge = bridge_module_to_nvfp4_linear(module)
    assert bridge is not None
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    output = bridge(x)

    dequantized = bridge.dequantize_weight()
    expected = torch.nn.functional.linear(x, dequantized, bridge.bias)
    assert torch.allclose(output, expected)
