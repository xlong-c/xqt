from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm import PackedWeight, PackedWeightMetadata
from xqt.runtime.modules import AWQW4A16Linear


def _weight(*, n: int = 64, k: int = 128) -> PackedWeight:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(73)
    codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.uint8)
    qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales = torch.rand((n, k // 64), generator=generator) * 0.1 + 0.01
    zero_points = torch.randint(
        0,
        16,
        tuple(scales.shape),
        generator=generator,
        dtype=torch.int16,
    ).float()
    return PackedWeight(
        qweight=qweight,
        scales=scales,
        zero_points=zero_points,
        metadata=PackedWeightMetadata(
            logical_shape=(n, k),
            storage_layout="xqt_int4_nk_v1",
            pack_version="xqt-w4a16-awq-v1",
            weight_dtype="int4",
            padded_k=k,
            group_size=64,
            packed_bits=4,
            nibble_order="low_high",
            nibble_signed=False,
        ),
    )


def _cuda_weight() -> PackedWeight:
    weight = _weight()
    return replace(
        weight,
        qweight=weight.qweight.cuda(),
        scales=weight.scales.cuda(),
        zero_points=weight.zero_points.cuda(),
    )


def test_constructor_rejects_wrong_bias_size_before_cuda_pack() -> None:
    with pytest.raises(ValueError, match="bias size"):
        AWQW4A16Linear(_weight(), bias=torch.zeros(63))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_rejects_training_and_autograd() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    module = AWQW4A16Linear(_cuda_weight()).train()
    inputs = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="requires eval mode"):
        with torch.no_grad():
            module(inputs)

    module.eval()
    with pytest.raises(RuntimeError, match="requires no_grad"):
        module(inputs.requires_grad_(True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warmup_bind_metadata_and_cache_invalidation() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    module = AWQW4A16Linear(_cuda_weight(), bias=bias).eval()
    module.warmup(rows=1, dtype=torch.float16)
    first_bound = module.bind(rows=1, dtype=torch.float16)
    inputs = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        first = module(inputs)
        bound_first = first_bound(inputs)
    torch.testing.assert_close(first, bound_first, rtol=0.0, atol=0.0)
    metadata = module.execution_metadata()
    assert metadata["native"] is True
    assert metadata["rows"] == 1
    assert metadata["dtype"] == "fp16"
    assert metadata["bias"] is True

    with torch.no_grad():
        module.bias.add_(0.25)
        updated = module(inputs)
    assert not torch.equal(first, updated)
    second_bound = module.bind(rows=1, dtype=torch.float16)
    assert second_bound is not first_bound

    module = module.to(dtype=torch.bfloat16)
    assert module.execution_metadata()["native"] is False
    module.warmup(rows=1, dtype=torch.bfloat16)
    third_bound = module.bind(rows=1, dtype=torch.bfloat16)
    with torch.no_grad():
        bf16_output = module(inputs.to(torch.bfloat16))
        bound_bf16 = third_bound(inputs.to(torch.bfloat16))
    torch.testing.assert_close(bf16_output, bound_bf16, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bound_residual_matches_unfused_add() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    module = AWQW4A16Linear(_cuda_weight()).eval()
    inputs = torch.randn(1, 128, device="cuda", dtype=torch.float16) * 0.1
    residual = torch.randn(1, 64, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        expected = residual + module(inputs)
    runner = module.bind_residual(rows=1, dtype=torch.float16)
    actual = residual.clone()
    runner(inputs, actual)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    with pytest.raises(XQTBackendError, match=r"contiguous \[1,64\]"):
        runner(inputs, torch.zeros(64, device="cuda", dtype=torch.float16))
    with pytest.raises(XQTBackendError, match="static shape/dtype/device"):
        runner(torch.randn(2, 128, device="cuda", dtype=torch.float16), actual)


def test_residual_binding_rejects_native_only_modules() -> None:
    module = object.__new__(AWQW4A16Linear)
    module._native_only = True
    with pytest.raises(RuntimeError, match="residual epilogue"):
        AWQW4A16Linear.bind_residual(module, rows=1, dtype=torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bound_input_contract_is_explicit() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    module = AWQW4A16Linear(_cuda_weight()).eval()
    bound = module.bind(rows=1, dtype=torch.float16)
    with pytest.raises(XQTBackendError, match="static shape/dtype/device"):
        bound(torch.randn(2, 128, device="cuda", dtype=torch.float16))


def test_warmup_rejects_unsupported_rows_and_dtype() -> None:
    module = object.__new__(AWQW4A16Linear)
    with pytest.raises(ValueError, match="between 1 and 8"):
        AWQW4A16Linear.warmup(module, rows=9, dtype=torch.float16)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        AWQW4A16Linear.warmup(module, rows=1, dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_freeze_native_inference_preserves_output_and_releases_canonical_state() -> (
    None
):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    module = AWQW4A16Linear(_cuda_weight(), bias=bias).eval()
    inputs = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    canonical_bytes = module._canonical_storage_bytes()

    with torch.inference_mode():
        expected = module(inputs).clone()
    released_bytes = module.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        rows=(1,),
    )
    with torch.inference_mode():
        actual = module(inputs).clone()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert released_bytes == canonical_bytes
    assert released_bytes > 0
    assert module.native_only is True
    assert list(module.parameters()) == []
    assert list(module.buffers()) == []
    assert module.qweight is None
    assert module.canonical_qweight is None
    assert module.weight_scale is None
    assert module.weight_zero_point is None
    assert module.bias is None
    metadata = module.execution_metadata()
    assert metadata["native"] is True
    assert metadata["native_only"] is True
    assert metadata["native_only_rows"] == [1]
    assert metadata["native_only_device"] == "cuda:0"
    assert metadata["native_only_dtype"] == "torch.float16"
    assert metadata["native_only_released_bytes"] == released_bytes
    assert metadata["bias"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_freeze_native_inference_rejects_mismatch_state_changes_and_mutation() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    module = AWQW4A16Linear(_cuda_weight()).eval()
    canonical_state = module.state_dict()
    inputs = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    module.freeze_native_inference(
        device="cuda",
        dtype=torch.float16,
        rows=1,
    )

    with torch.inference_mode(), pytest.raises(RuntimeError, match="input dtype"):
        module(inputs.bfloat16())
    with torch.inference_mode(), pytest.raises(RuntimeError, match="input device"):
        module(torch.randn(1, 128, dtype=torch.float16))
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="rows=2 were not frozen"),
    ):
        module(torch.randn(2, 128, device="cuda", dtype=torch.float16))
    with pytest.raises(RuntimeError, match="requires no_grad"):
        module(inputs)
    with pytest.raises(RuntimeError, match="cannot be moved or cast"):
        module.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="inference-only"):
        module.train()
    with pytest.raises(RuntimeError, match="cannot be serialized"):
        module.state_dict()
    with pytest.raises(RuntimeError, match="cannot load state"):
        module.load_state_dict(canonical_state)
    with pytest.raises(RuntimeError, match="cannot prepare canonical parameters"):
        module._prepared(dtype=torch.float16, spec=object())  # type: ignore[arg-type]

    with torch.no_grad():
        module._native_only_tensors[0].view(-1)[0].add_(1)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="state was mutated"):
        module(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_signed_groupwise_storages_match_individual_decodes() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires sm_89")
    from types import SimpleNamespace

    def storage(n: int, seed: int) -> SimpleNamespace:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        codes = torch.randint(0, 16, (n, 128), generator=generator, dtype=torch.uint8)
        qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        scales = torch.rand((n, 2), generator=generator) * 0.1 + 0.01
        return SimpleNamespace(
            quantized_weight=qweight.cuda(),
            weight_scale=scales.cuda(),
            bias=None,
            bits=4,
            group_size=64,
            padded_input_features=128,
            input_features=128,
            output_features=n,
        )

    first_storage = storage(64, 11)
    second_storage = storage(32, 12)
    fused = AWQW4A16Linear.from_signed_groupwise_storages(
        [first_storage, second_storage]
    ).eval()
    parts = [
        AWQW4A16Linear.from_signed_groupwise_storage(first_storage).eval(),
        AWQW4A16Linear.from_signed_groupwise_storage(second_storage).eval(),
    ]

    inputs = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        expected = torch.cat([part(inputs) for part in parts], dim=-1)
        actual = fused(inputs)
    assert fused.output_features == 96
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_signed_groupwise_storages_reject_mismatched_input_features() -> None:
    from types import SimpleNamespace

    def storage(n: int, k: int) -> SimpleNamespace:
        codes = torch.zeros((n, k), dtype=torch.uint8)
        qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        return SimpleNamespace(
            quantized_weight=qweight,
            weight_scale=torch.ones((n, k // 64)),
            bias=None,
            bits=4,
            group_size=64,
            padded_input_features=k,
            input_features=k,
            output_features=n,
        )

    with pytest.raises(XQTBackendError, match="matching input features"):
        AWQW4A16Linear.from_signed_groupwise_storages(
            [storage(64, 128), storage(32, 192)]
        )
