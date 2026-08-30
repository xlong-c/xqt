from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops._impl.tilelang.attention import fused_attention_forward_tilelang
from xqt.kernels.ops._impl.tilelang.linear import dense_linear_epilogue_tilelang
from xqt.kernels.wrappers.runtime import replay_cuda_graph_tensor_callable
from xqt.kernels.wrappers.triton_wrappers import _TritonLinearWrapper
from xqt.kernels.wrappers.attention import _TileLangAttentionWrapper
from xqt.runtime.modules import KvScaleAttention


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this P3 boundary test",
)


def _other_sm() -> str:
    major, minor = torch.cuda.get_device_capability()
    actual = f"sm_{major}{minor}"
    return "sm_90" if actual != "sm_90" else "sm_89"


def _attention_wrapper(
    attention: nn.MultiheadAttention,
    *,
    target_arch: str | None = None,
) -> _TileLangAttentionWrapper:
    settings: dict[str, object] = {
        "attention_fastpath": "tilelang",
        "preferred_patterns": ["attention"],
    }
    if target_arch is not None:
        settings["target_arch"] = target_arch
    return _TileLangAttentionWrapper(
        attention,
        fallback="eager",
        settings=settings,
    )


@pytest.mark.parametrize("mask_kind", ["attn_mask", "key_padding_mask"])
def test_attention_wrapper_mask_fallback_preserves_mha_contract(
    mask_kind: str,
) -> None:
    torch.manual_seed(11)
    attention = nn.MultiheadAttention(
        16,
        4,
        batch_first=True,
        dropout=0.0,
    ).eval()
    wrapper = _attention_wrapper(attention)
    query = torch.randn(2, 5, 16)
    if mask_kind == "attn_mask":
        kwargs: dict[str, torch.Tensor] = {
            "attn_mask": torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1),
        }
    else:
        kwargs = {
            "key_padding_mask": torch.tensor(
                [[False, False, False, True, True], [False, True, False, False, True]],
                dtype=torch.bool,
            ),
        }

    with torch.no_grad():
        actual, actual_weights = wrapper(
            query,
            need_weights=True,
            average_attn_weights=False,
            **kwargs,
        )
        expected, expected_weights = attention(
            query,
            query,
            query,
            need_weights=True,
            average_attn_weights=False,
            **kwargs,
        )

    torch.testing.assert_close(actual, expected)
    assert actual_weights is not None
    assert expected_weights is not None
    torch.testing.assert_close(actual_weights, expected_weights)
    metadata = wrapper.execution_metadata()
    assert metadata["execution_mode"] == "reference_fallback"
    assert metadata["execution_reason"] == "attention_mask_requires_reference"
    assert metadata["selected_fastpath"] == "eager_reference_fallback"


def test_attention_wrapper_self_attention_and_autograd_fallback() -> None:
    torch.manual_seed(12)
    attention = nn.MultiheadAttention(
        16,
        4,
        batch_first=True,
        dropout=0.0,
    ).eval()
    wrapper = _attention_wrapper(attention)
    query = torch.randn(2, 4, 16, requires_grad=True)

    actual, _ = wrapper(query, need_weights=False)
    assert actual.requires_grad
    assert wrapper.execution_metadata()["execution_reason"] == "autograd_unsupported"
    actual.sum().backward()
    assert query.grad is not None


def test_attention_wrapper_handles_noncontiguous_dynamic_model_inputs() -> None:
    torch.manual_seed(13)
    attention = nn.MultiheadAttention(
        16,
        4,
        batch_first=True,
        dropout=0.0,
    ).eval()
    wrapper = _attention_wrapper(attention)
    storage = torch.randn(2, 12, 16)
    query = storage[:, ::2, :]
    assert not query.is_contiguous()

    with torch.no_grad():
        actual, _ = wrapper(query, need_weights=False)
        expected, _ = attention(query, query, query, need_weights=False)

    torch.testing.assert_close(actual, expected)
    assert wrapper.execution_metadata()["execution_mode"] == "reference_fallback"


def test_attention_wrapper_dynamic_model_block_matches_reference() -> None:
    class _Block(nn.Module):
        def __init__(self, attention: nn.Module) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(16)
            self.attention = attention
            self.proj = nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normalized = self.norm(x)
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            return x + self.proj(attended)

    torch.manual_seed(14)
    reference_attention = nn.MultiheadAttention(
        16,
        4,
        batch_first=True,
        dropout=0.0,
    ).eval()
    optimized_attention = nn.MultiheadAttention(
        16,
        4,
        batch_first=True,
        dropout=0.0,
    ).eval()
    optimized_attention.load_state_dict(reference_attention.state_dict())
    reference = _Block(reference_attention).eval()
    optimized = _Block(_attention_wrapper(optimized_attention)).eval()
    optimized.proj.load_state_dict(reference.proj.state_dict())
    optimized.norm.load_state_dict(reference.norm.state_dict())

    with torch.no_grad():
        for seq_len in (3, 7):
            inputs = torch.randn(2, seq_len, 16)
            actual = optimized(inputs)
            expected = reference(inputs)
            torch.testing.assert_close(actual, expected)


class _FakeGraph:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def replay(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
        sleep(0.005)
        with self._lock:
            self.active -= 1


def test_cuda_graph_replay_serializes_shared_state_across_threads() -> None:
    fake_graph = _FakeGraph()
    output = torch.zeros(4)
    state = {
        "graph": fake_graph,
        "static_args": (torch.zeros(4),),
        "static_output": output,
    }

    def replay(index: int) -> torch.Tensor:
        return replay_cuda_graph_tensor_callable(
            state,
            (torch.full((4,), float(index)),),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(replay, range(12)))

    assert len(results) == 12
    assert all(result is output for result in results)
    assert fake_graph.calls == 12
    assert fake_graph.max_active == 1


def test_kv_scale_attention_reports_offline_phase_and_cache_boundary() -> None:
    torch.manual_seed(15)
    entity = KvScaleAttention(
        8,
        heads=2,
        k_scale=1.0,
        v_scale=1.0,
    ).eval()
    prefill_input = torch.randn(2, 5, 8)
    decode_input = torch.randn(2, 1, 8)

    prefill_output = entity.prefill(prefill_input)
    prefill_report = entity.report()
    decode_output = entity.decode(decode_input)
    decode_report = entity.report()

    assert prefill_output.shape == prefill_input.shape
    assert decode_output.shape == decode_input.shape
    assert prefill_report["phase"] == "prefill"
    assert decode_report["phase"] == "decode"
    assert decode_report["serving_cache"] == {
        "implemented": False,
        "reason": "model_side_entity_does_not_manage_serving_cache",
    }


@requires_cuda
def test_attention_wrapper_explicit_other_sm_uses_reference_fallback() -> None:
    dtype = torch.float16
    attention = nn.MultiheadAttention(
        32,
        4,
        batch_first=True,
        dropout=0.0,
        device="cuda",
        dtype=dtype,
    ).eval()
    wrapper = _attention_wrapper(attention, target_arch=_other_sm()).to(
        device="cuda",
        dtype=dtype,
    )
    query = torch.randn(1, 5, 32, device="cuda", dtype=dtype)

    with torch.inference_mode():
        actual, _ = wrapper(query, need_weights=False)
        expected, _ = attention(query, query, query, need_weights=False)

    torch.testing.assert_close(actual, expected)
    reason = wrapper.execution_metadata()["execution_reason"]
    assert str(reason).startswith("target_arch_mismatch:")


@requires_cuda
def test_triton_linear_other_sm_and_autograd_boundaries() -> None:
    dtype = torch.float16
    linear = nn.Linear(16, 24, device="cuda", dtype=dtype).eval()
    mismatch_wrapper = _TritonLinearWrapper(
        linear,
        fallback="eager",
        settings={
            "precision": "auto",
            "preferred_patterns": ["linear"],
            "target_arch": _other_sm(),
        },
    )
    query = torch.randn(2, 16, device="cuda", dtype=dtype)
    with torch.inference_mode():
        actual = mismatch_wrapper(query)
        expected = F.linear(query, linear.weight, linear.bias)
    torch.testing.assert_close(actual, expected)
    assert str(mismatch_wrapper.execution_metadata()["execution_reason"]).startswith(
        "Triton linear runtime fallback: target_arch_mismatch:"
    )

    autograd_wrapper = _TritonLinearWrapper(
        linear,
        fallback="eager",
        settings={
            "precision": "auto",
            "preferred_patterns": ["linear"],
        },
    )
    grad_input = torch.randn(2, 16, device="cuda", dtype=dtype, requires_grad=True)
    output = autograd_wrapper(grad_input)
    output.float().sum().backward()
    assert grad_input.grad is not None
    assert autograd_wrapper.execution_metadata()["execution_reason"] == (
        "Triton linear kernel is inference-only; autograd input requires reference fallback."
    )


@requires_cuda
def test_tilelang_low_level_other_sm_is_rejected_before_compile() -> None:
    wrong_arch = _other_sm()
    query = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    with pytest.raises(XQTBackendError, match="target architecture is not executable"):
        fused_attention_forward_tilelang(query, key, value, target_arch=wrong_arch)

    linear_input = torch.randn(1, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(24, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(XQTBackendError, match="target architecture is not executable"):
        dense_linear_epilogue_tilelang(
            linear_input,
            weight,
            target_arch=wrong_arch,
        )


@requires_cuda
def test_tilelang_attention_rejects_caller_owned_autograd_input() -> None:
    query = torch.randn(
        1,
        1,
        8,
        16,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    with pytest.raises(XQTBackendError, match="inference-only"):
        fused_attention_forward_tilelang(query, query, query)
