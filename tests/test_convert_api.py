from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import xqt
from xqt.conversion import ConvertResult, FeedForwardPrecisionPolicy, PrecisionPolicy
from xqt.quant import ReferenceFP4Linear


def test_xqt_convert_is_available_via_lazy_top_level_attribute() -> None:
    assert callable(xqt.convert)


def test_xqt_nn_exposes_operator_facades() -> None:
    assert xqt.nn.Linear is nn.Linear
    assert xqt.nn.Conv2d is nn.Conv2d
    assert xqt.nn.LayerNorm is nn.LayerNorm
    assert xqt.nn.FeedForward.__name__ == "FeedForward"
    assert xqt.nn.RMSNorm.__name__ == "RMSNorm"


def test_xqt_top_level_import_remains_lightweight_with_lazy_convert() -> None:
    script = (
        "import xqt; "
        "assert 'convert' not in xqt.__all__; "
        "assert callable(xqt.convert); "
        "assert xqt.nn.Linear.__name__ == 'Linear'"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_convert_linear_torch_engine_returns_copy_by_default() -> None:
    module = nn.Linear(4, 3)
    converted = xqt.convert(module, engine="torch")

    assert isinstance(converted, nn.Linear)
    assert converted is not module
    sample = torch.randn(2, 4)
    torch.testing.assert_close(converted(sample), module(sample))


def test_convert_linear_torch_engine_can_return_result() -> None:
    module = nn.Linear(4, 3)
    result = xqt.convert(
        module,
        engine="torch",
        policy=PrecisionPolicy(
            activation="fp16",
            weight="fp16",
            mma="fp16",
            accum="fp32",
            output="fp16",
        ),
        return_result=True,
    )

    assert isinstance(result, ConvertResult)
    assert result.converted is False
    assert result.contract.operator_kind == "linear"
    assert result.contract.policy.output == "fp16"
    assert result.engine == "torch"
    assert result.to_dict()["engine"] == "torch"
    assert (
        result.report["reason"]
        == "torch engine uses a runtime-configured Linear wrapper"
    )
    assert result.report["runtime_config"]["mma"] == "fp16"


def test_convert_reference_fp4_linear_tilelang_returns_operator_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    module = ReferenceFP4Linear.from_linear(nn.Linear(8, 4), group_size=4)

    def fake_materialize(
        module_arg: nn.Module,
        target: object,
    ) -> tuple[nn.Module, float | None]:
        captured["module"] = module_arg
        captured["target"] = target
        return module_arg, None

    monkeypatch.setattr(
        "xqt.conversion.materialize_operator_candidate_model",
        fake_materialize,
    )

    result = xqt.convert(
        module,
        engine="tilelang",
        target_arch="sm_89",
        policy=PrecisionPolicy(weight="fp4", output="fp16"),
        return_result=True,
    )

    assert isinstance(result, ConvertResult)
    assert result.converted is True
    assert result.contract.operator_kind == "linear"
    assert result.contract.weight_spec.storage_dtype == "fp4_packed"
    target = captured["target"]
    assert getattr(target, "engine") == "tilelang"
    assert getattr(target, "patterns") == ["fp4_packed_dequant_gemm_epilogue"]
    assert getattr(target, "tilelang")["target_arch"] == "sm_89"


def test_convert_linear_accepts_abco_precision_mapping() -> None:
    module = nn.Linear(8, 4)

    result = xqt.convert(
        module,
        engine="torch",
        policy={
            "A": "nvfp4",
            "B": "fp16",
            "C": "fp32",
            "MMA": "fp16",
            "ACCUM": "fp32",
            "O": "fp16",
        },
        return_result=True,
    )

    assert isinstance(result, ConvertResult)
    assert result.contract.policy.activation == "nvfp4"
    assert result.contract.policy.weight == "fp16"
    assert result.contract.policy.bias == "fp32"
    assert result.report["runtime_config"] == {
        "engine": "torch",
        "activation": "nvfp4",
        "weight": "fp16",
        "bias": "fp32",
        "mma": "fp16",
        "accum": "fp32",
        "output": "fp16",
    }


def test_precision_policy_from_matmul_accepts_abco_roles() -> None:
    policy = PrecisionPolicy.from_matmul(
        A="nvfp4",
        B="fp16",
        C="fp32",
        mma="fp16",
        accum="fp32",
        O="bf16",
    )

    assert policy.to_dict() == {
        "activation": "nvfp4",
        "weight": "fp16",
        "bias": "fp32",
        "mma": "fp16",
        "accum": "fp32",
        "output": "bf16",
    }


def test_convert_conv2d_tilelang_delegates_to_materializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    module = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def fake_materialize(
        module_arg: nn.Module,
        target: object,
    ) -> tuple[nn.Module, float | None]:
        captured["module"] = module_arg
        captured["target"] = target
        return module_arg, None

    monkeypatch.setattr(
        "xqt.conversion.materialize_operator_candidate_model",
        fake_materialize,
    )

    result = xqt.convert(module, engine="tilelang", return_result=True)

    assert isinstance(result, ConvertResult)
    assert result.contract.operator_kind == "conv2d"
    assert getattr(captured["target"], "patterns") == ["conv"]


def test_convert_layernorm_non_tilelang_engine_rejects() -> None:
    module = nn.LayerNorm(8)
    result = xqt.convert(module, engine="torch", return_result=True)

    assert isinstance(result, ConvertResult)
    assert result.converted is False
    assert result.contract.operator_kind == "layernorm"
    assert (
        result.report["reason"]
        == "torch engine keeps the original LayerNorm implementation"
    )


def test_convert_reference_fp4_linear_torch_preserves_forward() -> None:
    base = nn.Linear(8, 4)
    module = ReferenceFP4Linear.from_linear(base, group_size=4)
    sample = torch.randn(2, 8)

    converted = xqt.convert(module, engine="torch")

    assert isinstance(converted, ReferenceFP4Linear)
    torch.testing.assert_close(converted(sample), module(sample))


def test_convert_feedforward_torch_returns_runtime_configured_result() -> None:
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="swiglu")

    result = xqt.convert(
        module,
        engine="torch",
        policy=PrecisionPolicy(
            activation="bf16",
            weight="fp16",
            mma="bf16",
            accum="fp32",
            output="fp16",
        ),
        return_result=True,
    )

    assert isinstance(result, ConvertResult)
    assert result.contract.operator_kind == "feedforward"
    assert result.converted is False
    assert result.report["runtime_config"]["engine"] == "torch"
    assert result.report["runtime_config"]["activation"] == "bf16"
    assert result.report["runtime_config"]["bias"] == "fp16"
    assert result.report["runtime_config"]["mma"] == "bf16"
    assert result.report["runtime_config"]["output"] == "fp16"
    assert result.report["runtime_config"]["projections"]["proj_in"]["mma"] == "bf16"
    assert (
        result.report["runtime_config"]["projections"]["proj_gate"]["output"] == "bf16"
    )
    assert (
        result.report["runtime_config"]["projections"]["proj_out"]["output"] == "fp16"
    )
    assert result.report["fusion"]["realized_patterns"] == ["swiglu"]


def test_convert_feedforward_triton_updates_runtime_engine() -> None:
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="gelu")

    result = xqt.convert(
        module,
        engine="triton",
        policy=PrecisionPolicy(
            activation="fp16",
            weight="fp16",
            mma="fp16",
            accum="fp32",
            output="bf16",
        ),
        return_result=True,
    )

    assert isinstance(result, ConvertResult)
    assert result.contract.operator_kind == "feedforward"
    assert result.converted is True
    assert result.model.runtime_config()["engine"] == "triton"
    assert result.model.runtime_config()["output"] == "bf16"
    assert result.model.runtime_config()["projections"]["proj_in"]["output"] == "fp16"
    assert result.model.runtime_config()["projections"]["proj_out"]["output"] == "bf16"


def test_convert_feedforward_projection_policies_override_selected_matmuls() -> None:
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="swiglu")

    result = xqt.convert(
        module,
        engine="torch",
        policy=PrecisionPolicy(
            activation="bf16",
            weight="fp16",
            mma="bf16",
            accum="fp32",
            output="fp16",
        ),
        projection_policies={
            "proj_in": {"activation": "fp32", "output": "fp32"},
            "proj_gate": PrecisionPolicy(
                activation="fp16",
                weight="bf16",
                mma="fp16",
                accum="fp32",
                output="bf16",
            ),
            "proj_out": {"output": "fp32"},
        },
        return_result=True,
    )

    assert isinstance(result, ConvertResult)
    runtime = result.model.runtime_config()
    assert runtime["activation"] == "bf16"
    assert runtime["projections"]["proj_in"] == {
        "activation": "fp32",
        "weight": "fp16",
        "bias": "fp16",
        "mma": "bf16",
        "accum": "fp32",
        "output": "fp32",
    }
    assert runtime["projections"]["proj_gate"] == {
        "activation": "fp16",
        "weight": "bf16",
        "bias": "fp16",
        "mma": "fp16",
        "accum": "fp32",
        "output": "bf16",
    }
    assert runtime["projections"]["proj_out"] == {
        "activation": "bf16",
        "weight": "fp16",
        "bias": "fp16",
        "mma": "bf16",
        "accum": "fp32",
        "output": "fp32",
    }


def test_convert_feedforward_structured_precision_policy_sets_default_and_projection_overrides() -> (
    None
):
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="swiglu")

    result = xqt.convert(
        module,
        engine="torch",
        policy=FeedForwardPrecisionPolicy(
            default=PrecisionPolicy(
                activation="bf16",
                weight="fp16",
                mma="bf16",
                accum="fp32",
                output="fp16",
            ),
            proj_in=PrecisionPolicy(
                activation="fp32",
                weight="fp16",
                mma="bf16",
                accum="fp32",
                output="fp32",
            ),
            proj_gate=PrecisionPolicy(
                activation="fp16",
                weight="bf16",
                mma="fp16",
                accum="fp32",
                output="bf16",
            ),
            proj_out=PrecisionPolicy(
                activation="bf16",
                weight="fp16",
                mma="bf16",
                accum="fp32",
                output="fp32",
            ),
        ),
        return_result=True,
    )

    assert isinstance(result, ConvertResult)
    runtime = result.model.runtime_config()
    assert runtime["activation"] == "bf16"
    assert runtime["projections"]["proj_in"]["output"] == "fp32"
    assert runtime["projections"]["proj_gate"]["weight"] == "bf16"
    assert runtime["projections"]["proj_out"]["output"] == "fp32"


def test_convert_feedforward_rejects_structured_policy_plus_projection_policies() -> (
    None
):
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="gelu")

    with pytest.raises(
        Exception,
        match="does not allow both FeedForwardPrecisionPolicy and projection_policies",
    ):
        xqt.convert(
            module,
            engine="torch",
            policy=FeedForwardPrecisionPolicy(default=PrecisionPolicy()),
            projection_policies={"proj_out": {"output": "fp32"}},
        )


def test_convert_feedforward_rejects_unsupported_engine() -> None:
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="gelu")

    with pytest.raises(
        Exception,
        match="FeedForward currently supports only engine='torch' or engine='triton'",
    ):
        xqt.convert(module, engine="tilelang")


def test_convert_non_feedforward_rejects_projection_policies() -> None:
    module = nn.Linear(8, 4)

    with pytest.raises(
        Exception,
        match="projection_policies are supported only for FeedForward modules",
    ):
        xqt.convert(
            module,
            engine="torch",
            projection_policies={"proj_out": {"output": "fp32"}},
        )


def test_xqt_feedforward_gelu_matches_manual_reference() -> None:
    torch.manual_seed(0)
    module = xqt.nn.FeedForward(
        8,
        inner_dim=12,
        activation="gelu",
        norm="layernorm",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
    )
    sample = torch.randn(2, 3, 8)

    normed = module.norm(sample)
    expected_hidden = F.gelu(module.proj_in(normed))
    expected = module.proj_out(expected_hidden)

    torch.testing.assert_close(module(sample), expected)


def test_xqt_feedforward_swiglu_matches_manual_reference() -> None:
    torch.manual_seed(0)
    module = xqt.nn.FeedForward(
        8,
        inner_dim=10,
        activation="swiglu",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
    )
    sample = torch.randn(4, 8)

    up = module.proj_in(sample)
    gate = module.proj_gate(sample)
    expected = module.proj_out(F.silu(gate) * up)

    torch.testing.assert_close(module(sample), expected)


def test_xqt_feedforward_fusion_can_be_disabled() -> None:
    torch.manual_seed(0)
    module = xqt.nn.FeedForward(
        8,
        inner_dim=10,
        activation="swiglu",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
        fusion=False,
    )
    sample = torch.randn(4, 8)

    up = module.proj_in(sample)
    gate = module.proj_gate(sample)
    expected = module.proj_out(F.silu(gate) * up)

    assert module.runtime_config()["fusion"]["enabled"] is False
    assert module.runtime_config()["fusion"]["realized_patterns"] == []
    torch.testing.assert_close(module(sample), expected)


def test_xqt_feedforward_geglu_matches_manual_reference() -> None:
    torch.manual_seed(0)
    module = xqt.nn.FeedForward(
        8,
        inner_dim=10,
        activation="geglu",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
    )
    sample = torch.randn(4, 8)

    up = module.proj_in(sample)
    gate = module.proj_gate(sample)
    expected = module.proj_out(F.gelu(gate) * up)

    torch.testing.assert_close(module(sample), expected)


def test_xqt_feedforward_approximate_gelu_matches_manual_reference() -> None:
    torch.manual_seed(0)
    module = xqt.nn.FeedForward(
        8,
        inner_dim=12,
        activation="geglu-approximate",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
    )
    sample = torch.randn(2, 8)

    hidden = module.proj_in(sample)
    expected = module.proj_out(hidden * torch.sigmoid(1.702 * hidden))

    torch.testing.assert_close(module(sample), expected)


def test_xqt_feedforward_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unsupported engine"):
        xqt.nn.FeedForward(8, engine="tilelang")  # type: ignore[arg-type]


def test_xqt_feedforward_runtime_configure_updates_precision_policy() -> None:
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="gelu")
    module.configure_runtime(
        engine="torch",
        activation_dtype="bf16",
        weight_dtype="fp16",
        mma_dtype="bf16",
        accum_dtype="fp32",
        output_dtype="bf16",
    )

    runtime = module.runtime_config()
    assert runtime["engine"] == "torch"
    assert runtime["activation"] == "bf16"
    assert runtime["weight"] == "fp16"
    assert runtime["bias"] == "auto"
    assert runtime["mma"] == "bf16"
    assert runtime["accum"] == "fp32"
    assert runtime["output"] == "bf16"
    assert runtime["fusion"]["realized_patterns"] == ["proj_in_gelu_epilogue"]
    assert runtime["projections"] == {
        "proj_in": {
            "activation": "bf16",
            "weight": "fp16",
            "bias": "auto",
            "mma": "bf16",
            "accum": "fp32",
            "output": "bf16",
        },
        "proj_out": {
            "activation": "bf16",
            "weight": "fp16",
            "bias": "auto",
            "mma": "bf16",
            "accum": "fp32",
            "output": "bf16",
        },
    }


def test_xqt_feedforward_runtime_configure_accepts_projection_policy_overrides() -> (
    None
):
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="swiglu")
    module.configure_runtime(
        engine="torch",
        activation_dtype="bf16",
        weight_dtype="fp16",
        mma_dtype="bf16",
        accum_dtype="fp32",
        output_dtype="fp16",
        projection_policies={
            "proj_in": {"activation_dtype": "fp32", "output_dtype": "fp32"},
            "proj_gate": {"mma": "fp16"},
            "proj_out": {"output": "fp32"},
        },
    )

    assert module.runtime_config()["projections"] == {
        "proj_in": {
            "activation": "fp32",
            "weight": "fp16",
            "bias": "auto",
            "mma": "bf16",
            "accum": "fp32",
            "output": "fp32",
        },
        "proj_gate": {
            "activation": "bf16",
            "weight": "fp16",
            "bias": "auto",
            "mma": "fp16",
            "accum": "fp32",
            "output": "bf16",
        },
        "proj_out": {
            "activation": "bf16",
            "weight": "fp16",
            "bias": "auto",
            "mma": "bf16",
            "accum": "fp32",
            "output": "fp32",
        },
    }


def test_xqt_feedforward_runtime_configure_projection_none_resets_override() -> None:
    module = xqt.nn.FeedForward(8, inner_dim=12, activation="gelu")
    module.configure_runtime(
        engine="torch",
        activation_dtype="bf16",
        weight_dtype="fp16",
        mma_dtype="bf16",
        accum_dtype="fp32",
        output_dtype="fp16",
        projection_policies={"proj_in": {"output": "fp32"}},
    )
    module.configure_runtime(projection_policies={"proj_in": None})

    assert module.runtime_config()["projections"]["proj_in"] == {
        "activation": "bf16",
        "weight": "fp16",
        "bias": "auto",
        "mma": "bf16",
        "accum": "fp32",
        "output": "bf16",
    }


def test_xqt_feedforward_proj_out_output_precision_controls_cpu_reference_dtype() -> (
    None
):
    module = xqt.nn.FeedForward(
        8,
        inner_dim=12,
        activation="gelu",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
    )
    module.configure_runtime(
        output_dtype="bf16",
        projection_policies={"proj_out": {"output": "fp32"}},
    )
    sample = torch.randn(2, 8, dtype=torch.float32)

    output = module(sample)

    assert output.dtype == torch.float32


def test_xqt_feedforward_projection_hidden_precision_applies_per_projection() -> None:
    module = xqt.nn.FeedForward(
        8,
        inner_dim=10,
        activation="swiglu",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
    )
    module.configure_runtime(
        activation_dtype="bf16",
        projection_policies={
            "proj_in": {"output": "fp32"},
            "proj_gate": {"output": "fp16"},
        },
    )
    sample = torch.randn(2, 8, dtype=torch.float32)

    up = module._linear_epilogue(
        sample,
        module.proj_in,
        precision=module.runtime_config()["projections"]["proj_in"],
    )
    gate = module._linear_epilogue(
        sample,
        module.proj_gate,
        precision=module.runtime_config()["projections"]["proj_gate"],
    )

    assert up.dtype == torch.float32
    assert gate.dtype == torch.float16


def test_xqt_feedforward_rejects_low_bit_mma_without_runtime_kernel() -> None:
    module = xqt.nn.FeedForward(
        8,
        inner_dim=12,
        activation="gelu",
        dropout=0.0,
        final_dropout=False,
        engine="torch",
    )
    module.configure_runtime(
        activation_dtype="fp16",
        weight_dtype="fp16",
        mma_dtype="nvfp4",
        output_dtype="fp16",
    )

    with pytest.raises(ValueError, match="mma precision nvfp4"):
        module(torch.randn(2, 8))
