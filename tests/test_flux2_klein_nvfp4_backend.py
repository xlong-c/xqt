from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from typing import Sequence

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.model import (
    FLUX2_KLEIN_4B_REPO_ID,
    benchmark_flux2_klein_nvfp4_transformer_paired,
    benchmark_flux2_klein_nvfp4_transformer_forward,
    capture_flux2_klein_nvfp4_transformer_cuda_graph,
    compile_flux2_klein_nvfp4_transformer,
    collect_flux2_klein_nvfp4_engine_targets,
    flux2_klein_nvfp4_single_file_url,
    load_flux2_klein_nvfp4_transformer,
    materialize_flux2_klein_nvfp4_engine,
    normalize_flux2_klein_nvfp4_engine,
    optimize_flux2_klein_nvfp4_transformer,
    run_flux2_klein_nvfp4_inference,
    warmup_flux2_klein_nvfp4_transformer,
)
from xqt.quant import expand_group_scale, unpack_nvfp4e2m1


class _FakeFlux2NVFP4Linear(nn.Module):
    def __init__(
        self,
        *,
        input_features: int = 8,
        output_features: int = 4,
        group_size: int = 4,
    ) -> None:
        super().__init__()
        if input_features % group_size != 0:
            raise ValueError("input_features must be divisible by group_size")
        packed_cols = input_features // 2
        groups = input_features // group_size
        self.in_features = input_features
        self.out_features = output_features
        packed = torch.arange(output_features * packed_cols, dtype=torch.uint8).reshape(
            output_features,
            packed_cols,
        )
        packed = (packed * 17 + 3).to(torch.uint8)
        scale = torch.full((output_features, groups, 1), 0.125, dtype=torch.float32)
        bias = torch.linspace(-0.2, 0.2, output_features, dtype=torch.float32)
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("weight_global_scale", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes = unpack_nvfp4e2m1(self.weight_packed, input_features=self.in_features)
        scale = expand_group_scale(
            self.weight_scale,
            group_size=self.in_features // self.weight_scale.shape[1],
            input_features=self.in_features,
        )
        weight = (codes * scale).to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device)
        return F.linear(x, weight, bias)


class _TinyFlux2Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = _FakeFlux2NVFP4Linear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _TinyFlux2Pipeline:
    def __init__(self) -> None:
        self.transformer = _TinyFlux2Transformer().eval()
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        prompt: str | list[str],
        input_tensor: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return {
            "prompt": prompt,
            "output": self.transformer(input_tensor),
            "transformer_type": type(self.transformer.proj).__name__,
        }


def test_flux2_klein_nvfp4_engine_aliases_and_url() -> None:
    assert normalize_flux2_klein_nvfp4_engine("cutedsl") == "cute_dsl"
    assert normalize_flux2_klein_nvfp4_engine("cute-dsl") == "cute_dsl"
    assert normalize_flux2_klein_nvfp4_engine("cutile") == "cutile"
    assert normalize_flux2_klein_nvfp4_engine("tilelang") == "tilelang"
    assert flux2_klein_nvfp4_single_file_url().endswith(
        "/black-forest-labs/FLUX.2-klein-4b-nvfp4/resolve/main/flux-2-klein-4b-nvfp4.safetensors"
    )


def test_collect_flux2_klein_nvfp4_engine_targets_for_three_engines() -> None:
    model = _TinyFlux2Transformer()

    targets = collect_flux2_klein_nvfp4_engine_targets(
        model,
        engines=("cutedsl", "cutile", "tilelang"),
        target_arch="sm_89",
    )

    assert set(targets) == {"cute_dsl", "cutile", "tilelang"}
    assert targets["cute_dsl"][0][0].patterns == ["gemm_epilogue"]
    assert targets["cutile"][0][0].patterns == ["nvfp4_packed_dequant_gemm_epilogue"]
    assert targets["tilelang"][0][0].patterns == ["dequant_gemm_epilogue"]
    for engine, (_, summaries) in targets.items():
        assert summaries[0].engine == engine
        assert summaries[0].name == "proj"
        assert summaries[0].group_size == 4


def test_collect_flux2_klein_nvfp4_engine_targets_accepts_engines() -> None:
    model = _TinyFlux2Transformer()

    targets = collect_flux2_klein_nvfp4_engine_targets(
        model,
        engines=("cutedsl", "tilelang"),
        target_arch="sm_89",
    )

    assert set(targets) == {"cute_dsl", "tilelang"}


def test_load_flux2_klein_nvfp4_transformer_passes_diffusers_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class _FakeFlux2Transformer2DModel:
        @classmethod
        def from_single_file(cls, source: str, **kwargs: object) -> nn.Module:
            calls["source"] = source
            calls["kwargs"] = dict(kwargs)
            return nn.Linear(1, 1)

    fake_diffusers = types.SimpleNamespace(
        Flux2Transformer2DModel=_FakeFlux2Transformer2DModel
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    model = load_flux2_klein_nvfp4_transformer(
        model_file="local.safetensors",
        dtype=torch.float32,
        local_files_only=True,
    )

    assert isinstance(model, nn.Linear)
    assert calls["source"] == "local.safetensors"
    assert calls["kwargs"] == {
        "torch_dtype": torch.float32,
        "local_files_only": True,
        "config": FLUX2_KLEIN_4B_REPO_ID,
        "subfolder": "transformer",
    }


def test_load_flux2_klein_nvfp4_transformer_maps_modelopt_qkv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_file = tmp_path / "flux2-nvfp4.safetensors"
    save_file(
        {
            "double_blocks.0.img_attn.norm.key_norm.scale": torch.tensor(
                [1.25, 1.5],
                dtype=torch.float32,
            ),
            "double_blocks.0.img_attn.norm.query_norm.scale": torch.tensor(
                [1.75, 2.0],
                dtype=torch.float32,
            ),
            "double_blocks.0.img_attn.qkv.input_scale": torch.tensor(1.0),
            "double_blocks.0.img_attn.qkv.weight": torch.tensor(
                [
                    [0x10, 0x32],
                    [0x54, 0x76],
                    [0x10, 0x32],
                    [0x54, 0x76],
                    [0x10, 0x32],
                    [0x54, 0x76],
                ],
                dtype=torch.uint8,
            ),
            "double_blocks.0.img_attn.qkv.weight_scale": torch.ones(
                6,
                1,
                dtype=torch.float32,
            ).to(torch.float8_e4m3fn),
            "double_blocks.0.img_attn.qkv.weight_scale_2": torch.tensor(2.0),
        },
        str(model_file),
        metadata={
            "_quantization_metadata": json.dumps(
                {
                    "format_version": "1.0",
                    "layers": {
                        "double_blocks.0.img_attn.qkv": {"format": "nvfp4"},
                    },
                }
            )
        },
    )

    class _FakeAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Linear(4, 2, bias=False)
            self.to_k = nn.Linear(4, 2, bias=False)
            self.to_v = nn.Linear(4, 2, bias=False)
            self.norm_q = nn.LayerNorm(2)
            self.norm_k = nn.LayerNorm(2)

    class _FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = _FakeAttention()

    class _FakeFlux2Transformer2DModel(nn.Module):
        @classmethod
        def load_config(cls, **kwargs: object) -> dict[str, object]:
            assert kwargs["pretrained_model_name_or_path"] == FLUX2_KLEIN_4B_REPO_ID
            assert kwargs["subfolder"] == "transformer"
            return {}

        @classmethod
        def from_config(cls, config: dict[str, object]) -> "_FakeFlux2Transformer2DModel":
            del config
            return cls()

        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList([_FakeBlock()])

    fake_diffusers = types.SimpleNamespace(
        Flux2Transformer2DModel=_FakeFlux2Transformer2DModel
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    model = load_flux2_klein_nvfp4_transformer(
        model_file=str(model_file),
        dtype=torch.float16,
        local_files_only=True,
    )
    targets = collect_flux2_klein_nvfp4_engine_targets(
        model,
        engines=("cutile",),
        target_arch="sm_89",
    )

    summaries = targets["cutile"][1]
    assert [summary.name for summary in summaries] == [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
    ]
    assert {summary.input_features for summary in summaries} == {4}
    assert {summary.output_features for summary in summaries} == {2}
    torch.testing.assert_close(
        model.transformer_blocks[0].attn.norm_q.weight,
        torch.tensor([1.75, 2.0], dtype=torch.float16),
    )
    torch.testing.assert_close(
        model.transformer_blocks[0].attn.norm_k.weight,
        torch.tensor([1.25, 1.5], dtype=torch.float16),
    )


def test_materialize_flux2_klein_nvfp4_engine_runs_three_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xqt.operator_opt.reference_wrappers.cutile_available",
        lambda: False,
    )
    x = torch.randn(3, 8, dtype=torch.float32)

    for engine in ("cutedsl", "cutile", "tilelang"):
        model = _TinyFlux2Transformer().eval()
        expected = model(x)

        result = materialize_flux2_klein_nvfp4_engine(
            model,
            engine=engine,
            target_arch="sm_89",
            max_targets=1,
            inplace=False,
        )
        actual = result.model(x)

        torch.testing.assert_close(actual, expected)
        assert result.target_count == 1
        assert result.engine == result.engine
        metadata = result.model.proj.execution_metadata()
        assert metadata["operator_family"] == "linear"
        assert metadata["execution_mode"] == "reference_fallback"
        if result.engine == "cute_dsl":
            assert metadata["kernel_pattern"] == "gemm_epilogue"
            assert metadata["consumes_packed_weight"] is False
        elif result.engine == "cutile":
            assert metadata["kernel_pattern"] == "dense_linear_epilogue"
            assert metadata["consumes_packed_weight"] is False
            assert metadata["weight_representation"] == "dense_dequantized_weight_cache"
        else:
            assert metadata["kernel_pattern"] in {
                "dense_linear_epilogue",
                "nvfp4_packed_dequant_gemm_epilogue",
            }


def test_materialize_flux2_klein_nvfp4_engine_accepts_engine_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xqt.operator_opt.reference_wrappers.cutile_available",
        lambda: False,
    )
    model = _TinyFlux2Transformer().eval()

    result = materialize_flux2_klein_nvfp4_engine(
        model,
        engine="cutedsl",
        target_arch="sm_89",
        max_targets=1,
        inplace=False,
    )

    assert result.engine == "cute_dsl"


def test_cutile_uses_dense_cache_when_packed_runtime_is_reference_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xqt.operator_opt.reference_wrappers.cutile_available",
        lambda: True,
    )
    model = _TinyFlux2Transformer().eval()
    x = torch.randn(3, 8, dtype=torch.float32)
    expected = model(x)

    result = materialize_flux2_klein_nvfp4_engine(
        model,
        engine="cutile",
        target_arch="sm_89",
        max_targets=1,
        inplace=False,
    )
    actual = result.model(x)

    torch.testing.assert_close(actual, expected)
    metadata = result.model.proj.execution_metadata()
    assert metadata["kernel_pattern"] == "dense_linear_epilogue"
    assert metadata["consumes_packed_weight"] is False
    assert metadata["weight_representation"] == "dense_dequantized_weight_cache"


def test_cutile_uses_packed_nvfp4_when_runtime_kernel_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xqt.operator_opt.reference_wrappers.cutile_available",
        lambda: True,
    )

    def fake_get_cutile_kernel_spec(pattern: str) -> types.SimpleNamespace:
        metadata = {
            "production_status": "runtime_kernel",
            "fusion_status": "cutile_packed_nvfp4_runtime_kernel",
        }
        return types.SimpleNamespace(metadata=metadata)

    monkeypatch.setattr(
        "xqt.operator_opt.reference_wrappers.get_cutile_kernel_spec",
        fake_get_cutile_kernel_spec,
    )
    model = _TinyFlux2Transformer().eval()
    x = torch.randn(3, 8, dtype=torch.float32)
    expected = model(x)

    result = materialize_flux2_klein_nvfp4_engine(
        model,
        engine="cutile",
        target_arch="sm_89",
        max_targets=1,
        inplace=False,
    )
    actual = result.model(x)

    torch.testing.assert_close(actual, expected)
    metadata = result.model.proj.execution_metadata()
    assert metadata["kernel_pattern"] == "nvfp4_packed_dequant_gemm_epilogue"
    assert metadata["consumes_packed_weight"] is True


def test_cutile_dense_path_flattens_rank3_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run_cutile_kernel(
        pattern: str,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        seen["pattern"] = pattern
        seen["shape"] = tuple(x.shape)
        return F.linear(x, weight, bias)

    monkeypatch.setattr(
        "xqt.operator_opt.reference_wrappers.cutile_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "xqt.operator_opt.reference_wrappers.run_cutile_kernel",
        fake_run_cutile_kernel,
    )
    model = _TinyFlux2Transformer().eval()
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    expected = model(x)

    result = materialize_flux2_klein_nvfp4_engine(
        model,
        engine="cutile",
        target_arch="sm_89",
        max_targets=1,
        inplace=False,
    )
    actual = result.model(x)

    torch.testing.assert_close(actual, expected)
    assert seen == {"pattern": "dense_linear_epilogue", "shape": (6, 8)}
    assert tuple(actual.shape) == (2, 3, 4)


def test_materialize_flux2_klein_nvfp4_engine_replaces_pipeline_transformer() -> None:
    pipeline = _TinyFlux2Pipeline()
    x = torch.randn(2, 8, dtype=torch.float32)
    expected = pipeline.transformer(x)

    result = materialize_flux2_klein_nvfp4_engine(
        pipeline,
        engine="cutile",
        target_arch="sm_89",
        max_targets=1,
        inplace=False,
    )
    actual = result.model.transformer(x)

    assert result.model is not pipeline
    assert isinstance(pipeline.transformer.proj, _FakeFlux2NVFP4Linear)
    assert type(result.model.transformer.proj).__name__ == "_ReferenceGuardedLinearWrapper"
    torch.testing.assert_close(actual, expected)


def test_run_flux2_klein_nvfp4_inference_materializes_pipeline_inplace() -> None:
    pipeline = _TinyFlux2Pipeline()
    x = torch.randn(2, 8, dtype=torch.float32)
    expected = pipeline.transformer(x)

    output = run_flux2_klein_nvfp4_inference(
        pipeline,
        prompt="test prompt",
        engine="cutedsl",
        target_arch="sm_89",
        max_targets=1,
        input_tensor=x,
        output_type="latent",
    )

    assert output["prompt"] == "test prompt"
    assert output["transformer_type"] == "_ReferenceGuardedLinearWrapper"
    assert pipeline.calls == [
        {"prompt": "test prompt", "kwargs": {"output_type": "latent"}}
    ]
    torch.testing.assert_close(output["output"], expected)


def test_compile_flux2_klein_nvfp4_transformer_delegates_to_compile_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    model = nn.Linear(4, 4)

    def fake_compile_with_torch(
        module: nn.Module,
        plan: object,
    ) -> tuple[nn.Module, float]:
        captured["module"] = module
        captured["plan_engine"] = getattr(plan, "engine")
        captured["plan_mode"] = getattr(plan, "mode")
        captured["plan_options"] = dict(getattr(plan, "options"))
        captured["plan_fullgraph"] = getattr(plan, "fullgraph")
        captured["plan_dynamic"] = getattr(plan, "dynamic")
        return module, 12.5

    monkeypatch.setattr(
        "xqt.model.flux2_klein.runtime.compile_with_torch",
        fake_compile_with_torch,
    )

    result = compile_flux2_klein_nvfp4_transformer(
        model,
        compile_engine="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )

    assert result.model is model
    assert result.compile_time_ms == 12.5
    assert result.compile_engine == "inductor"
    assert result.compile_mode == "reduce-overhead"
    assert captured == {
        "module": model,
        "plan_engine": "torch_compile",
        "plan_mode": "reduce-overhead",
        "plan_options": {
            "engine": "inductor",
        },
        "plan_fullgraph": False,
        "plan_dynamic": False,
    }


def test_compile_flux2_klein_nvfp4_transformer_rejects_mode_plus_options() -> None:
    with pytest.raises(XQTBackendError, match="mode and options"):
        compile_flux2_klein_nvfp4_transformer(
            nn.Linear(4, 4),
            mode="reduce-overhead",
            options={"triton.cudagraphs": True},
        )


def test_optimize_flux2_klein_nvfp4_transformer_materializes_then_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 4)
    captured: dict[str, object] = {}

    def fake_materialize(
        module: nn.Module,
        *,
        engine: str,
        target_arch: str | None,
        max_targets: int | None,
        include_names: Sequence[str] | None,
        exclude_names: Sequence[str] | None,
        inplace: bool,
        min_speedup: float,
    ) -> object:
        del target_arch, max_targets, include_names, exclude_names, min_speedup
        captured["materialize_engine"] = engine
        captured["materialize_inplace"] = inplace
        return types.SimpleNamespace(
            model=module,
            engine="tilelang",
            target_count=3,
        )

    def fake_compile(
        module: nn.Module,
        **kwargs: object,
    ) -> object:
        captured["compile_kwargs"] = dict(kwargs)
        return types.SimpleNamespace(
            model=module,
            engine=kwargs["engine_name"],
            materialized_target_count=kwargs["materialized_target_count"],
            compile_engine=kwargs["compile_engine"],
            compile_mode=kwargs["mode"],
            compile_time_ms=7.5,
            warmup_iterations=0,
            warmup_time_ms=0.0,
        )

    monkeypatch.setattr(
        "xqt.model.flux2_klein.optimize.materialize_flux2_klein_nvfp4_engine",
        fake_materialize,
    )
    monkeypatch.setattr(
        "xqt.model.flux2_klein.optimize.compile_flux2_klein_nvfp4_transformer",
        fake_compile,
    )

    result = optimize_flux2_klein_nvfp4_transformer(
        model,
        engine="tilelang",
        compile_engine="inductor",
        compile_mode=None,
        inplace=True,
    )

    assert result.model is model
    assert result.engine == "tilelang"
    assert result.materialized_target_count == 3
    assert result.compile_engine == "inductor"
    assert result.compile_mode is None
    assert result.compile_time_ms == 7.5
    assert captured["materialize_engine"] == "tilelang"
    assert captured["materialize_inplace"] is True
    assert captured["compile_kwargs"] == {
        "engine_name": "tilelang",
        "materialized_target_count": 3,
        "compile_engine": "inductor",
        "mode": None,
        "fullgraph": False,
        "dynamic": False,
        "options": None,
    }


def test_optimize_flux2_klein_nvfp4_transformer_requires_inputs_for_warmup() -> None:
    with pytest.raises(XQTBackendError, match="warmup requires"):
        optimize_flux2_klein_nvfp4_transformer(
            nn.Linear(4, 4),
            compile_mode=None,
            warmup_iterations=1,
        )


def test_optimize_flux2_klein_nvfp4_transformer_requires_inputs_for_cuda_graph() -> None:
    with pytest.raises(XQTBackendError, match="cuda_graph optimization requires"):
        optimize_flux2_klein_nvfp4_transformer(
            nn.Linear(4, 4),
            optimization_kind="cuda_graph",
            warmup_iterations=1,
        )


def test_optimize_flux2_klein_nvfp4_transformer_skips_tilelang_materialization_for_sm89_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 4)
    captured: dict[str, object] = {}
    hidden_states = torch.randn(1, 2, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    timestep = torch.tensor([0.5])
    img_ids = torch.zeros(2, 4)
    txt_ids = torch.zeros(3, 4)

    def fake_materialize(**_: object) -> object:
        raise AssertionError("tilelang materialization should be skipped for sm_89 cuda_graph")

    def fake_capture(
        module: nn.Module,
        **kwargs: object,
    ) -> object:
        captured["module"] = module
        captured["capture_kwargs"] = dict(kwargs)
        return types.SimpleNamespace(
            model=module,
            engine=kwargs["engine_name"],
            materialized_target_count=kwargs["materialized_target_count"],
            graph_state={},
            input_signature=(),
            warmup_iterations=kwargs["warmup_iterations"],
            capture_time_ms=1.25,
        )

    monkeypatch.setattr(
        "xqt.model.flux2_klein.optimize.materialize_flux2_klein_nvfp4_engine",
        fake_materialize,
    )
    monkeypatch.setattr(
        "xqt.model.flux2_klein.optimize.capture_flux2_klein_nvfp4_transformer_cuda_graph",
        fake_capture,
    )

    result = optimize_flux2_klein_nvfp4_transformer(
        model,
        engine="tilelang",
        optimization_kind="cuda_graph",
        target_arch="sm_89",
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        warmup_iterations=2,
        inplace=True,
    )

    assert result.model is model
    assert result.engine == "tilelang"
    assert result.materialized_target_count == 0
    assert captured["module"] is model
    assert captured["capture_kwargs"]["engine_name"] == "tilelang"
    assert captured["capture_kwargs"]["materialized_target_count"] == 0
    assert captured["capture_kwargs"]["hidden_states"] is hidden_states
    assert captured["capture_kwargs"]["encoder_hidden_states"] is encoder_hidden_states
    assert captured["capture_kwargs"]["timestep"] is timestep
    assert captured["capture_kwargs"]["img_ids"] is img_ids
    assert captured["capture_kwargs"]["txt_ids"] is txt_ids
    assert captured["capture_kwargs"]["warmup_iterations"] == 2


def test_optimize_flux2_klein_nvfp4_transformer_materializes_tilelang_for_non_sm89_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(4, 4)
    captured: dict[str, object] = {}

    def fake_materialize(
        module: nn.Module,
        *,
        engine: str,
        target_arch: str | None,
        max_targets: int | None,
        include_names: Sequence[str] | None,
        exclude_names: Sequence[str] | None,
        inplace: bool,
        min_speedup: float,
    ) -> object:
        del max_targets, include_names, exclude_names, min_speedup
        captured["materialize_engine"] = engine
        captured["materialize_target_arch"] = target_arch
        captured["materialize_inplace"] = inplace
        return types.SimpleNamespace(
            model=module,
            engine=engine,
            target_count=5,
        )

    def fake_capture(
        module: nn.Module,
        **kwargs: object,
    ) -> object:
        captured["capture_module"] = module
        captured["capture_kwargs"] = dict(kwargs)
        return types.SimpleNamespace(
            model=module,
            engine=kwargs["engine_name"],
            materialized_target_count=kwargs["materialized_target_count"],
            graph_state={},
            input_signature=(),
            warmup_iterations=kwargs["warmup_iterations"],
            capture_time_ms=2.0,
        )

    hidden_states = torch.randn(1, 2, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    timestep = torch.tensor([0.5])
    img_ids = torch.zeros(2, 4)
    txt_ids = torch.zeros(3, 4)

    monkeypatch.setattr(
        "xqt.model.flux2_klein.optimize.materialize_flux2_klein_nvfp4_engine",
        fake_materialize,
    )
    monkeypatch.setattr(
        "xqt.model.flux2_klein.optimize.capture_flux2_klein_nvfp4_transformer_cuda_graph",
        fake_capture,
    )

    result = optimize_flux2_klein_nvfp4_transformer(
        model,
        engine="tilelang",
        optimization_kind="cuda_graph",
        target_arch="sm_90",
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        warmup_iterations=3,
        inplace=True,
    )

    assert result.model is model
    assert result.engine == "tilelang"
    assert result.materialized_target_count == 5
    assert captured["materialize_engine"] == "tilelang"
    assert captured["materialize_target_arch"] == "sm_90"
    assert captured["materialize_inplace"] is True
    assert captured["capture_module"] is model
    assert captured["capture_kwargs"]["engine_name"] == "tilelang"
    assert captured["capture_kwargs"]["materialized_target_count"] == 5
    assert captured["capture_kwargs"]["hidden_states"] is hidden_states
    assert captured["capture_kwargs"]["encoder_hidden_states"] is encoder_hidden_states
    assert captured["capture_kwargs"]["timestep"] is timestep
    assert captured["capture_kwargs"]["img_ids"] is img_ids
    assert captured["capture_kwargs"]["txt_ids"] is txt_ids
    assert captured["capture_kwargs"]["warmup_iterations"] == 3


def test_warmup_flux2_klein_nvfp4_transformer_runs_requested_iterations() -> None:
    class _CountingTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: torch.Tensor,
            guidance: torch.Tensor | None = None,
            joint_attention_kwargs: dict[str, object] | None = None,
            return_dict: bool = False,
        ) -> tuple[torch.Tensor]:
            del (
                encoder_hidden_states,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                joint_attention_kwargs,
                return_dict,
            )
            self.calls += 1
            return (hidden_states + 1.0,)

    model = _CountingTransformer()
    hidden_states = torch.randn(1, 2, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    timestep = torch.tensor([0.5])
    img_ids = torch.zeros(2, 4)
    txt_ids = torch.zeros(3, 4)

    elapsed_ms = warmup_flux2_klein_nvfp4_transformer(
        model,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        warmup_iterations=4,
        sync_cuda=False,
    )

    assert elapsed_ms >= 0.0
    assert model.calls == 4


def test_benchmark_flux2_klein_nvfp4_transformer_forward_uses_explicit_warmup() -> None:
    class _CountingTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: torch.Tensor,
            guidance: torch.Tensor | None = None,
            joint_attention_kwargs: dict[str, object] | None = None,
            return_dict: bool = False,
        ) -> tuple[torch.Tensor]:
            del (
                encoder_hidden_states,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                joint_attention_kwargs,
                return_dict,
            )
            self.calls += 1
            return (hidden_states + 1.0,)

    model = _CountingTransformer()
    hidden_states = torch.randn(1, 2, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    timestep = torch.tensor([0.5])
    img_ids = torch.zeros(2, 4)
    txt_ids = torch.zeros(3, 4)

    report = benchmark_flux2_klein_nvfp4_transformer_forward(
        model,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        warmup=3,
        iterations=5,
        sync_cuda=False,
    )

    assert report["warmup"] == 3
    assert report["iterations"] == 5
    assert model.calls == 8


def test_benchmark_flux2_klein_nvfp4_transformer_paired_reports_close_outputs() -> None:
    class _CountingTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: torch.Tensor,
            guidance: torch.Tensor | None = None,
            joint_attention_kwargs: dict[str, object] | None = None,
            return_dict: bool = False,
        ) -> tuple[torch.Tensor]:
            del encoder_hidden_states, timestep, img_ids, txt_ids, joint_attention_kwargs, return_dict
            self.calls += 1
            bias = 0.0 if guidance is None else guidance.reshape(-1, 1, 1)
            return (hidden_states + 1.0 + bias.to(hidden_states.dtype),)

    reference = _CountingTransformer()
    candidate = _CountingTransformer()
    hidden_states = torch.randn(1, 2, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    timestep = torch.tensor([0.5])
    img_ids = torch.zeros(2, 4)
    txt_ids = torch.zeros(3, 4)
    guidance = torch.tensor([0.25], dtype=torch.float32)

    result = benchmark_flux2_klein_nvfp4_transformer_paired(
        reference_transformer=reference,
        candidate_transformer=candidate,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        warmup=2,
        iterations=3,
        sync_cuda=False,
    )

    assert result.allclose_vs_eager is True
    assert result.max_abs_vs_eager == 0.0
    assert result.reference_report["warmup"] == 2
    assert result.reference_report["iterations"] == 3
    assert result.candidate_report["warmup"] == 2
    assert result.candidate_report["iterations"] == 3
    assert len(result.paired_speedup_ratios) == 3
    assert reference.calls == 6
    assert candidate.calls == 6


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for FLUX.2 CUDA Graph helper test",
)
def test_capture_flux2_klein_nvfp4_transformer_cuda_graph_replays_on_second_call() -> None:
    class _CudaTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: torch.Tensor,
            guidance: torch.Tensor | None = None,
            joint_attention_kwargs: dict[str, object] | None = None,
            return_dict: bool = False,
        ) -> tuple[torch.Tensor]:
            del joint_attention_kwargs, return_dict
            self.calls += 1
            total = hidden_states
            total = total + encoder_hidden_states.mean(dim=1, keepdim=True)[..., : hidden_states.shape[-1]]
            total = total + timestep.reshape(-1, 1, 1).to(hidden_states.dtype)
            total = total + img_ids.mean().to(hidden_states.dtype)
            total = total + txt_ids.mean().to(hidden_states.dtype)
            if guidance is not None:
                total = total + guidance.reshape(-1, 1, 1).to(hidden_states.dtype)
            return (total,)

    model = _CudaTransformer().eval().to(device="cuda", dtype=torch.float16)
    hidden_states = torch.randn(1, 2, 4, device="cuda", dtype=torch.float16)
    encoder_hidden_states = torch.randn(1, 3, 8, device="cuda", dtype=torch.float16)
    timestep = torch.tensor([0.5], device="cuda", dtype=torch.float16)
    img_ids = torch.zeros(2, 4, device="cuda", dtype=torch.float16)
    txt_ids = torch.zeros(3, 4, device="cuda", dtype=torch.float16)
    guidance = torch.tensor([0.125], device="cuda", dtype=torch.float16)
    capture_result = capture_flux2_klein_nvfp4_transformer_cuda_graph(
        model,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        warmup_iterations=2,
    )
    wrapped = capture_result.model

    first = wrapped(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )[0].clone()
    second_hidden_states = hidden_states + 1
    second = wrapped(
        hidden_states=second_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )[0]
    expected_first = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )[0]
    expected_second = model(
        hidden_states=second_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )[0]

    torch.testing.assert_close(first.float(), expected_first.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(second.float(), expected_second.float(), atol=1e-2, rtol=1e-2)
    assert capture_result.warmup_iterations == 2
    assert capture_result.capture_time_ms >= 0.0
    assert model.calls >= 4


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for FLUX.2 CUDA Graph helper test",
)
def test_capture_flux2_klein_nvfp4_transformer_cuda_graph_rejects_shape_change() -> None:
    class _CudaTransformer(nn.Module):
        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: torch.Tensor,
            guidance: torch.Tensor | None = None,
            joint_attention_kwargs: dict[str, object] | None = None,
            return_dict: bool = False,
        ) -> tuple[torch.Tensor]:
            del encoder_hidden_states, timestep, img_ids, txt_ids, guidance, joint_attention_kwargs, return_dict
            return (hidden_states + 1.0,)

    model = _CudaTransformer().eval().to(device="cuda", dtype=torch.float16)
    hidden_states = torch.randn(1, 2, 4, device="cuda", dtype=torch.float16)
    encoder_hidden_states = torch.randn(1, 3, 8, device="cuda", dtype=torch.float16)
    timestep = torch.tensor([0.5], device="cuda", dtype=torch.float16)
    img_ids = torch.zeros(2, 4, device="cuda", dtype=torch.float16)
    txt_ids = torch.zeros(3, 4, device="cuda", dtype=torch.float16)
    capture_result = capture_flux2_klein_nvfp4_transformer_cuda_graph(
        model,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        warmup_iterations=1,
    )

    with pytest.raises(XQTBackendError, match="matching shape/stride/dtype/device"):
        capture_result.model(
            hidden_states=torch.randn(1, 3, 4, device="cuda", dtype=torch.float16),
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )
