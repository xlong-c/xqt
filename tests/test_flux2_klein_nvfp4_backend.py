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
    benchmark_flux2_klein_nvfp4_transformer_forward,
    compile_flux2_klein_nvfp4_transformer,
    collect_flux2_klein_nvfp4_backend_targets,
    flux2_klein_nvfp4_single_file_url,
    load_flux2_klein_nvfp4_transformer,
    materialize_flux2_klein_nvfp4_backend,
    normalize_flux2_klein_nvfp4_backend,
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


def test_flux2_klein_nvfp4_backend_aliases_and_url() -> None:
    assert normalize_flux2_klein_nvfp4_backend("cutedsl") == "cute_dsl"
    assert normalize_flux2_klein_nvfp4_backend("cute-dsl") == "cute_dsl"
    assert normalize_flux2_klein_nvfp4_backend("cutile") == "cutile"
    assert normalize_flux2_klein_nvfp4_backend("tilelang") == "tilelang"
    assert flux2_klein_nvfp4_single_file_url().endswith(
        "/black-forest-labs/FLUX.2-klein-4b-nvfp4/resolve/main/flux-2-klein-4b-nvfp4.safetensors"
    )


def test_collect_flux2_klein_nvfp4_backend_targets_for_three_backends() -> None:
    model = _TinyFlux2Transformer()

    targets = collect_flux2_klein_nvfp4_backend_targets(
        model,
        backends=("cutedsl", "cutile", "tilelang"),
        target_arch="sm_89",
    )

    assert set(targets) == {"cute_dsl", "cutile", "tilelang"}
    assert targets["cute_dsl"][0][0].patterns == ["gemm_epilogue"]
    assert targets["cutile"][0][0].patterns == ["nvfp4_packed_dequant_gemm_epilogue"]
    assert targets["tilelang"][0][0].patterns == ["dequant_gemm_epilogue"]
    for backend, (_, summaries) in targets.items():
        assert summaries[0].backend == backend
        assert summaries[0].name == "proj"
        assert summaries[0].group_size == 4


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
    targets = collect_flux2_klein_nvfp4_backend_targets(
        model,
        backends=("cutile",),
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


def test_materialize_flux2_klein_nvfp4_backend_runs_three_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("xqt.operator_opt.executor.cutile_available", lambda: False)
    x = torch.randn(3, 8, dtype=torch.float32)

    for backend in ("cutedsl", "cutile", "tilelang"):
        model = _TinyFlux2Transformer().eval()
        expected = model(x)

        result = materialize_flux2_klein_nvfp4_backend(
            model,
            backend=backend,
            target_arch="sm_89",
            max_targets=1,
            inplace=False,
        )
        actual = result.model(x)

        torch.testing.assert_close(actual, expected)
        assert result.target_count == 1
        metadata = result.model.proj.execution_metadata()
        assert metadata["operator_family"] == "linear"
        assert metadata["execution_mode"] == "reference_fallback"
        if result.backend == "cute_dsl":
            assert metadata["kernel_pattern"] == "gemm_epilogue"
            assert metadata["consumes_packed_weight"] is False
        elif result.backend == "cutile":
            assert metadata["kernel_pattern"] == "dense_linear_epilogue"
            assert metadata["consumes_packed_weight"] is False
            assert metadata["weight_representation"] == "dense_dequantized_weight_cache"
        else:
            assert metadata["kernel_pattern"] in {
                "dense_linear_epilogue",
                "nvfp4_packed_dequant_gemm_epilogue",
            }


def test_cutile_uses_dense_cache_when_packed_runtime_is_reference_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("xqt.operator_opt.executor.cutile_available", lambda: True)
    model = _TinyFlux2Transformer().eval()
    x = torch.randn(3, 8, dtype=torch.float32)
    expected = model(x)

    result = materialize_flux2_klein_nvfp4_backend(
        model,
        backend="cutile",
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
    monkeypatch.setattr("xqt.operator_opt.executor.cutile_available", lambda: True)

    def fake_get_cutile_kernel_spec(pattern: str) -> types.SimpleNamespace:
        metadata = {
            "production_status": "runtime_kernel",
            "fusion_status": "cutile_packed_nvfp4_runtime_kernel",
        }
        return types.SimpleNamespace(metadata=metadata)

    monkeypatch.setattr(
        "xqt.operator_opt.executor.get_cutile_kernel_spec",
        fake_get_cutile_kernel_spec,
    )
    model = _TinyFlux2Transformer().eval()
    x = torch.randn(3, 8, dtype=torch.float32)
    expected = model(x)

    result = materialize_flux2_klein_nvfp4_backend(
        model,
        backend="cutile",
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

    monkeypatch.setattr("xqt.operator_opt.executor.cutile_available", lambda: True)
    monkeypatch.setattr(
        "xqt.operator_opt.executor.run_cutile_kernel",
        fake_run_cutile_kernel,
    )
    model = _TinyFlux2Transformer().eval()
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    expected = model(x)

    result = materialize_flux2_klein_nvfp4_backend(
        model,
        backend="cutile",
        target_arch="sm_89",
        max_targets=1,
        inplace=False,
    )
    actual = result.model(x)

    torch.testing.assert_close(actual, expected)
    assert seen == {"pattern": "dense_linear_epilogue", "shape": (6, 8)}
    assert tuple(actual.shape) == (2, 3, 4)


def test_materialize_flux2_klein_nvfp4_backend_replaces_pipeline_transformer() -> None:
    pipeline = _TinyFlux2Pipeline()
    x = torch.randn(2, 8, dtype=torch.float32)
    expected = pipeline.transformer(x)

    result = materialize_flux2_klein_nvfp4_backend(
        pipeline,
        backend="cutile",
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
        backend="cutedsl",
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


def test_compile_flux2_klein_nvfp4_transformer_delegates_to_compile_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    model = nn.Linear(4, 4)

    def fake_compile_with_torch(
        module: nn.Module,
        plan: object,
    ) -> tuple[nn.Module, float]:
        captured["module"] = module
        captured["plan_backend"] = getattr(plan, "backend")
        captured["plan_mode"] = getattr(plan, "mode")
        captured["plan_options"] = dict(getattr(plan, "options"))
        captured["plan_fullgraph"] = getattr(plan, "fullgraph")
        captured["plan_dynamic"] = getattr(plan, "dynamic")
        return module, 12.5

    monkeypatch.setattr(
        "xqt.model.flux2_klein_nvfp4.compile_with_torch",
        fake_compile_with_torch,
    )

    result = compile_flux2_klein_nvfp4_transformer(
        model,
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )

    assert result.model is model
    assert result.compile_time_ms == 12.5
    assert result.compile_backend == "inductor"
    assert result.compile_mode == "reduce-overhead"
    assert captured == {
        "module": model,
        "plan_backend": "torch_compile",
        "plan_mode": "reduce-overhead",
        "plan_options": {
            "backend": "inductor",
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
        backend: str,
        target_arch: str | None,
        max_targets: int | None,
        include_names: Sequence[str] | None,
        exclude_names: Sequence[str] | None,
        inplace: bool,
        min_speedup: float,
    ) -> object:
        del target_arch, max_targets, include_names, exclude_names, min_speedup
        captured["materialize_backend"] = backend
        captured["materialize_inplace"] = inplace
        return types.SimpleNamespace(
            model=module,
            backend="tilelang",
            target_count=3,
        )

    def fake_compile(
        module: nn.Module,
        **kwargs: object,
    ) -> object:
        captured["compile_kwargs"] = dict(kwargs)
        return types.SimpleNamespace(
            model=module,
            backend=kwargs["backend_name"],
            materialized_target_count=kwargs["materialized_target_count"],
            compile_backend=kwargs["backend"],
            compile_mode=kwargs["mode"],
            compile_time_ms=7.5,
            warmup_iterations=0,
            warmup_time_ms=0.0,
        )

    monkeypatch.setattr(
        "xqt.model.flux2_klein_nvfp4.materialize_flux2_klein_nvfp4_backend",
        fake_materialize,
    )
    monkeypatch.setattr(
        "xqt.model.flux2_klein_nvfp4.compile_flux2_klein_nvfp4_transformer",
        fake_compile,
    )

    result = optimize_flux2_klein_nvfp4_transformer(
        model,
        backend="tilelang",
        compile_backend="inductor",
        compile_mode=None,
        inplace=True,
    )

    assert result.model is model
    assert result.backend == "tilelang"
    assert result.materialized_target_count == 3
    assert result.compile_backend == "inductor"
    assert result.compile_mode is None
    assert result.compile_time_ms == 7.5
    assert captured["materialize_backend"] == "tilelang"
    assert captured["materialize_inplace"] is True
    assert captured["compile_kwargs"] == {
        "backend_name": "tilelang",
        "materialized_target_count": 3,
        "backend": "inductor",
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
