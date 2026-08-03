from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.model import (
    WAN21_VAE_OPTIMIZATION_KINDS,
    WAN21_VAE_REPO_ID,
    WAN21_VAE_RUN_MODES,
    WAN21_VAE_SUBFOLDER,
    benchmark_wan21_vae_paired,
    benchmark_wan21_vae_runner,
    build_wan21_vae_runner,
    capture_wan21_vae_cuda_graph,
    collect_wan21_vae_conv3d_targets,
    collect_wan21_vae_rmsnorm_targets,
    compile_wan21_vae_runner,
    load_wan21_pipeline_with_vae,
    load_wan21_vae,
    materialize_wan21_vae_conv3d_fastpath,
    materialize_wan21_vae_rmsnorm_fastpath,
    optimize_wan21_vae,
    run_wan21_vae_inference,
    warmup_wan21_vae_runner,
)


class _FakeLatentDistribution:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def mode(self) -> torch.Tensor:
        return self._tensor


class _FakeEncodeOutput:
    def __init__(self, latent_tensor: torch.Tensor) -> None:
        self.latent_dist = _FakeLatentDistribution(latent_tensor)


class _FakeDecodeOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _FakeWanRMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = float(dim) ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gamma.to(device=x.device, dtype=x.dtype).reshape(1, -1, 1, 1, 1)
        variance = x.pow(2).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * weight * self.scale


class _FakeAutoencoderKLWan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.conv_in = nn.Conv3d(3, 64, kernel_size=1)
        self.norm_in = _FakeWanRMSNorm(64)
        self.conv_out = nn.Conv3d(64, 3, kernel_size=1)
        self.conv_skip = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.norm_skip = _FakeWanRMSNorm(64)
        self.use_tiling = False
        self.use_slicing = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: object) -> "_FakeAutoencoderKLWan":
        del kwargs
        if repo_id != WAN21_VAE_REPO_ID:
            raise AssertionError(f"unexpected repo_id: {repo_id}")
        return cls()

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
    ) -> None:
        self.use_tiling = True
        if tile_sample_min_height is not None:
            self.tile_sample_min_height = tile_sample_min_height
        if tile_sample_min_width is not None:
            self.tile_sample_min_width = tile_sample_min_width
        if tile_sample_stride_height is not None:
            self.tile_sample_stride_height = tile_sample_stride_height
        if tile_sample_stride_width is not None:
            self.tile_sample_stride_width = tile_sample_stride_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> _FakeEncodeOutput | tuple[_FakeLatentDistribution]:
        hidden = self.conv_in(x)
        hidden = self.norm_in(hidden)
        hidden = self.conv_skip(hidden)
        hidden = self.norm_skip(hidden)
        latent = hidden.mean(dim=1, keepdim=True) * self.weight
        if not return_dict:
            return (_FakeLatentDistribution(latent),)
        return _FakeEncodeOutput(latent)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> _FakeDecodeOutput | tuple[torch.Tensor]:
        hidden = z.repeat(1, 64, 1, 1, 1)
        hidden = self.norm_in(hidden)
        sample = self.conv_out(hidden) + self.bias.to(dtype=z.dtype, device=z.device)
        if not return_dict:
            return (sample,)
        return _FakeDecodeOutput(sample)


class _FakeWanPipeline:
    def __init__(self, vae: nn.Module | None = None) -> None:
        self.vae = _FakeAutoencoderKLWan() if vae is None else vae

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: object) -> "_FakeWanPipeline":
        assert repo_id == WAN21_VAE_REPO_ID
        return cls(vae=kwargs["vae"])


def _video_input() -> torch.Tensor:
    return torch.arange(2 * 3 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 3, 8, 8) / 100.0


def _latent_input() -> torch.Tensor:
    return torch.arange(2 * 1 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 1, 3, 8, 8) / 100.0


def test_wan21_vae_constants_are_exported() -> None:
    assert WAN21_VAE_REPO_ID == "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    assert WAN21_VAE_SUBFOLDER == "vae"
    assert WAN21_VAE_RUN_MODES == ("decode", "encode")
    assert WAN21_VAE_OPTIMIZATION_KINDS == ("compile", "cuda_graph")


def test_load_wan21_vae_uses_diffusers_autoencoder(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_diffusers = types.SimpleNamespace(AutoencoderKLWan=_FakeAutoencoderKLWan)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    vae = load_wan21_vae(dtype=torch.float32, local_files_only=True)

    assert isinstance(vae, _FakeAutoencoderKLWan)


def test_load_wan21_pipeline_with_vae_replaces_pipeline_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_diffusers = types.SimpleNamespace(AutoencoderKLWan=_FakeAutoencoderKLWan)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipeline = load_wan21_pipeline_with_vae(
        pipeline_cls=_FakeWanPipeline,
        local_files_only=True,
    )

    assert isinstance(pipeline, _FakeWanPipeline)
    assert isinstance(pipeline.vae, _FakeAutoencoderKLWan)


def test_build_wan21_vae_runner_configures_tiling_and_slicing() -> None:
    pipeline = _FakeWanPipeline()

    runner, summary = build_wan21_vae_runner(
        pipeline,
        run_mode="decode",
        enable_tiling=True,
        enable_slicing=True,
        tile_sample_min_height=320,
        tile_sample_min_width=640,
        tile_sample_stride_height=256,
        tile_sample_stride_width=512,
        inplace=False,
    )

    assert runner.run_mode == "decode"
    assert summary.run_mode == "decode"
    assert summary.tiled is True
    assert summary.sliced is True
    assert summary.tile_sample_min_height == 320
    assert summary.tile_sample_min_width == 640
    assert summary.tile_sample_stride_height == 256
    assert summary.tile_sample_stride_width == 512
    assert pipeline.vae.use_tiling is False
    assert pipeline.vae.use_slicing is False


def test_collect_wan21_vae_conv3d_targets_only_keeps_1x1x1_layers() -> None:
    pipeline = _FakeWanPipeline()

    targets = collect_wan21_vae_conv3d_targets(pipeline, target_arch="sm_89")

    assert [target.target_path for target in targets] == ["conv_in", "conv_out"]
    assert all(target.patterns == ["conv3d_1x1x1"] for target in targets)


def test_materialize_wan21_vae_conv3d_fastpath_replaces_eligible_layers() -> None:
    pipeline = _FakeWanPipeline()

    optimized, target_count = materialize_wan21_vae_conv3d_fastpath(
        pipeline,
        target_arch="sm_89",
        inplace=False,
    )

    assert target_count == 2
    assert type(optimized.vae.conv_in).__name__ == "_TileLangConv3dWrapper"
    assert type(optimized.vae.conv_out).__name__ == "_TileLangConv3dWrapper"
    assert isinstance(optimized.vae.conv_skip, nn.Conv3d)
    assert isinstance(pipeline.vae.conv_in, nn.Conv3d)


def test_collect_wan21_vae_rmsnorm_targets_only_keeps_gamma_scale_layers() -> None:
    pipeline = _FakeWanPipeline()

    targets = collect_wan21_vae_rmsnorm_targets(pipeline)

    assert [target.target_path for target in targets] == ["norm_in", "norm_skip"]
    assert all(target.engine == "triton" for target in targets)
    assert all(target.patterns == ["rmsnorm"] for target in targets)


def test_materialize_wan21_vae_rmsnorm_fastpath_replaces_eligible_layers() -> None:
    pipeline = _FakeWanPipeline()

    optimized, target_count = materialize_wan21_vae_rmsnorm_fastpath(
        pipeline,
        inplace=False,
    )

    assert target_count == 2
    assert type(optimized.vae.norm_in).__name__ == "_TritonRMSNormWrapper"
    assert type(optimized.vae.norm_skip).__name__ == "_TritonRMSNormWrapper"
    assert isinstance(pipeline.vae.norm_in, _FakeWanRMSNorm)


def test_build_wan21_vae_runner_rejects_unknown_mode() -> None:
    with pytest.raises(XQTBackendError, match="Unsupported Wan 2.1 VAE run mode"):
        build_wan21_vae_runner(_FakeWanPipeline(), run_mode="sample")


def test_compile_wan21_vae_runner_delegates_to_compile_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_compile_with_torch(module: nn.Module, plan: object) -> tuple[nn.Module, float]:
        captured["module"] = module
        captured["engine"] = getattr(plan, "engine")
        captured["mode"] = getattr(plan, "mode")
        captured["options"] = dict(getattr(plan, "options"))
        return module, 5.5

    runner, _ = build_wan21_vae_runner(_FakeWanPipeline(), run_mode="decode")
    monkeypatch.setattr("xqt.model.wan21.runtime.compile_with_torch", fake_compile_with_torch)

    result = compile_wan21_vae_runner(
        runner,
        run_mode="decode",
        compile_engine="inductor",
        mode="reduce-overhead",
    )

    assert result.model is runner
    assert result.compile_time_ms == 5.5
    assert result.compile_engine == "inductor"
    assert result.compile_mode == "reduce-overhead"
    assert captured["module"] is runner
    assert captured["engine"] == "torch_compile"
    assert captured["mode"] == "reduce-overhead"
    assert captured["options"] == {"engine": "inductor"}


def test_compile_wan21_vae_runner_rejects_mode_plus_options() -> None:
    runner, _ = build_wan21_vae_runner(_FakeWanPipeline(), run_mode="decode")
    with pytest.raises(XQTBackendError, match="mode and options"):
        compile_wan21_vae_runner(
            runner,
            run_mode="decode",
            mode="reduce-overhead",
            options={"triton.cudagraphs": True},
        )


def test_build_wan21_vae_runner_materializes_rmsnorm_fastpath() -> None:
    pipeline = _FakeWanPipeline()

    runner, summary = build_wan21_vae_runner(
        pipeline,
        run_mode="decode",
        materialize_rmsnorm_fastpath=True,
        inplace=False,
    )

    assert summary.materialized_rmsnorm_targets == 2
    assert type(runner.vae.norm_in).__name__ == "_TritonRMSNormWrapper"  # type: ignore[attr-defined]
    assert isinstance(pipeline.vae.norm_in, _FakeWanRMSNorm)


def test_optimize_wan21_vae_requires_tensor_for_warmup() -> None:
    with pytest.raises(XQTBackendError, match="warmup requires tensor"):
        optimize_wan21_vae(_FakeWanPipeline(), run_mode="decode", warmup_iterations=1)


def test_optimize_wan21_vae_requires_tensor_for_cuda_graph() -> None:
    with pytest.raises(XQTBackendError, match="cuda_graph optimization requires tensor"):
        optimize_wan21_vae(
            _FakeWanPipeline(),
            run_mode="decode",
            optimization_kind="cuda_graph",
        )


def test_optimize_wan21_vae_compiles_and_warms_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakeWanPipeline()
    tensor = _latent_input()
    captured: dict[str, object] = {}

    def fake_compile(
        runner: nn.Module,
        **kwargs: object,
    ) -> object:
        captured["compile_kwargs"] = dict(kwargs)
        return types.SimpleNamespace(
            model=runner,
            run_mode="decode",
            compile_engine=kwargs["compile_engine"],
            compile_mode=kwargs["mode"],
            compile_time_ms=3.0,
            warmup_iterations=0,
            warmup_time_ms=0.0,
        )

    def fake_warmup(
        runner: nn.Module,
        *,
        tensor: torch.Tensor,
        warmup_iterations: int,
        sync_cuda: bool = True,
    ) -> float:
        del runner, sync_cuda
        captured["warmup_tensor"] = tensor
        captured["warmup_iterations"] = warmup_iterations
        return 7.0

    monkeypatch.setattr("xqt.model.wan21.optimize.compile_wan21_vae_runner", fake_compile)
    monkeypatch.setattr("xqt.model.wan21.optimize.warmup_wan21_vae_runner", fake_warmup)

    result, summary = optimize_wan21_vae(
        pipeline,
        run_mode="decode",
        tensor=tensor,
        warmup_iterations=2,
        inplace=False,
    )

    assert result.run_mode == "decode"
    assert result.compile_time_ms == 3.0
    assert result.warmup_time_ms == 7.0
    assert summary.run_mode == "decode"
    assert summary.optimization_kind == "compile"
    assert captured["compile_kwargs"]["compile_engine"] == "inductor"
    assert captured["warmup_tensor"] is tensor
    assert captured["warmup_iterations"] == 2


def test_optimize_wan21_vae_captures_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakeWanPipeline()
    tensor = _latent_input().cuda() if torch.cuda.is_available() else None
    if tensor is None:
        pytest.skip("CUDA required for CUDA Graph capture test")

    captured: dict[str, object] = {}

    def fake_capture(
        runner: nn.Module,
        *,
        tensor: torch.Tensor,
        run_mode: str,
        warmup_iterations: int,
    ) -> object:
        captured["runner"] = runner
        captured["tensor"] = tensor
        captured["run_mode"] = run_mode
        captured["warmup_iterations"] = warmup_iterations
        return types.SimpleNamespace(
            model=runner,
            run_mode=run_mode,
            graph_state={},
            input_signature=(),
            warmup_iterations=warmup_iterations,
            capture_time_ms=9.0,
            to_dict=lambda: {"capture_time_ms": 9.0, "run_mode": run_mode},
        )

    monkeypatch.setattr("xqt.model.wan21.optimize.capture_wan21_vae_cuda_graph", fake_capture)

    result, summary = optimize_wan21_vae(
        pipeline,
        run_mode="decode",
        optimization_kind="cuda_graph",
        tensor=tensor,
        warmup_iterations=3,
        inplace=False,
    )

    assert result.capture_time_ms == 9.0
    assert summary.optimization_kind == "cuda_graph"
    assert captured["run_mode"] == "decode"
    assert captured["tensor"] is tensor
    assert captured["warmup_iterations"] == 3


def test_warmup_wan21_vae_runner_runs_requested_iterations() -> None:
    class _CountingRunner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return tensor + 1.0

    runner = _CountingRunner()
    tensor = _latent_input()
    elapsed_ms = warmup_wan21_vae_runner(
        runner,
        tensor=tensor,
        warmup_iterations=4,
        sync_cuda=False,
    )
    assert elapsed_ms >= 0.0
    assert runner.calls == 4


def test_benchmark_wan21_vae_runner_uses_explicit_warmup() -> None:
    class _CountingRunner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return tensor + 1.0

    runner = _CountingRunner()
    tensor = _latent_input()
    report = benchmark_wan21_vae_runner(
        runner,
        tensor=tensor,
        run_mode="decode",
        warmup=2,
        iterations=3,
        sync_cuda=False,
    )
    assert report["run_mode"] == "decode"
    assert report["warmup"] == 2
    assert report["iterations"] == 3
    assert runner.calls == 5


def test_benchmark_wan21_vae_paired_compares_candidate_against_eager() -> None:
    torch.manual_seed(0)
    eager_runner, _ = build_wan21_vae_runner(_FakeWanPipeline(), run_mode="decode", inplace=False)
    torch.manual_seed(0)
    candidate_runner, _ = build_wan21_vae_runner(_FakeWanPipeline(), run_mode="decode", inplace=False)
    tensor = _latent_input()

    result = benchmark_wan21_vae_paired(
        eager_runner=eager_runner,
        candidate_runner=candidate_runner,
        tensor=tensor,
        run_mode="decode",
        warmup=1,
        iterations=2,
        sync_cuda=False,
        atol=1e-5,
        rtol=1e-5,
    )

    assert result.run_mode == "decode"
    assert result.allclose_vs_eager is True
    assert result.max_abs_vs_eager == 0.0
    assert len(result.paired_speedup_ratios) == 2


def test_run_wan21_vae_inference_returns_summary_and_output() -> None:
    pipeline = _FakeWanPipeline()
    tensor = _latent_input()

    result = run_wan21_vae_inference(
        pipeline,
        tensor=tensor,
        run_mode="decode",
        warmup_iterations=0,
        inplace=False,
    )

    assert result["summary"]["run_mode"] == "decode"
    assert result["summary"]["optimization_kind"] == "compile"
    assert isinstance(result["output"], torch.Tensor)
    assert tuple(result["output"].shape) == (2, 3, 3, 8, 8)
