import torch

from xqt.core.artifact import ArtifactManifest
from xqt.core.config import load_xqt_config
from xqt.core.types import XQTContext
from xqt.pipeline.passes import OperatorOptimizationPass


def test_operator_optimization_pass_skips_when_speedup_threshold_not_met(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "operator_opt")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                        "min_speedup": 10.0,
                    }
                ],
            },
        }
    )
    context = XQTContext(
        config=config,
        model=model,
        data={"validation": [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = OperatorOptimizationPass().run(context)

    assert output is context
    metrics = output.metrics["operator_optimization"]
    assert metrics["target_count"] == 1
    assert metrics["applied_count"] == 0
    assert metrics["skipped_count"] == 1
    target = metrics["targets"][0]
    assert target["target_name"] == "model"
    assert target["backend"] == "torch_compile"
    assert target["fallback"] == "eager"
    assert "min_speedup" in target["skip_reason"]
    assert target["numeric_diff"]["allclose"] is True
    assert output.artifacts["operator_optimization_report"].is_file()
    assert output.manifest is not None
    assert any(
        metric.name == "operator_optimization.model.applied"
        for metric in output.manifest.metrics
    )


def test_operator_optimization_pass_skips_non_pytorch_quant_runtime(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "operator_opt_qdq_guard")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                        "min_speedup": 1.000001,
                    }
                ],
            },
        }
    )
    context = XQTContext(
        config=config,
        model=model,
        data={"validation": [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))]},
        metrics={"quant": {"backend": "onnxruntime_qdq", "runtime": "onnxruntime"}},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = OperatorOptimizationPass().run(context)

    target = output.metrics["operator_optimization"]["targets"][0]
    assert target["applied"] is False
    assert "onnxruntime_qdq" in target["skip_reason"]


def test_operator_optimization_pass_emits_candidate_reports(tmp_path) -> None:
    class SwiGLUToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin1 = torch.nn.Linear(8, 16)
            self.lin2 = torch.nn.Linear(8, 16)
            self.out = torch.nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out(torch.nn.functional.silu(self.lin1(x)) * self.lin2(x))

    model = SwiGLUToy()
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "operator_opt_candidates")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                        "min_speedup": 10.0,
                    }
                ],
            },
        }
    )
    context = XQTContext(
        config=config,
        model=model,
        data={"validation": [(torch.randn(2, 8), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = OperatorOptimizationPass().run(context)

    assert "operator_optimization_candidates" in output.artifacts
    candidates = output.artifacts["operator_optimization_candidates"]
    assert candidates["fx"]["candidate_count"] >= 1
    assert "swiglu" in candidates["fx"]["patterns"]
    assert isinstance(candidates["torch_export"]["candidates"], list)
    assert output.metrics["operator_optimization"]["candidates"]["fx"]["candidate_count"] >= 1


def test_operator_optimization_pass_preserves_candidate_scan_errors(tmp_path) -> None:
    class FxUntraceableToy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            chunks = list(x.chunk(2, dim=1))
            return chunks[0] + chunks[1]

    model = FxUntraceableToy()
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "operator_opt_candidate_error")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "model",
                        "backend": "torch_compile",
                        "min_speedup": 10.0,
                    }
                ],
            },
        }
    )
    context = XQTContext(
        config=config,
        model=model,
        data={"validation": [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = OperatorOptimizationPass().run(context)

    candidates = output.artifacts["operator_optimization_candidates"]
    assert candidates["fx"]["status"] == "error"
    assert candidates["fx"]["candidate_count"] == 0
    assert "Proxy object cannot be iterated" in candidates["fx"]["error"]
    assert candidates["fx"]["candidates"] == []
    assert candidates["torch_export"]["status"] in {"ok", "error"}
    assert output.metrics["operator_optimization"]["target_count"] == 1
    assert output.artifacts["operator_optimization_report"].is_file()


def test_operator_optimization_pass_records_custom_backend_metadata(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "operator_custom_backend_metadata")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "mlp_triton",
                        "target": "",
                        "backend": "triton",
                        "patterns": ["swiglu"],
                    },
                    {
                        "name": "attention_tilelang",
                        "target": "",
                        "backend": "tilelang",
                        "patterns": ["attention", "dequant_gemm_epilogue"],
                        "tilelang": {
                            "target": "cuda",
                            "target_arch": "sm_89",
                            "cache_dir": str(tmp_path / "tilelang-cache"),
                            "pass_configs": {"TL_ENABLE_FAST_MATH": True},
                        },
                    },
                ],
            },
        }
    )
    context = XQTContext(
        config=config,
        model=model,
        data={"validation": [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = OperatorOptimizationPass().run(context)
    targets = {
        target["target_name"]: target
        for target in output.metrics["operator_optimization"]["targets"]
    }

    triton = targets["mlp_triton"]
    assert triton["applied"] is False
    assert (
        "requires CUDA-capable hardware" in triton["skip_reason"]
        or "not implemented in the built-in executor" in triton["skip_reason"]
    )
    assert "swiglu" in triton["metadata"]["kernel_registry"]

    tilelang = targets["attention_tilelang"]
    assert tilelang["applied"] is False
    assert (
        "requires CUDA-capable hardware" in tilelang["skip_reason"]
        or "not implemented in the built-in executor" in tilelang["skip_reason"]
    )
    assert "attention" in tilelang["metadata"]["kernel_registry"]
    assert "dequant_gemm_epilogue" in tilelang["metadata"]["tilelang_artifacts"]
    assert (
        tilelang["metadata"]["tilelang_artifacts"]["attention"]["compile"]["target_arch"]
        == "sm_89"
    )
    assert (
        tilelang["artifact_paths"]["tilelang.attention"]
        == str(tmp_path / "tilelang-cache" / "attention.tilelang.json")
    )
    assert tilelang["metadata"]["latency"]["status"] == "not_executed"
    assert tilelang["metadata"]["latency"]["compile_latency_ms"] is None
    assert tilelang["metadata"]["validation_thresholds"] == {"atol": 1e-5, "rtol": 1e-5}
    assert (
        tilelang["metadata"]["tilelang_artifacts"]["attention"]["artifact_path"]
        == str(tmp_path / "tilelang-cache" / "attention.tilelang.json")
    )
