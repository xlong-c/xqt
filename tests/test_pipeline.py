from dataclasses import dataclass

import pytest
import torch
from torch import nn

from xqt.core.artifact import ArtifactManifest
from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTPipelineError
from xqt.core.types import XQTContext
from xqt.pipeline.pass_manager import SequentialPipeline
from xqt.pipeline.passes import AnalyzePass


@dataclass
class SetMetricPass:
    name: str
    metric_name: str
    value: float

    def run(self, context: XQTContext) -> XQTContext:
        context.metrics[self.metric_name] = self.value
        return context


class FailingPass:
    name = "failing"

    def run(self, context: XQTContext) -> XQTContext:
        raise RuntimeError("boom")


class TinyPipelineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def test_sequential_pipeline_runs_passes_and_records_manifest() -> None:
    config = load_xqt_config({"project": {"name": "pipeline"}})
    context = XQTContext(
        config=config,
        manifest=ArtifactManifest(project_name=config.project.name),
    )
    pipeline = SequentialPipeline.from_iterable(
        [
            SetMetricPass(name="baseline_eval", metric_name="baseline.top1", value=0.8),
            SetMetricPass(name="benchmark", metric_name="latency.p50", value=1.2),
        ]
    )

    output = pipeline.run(context)

    assert output is context
    assert output.metrics == {"baseline.top1": 0.8, "latency.p50": 1.2}
    assert output.manifest is not None
    assert output.manifest.passes == ["baseline_eval", "benchmark"]


def test_sequential_pipeline_wraps_pass_errors() -> None:
    config = load_xqt_config({})
    context = XQTContext(config=config)
    pipeline = SequentialPipeline([FailingPass()])

    with pytest.raises(XQTPipelineError, match="Pass 'failing' failed: boom"):
        pipeline.run(context)


def test_xqt_context_require_model_returns_model_or_raises() -> None:
    config = load_xqt_config({})
    context = XQTContext(config=config, model=object())

    assert context.require_model() is context.model

    context.model = None
    with pytest.raises(ValueError, match="XQTContext.model is required"):
        context.require_model()


def test_analyze_pass_writes_analysis_artifacts(tmp_path) -> None:
    reference = TinyPipelineModel()
    candidate = TinyPipelineModel()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        candidate.features[0].bias.add_(0.25)

    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "artifacts")},
            "model": {"device": "cpu"},
            "analysis": {"enabled": True, "top_k": 1},
        }
    )
    context = XQTContext(
        config=config,
        model=candidate,
        reference_model=reference,
        data={"validation": [(torch.ones(2, 3), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = AnalyzePass().run(context)

    assert output is context
    assert output.metrics["analysis"]["compare_to"] == "baseline"
    assert output.metrics["analysis"]["metrics"] == [
        "max_abs",
        "mean_abs",
        "cosine_similarity",
    ]
    assert output.metrics["analysis"]["record_count"] == 1
    assert len(output.metrics["analysis"]["rows"]) == 1
    assert len(output.metrics["analysis"]["activation_drift"]) >= 1
    assert len(output.metrics["analysis"]["importance"]) >= 1
    assert len(output.metrics["analysis"]["prune_candidates"]) >= 1
    assert any(
        record["name"] == "features.0" for record in output.metrics["analysis"]["activation_drift"]
    )
    assert output.metrics["analysis"]["recommended_high_precision_modules"] == [
        output.metrics["analysis"]["records"][0]["name"]
    ]
    assert output.artifacts["analysis_json"].is_file()
    assert output.artifacts["analysis_csv"].is_file()
    assert output.artifacts["analysis_markdown"].is_file()
    assert output.manifest is not None
    assert output.manifest.metrics[-1].name == "analysis.record_count"


def test_analyze_pass_reports_teacher_student_alignment_when_teacher_exists(tmp_path) -> None:
    teacher = TinyPipelineModel()
    reference = TinyPipelineModel()
    candidate = TinyPipelineModel()
    reference.load_state_dict(teacher.state_dict())
    candidate.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        candidate.features[0].bias.add_(0.3)

    config = load_xqt_config(
        {
            "project": {"artifact_dir": str(tmp_path / "teacher_alignment")},
            "model": {"device": "cpu"},
            "analysis": {"enabled": True, "top_k": 1},
        }
    )
    context = XQTContext(
        config=config,
        model=candidate,
        reference_model=reference,
        teacher=teacher,
        data={"validation": [(torch.ones(2, 3), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    output = AnalyzePass().run(context)

    assert len(output.metrics["analysis"]["teacher_student_alignment"]) == 1
    alignment = output.metrics["analysis"]["teacher_student_alignment"][0]
    assert alignment["teacher_name"] == "features.0"
    assert alignment["student_name"] == "features.0"


def test_benchmark_pass_populates_analysis_pareto_points(tmp_path) -> None:
    reference = TinyPipelineModel()
    candidate = TinyPipelineModel()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        candidate.features[0].bias.add_(0.25)

    config = load_xqt_config(
        {
            "project": {"name": "pareto_case", "artifact_dir": str(tmp_path / "pareto")},
            "model": {"device": "cpu"},
            "benchmark": {"warmup": 0, "iterations": 1},
            "analysis": {"enabled": True, "top_k": 1},
        }
    )
    context = XQTContext(
        config=config,
        model=candidate,
        reference_model=reference,
        data={"validation": [(torch.ones(2, 3), torch.zeros(2, dtype=torch.long))]},
        manifest=ArtifactManifest(project_name=config.project.name),
    )

    from xqt.pipeline.passes import BenchmarkPass

    AnalyzePass().run(context)
    BenchmarkPass().run(context)

    assert len(context.metrics["analysis"]["pareto_points"]) == 1
    point = context.metrics["analysis"]["pareto_points"][0]
    assert point["config_id"] == "pareto_case"
    assert point["latency_ms"] is not None
    assert point["error"] is not None
