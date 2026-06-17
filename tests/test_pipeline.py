from dataclasses import dataclass

import pytest

from xqt.core.artifact import ArtifactManifest
from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTPipelineError
from xqt.core.types import XQTContext
from xqt.pipeline.pass_manager import SequentialPipeline


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
