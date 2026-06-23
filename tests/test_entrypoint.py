from pathlib import Path

from examples import yolo_detection_practice
from xqt.entrypoints import preflight
from xqt.entrypoints.run_recipe import CONFIG_ENV, WRITE_MANIFEST_ENV, main
from xqt.entrypoints.run_workflow import (
    CONFIG_ENV as WORKFLOW_CONFIG_ENV,
    main as workflow_main,
)


def test_run_recipe_entrypoint_uses_env_config(monkeypatch, tmp_path, capsys) -> None:
    recipe = tmp_path / "recipe.yaml"
    artifact_dir = tmp_path / "artifacts"
    recipe.write_text(
        f"""
project:
  name: entrypoint_smoke
  artifact_dir: {artifact_dir}
model:
  target: torch.nn.Linear
  params:
    in_features: 4
    out_features: 2
data:
  validation:
    target: synthetic_classification
    sample_limit: 2
    batch_size: 1
benchmark:
  warmup: 0
  iterations: 1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV, str(recipe))

    assert main([]) == 0

    output = capsys.readouterr().out
    assert "project: entrypoint_smoke" in output
    assert (artifact_dir / "manifest.json").is_file()


def test_run_recipe_entrypoint_can_skip_manifest(monkeypatch, tmp_path, capsys) -> None:
    recipe = tmp_path / "recipe.yaml"
    artifact_dir = tmp_path / "artifacts"
    recipe.write_text(
        f"""
project:
  name: entrypoint_no_manifest
  artifact_dir: {artifact_dir}
model:
  target: torch.nn.Linear
  params:
    in_features: 4
    out_features: 2
data:
  validation:
    target: synthetic_classification
    sample_limit: 2
    batch_size: 1
benchmark:
  warmup: 0
  iterations: 1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV, str(recipe))
    monkeypatch.setenv(WRITE_MANIFEST_ENV, "0")

    assert main([]) == 0

    output = capsys.readouterr().out
    assert "project: entrypoint_no_manifest" in output
    assert not (artifact_dir / "manifest.json").exists()


def test_preflight_entrypoint_outputs_json_and_status(monkeypatch, tmp_path, capsys) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """
model:
  target: missing.module.Target
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(preflight.CONFIG_ENV, str(recipe))

    assert preflight.main([]) == 1

    output = capsys.readouterr().out
    assert '"passed": false' in output
    assert "model.target" in output


def test_run_workflow_entrypoint_uses_env_config(monkeypatch, tmp_path, capsys) -> None:
    recipe = tmp_path / "workflow.yaml"
    artifact_dir = tmp_path / "workflow_artifacts"
    recipe.write_text(
        f"""
project:
  name: workflow_smoke
  artifact_dir: {artifact_dir}
task:
  type: classification
model:
  target: torch.nn.Linear
  params:
    in_features: 4
    out_features: 2
data_splits:
  validation:
    target: synthetic_classification
    sample_limit: 2
    batch_size: 1
stages:
  - name: baseline_eval
    kind: eval
    split: validation
    params:
      baseline: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(WORKFLOW_CONFIG_ENV, str(recipe))

    assert workflow_main([]) == 0

    output = capsys.readouterr().out
    assert "project: workflow_smoke_baseline_eval" in output
    assert "stages: baseline_eval" in output
    assert "workflow_manifest:" in output
    assert "workflow_result:" in output
    assert (artifact_dir / "baseline_eval" / "workflow_manifest.json").is_file()
    assert (artifact_dir / "baseline_eval" / "workflow_result.json").is_file()


class _StageRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def eval(self, **kwargs: object) -> None:
        self.calls.append(("eval", kwargs))

    def benchmark(self, **kwargs: object) -> None:
        self.calls.append(("benchmark", kwargs))

    def export(self, **kwargs: object) -> None:
        self.calls.append(("export", kwargs))

    def runtime_eval(self, **kwargs: object) -> None:
        self.calls.append(("runtime_eval", kwargs))

    def quant(self, **kwargs: object) -> None:
        self.calls.append(("quant", kwargs))

    def prune(self, **kwargs: object) -> None:
        self.calls.append(("prune", kwargs))


def test_yolo_detection_practice_example_uses_session_stage_calls() -> None:
    recorder = _StageRecorder()

    yolo_detection_practice._run_default_stages(recorder)  # type: ignore[arg-type]

    assert "ultralytics" not in repr(recorder.calls).lower()
    assert [kind for kind, _kwargs in recorder.calls] == [
        "eval",
        "benchmark",
        "export",
        "runtime_eval",
        "quant",
        "runtime_eval",
        "prune",
        "eval",
        "benchmark",
        "prune",
    ]
    assert recorder.calls[6][1]["method"] == "global_l1_unstructured"
    assert recorder.calls[-1][1]["method"] == "structured"


def test_yolo_detection_practice_example_builds_session_without_ultralytics() -> None:
    session = yolo_detection_practice._build_default_session()

    assert session.config.project["name"] == "yolo_detection_practice_example"
    assert session.config.task.type == "detection"
    assert "ultralytics" not in repr(session.config).lower()
    assert set(session.context.data) == {"calibration", "validation"}
