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


def test_yolo_detection_practice_example_embeds_lightweight_workflow() -> None:
    config = yolo_detection_practice._default_config()

    assert config["model"]["target"] == "xqt.model.build_toy_detection_module"
    assert "ultralytics" not in repr(config).lower()
    assert [stage["kind"] for stage in config["stages"]] == [
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
    assert config["stages"][6]["params"]["method"] == "global_l1_unstructured"
    assert config["stages"][-1]["params"]["method"] == "structured"
