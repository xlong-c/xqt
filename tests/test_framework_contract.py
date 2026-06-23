import tomllib

import xqt
from xqt.run_workflow import DEFAULT_CONFIG
from xqt.workflows import load_optimization_config, optimize_model


def test_top_level_xqt_api_matches_framework_contract() -> None:
    expected = {
        "XQTOptimizationSession",
        "OptimizedModelResult",
        "OptimizationConfig",
        "OptimizationStageConfig",
        "OptimizationStageResult",
        "StageAcceptanceConfig",
        "load_optimization_config",
        "optimize_model",
        "ArtifactManifest",
        "ArtifactRecord",
        "MetricRecord",
        "load_checkpoint_into_model",
        "xdl_checkpoint_to_xqt_context",
        "xdl_setup_to_xqt_context",
    }
    removed = {
        "load_xqt_config",
        "run_xqt_recipe",
        "preflight_xqt_config",
        "XQTConfig",
        "XQTRegistry",
        "PASS_REGISTRY",
        "RECIPE_REGISTRY",
        "EXPORTER_REGISTRY",
        "register_pass",
        "register_recipe",
        "register_exporter",
    }

    assert set(xqt.__all__) == expected
    assert removed.isdisjoint(set(xqt.__all__))
    for name in removed:
        assert not hasattr(xqt, name)


def test_default_workflow_config_is_stage_workflow() -> None:
    config = load_optimization_config(DEFAULT_CONFIG)

    assert DEFAULT_CONFIG.name == "smoke_workflow.yaml"
    assert config.project["name"] == "smoke_workflow"
    assert [stage.kind for stage in config.stages] == ["prune"]


def test_default_workflow_runs_without_external_inputs() -> None:
    result = optimize_model(DEFAULT_CONFIG, write_outputs=False)

    assert [stage.name for stage in result.stages] == ["prune_l1"]
    assert result.stages[0].accepted is True
    assert result.best_stage == "prune_l1"


def test_pyproject_exposes_only_workflow_console_script() -> None:
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    scripts = pyproject["project"]["scripts"]
    assert scripts["xqt-run-workflow"] == "xqt.run_workflow:main"
    assert "xqt-run-recipe" not in scripts
    assert "xqt-preflight" not in scripts

    package_data = pyproject["tool"]["setuptools"]["package-data"]["xqt"]
    assert "recipes/*/*.yaml" in package_data
    assert "recipes/*/*/*.yaml" in package_data
