import tomllib
from pathlib import Path


def test_xqt_optional_dependency_groups_are_declared() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert "xqt" in optional
    assert "xqt-hf" in optional
    assert "xqt-yolo" in optional
    assert "xqt-diffusion" in optional
    assert "xqt-all" in optional
    assert any(dep.startswith("onnxruntime") for dep in optional["xqt"])
    assert any(dep.startswith("datasets") for dep in optional["xqt-hf"])
    assert any(dep.startswith("ultralytics") for dep in optional["xqt-yolo"])
    assert any(dep.startswith("diffusers") for dep in optional["xqt-diffusion"])
    assert "xdl[xqt-all]" in optional["all"]
