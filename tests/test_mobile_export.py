import pytest
import torch
from torch import nn

from xqt.core.config import load_xqt_config
from xqt.core.errors import XQTBackendError
from xqt.export import (
    build_mnnconvert_command,
    build_onnx2ncnn_command,
    build_pnnx_command,
    export_executorch_program,
    export_mnn_from_onnx,
    export_ncnn_from_onnx,
    export_ncnn_with_pnnx,
)
from xqt.pipeline.runner import run_xqt_recipe


def test_build_mobile_conversion_commands() -> None:
    assert build_onnx2ncnn_command("model.onnx", "model.param", "model.bin") == [
        "onnx2ncnn",
        "model.onnx",
        "model.param",
        "model.bin",
    ]
    assert build_mnnconvert_command("model.onnx", "model.mnn") == [
        "MNNConvert",
        "-f",
        "ONNX",
        "--modelFile",
        "model.onnx",
        "--MNNModel",
        "model.mnn",
    ]
    assert build_pnnx_command("model.onnx", extra_args=["inputshape=[1,3,224,224]"]) == [
        "pnnx",
        "model.onnx",
        "inputshape=[1,3,224,224]",
    ]


def test_ncnn_and_mnn_dry_run_require_existing_onnx(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    ncnn = export_ncnn_from_onnx(
        onnx_path,
        tmp_path / "model.param",
        tmp_path / "model.bin",
        dry_run=True,
    )
    mnn = export_mnn_from_onnx(
        onnx_path,
        tmp_path / "model.mnn",
        dry_run=True,
    )

    assert ncnn.dry_run is True
    assert ncnn.output_paths == [tmp_path / "model.param", tmp_path / "model.bin"]
    assert ncnn.command[0] == "onnx2ncnn"
    assert mnn.dry_run is True
    assert mnn.output_paths == [tmp_path / "model.mnn"]
    assert mnn.command[0] == "MNNConvert"

    with pytest.raises(XQTBackendError, match="ONNX file not found"):
        export_ncnn_from_onnx(tmp_path / "missing.onnx", "model.param", "model.bin")


def test_pnnx_ncnn_dry_run_uses_default_outputs(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    result = export_ncnn_with_pnnx(onnx_path, dry_run=True)

    assert result.dry_run is True
    assert result.command == ["pnnx", str(onnx_path)]
    assert result.output_paths == [
        tmp_path / "model.ncnn.param",
        tmp_path / "model.ncnn.bin",
    ]
    assert result.metadata["converter"] == "pnnx"


def test_executorch_dry_run_returns_pte_path(tmp_path) -> None:
    result = export_executorch_program(
        nn.Linear(2, 2),
        torch.randn(1, 2),
        tmp_path / "model.pte",
        dry_run=True,
        metadata={"source": "unit"},
    )

    assert result.dry_run is True
    assert result.pte_path == tmp_path / "model.pte"
    assert result.checksum is None
    assert result.metadata["source"] == "unit"


def test_pipeline_supports_mobile_dry_run_targets_after_onnx(tmp_path) -> None:
    onnx_path = tmp_path / "model.onnx"
    config = load_xqt_config(
        "xqt/recipes/smoke/smoke_cpu.yaml",
        overrides={
            "project": {"artifact_dir": str(tmp_path / "artifacts")},
            "compression": {"prune": {"enabled": False}},
            "export": {
                "targets": [
                    {
                        "format": "onnx",
                        "output_path": str(onnx_path),
                        "params": {"runtime_diff": False},
                    },
                    {
                        "format": "ncnn",
                        "output_path": str(tmp_path / "model.param"),
                        "params": {
                            "bin_path": str(tmp_path / "model.bin"),
                            "dry_run": True,
                        },
                    },
                    {
                        "format": "mnn",
                        "output_path": str(tmp_path / "model.mnn"),
                        "params": {"dry_run": True},
                    },
                    {
                        "format": "executorch",
                        "output_path": str(tmp_path / "model.pte"),
                        "params": {"dry_run": True},
                    },
                ]
            },
        },
    )

    context = run_xqt_recipe(config)
    formats = [
        artifact["format"]
        for artifact in context.metrics["export"]["artifacts"]
    ]

    assert formats == ["onnx", "ncnn", "mnn", "executorch"]
    assert context.metrics["export"]["artifacts"][1]["dry_run"] is True
    assert context.metrics["export"]["artifacts"][2]["dry_run"] is True
    assert context.metrics["export"]["artifacts"][3]["dry_run"] is True
