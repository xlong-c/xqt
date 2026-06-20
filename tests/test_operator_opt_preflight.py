from xqt.pipeline.preflight import preflight_xqt_config


def test_preflight_reports_operator_optimization_capability(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    monkeypatch.setattr(
        "xqt.pipeline.preflight._package_available",
        lambda package_name: False if package_name == "tilelang" else True,
    )
    monkeypatch.setattr(
        "xqt.operator_opt.capability._package_available",
        lambda package_name: False if package_name == "tilelang" else True,
    )

    report = preflight_xqt_config(
        {
            "model": {
                "target": "torch.nn.Linear",
                "params": {"in_features": 4, "out_features": 2},
            },
            "data": {
                "validation": {
                    "target": "synthetic_classification",
                    "sample_limit": 1,
                    "batch_size": 1,
                }
            },
            "operator_optimization": {
                "enabled": True,
                "targets": [
                    {
                        "name": "compile_model",
                        "backend": "torch_compile",
                        "target": "model",
                    },
                    {
                        "name": "tilelang_attention",
                        "backend": "tilelang",
                        "target": "model.attn",
                    },
                ],
            },
        }
    )
    checks = {check.name: check for check in report.checks}

    assert checks["operator_optimization.targets"].passed is True
    assert checks["operator_optimization.torch_compile"].metadata["torch_version"]
    assert checks["operator_optimization.hardware.cuda"].passed is False
    assert checks["operator_optimization.targets.0.capability"].passed is True
    assert checks["operator_optimization.targets.0.capability"].metadata["backend"] == "torch_compile"
    assert checks["operator_optimization.targets.1.capability"].metadata["requires_cuda"] is True
    assert checks["operator_optimization.targets.1.tilelang.runtime"].passed is False
    assert checks["operator_optimization.targets.1.tilelang.runtime"].level == "warning"
    assert checks["operator_optimization.targets.1.tilelang.config"].passed is True
    assert checks["operator_optimization.targets.1.tilelang.config"].metadata["target"] == "cuda"
    assert checks["operator_optimization.targets.1.tilelang.config"].metadata["target_arch"] is None
