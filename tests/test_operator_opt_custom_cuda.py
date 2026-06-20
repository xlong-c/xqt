import torch

from xqt.operator_opt.cuda_extension import (
    CUSTOM_CUDA_BUILD_ENV,
    CUSTOM_CUDA_EXTENSION_NAME,
    custom_cuda_build_requested,
    describe_custom_cuda_extension_capability,
    fused_bias_gelu_custom_cuda,
    run_custom_cuda_opcheck,
)
from xqt.pipeline.preflight import preflight_xqt_config


def test_custom_cuda_placeholder_op_matches_reference() -> None:
    x = torch.randn(2, 4)
    bias = torch.randn(4)

    output = fused_bias_gelu_custom_cuda(x, bias)

    assert torch.allclose(output, torch.nn.functional.gelu(x + bias))


def test_custom_cuda_placeholder_op_supports_faketensor() -> None:
    with torch._subclasses.fake_tensor.FakeTensorMode():
        x = torch.empty(2, 4)
        bias = torch.empty(4)
        output = fused_bias_gelu_custom_cuda(x, bias)

    assert output.shape == torch.Size([2, 4])
    assert output.dtype == x.dtype


def test_custom_cuda_placeholder_opcheck_passes() -> None:
    result = run_custom_cuda_opcheck()

    assert result
    assert all(status == "SUCCESS" for status in result.values())


def test_custom_cuda_capability_does_not_compile_extension_by_default() -> None:
    capability = describe_custom_cuda_extension_capability(env={})

    assert capability.extension_name == CUSTOM_CUDA_EXTENSION_NAME
    assert capability.build_env_var == CUSTOM_CUDA_BUILD_ENV
    assert capability.build_requested is False
    assert capability.compiled is False
    assert capability.available is False
    assert "bias_gelu" in capability.registered_ops


def test_custom_cuda_build_switch_is_explicit() -> None:
    assert custom_cuda_build_requested({CUSTOM_CUDA_BUILD_ENV: "1"}) is True
    assert custom_cuda_build_requested({CUSTOM_CUDA_BUILD_ENV: "true"}) is True
    assert custom_cuda_build_requested({CUSTOM_CUDA_BUILD_ENV: "0"}) is False


def test_preflight_reports_custom_cuda_extension_metadata(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

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
                        "name": "custom_bias_gelu",
                        "backend": "custom_cuda",
                        "target": "model",
                        "patterns": ["bias_gelu"],
                    }
                ],
            },
        }
    )
    checks = {check.name: check for check in report.checks}

    extension = checks["operator_optimization.targets.0.custom_cuda.extension"]
    assert extension.passed is False
    assert extension.level == "warning"
    assert extension.metadata["compiled"] is False
    assert extension.metadata["available"] is False
    assert "bias_gelu" in extension.metadata["registered_ops"]
