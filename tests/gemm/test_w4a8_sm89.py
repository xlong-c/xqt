from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    W4A8Contract,
    build_w4a8_gemm_spec,
    calibrate_fp8_scale,
    default_registry,
    dispatch_gemm,
    pack_w4a8_weight,
    reference_w4a8_gemm,
)
from xqt.kernels.ops._impl.gemm_backends.sm89 import w4a8_sm89 as backend


_CUDA_SM89 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 9)


def _packed(
    *,
    contract: W4A8Contract,
    n: int,
    k: int,
    device: torch.device | str = "cpu",
) -> object:
    packed = pack_w4a8_weight(torch.randn(n, k), contract=contract)
    if device == "cpu":
        return packed
    return type(packed)(
        qweight=packed.qweight.to(device),
        scales=packed.scales.to(device),
        zero_points=None,
        metadata=packed.metadata,
    )


def _spec(
    *,
    contract: W4A8Contract,
    m: int,
    n: int,
    k: int,
    device: str,
    has_bias: bool = False,
) -> GemmSpec:
    built = build_w4a8_gemm_spec(
        GemmProblem(m=m, n=n, k=k, sm=89, device=device),
        contract,
        has_bias=has_bias,
    )
    return GemmSpec(
        problem=built.problem,
        quant=built.quant,
        epilogue=EpilogueSpec(output_dtype=contract.output_dtype, has_bias=has_bias),
    )


def test_cpu_dispatch_uses_w4a8_reference_contract() -> None:
    contract = W4A8Contract()
    packed = _packed(contract=contract, n=8, k=32)
    spec = _spec(contract=contract, m=4, n=8, k=32, device="cpu")
    result = dispatch_gemm(torch.randn(4, 32), packed, spec=spec)

    assert result.report.selected_kernel == "w4a8_reference"
    assert result.report.native is False


def test_w4a8_executor_rejects_cpu_contract() -> None:
    contract = W4A8Contract()
    with pytest.raises(XQTBackendError, match="CUDA"):
        backend.sm89_w4a8_executor(
            torch.ones((16, 32), dtype=torch.float16),
            _packed(contract=contract, n=16, k=32),
            spec=_spec(contract=contract, m=16, n=16, k=32, device="cpu"),
            artifact=Path("missing.so"),
        )


def test_missing_w4a8_artifact_does_not_promote_registry(tmp_path: Path) -> None:
    artifact = tmp_path / "missing-w4a8.so"
    registry = default_registry()

    assert backend.sm89_w4a8_artifact_available(artifact) is False
    assert backend.install_sm89_w4a8_executors(registry, artifact=artifact) is False
    assert registry.get("sm89_w4a8_int8_cutlass").maturity == "metadata_only"
    assert registry.get("sm89_w4a8_fp8_cutlass").maturity == "metadata_only"


def test_w4a8_install_requires_correctness_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = default_registry()
    artifact = tmp_path / "w4a8.so"
    monkeypatch.setattr(backend, "sm89_w4a8_artifact_available", lambda _: True)
    monkeypatch.setattr(backend, "artifact_ready_for_execution", lambda *args, **kwargs: False)

    assert backend.install_sm89_w4a8_executors(registry, artifact=artifact) is False
    assert registry.get("sm89_w4a8_int8_cutlass").maturity == "metadata_only"


def test_w4a8_install_promotes_after_correctness_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = default_registry()
    artifact = tmp_path / "w4a8.so"
    monkeypatch.setattr(backend, "sm89_w4a8_artifact_available", lambda _: True)
    monkeypatch.setattr(backend, "artifact_ready_for_execution", lambda *args, **kwargs: True)

    assert backend.install_sm89_w4a8_executors(registry, artifact=artifact) is True
    for name in ("sm89_w4a8_int8_cutlass", "sm89_w4a8_fp8_cutlass"):
        entry = registry.get(name)
        assert entry.maturity == "executable"
        assert entry.executor is not None


@pytest.mark.skipif(not _CUDA_SM89, reason="requires an SM89 CUDA device")
@pytest.mark.skipif(
    not backend.sm89_w4a8_artifact_available(),
    reason="SM89 W4A8 artifact is not built",
)
@pytest.mark.parametrize("granularity", ("per_tensor", "per_token"))
def test_native_int8_w4a8_matches_reference(granularity: str) -> None:
    device = torch.device("cuda:0")
    m, n, k = 16, 16, 64
    contract = W4A8Contract(weight_group_size=32, activation_granularity=granularity)
    packed = _packed(contract=contract, n=n, k=k, device=device)
    spec = _spec(contract=contract, m=m, n=n, k=k, device=str(device))
    activation = torch.randn(m, k, dtype=torch.float16, device=device)

    actual = backend.sm89_w4a8_executor(activation, packed, spec=spec)
    expected = reference_w4a8_gemm(activation, packed, contract=contract, spec=spec)

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.5, rtol=2.0e-2)


@pytest.mark.skipif(not _CUDA_SM89, reason="requires an SM89 CUDA device")
@pytest.mark.skipif(
    not backend.sm89_w4a8_artifact_available(),
    reason="SM89 W4A8 artifact is not built",
)
@pytest.mark.parametrize("format_name", ("fp8_e4m3", "fp8_e5m2"))
@pytest.mark.parametrize("granularity", ("per_tensor", "per_token", "blockwise"))
def test_native_fp8_w4a8_matches_reference(format_name: str, granularity: str) -> None:
    device = torch.device("cuda:0")
    m, n, k = 16, 16, 64
    contract = W4A8Contract(
        weight_group_size=32,
        activation_dtype=format_name,
        activation_granularity=granularity,
        activation_scale_source="activation_static",
    )
    packed = _packed(contract=contract, n=n, k=k, device=device)
    spec = _spec(contract=contract, m=m, n=n, k=k, device=str(device), has_bias=True)
    activation = torch.randn(m, k, dtype=torch.float16, device=device)
    scale = calibrate_fp8_scale(
        activation,
        format_name=format_name,
        granularity=granularity,
        role="activation",
        block_k=32 if granularity == "blockwise" else None,
    )
    bias = torch.randn(n, device=device)

    actual = backend.sm89_w4a8_executor(
        activation, packed, spec=spec, activation_scales=scale, bias=bias
    )
    expected = reference_w4a8_gemm(
        activation, packed, contract=contract, spec=spec, activation_scales=scale, bias=bias
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.5, rtol=2.0e-2)


@pytest.mark.skipif(not _CUDA_SM89, reason="requires an SM89 CUDA device")
@pytest.mark.skipif(
    not backend.sm89_w4a8_artifact_available(),
    reason="SM89 W4A8 artifact is not built",
)
def test_native_w4a8_shape_gates() -> None:
    device = torch.device("cuda:0")
    contract = W4A8Contract(weight_group_size=32)
    packed = _packed(contract=contract, n=16, k=64, device=device)
    spec = _spec(contract=contract, m=16, n=16, k=64, device=str(device))

    with pytest.raises(XQTBackendError, match="M % 16 == 0"):
        backend.sm89_w4a8_executor(
            torch.randn(15, 64, dtype=torch.float16, device=device),
            packed,
            spec=_spec(contract=contract, m=15, n=16, k=64, device=str(device)),
        )


@pytest.mark.skipif(not _CUDA_SM89, reason="requires an SM89 CUDA device")
@pytest.mark.skipif(
    not backend.sm89_w4a8_artifact_available(),
    reason="SM89 W4A8 artifact is not built",
)
def test_dispatch_selects_native_w4a8_after_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    registry = default_registry()
    monkeypatch.setattr(backend, "sm89_w4a8_artifact_available", lambda _: True)
    monkeypatch.setattr(backend, "artifact_ready_for_execution", lambda *args, **kwargs: True)
    assert backend.install_sm89_w4a8_executors(registry)

    contract = W4A8Contract(weight_group_size=32)
    m, n, k = 16, 16, 64
    packed = _packed(contract=contract, n=n, k=k, device=device)
    spec = _spec(contract=contract, m=m, n=n, k=k, device=str(device))
    result = dispatch_gemm(
        torch.randn(m, k, dtype=torch.float16, device=device),
        packed,
        spec=spec,
        registry=registry,
    )

    assert result.report.selected_kernel == "sm89_w4a8_int8_cutlass"
    assert result.report.native is True
