from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    PackedWeight,
    QuantSpec,
    build_packed_weight,
    default_registry,
    dispatch_gemm,
    reference_w8a16_gemm,
)
from xqt.gemm.backends import w8a16_sm89 as backend


def _quant(dtype: str = "fp16") -> QuantSpec:
    return QuantSpec(
        weight_dtype="int8",
        activation_dtype=dtype,
        output_dtype=dtype,
        weight_granularity="per_channel",
        activation_granularity="per_tensor",
        weight_scale_source="weight_offline",
        storage_layout="xqt_int8_nk_v1",
        pack_version="xqt-int8-v1",
    )


def _packed(
    *, n: int = 4, k: int = 8, device: torch.device | str = "cpu"
) -> PackedWeight:
    spec = _quant()
    qweight = torch.arange(n * k, dtype=torch.int8, device=device).reshape(n, k)
    scales = torch.ones(n, dtype=torch.float32, device=device)
    return build_packed_weight(
        qweight,
        logical_shape=(n, k),
        spec=spec,
        scales=scales,
        padded_k=k,
        storage_layout="xqt_int8_nk_v1",
        pack_version="xqt-int8-v1",
    )


def _spec(*, m: int = 2, n: int = 4, k: int = 8, dtype: str = "fp16") -> GemmSpec:
    quant = _quant(dtype)
    return GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, phase="decode", sm=89, device="cpu"),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype=dtype),
    )


def test_cpu_dispatch_uses_w8a16_reference_contract() -> None:
    packed = _packed()
    spec = _spec()
    activation = torch.ones((2, 8), dtype=torch.float16)
    result = dispatch_gemm(activation, packed, spec=spec, registry=default_registry())
    expected = reference_w8a16_gemm(activation, packed, spec=spec)

    assert result.report.selected_kernel == "w8a16_packed_reference"
    assert result.report.native is False
    torch.testing.assert_close(result.output, expected)


def test_w8a16_executor_rejects_cpu_contract() -> None:
    with pytest.raises(XQTBackendError, match="CUDA"):
        backend.sm89_w8a16_executor(
            torch.ones((2, 8), dtype=torch.float16),
            _packed(),
            spec=_spec(),
            artifact=Path("missing.so"),
        )


def test_missing_w8a16_artifact_does_not_promote_registry(tmp_path: Path) -> None:
    artifact = tmp_path / "missing-w8a16.so"
    registry = default_registry()

    assert backend.sm89_w8a16_artifact_available(artifact) is False
    assert backend.install_sm89_w8a16_executor(registry, artifact=artifact) is False
    assert registry.get("sm89_w8a16_cutlass").maturity == "metadata_only"


def test_w8a16_install_requires_correctness_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = default_registry()
    artifact = tmp_path / "w8a16.so"
    monkeypatch.setattr(backend, "sm89_w8a16_artifact_available", lambda _: True)
    monkeypatch.setattr(backend, "artifact_ready_for_execution", lambda *args, **kwargs: False)

    assert backend.install_sm89_w8a16_executor(registry, artifact=artifact) is False
    assert registry.get("sm89_w8a16_cutlass").maturity == "metadata_only"


def test_w8a16_install_promotes_after_correctness_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = default_registry()
    artifact = tmp_path / "w8a16.so"
    monkeypatch.setattr(backend, "sm89_w8a16_artifact_available", lambda _: True)
    monkeypatch.setattr(backend, "artifact_ready_for_execution", lambda *args, **kwargs: True)

    assert backend.install_sm89_w8a16_executor(registry, artifact=artifact) is True
    entry = registry.get("sm89_w8a16_cutlass")
    assert entry.maturity == "executable"
    assert entry.implementation == "cutlass_sm89_w8a16_dynamic_int8"
    assert entry.executor is not None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9),
    reason="requires an SM89 CUDA device",
)
def test_m1_nonaligned_k_is_explicit_reference_fallback() -> None:
    device = torch.device("cuda:0")
    n, k = 129, 1001
    quant = _quant()
    packed = build_packed_weight(
        torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device),
        logical_shape=(n, k),
        spec=quant,
        scales=torch.ones(n, dtype=torch.float32, device=device),
        padded_k=k,
        storage_layout="xqt_int8_nk_v1",
        pack_version="xqt-int8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=1, n=n, k=k, phase="decode", sm=89, device=str(device)),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )

    with pytest.raises(XQTBackendError, match="K divisible by four"):
        backend.sm89_w8a16_executor(
            torch.randn((1, k), dtype=torch.float16, device=device),
            packed,
            spec=spec,
            artifact=Path.home() / ".cache/xqt/gemm/sm89/w8a16_sm89.so",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9),
    reason="requires an SM89 CUDA device",
)
def test_small_m_w8a16_is_explicit_reference_fallback() -> None:
    device = torch.device("cuda:0")
    n, k, m = 64, 128, 4
    quant = _quant()
    packed = build_packed_weight(
        torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device),
        logical_shape=(n, k),
        spec=quant,
        scales=torch.ones(n, dtype=torch.float32, device=device),
        padded_k=k,
        storage_layout="xqt_int8_nk_v1",
        pack_version="xqt-int8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, phase="decode", sm=89, device=str(device)),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )

    with pytest.raises(XQTBackendError, match="M=2..31"):
        backend.sm89_w8a16_executor(
            torch.randn((m, k), dtype=torch.float16, device=device),
            packed,
            spec=spec,
            artifact=Path.home() / ".cache/xqt/gemm/sm89/w8a16_sm89.so",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9),
    reason="requires an SM89 CUDA device",
)
def test_native_fp16_w8a16_matches_reference_with_quantization_tolerance() -> None:
    device = torch.device("cuda:0")
    n, k, m = 64, 128, 32
    quant = _quant()
    packed = build_packed_weight(
        torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device),
        logical_shape=(n, k),
        spec=quant,
        scales=torch.rand(n, dtype=torch.float32, device=device) * 0.004 + 0.001,
        padded_k=k,
        storage_layout="xqt_int8_nk_v1",
        pack_version="xqt-int8-v1",
    )
    spec = GemmSpec(
        problem=GemmProblem(m=m, n=n, k=k, phase="prefill", sm=89, device=str(device)),
        quant=quant,
        epilogue=EpilogueSpec(output_dtype="fp16"),
    )
    activation = torch.randn((m, k), dtype=torch.float16, device=device)
    actual = backend.sm89_w8a16_executor(
        activation,
        packed,
        spec=spec,
        artifact=Path.home() / ".cache/xqt/gemm/sm89/w8a16_sm89.so",
    )
    expected = reference_w8a16_gemm(activation, packed, spec=spec)

    assert actual.shape == (m, n)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.5, rtol=2.0e-2)
