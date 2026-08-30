"""flashinfer S6 backend contract."""

from __future__ import annotations

import pytest

from xqt.core.base.errors import XQTBackendError
import xqt.kernels.ops  # noqa: F401  # trigger group registration

from xqt.kernels import KernelBackend, select_kernel
from xqt.kernels.selector import clear_cache


def test_flashinfer_spec_registered() -> None:
    spec = select_kernel("gemm.bmm_fp8", KernelBackend.FLASHINFER)
    assert spec.backend == KernelBackend.FLASHINFER
    assert spec.op == "gemm.bmm_fp8"


def test_flashinfer_load_raises_xqt_backend_error_when_not_installed() -> None:
    spec = select_kernel("gemm.bmm_fp8", KernelBackend.FLASHINFER)
    with pytest.raises(XQTBackendError, match="not available"):
        spec.load()


def test_flashinfer_get_kernel_raises_xqt_backend_error() -> None:
    from xqt.kernels import get_kernel

    clear_cache()
    with pytest.raises(XQTBackendError):
        get_kernel("gemm.bmm_fp8", KernelBackend.FLASHINFER)
    clear_cache()
