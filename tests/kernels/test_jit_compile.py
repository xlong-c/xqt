"""CPU-only contracts for the unified JIT compilation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from xqt.kernels.jit.utils.arch import make_jit_cuda_arch, normalize_cuda_arch
from xqt.kernels.jit.utils.compile import (
    CompileSpec,
    build_directory,
    compile_cache_key,
    csrc_path,
    extension_load_kwargs,
    serialize_compile_settings,
)


def test_arch_normalization_accepts_common_spellings() -> None:
    assert normalize_cuda_arch("sm_89") == "sm_89"
    assert normalize_cuda_arch("8.9") == "sm_89"
    assert normalize_cuda_arch((8, 9)) == "sm_89"
    assert make_jit_cuda_arch("sm_120") == "12.0"
    assert normalize_cuda_arch("8.9+PTX") == "sm_89"


def test_arch_environment_accepts_torch_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    from xqt.kernels.jit.utils.arch import get_jit_cuda_arch

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9;9.0+PTX")
    monkeypatch.delenv("XQT_JIT_CUDA_ARCH", raising=False)
    assert get_jit_cuda_arch() == "sm_89"


def test_migrated_sources_are_grouped() -> None:
    assert csrc_path("gemm", "dense_sm89.cu").is_file()
    assert csrc_path("quantization", "int8mma_kernel.cu").is_file()
    with pytest.raises(FileNotFoundError):
        csrc_path("gemm", "int8mma_kernel.cu")


def test_compile_spec_cache_and_load_kwargs_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("// smoke\n", encoding="utf-8")
    spec = CompileSpec(
        name="smoke_ext",
        sources=(source,),
        include_dirs=(tmp_path,),
        target_arch="sm_89",
        cache_dir=tmp_path / "cache",
        cuda_flags=("-O3",),
    )
    original_key = compile_cache_key(spec)
    assert original_key == compile_cache_key(spec)
    assert build_directory(spec).parent == tmp_path / "cache"
    kwargs = extension_load_kwargs(spec)
    assert kwargs["sources"] == [str(source.resolve())]
    assert "-gencode=arch=compute_89,code=sm_89" in kwargs["extra_cuda_cflags"]

    source.write_text("// changed\n", encoding="utf-8")
    assert compile_cache_key(spec) != original_key

    env_spec = CompileSpec(
        name="smoke_ext",
        sources=(source,),
        env={"XQT_TEST_DEFINE": "1"},
    )
    env_spec_changed = CompileSpec(
        name="smoke_ext",
        sources=(source,),
        env={"XQT_TEST_DEFINE": "2"},
    )
    assert compile_cache_key(env_spec) != compile_cache_key(env_spec_changed)


def test_compile_settings_are_json_safe() -> None:
    result = serialize_compile_settings(
        cache_dir=Path("cache"),
        tile_shape=(128, 128, 64),
        pass_configs={"threads": 128, "include": Path("headers")},
    )
    assert result == {
        "cache_dir": "cache",
        "tile_shape": [128, 128, 64],
        "pass_configs": {"threads": 128, "include": "headers"},
    }
