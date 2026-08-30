"""Shared lazy extension compilation primitives."""

from .cache import build_directory, compile_cache_key, metadata_path
from .cpp_args import extension_load_kwargs
from .loader import load_extension, load_jit
from .paths import (
    artifact_metadata_path,
    cache_dir,
    csrc_path,
    csrc_root,
    default_cache_dir,
    include_root,
    jit_root,
)
from .spec import CompileSpec, ExtensionSpec, serialize_compile_settings
from .toolchain import ToolchainInfo, detect_toolchain

__all__ = [
    "CompileSpec",
    "ExtensionSpec",
    "ToolchainInfo",
    "artifact_metadata_path",
    "build_directory",
    "cache_dir",
    "compile_cache_key",
    "csrc_path",
    "csrc_root",
    "default_cache_dir",
    "detect_toolchain",
    "extension_load_kwargs",
    "include_root",
    "jit_root",
    "load_extension",
    "load_jit",
    "metadata_path",
    "serialize_compile_settings",
]
