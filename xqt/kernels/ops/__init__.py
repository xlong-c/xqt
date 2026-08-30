"""Public operator groups for the xqt.kernels namespace."""
from importlib import import_module as _import_module

from xqt.kernels.deprecation import suppress_legacy_warnings as _suppress_legacy_warnings

_GROUPS = (
    "activation",
    "layernorm",
    "attention",
    "gemm",
    "quantization",
    "kvcache",
    "moe",
    "mamba",
    "diffusion",
    "sampling",
    "communication",
    "memory",
    "speculative",
    "norm",
    "elementwise",
    "embeddings",
    "grammar",
    "lplb",
    "kv_canary",
)

for _group in _GROUPS:
    _import_module(f"{__name__}.{_group}")

# Phase 2 bridge: inventory legacy registries (metadata only, real targets).
with _suppress_legacy_warnings():
    _import_module(f"{__name__}._legacy")

del _import_module, _group, _suppress_legacy_warnings

__all__ = list(_GROUPS)
