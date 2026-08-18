"""Generate offline vLLM-style quantization launch fragments (C9).

Produces vLLM-style parameter dicts from a quant pair directory or an in-memory
compute/quant metadata mapping. SGLang, when requested, is treated as a consumer
of the same quantization style rather than a separate internal schema. Never
starts an engine.

Optional external serving-config adapter (C9); not internal gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

from xqt.core.errors import XQTArtifactError
from xqt.contracts.external import probe_external_quant_config
from xqt.contracts.quant_pair_schema import (
    DEFAULT_SIDECAR_NAME,
    QuantPairManifest,
)

ServingEngine = Literal["vllm", "sglang"]

_STRATEGY_TO_QUANTIZATION: dict[str, str] = {
    "w4a16_int4": "compressed-tensors",
    "w8a16_int8": "compressed-tensors",
    "w8a8_int8": "compressed-tensors",
    "w4a16_fp4": "compressed-tensors",
    "w4a16_nvfp4": "compressed-tensors",
    "w4a4_nvfp4": "compressed-tensors",
    "w4a4_mxfp4": "compressed-tensors",
    "kv_scale": "compressed-tensors",
}

_METHOD_TO_QUANTIZATION: dict[str, str] = {
    "awq": "awq",
    "gptq": "gptq",
    "compressed_tensors": "compressed-tensors",
    "compressed-tensors": "compressed-tensors",
    "moe_weight_only": "compressed-tensors",
    "kv_scale": "compressed-tensors",
}


def _load_quant_pair_sidecar(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if root.is_file():
        sidecar = root
    else:
        sidecar = root / DEFAULT_SIDECAR_NAME
        if not sidecar.is_file():
            # Allow HF export dir (config.json only).
            return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise XQTArtifactError(f"quant sidecar must be a JSON object: {sidecar}")
    # Prefer schema validation when it is an XQT quant pair.
    if payload.get("artifact_type") == "xqt_quant_sidecar":
        manifest = QuantPairManifest.from_dict(payload)
        return manifest.to_dict()
    return dict(payload)


def _resolve_quantization_name(
    *,
    method: str | None,
    strategy: str | None,
    probe_format: str | None,
    explicit: str | None,
) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if method is not None:
        mapped = _METHOD_TO_QUANTIZATION.get(str(method).strip().lower())
        if mapped is not None:
            return mapped
    if strategy is not None:
        mapped = _STRATEGY_TO_QUANTIZATION.get(str(strategy).strip().lower())
        if mapped is not None:
            return mapped
    if probe_format is not None:
        if probe_format == "compressed_tensors":
            return "compressed-tensors"
        if probe_format in {"awq", "gptq"}:
            return probe_format
    raise XQTArtifactError(
        "cannot derive serving --quantization from method="
        f"{method!r} strategy={strategy!r} probe_format={probe_format!r}"
    )


def _kv_cache_dtype_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    kv = metadata.get("kv_cache_quant")
    if not isinstance(kv, Mapping):
        lineage = metadata.get("kv_scale_lineage")
        if isinstance(lineage, Mapping):
            kv = lineage.get("kv_cache_quant")
    if not isinstance(kv, Mapping):
        compute = metadata.get("compute_config")
        if isinstance(compute, Mapping):
            compute_meta = compute.get("metadata")
            if isinstance(compute_meta, Mapping):
                kv = compute_meta.get("kv_cache_quant")
    if not isinstance(kv, Mapping):
        return None
    dtype = kv.get("dtype") or kv.get("kv_cache_dtype")
    if dtype is None:
        return None
    text = str(dtype).strip().lower()
    if text in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2", "auto"}:
        if text == "fp8":
            return "fp8_e4m3"
        return text
    raise XQTArtifactError(
        f"unsupported kv_cache_dtype {dtype!r}; "
        "supported: int8, fp8, fp8_e4m3, fp8_e5m2, auto"
    )


def generate_serving_config(
    quant_pair_path: str | Path | None = None,
    *,
    engine: ServingEngine = "vllm",
    method: str | None = None,
    strategy: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    quantization: str | None = None,
    kv_cache_dtype: str | None = None,
) -> dict[str, Any]:
    """Build a pure-file vLLM-style quantization config fragment.

    Returns a dict such as::

        {"engine": "vllm", "quantization": "compressed-tensors", "kv_cache_dtype": "fp8_e4m3"}

    ``engine`` is only a consumer label. The emitted quantization schema follows
    vLLM-style names; SGLang does not get a separate XQT-internal mapping.
    Unsupported combinations raise ``XQTArtifactError``.
    """

    if engine not in {"vllm", "sglang"}:
        raise XQTArtifactError(
            f"unsupported serving engine {engine!r}; supported: vllm, sglang"
        )

    resolved_method = method
    resolved_strategy = strategy
    resolved_meta: dict[str, Any] = dict(metadata or {})
    probe_format: str | None = None

    if quant_pair_path is not None:
        root = Path(quant_pair_path)
        sidecar = _load_quant_pair_sidecar(root)
        lineage = sidecar.get("lineage") if isinstance(sidecar.get("lineage"), Mapping) else {}
        side_meta = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), Mapping) else {}
        if resolved_method is None and isinstance(lineage, Mapping):
            resolved_method = lineage.get("method")  # type: ignore[assignment]
        if resolved_strategy is None and isinstance(lineage, Mapping):
            resolved_strategy = lineage.get("strategy")  # type: ignore[assignment]
        for source in (side_meta, lineage, sidecar):
            if isinstance(source, Mapping):
                resolved_meta = {**dict(source), **resolved_meta}
        compute = sidecar.get("compute_config")
        if isinstance(compute, Mapping):
            resolved_meta.setdefault("compute_config", dict(compute))
            compute_meta = compute.get("metadata")
            if isinstance(compute_meta, Mapping):
                resolved_meta = {**dict(compute_meta), **resolved_meta}
        try:
            info = probe_external_quant_config(root if root.is_dir() else root.parent)
        except XQTArtifactError:
            info = None
        if info is not None:
            probe_format = info.format

    quant_name = _resolve_quantization_name(
        method=str(resolved_method) if resolved_method is not None else None,
        strategy=str(resolved_strategy) if resolved_strategy is not None else None,
        probe_format=probe_format,
        explicit=quantization,
    )
    kv_dtype = kv_cache_dtype
    if kv_dtype is None:
        kv_dtype = _kv_cache_dtype_from_metadata(resolved_meta)

    payload: dict[str, Any] = {
        "engine": engine,
        "schema": "vllm_quantization",
        "quantization": quant_name,
    }
    if kv_dtype is not None:
        payload["kv_cache_dtype"] = kv_dtype

    # CLI fragment is intentionally shared: vLLM is the naming source of truth,
    # and SGLang remains a consumer label rather than a parallel schema.
    args = [f"--quantization {quant_name}"]
    if kv_dtype is not None:
        args.append(f"--kv-cache-dtype {kv_dtype}")
    payload["cli_args"] = args

    payload["source"] = {
        "method": resolved_method,
        "strategy": resolved_strategy,
        "probe_format": probe_format,
        "quant_pair_path": str(quant_pair_path) if quant_pair_path is not None else None,
    }
    return payload


def write_serving_config(
    output_path: str | Path,
    *,
    quant_pair_path: str | Path | None = None,
    engine: ServingEngine = "vllm",
    **kwargs: Any,
) -> Path:
    """Write ``generate_serving_config`` output as JSON."""

    payload = generate_serving_config(
        quant_pair_path,
        engine=engine,
        **kwargs,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "ServingEngine",
    "generate_serving_config",
    "write_serving_config",
]
