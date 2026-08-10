"""Tests for runtime feature metadata and the KV-scale attention entity."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xqt.benchmark.phase_latency import benchmark_prefill_decode
from xqt.contracts.runtime_features import (
    RUNTIME_FEATURES_KEY,
    RuntimeFeatureMetadata,
    build_runtime_feature_metadata,
    describe_runtime_feature,
    prefix_paged_kv_metadata,
    runtime_feature_report,
    runtime_feature_specs,
    speculative_decode_metadata,
)
from xqt.contracts.runtime_manifest import RuntimeManifest, build_runtime_manifest
from xqt.contracts.runtime_quant import (
    RuntimeQuantContract,
    build_runtime_quant_contract,
)
from xqt.core.errors import XQTConfigError
from xqt.runtime.modules import KvCacheMetadata, KvScaleAttention


def test_runtime_feature_specs_cover_canonical_features() -> None:
    specs = {item["name"]: item for item in runtime_feature_specs()}
    assert set(specs) == {
        "prefix_cache",
        "paged_kv",
        "kv_cache_quant",
        "chunked_prefill",
        "speculative_decode",
        "continuous_batching",
    }
    assert specs["prefix_cache"]["scope"] == "model_side_metadata"
    assert specs["paged_kv"]["scope"] == "runtime_capability"
    assert specs["kv_cache_quant"]["xqt_status"] == "reference_entity"
    assert specs["continuous_batching"]["owner"] == "external_runtime"
    assert describe_runtime_feature("PAGED_KV")["name"] == "paged_kv"
    with pytest.raises(ValueError, match="Unknown runtime feature"):
        describe_runtime_feature("not_a_feature")


def test_default_metadata_is_honest_about_xqt_ownership() -> None:
    metadata = build_runtime_feature_metadata()
    report = metadata.report()
    features = report["features"]

    assert features["kv_cache_quant"]["status"] == "unverified"
    assert "pending" in " ".join(features["kv_cache_quant"]["notes"])
    assert features["paged_kv"]["status"] == "metadata_only"
    assert "external runtime" in features["paged_kv"]["reason"]
    assert features["prefix_cache"]["status"] == "metadata_only"
    assert "external runtime" in features["prefix_cache"]["reason"]
    assert features["continuous_batching"]["status"] == "not_implemented"


def test_speculative_decode_records_relations_only() -> None:
    metadata = speculative_decode_metadata(
        draft_model="llama-68m-draft",
        target_model="llama-8b",
        backend="vllm",
        acceptance_rate=0.72,
    )
    entry = metadata.by_name("speculative_decode")
    assert entry is not None
    assert entry.enabled is True
    assert entry.status == "unverified"
    assert entry.provider == "vllm"
    assert entry.metrics["draft_model"] == "llama-68m-draft"
    assert entry.metrics["target_model"] == "llama-8b"
    assert entry.metrics["acceptance_rate"] == pytest.approx(0.72)

    with pytest.raises(XQTConfigError, match="acceptance_rate"):
        speculative_decode_metadata(
            draft_model="draft",
            target_model="target",
            backend="vllm",
            acceptance_rate=1.5,
        )
    with pytest.raises(XQTConfigError, match="draft_model"):
        speculative_decode_metadata(
            draft_model="",
            target_model="target",
            backend="vllm",
        )


def test_prefix_and_paged_kv_record_metrics_without_cache_management() -> None:
    metadata = prefix_paged_kv_metadata(
        prefix_enabled=True,
        paged_enabled=True,
        cache_block_size=16,
        hit_rate=0.42,
        backend="vllm",
    )
    prefix = metadata.by_name("prefix_cache")
    paged = metadata.by_name("paged_kv")
    assert prefix is not None and paged is not None
    assert prefix.enabled is True
    assert prefix.metrics["cache_block_size"] == 16
    assert prefix.metrics["hit_rate"] == pytest.approx(0.42)
    assert prefix.provider == "vllm"
    assert paged.enabled is True
    assert "serving engine" in " ".join(paged.notes)

    with pytest.raises(XQTConfigError, match="cache_block_size"):
        prefix_paged_kv_metadata(cache_block_size=-1)
    with pytest.raises(XQTConfigError, match="hit_rate"):
        prefix_paged_kv_metadata(hit_rate=1.2)


def test_runtime_feature_metadata_roundtrip_and_validation() -> None:
    metadata = speculative_decode_metadata(
        draft_model="draft",
        target_model="target",
        backend="vllm",
    )
    restored = RuntimeFeatureMetadata.from_dict(metadata.to_dict())
    assert restored.by_name("speculative_decode") == metadata.by_name(
        "speculative_decode"
    )

    with pytest.raises(XQTConfigError, match="Unknown runtime feature"):
        RuntimeFeatureMetadata.from_dict(
            {
                "entries": [
                    {
                        "name": "not_a_feature",
                        "enabled": False,
                        "status": "metadata_only",
                    }
                ]
            }
        )
    with pytest.raises(XQTConfigError, match="status"):
        RuntimeFeatureMetadata.from_dict(
            {
                "entries": [
                    {
                        "name": "paged_kv",
                        "enabled": False,
                        "status": "magical",
                    }
                ]
            }
        )


def test_runtime_manifest_roundtrip_preserves_runtime_features() -> None:
    metadata = prefix_paged_kv_metadata(
        prefix_enabled=True,
        cache_block_size=32,
    )
    manifest = RuntimeManifest(runtime_features=metadata)
    restored = RuntimeManifest.from_dict(manifest.to_dict())
    assert restored.runtime_features is not None
    assert restored.runtime_features.by_name("prefix_cache") is not None
    assert (
        restored.runtime_features.by_name("prefix_cache").metrics["cache_block_size"]
        == 32
    )

    from_metadata = build_runtime_manifest({RUNTIME_FEATURES_KEY: metadata.to_dict()})
    assert from_metadata.runtime_features is not None
    assert from_metadata.runtime_features.by_name("paged_kv") is not None


def test_kv_cache_metadata_from_artifacts_extracts_layer_scales() -> None:
    artifacts = {
        "mode": "per_tensor_scale",
        "dtype": "int8",
        "field_convention": "vllm.attn.k_scale",
        "layers": {
            "model.layers.0.self_attn": {
                "layer_path": "model.layers.0.self_attn",
                "k_scale": 0.05,
                "v_scale": 0.06,
                "attn.k_scale": 0.05,
                "attn.v_scale": 0.06,
            },
            "model.layers.1.self_attn": {
                "layer_path": "model.layers.1.self_attn",
                "k_scale": 0.07,
                "v_scale": 0.08,
                "attn.k_scale": 0.07,
                "attn.v_scale": 0.08,
            },
        },
    }
    metadata = KvCacheMetadata.from_artifacts(artifacts)
    assert metadata.dtype == "int8"
    assert len(metadata.layer_scales) == 2
    assert metadata.scales_for("model.layers.1.self_attn") == pytest.approx(
        (0.07, 0.08)
    )
    assert metadata.scales_for("missing.layer") is None


def _kv_contract() -> RuntimeQuantContract:
    return build_runtime_quant_contract(
        quant_spec={
            "weight_dtype": "int8",
            "weight_granularity": "per_tensor",
            "activation_mode": "none",
        },
        storage_layout="kv_scale",
        required_kernels=("torch_sdpa_kv_scale_reference",),
        global_shape=(8, 8),
        kv_cache_dtype="int8",
    )


def _float_reference_attention(
    entity: KvScaleAttention,
    x: torch.Tensor,
) -> torch.Tensor:
    batch, seq, _ = x.shape

    def reshape(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.reshape(batch, seq, entity.heads, entity.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    q_projection, k_projection, v_projection = entity._split_qkv(entity.qkv(x))
    q = reshape(q_projection)
    k = reshape(k_projection)
    v = reshape(v_projection)
    attn = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=entity.causal,
    )
    merged = attn.permute(0, 2, 1, 3).contiguous().reshape(batch, seq, entity.inner_dim)
    return entity.out_proj(merged)


def test_kv_scale_attention_reference_entity_runs_and_reports() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8) * 0.1
    probe = KvScaleAttention(8, heads=2, k_scale=1.0, v_scale=1.0)
    with torch.no_grad():
        _, k_probe, v_probe = probe._split_qkv(probe.qkv(x))
    k_scale = float(k_probe.abs().max().item()) / 127.0
    v_scale = float(v_probe.abs().max().item()) / 127.0
    entity = KvScaleAttention(
        8,
        heads=2,
        k_scale=k_scale,
        v_scale=v_scale,
        layer_path="model.layers.0.self_attn",
        contract=_kv_contract(),
    )

    output = entity(x)
    assert output.shape == (2, 5, 8)
    reference = _float_reference_attention(entity, x)
    # Int8 per-tensor K/V quantization has an inherent rounding error that is
    # amplified by softmax; keep the tolerance at reference-entity scale.
    assert torch.allclose(output, reference, rtol=0.1, atol=1e-2)

    report = entity.report()
    assert report["entity"] == "kv_scale_attention_reference"
    assert report["selected_kernel"] == "torch_sdpa_kv_scale_reference"
    assert report["contract_ok"] is True
    assert report["fallback_reason"] is None
    assert report["kv_cache_metadata"]["dtype"] == "int8"
    assert report["kv_cache_metadata"]["layer_scales"][0]["layer_path"] == (
        "model.layers.0.self_attn"
    )

    # The same entity without a contract reports an honest not-run reason.
    entity_no_contract = KvScaleAttention(
        8,
        heads=2,
        k_scale=10.0,
        v_scale=10.0,
    )
    report_no_contract = entity_no_contract.report()
    assert report_no_contract["contract_ok"] is False
    assert "runtime_quant_contract_absent" in report_no_contract["contract_errors"]
    assert report_no_contract["fallback_reason"] == "runtime_quant_contract_absent"


def test_kv_scale_attention_from_artifacts_and_offline_phases() -> None:
    artifacts = {
        "mode": "per_tensor_scale",
        "dtype": "int8",
        "layers": {
            "model.layers.0.self_attn": {
                "layer_path": "model.layers.0.self_attn",
                "k_scale": 10.0,
                "v_scale": 10.0,
            }
        },
    }
    entity = KvScaleAttention.from_artifacts(
        artifacts,
        dim=8,
        heads=2,
        layer_path="model.layers.0.self_attn",
        contract=_kv_contract(),
    )
    x = torch.randn(2, 4, 8) * 0.1
    assert entity(x).shape == (2, 4, 8)
    assert entity.kv_cache_metadata().dtype == "int8"

    phase = benchmark_prefill_decode(
        lambda: entity.prefill(x),
        lambda: entity.decode(x[:, :1]),
        warmup=0,
        iterations=2,
        sync_cuda=False,
    )
    assert phase.offline_estimate is True
    assert phase.prefill.mean_ms >= 0.0
    assert phase.decode.mean_ms >= 0.0


def test_readiness_includes_runtime_feature_metadata_scenario() -> None:
    from xqt.readiness import assess_xqt_readiness

    report = assess_xqt_readiness()
    names = {scenario.name for scenario in report.scenarios}
    assert "runtime_feature_metadata" in names
    scenario = next(
        item for item in report.scenarios if item.name == "runtime_feature_metadata"
    )
    assert scenario.status == "partial"
    assert scenario.checks["xqt_serving_engine"] is False
    assert scenario.checks["cache_management_in_xqt"] is False
    assert any("CUDA" in gap for gap in scenario.gaps)
