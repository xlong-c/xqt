"""Scenario-level readiness audit helpers for XQT."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from xqt.core.artifact import ArtifactManifest, ArtifactRecord, MetricRecord
from xqt.core.reporting import OptimizationCapability, reporting_schema_payload
from xqt.export.capability import deployment_capability_matrix
from xqt.export.tensorrt import validate_tensorrt_plugin_libraries
from xqt.operator_opt.backends.tilelang_validation import (
    TileLangFP4ValidationResult,
    validate_tilelang_packed_fp4_fused_gemm,
)
from xqt.operator_opt.capability import describe_operator_engine_capability
from xqt.prune.capability import describe_prune_runtime_capability
from xqt.quant.capability import describe_quant_backend_capability


@dataclass(frozen=True)
class XQTReadinessScenario:
    """Readiness result for one requested XQT usage scenario."""

    name: str
    status: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def distance_to_ready(self) -> dict[str, Any]:
        """Return compact counters that explain how far this scenario is from ready."""

        return {
            "ready": self.status == "ready",
            "gap_count": len(self.gaps),
            "required_action_count": len(self.required_actions),
            "evidence_count": len(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary for manifests or reports."""

        data = asdict(self)
        data["distance_to_ready"] = self.distance_to_ready
        return data


@dataclass(frozen=True)
class XQTReadinessReport:
    """Aggregated XQT readiness report."""

    overall_status: str
    scenarios: list[XQTReadinessScenario] = field(default_factory=list)
    capability_matrix: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reporting_schemas: dict[str, Any] = field(default_factory=reporting_schema_payload)

    @property
    def status_counts(self) -> dict[str, int]:
        """Return scenario counts by readiness status."""

        counts: dict[str, int] = {}
        for scenario in self.scenarios:
            counts[scenario.status] = counts.get(scenario.status, 0) + 1
        return counts

    @property
    def required_action_count(self) -> int:
        """Return the total number of required actions across scenarios."""

        return sum(len(scenario.required_actions) for scenario in self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary for manifests or reports."""

        return {
            "overall_status": self.overall_status,
            "status_counts": self.status_counts,
            "required_action_count": self.required_action_count,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "capability_matrix": self.capability_matrix,
            "reporting_schemas": self.reporting_schemas,
        }

    def to_markdown(self) -> str:
        """Return a compact Markdown readiness summary."""

        lines = [
            "# XQT Readiness Report",
            "",
            f"- Overall status: `{self.overall_status}`",
            f"- Status counts: `{self.status_counts}`",
            f"- Required actions: `{self.required_action_count}`",
            "",
            "| Scenario | Status | Gaps | Required actions | Evidence |",
            "| --- | --- | --- | --- | --- |",
        ]
        for scenario in self.scenarios:
            distance = scenario.distance_to_ready
            lines.append(
                "| "
                + " | ".join(
                    [
                        scenario.name,
                        f"`{scenario.status}`",
                        str(distance["gap_count"]),
                        str(distance["required_action_count"]),
                        str(distance["evidence_count"]),
                    ]
                )
                + " |"
            )
        if self.capability_matrix:
            lines.extend(
                [
                    "",
                    "## Capability matrix",
                    "",
                    "| Domain | Count | Statuses |",
                    "| --- | --- | --- |",
                ]
            )
            for domain, capabilities in sorted(self.capability_matrix.items()):
                statuses = sorted(
                    {
                        str(capability.get("status", "unknown"))
                        for capability in capabilities
                    }
                )
                lines.append(
                    f"| {domain} | {len(capabilities)} | `{', '.join(statuses)}` |"
                )
        for scenario in self.scenarios:
            lines.extend(
                [
                    "",
                    f"## {scenario.name}",
                    "",
                    f"Status: `{scenario.status}`",
                    "",
                    scenario.summary,
                    "",
                    "Evidence:",
                ]
            )
            lines.extend(f"- {item}" for item in scenario.evidence)
            lines.append("")
            lines.append("Gaps:")
            lines.extend(f"- {item}" for item in scenario.gaps)
            lines.append("")
            lines.append("Required actions:")
            if scenario.required_actions:
                lines.extend(f"- {item}" for item in scenario.required_actions)
            else:
                lines.append("- None")
        lines.append("")
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        """Write the readiness report to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def write_markdown(self, path: str | Path) -> Path:
        """Write the readiness report to Markdown."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def write_artifacts(
        self,
        output_dir: str | Path,
        *,
        stem: str = "xqt_readiness",
    ) -> dict[str, Path]:
        """Write JSON and Markdown readiness artifacts to a directory."""

        output_path = Path(output_dir)
        return {
            "json": self.write_json(output_path / f"{stem}.json"),
            "markdown": self.write_markdown(output_path / f"{stem}.md"),
        }

    def add_to_manifest(
        self,
        manifest: ArtifactManifest,
        *,
        artifact_paths: Mapping[str, str | Path] | None = None,
    ) -> ArtifactManifest:
        """Attach readiness metrics and optional report artifacts to a manifest."""

        manifest.add_metric(
            MetricRecord(
                name="readiness.overall_status",
                value=self.overall_status,
                passed=self.overall_status == "ready",
                metadata={
                    "status_counts": self.status_counts,
                    "required_action_count": self.required_action_count,
                    "capability_domains": sorted(self.capability_matrix),
                    "reporting_schemas": self.reporting_schemas,
                },
            )
        )
        capability_count = sum(
            len(capabilities) for capabilities in self.capability_matrix.values()
        )
        manifest.add_metric(
            MetricRecord(
                name="readiness.capability_matrix.count",
                value=capability_count,
                passed=True,
                metadata={
                    "capability_matrix": self.capability_matrix,
                    "reporting_schemas": self.reporting_schemas,
                },
            )
        )
        for scenario in self.scenarios:
            manifest.add_metric(
                MetricRecord(
                    name=f"readiness.{scenario.name}.status",
                    value=scenario.status,
                    passed=scenario.status == "ready",
                    metadata={
                        "summary": scenario.summary,
                        "distance_to_ready": scenario.distance_to_ready,
                        "gaps": list(scenario.gaps),
                        "required_actions": list(scenario.required_actions),
                    },
                )
            )
        for artifact_kind, raw_path in sorted(dict(artifact_paths or {}).items()):
            path = Path(raw_path)
            manifest.add_artifact(
                ArtifactRecord.from_file(
                    path,
                    format=str(artifact_kind),
                    runtime="xqt",
                    metadata={"kind": "readiness_report"},
                )
            )
        return manifest


def _package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None or Path("/usr/local/cuda/bin/nvcc").exists()


def _cuda_device_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


def _scaled_mm_available() -> bool:
    return callable(getattr(torch, "_scaled_mm", None))


def _overall_status(scenarios: Sequence[XQTReadinessScenario]) -> str:
    statuses = {scenario.status for scenario in scenarios}
    if "blocked" in statuses:
        return "blocked"
    if statuses == {"ready"}:
        return "ready"
    if "partial" in statuses or "not_verified" in statuses:
        return "partial"
    return "unknown"


def _capability_dict(capability: OptimizationCapability) -> dict[str, Any]:
    return capability.to_dict()


def _inference_optimization_capability_matrix() -> dict[str, list[dict[str, Any]]]:
    quantization = [
        describe_quant_backend_capability(
            "torchao",
            method=None,
            strategy="w8a8_int8",
            compute="qdq_dynamic",
        ).to_optimization_capability(),
        describe_quant_backend_capability(
            "torchao",
            method=None,
            strategy="w4a16_int4",
            compute="dequant_fp16",
        ).to_optimization_capability(),
        describe_quant_backend_capability(
            "torchao",
            method=None,
            strategy="w8a8_fp8_e4m3",
            compute="fp8_mma",
        ).to_optimization_capability(),
        describe_quant_backend_capability(
            "pytorch",
            method="awq",
            strategy="w4a16_fp4",
            compute="dequant_fp16",
            policy={"dtype": "fp4", "scheme": "weight_only"},
        ).to_optimization_capability(),
        describe_quant_backend_capability(
            "onnxruntime_qdq",
            method=None,
            strategy="w8a8_int8",
            compute="qdq_static",
        ).to_optimization_capability(),
        describe_quant_backend_capability(
            "bitsandbytes",
            method=None,
            strategy="w4a16_int4",
            compute="dequant_fp16",
        ).to_optimization_capability(),
    ]
    operator = [
        describe_operator_engine_capability("torch_compile").to_optimization_capability(),
        describe_operator_engine_capability("tilelang").to_optimization_capability(),
        describe_operator_engine_capability("cute_dsl").to_optimization_capability(),
    ]
    pruning = [
        OptimizationCapability(
            kind="pruning",
            name="structured",
            engine="pytorch_rewrite",
            status="available",
            maturity="executable",
            runtime="pytorch",
            artifact_kind="pytorch_model",
            available=True,
            supported=True,
            methods=("structured",),
            target_module_types=("Conv2d", "Linear", "BatchNorm2d"),
            notes=("Structured pruning can rewrite selected PyTorch module dimensions.",),
            limitations=(
                "Speedup is model, shape, and backend dependent and must be benchmarked.",
            ),
            metadata={
                "granularities": [
                    "channel",
                    "filter",
                    "mlp_neuron",
                    "attention_head",
                    "block",
                    "token",
                ],
                "speedup_verified": False,
            },
        ),
        describe_prune_runtime_capability(
            method="nm_structured",
            device="cuda",
            pattern=(2, 4),
        ).to_optimization_capability(),
        describe_prune_runtime_capability(
            method="block_sparse",
            device=None,
            block_shape=(4, 4),
        ).to_optimization_capability(),
    ]
    export = [
        capability.to_optimization_capability()
        for capability in deployment_capability_matrix(implemented_only=False)
        if capability.format in {"onnx", "tensorrt", "openvino"}
    ]
    runtime_features = [
        OptimizationCapability(
            kind="runtime_feature",
            name="llm_runtime_metadata",
            engine="adapter_only",
            status="partial",
            maturity="metadata_only",
            runtime="external_serving",
            artifact_kind="metadata",
            available=False,
            supported=True,
            notes=(
                "XQT records canonical runtime feature metadata (prefix_cache, "
                "paged_kv, kv_cache_quant, chunked_prefill, speculative_decode, "
                "continuous_batching) and a reference KV-scale attention entity; "
                "it does not implement serving schedulers.",
            ),
            limitations=(
                "Paged KV, prefix cache, speculative decode, and continuous "
                "batching require an external runtime; CUDA kernel verification "
                "is pending.",
            ),
            metadata=reporting_schema_payload()["runtime_features"],
        )
    ]
    return {
        "quantization": [_capability_dict(capability) for capability in quantization],
        "pruning": [_capability_dict(capability) for capability in pruning],
        "operator": [_capability_dict(capability) for capability in operator],
        "export": [_capability_dict(capability) for capability in export],
        "runtime_features": [
            _capability_dict(capability) for capability in runtime_features
        ],
    }


def _fp4_tilelang_readiness(
    *,
    run_tilelang_probe: bool,
    tilelang_compile_only: bool,
    tilelang_target_arch: str | None,
    tilelang_warmup: int,
    tilelang_iterations: int,
) -> XQTReadinessScenario:
    quant_capability = describe_quant_backend_capability(
        "pytorch",
        method="awq",
        strategy="w4a16_fp4",
        compute="dequant_fp16",
        policy={"dtype": "fp4", "scheme": "weight_only"},
    )
    operator_capability = describe_operator_engine_capability("tilelang")
    sm = _cuda_device_capability()
    checks: dict[str, Any] = {
        "quantization": quant_capability.to_dict(),
        "operator_backend": operator_capability.to_dict(),
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device_capability": (
            None if sm is None else {"major": sm[0], "minor": sm[1], "sm": f"sm_{sm[0]}{sm[1]}"}
        ),
        "tilelang_package_available": _package_available("tilelang"),
        "nvcc_available": _nvcc_available(),
    }
    evidence = [
        "pytorch + w4a16_fp4 is classified as PSEUDO weight-only quantization",
        "FP4WeightOnlyLinear stores packed signed int4 codes plus per-group scale",
        "TileLang registry includes packed FP4 fused unpack/dequant/GEMM/bias/activation entry",
    ]
    gaps = [
        "full AWQ/GPTQ execution is still not complete",
        "runtime performance requires CUDA hardware validation",
    ]
    required_actions = [
        "capture TileLang packed FP4 runtime correctness and latency on the target CUDA GPU",
        "complete full AWQ/GPTQ execution if the production path must use those algorithms rather than the current fp4_weight_only path",
    ]
    validation: TileLangFP4ValidationResult | None = None
    if run_tilelang_probe:
        validation = validate_tilelang_packed_fp4_fused_gemm(
            target_arch=tilelang_target_arch,
            compile_only=tilelang_compile_only,
            warmup=tilelang_warmup,
            iterations=tilelang_iterations,
        )
        checks["tilelang_probe"] = validation.to_dict()
        if validation.status == "ok" and validation.compile_only:
            evidence.append("TileLang packed FP4 fused kernel compile-only probe passed")
            required_actions.append(
                "rerun assess_xqt_readiness(run_tilelang_probe=True, tilelang_compile_only=False) on the target CUDA GPU"
            )
        elif validation.status == "ok":
            evidence.append("TileLang packed FP4 fused kernel runtime correctness probe passed")
            required_actions = [
                action
                for action in required_actions
                if "runtime correctness and latency" not in action
            ]
        else:
            gaps.append(f"TileLang packed FP4 probe did not pass: {validation.reason}")
            required_actions.append("fix the TileLang packed FP4 probe failure before claiming runtime readiness")
    else:
        checks["tilelang_probe"] = {"status": "not_requested"}
        gaps.append("TileLang packed FP4 probe was not requested")
        required_actions.append(
            "run assess_xqt_readiness(run_tilelang_probe=True) to collect compile-only or CUDA runtime evidence"
        )

    if validation is not None and validation.status == "ok" and not validation.compile_only:
        status = "ready"
        summary = "FP4 packed storage and TileLang fused kernel path have runtime validation evidence."
    elif quant_capability.status == "available" and operator_capability.status == "available":
        status = "partial"
        summary = "FP4 quantization and TileLang fused-kernel plumbing exist, but runtime evidence is incomplete."
    else:
        status = "blocked"
        summary = "FP4 + TileLang path is missing required quantization or operator capability."

    return XQTReadinessScenario(
        name="fp4_tilelang_megakernel",
        status=status,
        summary=summary,
        evidence=evidence,
        gaps=gaps,
        required_actions=required_actions,
        checks=checks,
    )


def _tensorrt_plugin_readiness(
    *,
    tensorrt_plugin_libraries: Sequence[str | Path] | None,
    validate_tensorrt_plugin_loadability: bool,
    trtexec_path: str,
) -> XQTReadinessScenario:
    trtexec_resolved = shutil.which(trtexec_path)
    tensorrt_python_available = _package_available("tensorrt")
    evidence = [
        "TensorRT adapter supports plugin_libraries for trtexec and python_api paths",
        "preflight can check plugin path presence and optional ctypes loadability",
    ]
    gaps = [
        "TensorRT plugin ABI correctness needs a real TensorRT engine build or runtime load",
        "target deployment environment still needs end-to-end validation",
    ]
    required_actions = [
        "provide target TensorRT plugin .so paths through tensorrt_plugin_libraries",
        "build or inspect the TensorRT engine in the target deployment environment",
        "run runtime benchmark or inference with the plugin-loaded engine",
    ]

    validation = validate_tensorrt_plugin_libraries(
        tensorrt_plugin_libraries,
        validate_loadability=validate_tensorrt_plugin_loadability,
    )
    plugin_checks = [
        check.to_dict() for check in validation.plugin_libraries
    ]
    plugin_status = (
        "not_provided"
        if validation.status == "not_requested"
        else validation.status
    )

    if plugin_checks:
        if plugin_status in {"present", "ok"}:
            evidence.append("configured TensorRT plugin libraries exist on disk")
            required_actions = [
                action
                for action in required_actions
                if "provide target TensorRT plugin .so paths" not in action
            ]
        if plugin_status == "missing":
            gaps.append("one or more configured TensorRT plugin libraries are missing")
            required_actions.append("build or copy the missing TensorRT plugin libraries before export")
        if plugin_status == "ok":
            evidence.append("configured TensorRT plugin libraries loaded with ctypes RTLD_GLOBAL")
            required_actions = [
                action
                for action in required_actions
                if "ctypes loadability" not in action
            ]
        elif plugin_status == "load_failed":
            gaps.append("one or more configured TensorRT plugin libraries failed ctypes loading")
            required_actions.append("fix plugin shared-library dependencies until ctypes RTLD_GLOBAL load succeeds")
        elif plugin_status == "present":
            required_actions.append(
                "rerun with validate_tensorrt_plugin_loadability=True to check ctypes loadability"
            )
    else:
        gaps.append("no TensorRT plugin library path was provided for this readiness audit")

    checks = {
        "trtexec_path": trtexec_path,
        "trtexec_resolved": trtexec_resolved,
        "tensorrt_python_available": tensorrt_python_available,
        "plugin_status": plugin_status,
        "plugin_validation": validation.to_dict(),
        "plugin_libraries": plugin_checks,
    }
    if plugin_status in {"missing", "load_failed"}:
        status = "blocked"
        summary = "TensorRT plugin path or loadability checks failed."
    elif trtexec_resolved is not None or tensorrt_python_available:
        status = "partial"
        summary = "TensorRT plugin plumbing is present, but ABI and engine validation are still required."
    else:
        status = "partial"
        summary = "TensorRT plugin configuration support exists, but local TensorRT runtime is not available."
        gaps.append("local trtexec or tensorrt Python package is not available")
        required_actions.append("install trtexec or the tensorrt Python package in the target validation environment")

    return XQTReadinessScenario(
        name="tensorrt_so_plugin",
        status=status,
        summary=summary,
        evidence=evidence,
        gaps=gaps,
        required_actions=required_actions,
        checks=checks,
    )


def _analysis_readiness() -> XQTReadinessScenario:
    recipe_path = Path("xqt/recipes/analysis/quant_layer_statistics_workflow.yaml")
    checks = {
        "layer_statistics": "supported",
        "activation_drift": "supported",
        "weight_statistics": "supported",
        "prune_candidates": "supported",
        "analysis_recipe_exists": recipe_path.is_file(),
    }
    evidence = [
        "analyze stage can emit layer error, activation drift, importance, prune candidates, and layer_statistics",
        "layer statistics include output and weight distribution summaries",
        "FP4 FP4WeightOnlyLinear weights are materialized through dequantize_weight for analysis",
    ]
    gaps = [
        "task-level accuracy, mAP, perplexity, or business metrics remain external to XQT",
    ]
    return XQTReadinessScenario(
        name="prune_quant_error_analysis",
        status="ready",
        summary="Model-side pruning and quantization error analysis are available for XQT workflows.",
        evidence=evidence,
        gaps=gaps,
        checks=checks,
    )


def _fp8_mma_readiness() -> XQTReadinessScenario:
    """W8A8 FP8 path: hardware + torch._scaled_mm dependency (model-side only)."""

    quant_capability = describe_quant_backend_capability(
        "torchao",
        method=None,
        strategy="w8a8_fp8_e4m3",
        compute="fp8_mma",
    )
    sm = _cuda_device_capability()
    sm_ok = sm is not None and sm >= (8, 9)
    scaled_mm = _scaled_mm_available()
    cuda_ok = torch.cuda.is_available()
    checks: dict[str, Any] = {
        "quantization": quant_capability.to_dict(),
        "torch_cuda_available": cuda_ok,
        "cuda_device_capability": (
            None if sm is None else {"major": sm[0], "minor": sm[1], "sm": f"sm_{sm[0]}{sm[1]}"}
        ),
        "min_sm_for_fp8_mma": {"major": 8, "minor": 9},
        "sm_meets_fp8_threshold": sm_ok,
        "torch_scaled_mm_available": scaled_mm,
        "fp8_dtype_available": hasattr(torch, "float8_e4m3fn"),
    }
    evidence = [
        "Fp8MmaLinear uses torch._scaled_mm with tensorwise FP8 e4m3 on Ada sm_89+",
        "activation_scale_mode defaults to static (calibrated) to avoid per-forward amax cost",
        "small-M decode shapes may fall back to dense fp16 (min_fp8_rows)",
    ]
    gaps: list[str] = []
    required_actions: list[str] = []
    if not cuda_ok:
        gaps.append("CUDA is not available in this environment")
        required_actions.append("run readiness on a CUDA host to assess FP8 hardware")
    elif not sm_ok:
        gaps.append(
            f"device SM {sm} is below Ada sm_89 required for consumer FP8 MMA path"
            if sm is not None
            else "CUDA device capability could not be read"
        )
        required_actions.append("use sm_89+ GPU or keep FP8 as planned/reference only")
    if not scaled_mm:
        gaps.append("torch._scaled_mm is not available in this PyTorch build")
        required_actions.append("upgrade PyTorch to a build that exposes torch._scaled_mm")
    if not hasattr(torch, "float8_e4m3fn"):
        gaps.append("torch.float8_e4m3fn dtype is missing")
        required_actions.append("use a PyTorch build with float8 dtypes")

    if cuda_ok and sm_ok and scaled_mm and hasattr(torch, "float8_e4m3fn"):
        status = "partial"
        summary = (
            "FP8 MMA hardware and torch._scaled_mm are present; "
            "end-to-end latency/correctness still needs model-level validation."
        )
        required_actions.append(
            "benchmark Fp8MmaLinear vs fp16 on target shapes before claiming speedup"
        )
    elif not gaps:
        status = "partial"
        summary = "FP8 path is advertised but hardware or runtime dependencies are incomplete."
    else:
        status = "blocked" if (not cuda_ok or not scaled_mm) else "partial"
        summary = (
            "FP8 MMA path is blocked or incomplete for this host "
            "(missing CUDA, SM, or torch._scaled_mm)."
        )

    return XQTReadinessScenario(
        name="fp8_mma_hardware",
        status=status,
        summary=summary,
        evidence=evidence,
        gaps=gaps,
        required_actions=required_actions,
        checks=checks,
    )


def _kv_cache_quant_readiness() -> XQTReadinessScenario:
    """Model-side KV scale path, independent of weight-quant stages (T6 / GUIDE)."""

    from xqt.quant.registry import RouteQuery, resolve_quant_route
    import xqt.quant.quantizers  # noqa: F401  # register routes

    route = resolve_quant_route(
        RouteQuery(
            backend="pytorch",
            method="kv_scale",
            strategy="kv_scale",
            compute="dequant_fp16",
        )
    )
    route_available = route is not None and route.maturity == "executable"
    checks = {
        "kv_scale_route_name": None if route is None else route.name,
        "kv_scale_route_maturity": None if route is None else route.maturity,
        "runtime_quant_contract_kv_field": "kv_cache_dtype",
        "field_convention": ".attn.k_scale / .attn.v_scale",
        "cache_runtime_owned_by_xqt": False,
    }
    evidence = [
        "kv_scale quantizer produces per-tensor K/V scale artifacts on attention modules",
        "RuntimeQuantContract carries optional kv_cache_dtype without weight-quant coupling",
        "XQT does not implement paged KV, block pool, or cache eviction",
    ]
    gaps = [
        "serving engines own real KV storage dtype and attention kernels",
        "offline cos-threshold calibration still needs representative prompts per model family",
    ]
    required_actions = [
        "attach RuntimeQuantContract.kv_cache_dtype when exporting handoff metadata",
        "run offline cos similarity vs FP16 attention when claiming KV scale quality",
    ]
    if route_available:
        status = "partial"
        summary = (
            "Model-side KV scale artifacts are available; "
            "cache management stays outside XQT."
        )
        evidence.append(f"quant route {route.name!r} is registered as executable")
    else:
        status = "blocked"
        summary = "KV scale quant route is not registered as executable."
        gaps.append("register or repair the kv_scale quant route")
    return XQTReadinessScenario(
        name="kv_cache_quant",
        status=status,
        summary=summary,
        evidence=evidence,
        gaps=gaps,
        required_actions=required_actions,
        checks=checks,
    )


def _runtime_feature_metadata_readiness() -> XQTReadinessScenario:
    """Canonical runtime feature metadata + reference KV attention entity."""

    from xqt.contracts.runtime_features import runtime_feature_specs

    specs = runtime_feature_specs()
    checks = {
        "canonical_feature_count": len(specs),
        "canonical_features": [spec["name"] for spec in specs],
        "model_side_or_external": all(
            spec["scope"] == "model_side_metadata" or spec["owner"] == "external_runtime"
            for spec in specs
        ),
        "xqt_serving_engine": False,
        "cache_management_in_xqt": False,
    }
    evidence = [
        "RuntimeFeatureMetadata schema declares scope and XQT status per feature",
        "speculative_decode records draft/target model, acceptance rate, backend",
        "prefix_cache / paged_kv record switch, cache block size and hit rate only",
        "KvScaleAttention reference entity consumes KvScaleArtifact + RuntimeQuantContract",
        "report explains supported / unsupported / unverified reasons per feature",
    ]
    gaps = [
        "CUDA fused KV attention kernel verification pending (no CUDA in current environment)",
        "real serving engine startup is an optional milestone, not an XQT gate",
    ]
    required_actions = [
        "attach runtime_features metadata to quant pair / manifest when emitting handoff artifacts",
        "run KV-scale entity numerical verification on CUDA hardware when available",
    ]
    return XQTReadinessScenario(
        name="runtime_feature_metadata",
        status="partial",
        summary=(
            "Runtime feature metadata schema and the reference KV-scale "
            "attention entity are landed; CUDA kernel verification is pending."
        ),
        evidence=evidence,
        gaps=gaps,
        required_actions=required_actions,
        checks=checks,
    )


def assess_xqt_readiness(
    *,
    run_tilelang_probe: bool = False,
    tilelang_compile_only: bool = True,
    tilelang_target_arch: str | None = "sm_80",
    tilelang_warmup: int = 1,
    tilelang_iterations: int = 2,
    tensorrt_plugin_libraries: Sequence[str | Path] | None = None,
    validate_tensorrt_plugin_loadability: bool = False,
    trtexec_path: str = "trtexec",
) -> XQTReadinessReport:
    """Assess XQT readiness for the currently requested optimization scenarios."""

    scenarios = [
        _fp4_tilelang_readiness(
            run_tilelang_probe=run_tilelang_probe,
            tilelang_compile_only=tilelang_compile_only,
            tilelang_target_arch=tilelang_target_arch,
            tilelang_warmup=tilelang_warmup,
            tilelang_iterations=tilelang_iterations,
        ),
        _fp8_mma_readiness(),
        _tensorrt_plugin_readiness(
            tensorrt_plugin_libraries=tensorrt_plugin_libraries,
            validate_tensorrt_plugin_loadability=validate_tensorrt_plugin_loadability,
            trtexec_path=trtexec_path,
        ),
        _analysis_readiness(),
        _kv_cache_quant_readiness(),
        _runtime_feature_metadata_readiness(),
    ]
    return XQTReadinessReport(
        overall_status=_overall_status(scenarios),
        scenarios=scenarios,
        capability_matrix=_inference_optimization_capability_matrix(),
    )


__all__ = [
    "XQTReadinessReport",
    "XQTReadinessScenario",
    "assess_xqt_readiness",
]
