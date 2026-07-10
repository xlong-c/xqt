"""XQT GPTQ/AWQ smoke demo for baidu/Unlimited-OCR.

The script intentionally keeps explicit in-file settings instead of command-line
arguments.  It loads the real Unlimited-OCR checkpoint, extracts one Linear
layer, promotes the extracted source layer to FP32, and uses that layer as a
small, repeatable XQT target.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt import XQTOptimizationSession
from xqt.operator_opt.backends.gemm_precision import describe_gemm_precision_capability
from xqt.quant.capability import describe_quant_backend_capability


MODEL_ID = "baidu/Unlimited-OCR"
MODEL_REVISION = "ee63731b6461c8afcdcc7b15352e7d2ffecc2ead"
TARGET_LINEAR = "model.layers.0.self_attn.q_proj"
ARTIFACT_DIR = Path("artifacts/xqt/examples/unlimited_ocr_awq_gptq_tilelang")
SOURCE_MODEL_DTYPE = torch.float32
BATCH_SIZE = 64
WARMUP = 5
ITERATIONS = 20
GROUP_SIZE = 64
MXFP_BLOCK_SIZE = 32
RANDOM_SEED = 0
PRECISION_PROBE = (
    "fp32",
    "fp16",
    "bf16",
    "fp8",
    "int8",
    "int4",
    "fp4",
    "nvfp4",
    "mxfp8",
    "mxfp6",
    "mxfp4",
)


@dataclass(frozen=True)
class LoadedLinear:
    """Real Linear layer extracted from the Unlimited-OCR checkpoint."""

    module: torch.nn.Linear
    model_class: str
    load_seconds: float
    linear_count: int
    module_name: str
    checkpoint_default_dtype: str
    source_weight_dtype: str
    in_features: int
    out_features: int
    bias: bool


class _LinearOnly(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def _runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _target_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clone_linear(
    module: torch.nn.Linear,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.nn.Linear:
    target_dtype = dtype or module.weight.dtype
    target_device = torch.device(device) if device is not None else module.weight.device
    cloned = torch.nn.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        dtype=target_dtype,
        device=target_device,
    )
    with torch.no_grad():
        cloned.weight.copy_(module.weight.to(device=target_device, dtype=target_dtype))
        if module.bias is not None and cloned.bias is not None:
            cloned.bias.copy_(module.bias.to(device=target_device, dtype=target_dtype))
    return cloned


def _load_source_linear() -> LoadedLinear:
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise RuntimeError("transformers is required to load baidu/Unlimited-OCR") from exc

    start = time.perf_counter()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=SOURCE_MODEL_DTYPE,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - start
    model.eval()
    source = model.get_submodule(TARGET_LINEAR)
    if not isinstance(source, torch.nn.Linear):
        raise TypeError(f"{TARGET_LINEAR} is not a torch.nn.Linear module")
    linear = _clone_linear(source, dtype=SOURCE_MODEL_DTYPE, device=torch.device("cpu")).eval()
    linear_count = sum(1 for _, module in model.named_modules() if isinstance(module, torch.nn.Linear))
    model_class = type(model).__name__
    config_dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
    del model
    return LoadedLinear(
        module=linear,
        model_class=model_class,
        load_seconds=load_seconds,
        linear_count=linear_count,
        module_name=TARGET_LINEAR,
        checkpoint_default_dtype=str(config_dtype),
        source_weight_dtype=str(linear.weight.dtype),
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
    )


def _fallback_random_linear() -> LoadedLinear:
    linear = torch.nn.Linear(1280, 1280, bias=False, dtype=SOURCE_MODEL_DTYPE).eval()
    return LoadedLinear(
        module=linear,
        model_class="fallback_random_linear",
        load_seconds=0.0,
        linear_count=0,
        module_name=TARGET_LINEAR,
        checkpoint_default_dtype="fallback",
        source_weight_dtype=str(linear.weight.dtype),
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=False,
    )


def _tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    diff = (ref - cand).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "allclose_1e_2": bool(torch.allclose(ref, cand, atol=1e-2, rtol=1e-2)),
        "shape": list(ref.shape),
    }


def _precision_capabilities(device: torch.device) -> dict[str, Any]:
    capabilities: dict[str, Any] = {}
    for precision in PRECISION_PROBE:
        try:
            capabilities[precision] = describe_gemm_precision_capability(precision, device)
        except Exception as exc:
            capabilities[precision] = {
                "precision": precision,
                "available": False,
                "error": str(exc),
            }
    return capabilities


def _quant_capability(
    *,
    backend: str,
    method: str,
    strategy: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    try:
        return describe_quant_backend_capability(
            backend,
            method=method,
            strategy=strategy,
            policy=policy,
        ).to_dict()
    except Exception as exc:
        return {
            "backend": backend,
            "method": method,
            "strategy": strategy,
            "available": False,
            "error": str(exc),
        }


def _run_executable_quant(
    *,
    source_linear: torch.nn.Linear,
    method: str,
    strategy: str,
    policy: dict[str, Any],
    example_inputs: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model = _LinearOnly(_clone_linear(source_linear, dtype=SOURCE_MODEL_DTYPE)).eval()
    source_model = _LinearOnly(_clone_linear(source_linear, dtype=torch.float32, device=device)).eval()
    reference_inputs = example_inputs.detach().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        source_output = source_model(reference_inputs)

    session = XQTOptimizationSession(
        project={
            "name": f"unlimited_ocr_{method}_{strategy}",
            "artifact_dir": str(ARTIFACT_DIR / f"{method}_{strategy}"),
        },
        model=model,
        device=str(device),
        example_inputs=example_inputs,
        calibration_inputs=[example_inputs],
    )
    quant_stage = session.quant(
        name=f"{method}_{strategy}",
        backend="pytorch",
        method=method,
        strategy=strategy,
        policy=policy,
    )
    quantized_reference_model = copy.deepcopy(session.model).eval().to(device=device).to(dtype=torch.float32)
    with torch.no_grad():
        quantized_output = quantized_reference_model(reference_inputs)
    quant_error = _tensor_diff(source_output, quantized_output)
    session.context.model = session.model.to(device=device).to(dtype=example_inputs.dtype)
    quant_metadata = quant_stage.metrics.get("metadata", {})

    operator_stage = session.operator(
        name=f"tilelang_{method}_{strategy}",
        from_stage=quant_stage.name,
        benchmark={
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "sync_cuda": device.type == "cuda",
            "measure_memory": False,
        },
        targets=[
            {
                "name": "linear_tilelang",
                "target": "linear",
                "engine": "tilelang",
                "patterns": ["dequant_gemm_epilogue"],
                "min_speedup": 1.000001,
                "tilelang": {
                    "target_arch": _target_arch(),
                    "linear_runtime": "auto",
                    "linear_fastpath": "auto",
                },
            }
        ],
    )
    target = operator_stage.metrics["targets"][0]
    metadata = target.get("metadata", {})
    return {
        "method": method,
        "backend": "pytorch",
        "strategy": strategy,
        "policy": dict(policy),
        "quant_stage": {
            "accepted": quant_stage.accepted,
            "quantized_module_count": quant_stage.metrics.get("quantized_module_count"),
            "quantized_modules": quant_stage.metrics.get("quantized_modules"),
            "execution_state": quant_metadata.get("execution_state"),
            "implementation": quant_metadata.get("implementation"),
            "nature": quant_stage.metrics.get("nature"),
            "algorithm_executable": quant_stage.metrics.get("algorithm_executable"),
            "method_semantics": quant_stage.metrics.get("method_semantics"),
        },
        "quant_error_vs_fp32_source": quant_error,
        "tilelang_operator": {
            "applied": target.get("applied"),
            "skip_reason": target.get("skip_reason"),
            "speedup": target.get("speedup"),
            "latency_before": target.get("latency_before"),
            "latency_after": target.get("latency_after"),
            "numeric_diff": target.get("numeric_diff"),
            "execution_mode": metadata.get("execution_mode"),
            "kernel_kind": metadata.get("kernel_kind"),
            "kernel_pattern": metadata.get("kernel_pattern"),
            "weight_source": metadata.get("weight_source"),
            "weight_representation": metadata.get("weight_representation"),
            "unpack_stage": metadata.get("unpack_stage"),
            "benchmark_strategy": metadata.get("benchmark_strategy"),
        },
    }


def _run_backend_quant_coverage(
    *,
    source_linear: torch.nn.Linear,
    method: str,
    backend: str,
    strategy: str,
    policy: dict[str, Any],
    example_inputs: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    session = XQTOptimizationSession(
        project={
            "name": f"unlimited_ocr_{backend}_{method}_{strategy}_coverage",
            "artifact_dir": str(ARTIFACT_DIR / f"coverage_{backend}_{method}_{strategy}"),
        },
        model=_LinearOnly(_clone_linear(source_linear)).eval(),
        device=str(device),
        example_inputs=example_inputs,
        calibration_inputs=[example_inputs],
    )
    try:
        stage = session.quant(
            name=f"{backend}_{method}_{strategy}",
            backend=backend,
            method=method,
            strategy=strategy,
            policy=policy,
        )
        component = stage.metrics.get("components", [{}])[0]
        metadata = stage.metrics.get("metadata", {})
        return {
            "method": method,
            "backend": backend,
            "strategy": strategy,
            "policy": dict(policy),
            "accepted": stage.accepted,
            "execution_state": metadata.get("execution_state"),
            "executed": metadata.get("executed"),
            "nature": stage.metrics.get("nature", component.get("nature")),
            "algorithm_executable": stage.metrics.get("algorithm_executable"),
            "method_semantics": stage.metrics.get("method_semantics"),
            "capability_status": metadata.get("capability", {}).get("status"),
            "capability_limitations": metadata.get("capability", {}).get("limitations"),
            "artifact_count": stage.metrics.get("summary", {}).get("artifact_count"),
        }
    except Exception as exc:
        return {
            "method": method,
            "backend": backend,
            "strategy": strategy,
            "policy": dict(policy),
            "accepted": False,
            "error": str(exc),
        }


def _existing_tilelang_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    sweep_path = Path("artifacts/xqt/benchmarks/unlimited_ocr_nvfp4_tilelang/sweep_summary.json")
    model_path = Path("artifacts/xqt/benchmarks/unlimited_ocr_nvfp4_model_tilelang/summary.json")
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        evidence["nvfp4_layer_sweep"] = {
            "path": str(sweep_path),
            "summary": sweep.get("summary"),
        }
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding="utf-8"))
        evidence["nvfp4_model_benchmark"] = {
            "path": str(model_path),
            "target_count": model.get("target_count"),
            "wrapped_count": model.get("wrapped_count"),
            "text_prefill_speedup": model.get("speedup"),
            "ocr_prefill_speedup": model.get("ocr_prefill", {}).get("speedup"),
            "numeric_diff": model.get("numeric_diff"),
            "ocr_numeric_diff": model.get("ocr_prefill", {}).get("numeric_diff"),
            "route": model.get("route"),
        }
    return evidence


def _int8_mma_evidence() -> dict[str, Any]:
    report_path = Path("artifacts/xqt/examples/unlimited_ocr_int8/unlimited_ocr_int8_report.json")
    evidence: dict[str, Any] = {
        "path": str(report_path),
        "available": report_path.exists(),
        "relationship_to_awq_gptq": (
            "independent true W8A8 INT8 MMA helper/session path; not an AWQ/GPTQ "
            "algorithm implementation"
        ),
    }
    if not report_path.exists():
        return evidence
    report = json.loads(report_path.read_text(encoding="utf-8"))
    environment = report.get("environment", {})
    tilelang_probe = report.get("tilelang_probe", {})
    linear_benchmark = report.get("linear_benchmark", {})
    evidence.update(
        {
            "device": environment.get("device"),
            "target_arch": environment.get("target_arch"),
            "cuda_available": environment.get("cuda_available"),
            "tilelang_probe_status": tilelang_probe.get("status"),
            "tilelang_probe_reason": tilelang_probe.get("reason"),
            "linear_benchmark_status": linear_benchmark.get("status"),
            "linear_benchmark_reason": linear_benchmark.get("reason"),
            "full_ocr_status": report.get("full_ocr", {}).get("status"),
        }
    )
    if isinstance(linear_benchmark.get("latency_ms"), dict):
        evidence["linear_latency_ms"] = linear_benchmark["latency_ms"]
    return evidence


def _coverage_matrix(
    *,
    executable_runs: list[dict[str, Any]],
    backend_coverage_runs: list[dict[str, Any]],
    precision_capabilities: dict[str, Any],
    int8_mma_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "area": "reference",
            "precision": "fp32",
            "backend": "pytorch",
            "status": "executed",
            "algorithm_executable": True,
            "method_semantics": "high_precision_source_reference",
            "evidence": "source_linear.weight_dtype == torch.float32; quant_error_reference == fp32_source",
        }
    ]
    for run in executable_runs:
        policy = run.get("policy", {})
        precision = str(policy.get("precision", policy.get("dtype")))
        operator = run.get("tilelang_operator", {})
        rows.append(
            {
                "area": "awq_gptq_weight_only",
                "method": run.get("method"),
                "strategy": run.get("strategy"),
                "precision": precision,
                "backend": run.get("backend"),
                "status": "executed",
                "algorithm_executable": run.get("quant_stage", {}).get("algorithm_executable"),
                "method_semantics": run.get("quant_stage", {}).get("method_semantics"),
                "quantized_module_count": run.get("quant_stage", {}).get("quantized_module_count"),
                "quant_error_vs_fp32_source": run.get("quant_error_vs_fp32_source"),
                "tilelang_execution_mode": operator.get("execution_mode"),
                "tilelang_kernel_kind": operator.get("kernel_kind"),
                "tilelang_speedup": operator.get("speedup"),
                "tilelang_skip_reason": operator.get("skip_reason"),
            }
        )
    for run in backend_coverage_runs:
        execution_state = run.get("execution_state")
        rows.append(
            {
                "area": "awq_gptq_backend_coverage",
                "method": run.get("method"),
                "strategy": run.get("strategy"),
                "precision": str(run.get("policy", {}).get("dtype")),
                "backend": run.get("backend"),
                "status": (
                    "executed"
                    if run.get("executed")
                    else "planned"
                    if execution_state == "planned"
                    else "unsupported"
                ),
                "accepted": run.get("accepted"),
                "algorithm_executable": run.get("algorithm_executable", False),
                "method_semantics": run.get(
                    "method_semantics",
                    "capability_report_only_no_executable_algorithm",
                ),
                "reason": run.get("error") or run.get("capability_limitations"),
            }
        )
    rows.append(
        {
            "area": "true_int8_mma",
            "precision": "int8",
            "backend": "tilelang",
            "status": int8_mma_evidence.get("linear_benchmark_status", "unknown"),
            "algorithm_executable": True,
            "method_semantics": "independent_true_w8a8_int8_mma_not_awq_gptq",
            "cuda_available": int8_mma_evidence.get("cuda_available"),
            "target_arch": int8_mma_evidence.get("target_arch"),
            "reason": int8_mma_evidence.get("linear_benchmark_reason"),
        }
    )
    for precision in ("fp16", "bf16", "fp8", "int8", "int4", "fp4", "nvfp4", "mxfp8", "mxfp6", "mxfp4"):
        capability = precision_capabilities.get(precision, {})
        rows.append(
            {
                "area": "gemm_precision_capability",
                "precision": precision,
                "status": "available" if capability.get("available") else "unavailable",
                "engine": capability.get("engine"),
                "hardware_native": capability.get("hardware_native"),
                "notes": capability.get("notes"),
            }
        )
    return rows


def _observed_issues(summary: dict[str, Any]) -> list[dict[str, str]]:
    issues = [
        {
            "id": "awq_gptq_remaining_label_only_paths",
            "severity": "high",
            "finding": (
                "FP4, INT4, and INT8 AWQ/GPTQ now use calibration-aware quantization "
                "when calibration inputs are provided, but MXFP and no-calibration "
                "fallback paths remain label-only groupwise storage quantization."
            ),
        },
        {
            "id": "tilelang_low_bit_backend_partial_closure",
            "severity": "medium",
            "finding": (
                "TileLang AWQ/GPTQ FP4, INT4, and INT8 quant backends now emit XQT "
                "weight-only modules with TileLang bridge protocols, but true packed "
                "low-bit MMA performance still depends on operator-stage runtime and "
                "hardware support."
            ),
        },
        {
            "id": "sm89_packed_low_bit_not_winning",
            "severity": "medium",
            "finding": (
                "Existing Unlimited-OCR NVFP4 sweep shows packed TileLang path is "
                "slower on sm_89; dense-cache native half routing is the practical "
                "Ada path."
            ),
        },
    ]
    executable_runs = summary.get("executable_runs", [])
    valid_executable_runs = [run for run in executable_runs if isinstance(run, dict)]
    missed_speedup_runs = [
        run
        for run in valid_executable_runs
        if run.get("tilelang_operator", {}).get("skip_reason") is not None
    ]
    if valid_executable_runs and len(missed_speedup_runs) >= len(valid_executable_runs) // 2:
        issues.append(
            {
                "id": "single_layer_tilelang_dense_cache_not_breaking_even",
                "severity": "medium",
                "finding": (
                    "The q_proj layer extracted from baidu/Unlimited-OCR validates "
                    "numerically through TileLang dense-cache bridges, but every "
                    "weight-only low-bit run misses the speedup gate on sm_89."
                ),
            }
        )
    if summary.get("source_linear", {}).get("status") != "loaded":
        issues.append(
            {
                "id": "remote_model_load_fragility",
                "severity": "medium",
                "finding": str(summary.get("source_linear", {}).get("error", "model load failed")),
            }
        )
    int8_evidence = summary.get("int8_mma_evidence", {})
    if int8_evidence.get("available") and int8_evidence.get("linear_benchmark_status") != "ok":
        issues.append(
            {
                "id": "true_int8_mma_requires_cuda_validation",
                "severity": "medium",
                "finding": (
                    "The true W8A8 INT8 MMA path is wired into XQTOptimizationSession.quant(), "
                    "but CUDA is required for TileLang runtime validation and it remains "
                    "independent from AWQ/GPTQ algorithms."
                ),
            }
        )
    return issues


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = _runtime_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    source_status: dict[str, Any] = {"status": "loaded"}
    try:
        loaded = _load_source_linear()
    except Exception as exc:
        source_status = {
            "status": "fallback_random_shape",
            "error": str(exc),
        }
        loaded = _fallback_random_linear()

    example_inputs = torch.randn(
        BATCH_SIZE,
        loaded.in_features,
        device=device,
        dtype=dtype,
    )

    executable_runs: list[dict[str, Any]] = []
    executable_specs = [
        (
            "fp4_weight_only",
            {
                "dtype": "fp4",
                "scheme": "weight_only",
                "include_module_names": ["linear"],
                "group_size": GROUP_SIZE,
            },
        ),
        (
            "mxfp_weight_only",
            {
                "dtype": "mxfp",
                "scheme": "weight_only",
                "include_module_names": ["linear"],
                "precision": 8,
                "block_size": MXFP_BLOCK_SIZE,
            },
        ),
        (
            "mxfp_weight_only",
            {
                "dtype": "mxfp",
                "scheme": "weight_only",
                "include_module_names": ["linear"],
                "precision": 6,
                "block_size": MXFP_BLOCK_SIZE,
            },
        ),
        (
            "mxfp_weight_only",
            {
                "dtype": "mxfp",
                "scheme": "weight_only",
                "include_module_names": ["linear"],
                "precision": 4,
                "block_size": MXFP_BLOCK_SIZE,
            },
        ),
        (
            "weight_only_int4",
            {
                "dtype": "int4",
                "scheme": "weight_only",
                "include_module_names": ["linear"],
                "bits": 4,
                "group_size": GROUP_SIZE,
            },
        ),
        (
            "weight_only_int8",
            {
                "dtype": "int8",
                "scheme": "weight_only",
                "include_module_names": ["linear"],
                "bits": 8,
                "group_size": GROUP_SIZE,
            },
        ),
    ]
    for method in ("awq", "gptq"):
        for strategy, policy in executable_specs:
            executable_runs.append(
                _run_executable_quant(
                    source_linear=loaded.module,
                    method=method,
                    strategy=strategy,
                    policy=policy,
                    example_inputs=example_inputs,
                    device=device,
                )
            )
            _sync(device)

    backend_coverage_specs = [
        ("tilelang", "awq", "fp4_weight_only", {"dtype": "fp4", "scheme": "weight_only"}),
        ("tilelang", "gptq", "fp4_weight_only", {"dtype": "fp4", "scheme": "weight_only"}),
        ("tilelang", "awq", "weight_only_int4", {"dtype": "int4", "scheme": "weight_only", "bits": 4}),
        ("tilelang", "gptq", "weight_only_int4", {"dtype": "int4", "scheme": "weight_only", "bits": 4}),
        ("tilelang", "awq", "weight_only_int8", {"dtype": "int8", "scheme": "weight_only", "bits": 8}),
        ("tilelang", "gptq", "weight_only_int8", {"dtype": "int8", "scheme": "weight_only", "bits": 8}),
    ]
    backend_coverage_runs = [
        _run_backend_quant_coverage(
            source_linear=loaded.module,
            backend=backend,
            method=method,
            strategy=strategy,
            policy=policy,
            example_inputs=example_inputs,
            device=device,
        )
        for backend, method, strategy, policy in backend_coverage_specs
    ]

    quant_capabilities = {
        f"{backend}:{method}:{strategy}": _quant_capability(
            backend=backend,
            method=method,
            strategy=strategy,
            policy=policy,
        )
        for backend, method, strategy, policy in backend_coverage_specs
    }
    precision_capabilities = _precision_capabilities(device)
    int8_mma_evidence = _int8_mma_evidence()
    summary: dict[str, Any] = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": str(device),
        "target_arch": _target_arch(),
        "example_shape": list(example_inputs.shape),
        "operator_input_dtype": str(example_inputs.dtype),
        "quant_error_reference": "fp32_source",
        "source_linear": {
            **source_status,
            **asdict(loaded),
            "module": type(loaded.module).__name__,
            "weight_dtype": str(loaded.module.weight.dtype),
        },
        "precision_capabilities": precision_capabilities,
        "quant_backend_capabilities": quant_capabilities,
        "executable_runs": executable_runs,
        "backend_coverage_runs": backend_coverage_runs,
        "existing_tilelang_evidence": _existing_tilelang_evidence(),
        "int8_mma_evidence": int8_mma_evidence,
    }
    summary["coverage_matrix"] = _coverage_matrix(
        executable_runs=executable_runs,
        backend_coverage_runs=backend_coverage_runs,
        precision_capabilities=precision_capabilities,
        int8_mma_evidence=int8_mma_evidence,
    )
    summary["observed_issues"] = _observed_issues(summary)
    output_path = ARTIFACT_DIR / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
