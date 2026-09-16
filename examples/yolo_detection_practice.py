"""Single-file XQT detection practice entry.

The default path uses XQT's smoke detection module, explicit example inputs,
and the Pythonic ``XQTOptimizationSession`` API. It intentionally does not import
ultralytics and does not embed a full JSON-shaped workflow dict in Python code.

Set XQT_YOLO_PRACTICE_CONFIG to a YAML workflow path when you want to run a
different workflow, for example:

    XQT_YOLO_PRACTICE_CONFIG=xqt/recipes/detection/hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_eval.yaml \
        python examples/yolo_detection_practice.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xqt import XQTOptimizationSession
from xqt.kernels.nn.fixtures import build_smoke_detection_module
from xqt.workflows import OptimizedModelResult, optimize_model

CONFIG_ENV = "XQT_YOLO_PRACTICE_CONFIG"


def _repo_root() -> Path:
    return _REPO_ROOT


def _artifact_path(*parts: str) -> str:
    return str(_repo_root().joinpath("artifacts", "xqt", "detection", *parts))


def _default_example_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    return {"images": torch.rand(1, 3, 64, 64, generator=generator)}


def _build_default_session() -> XQTOptimizationSession:
    artifact_dir = _artifact_path("yolo_practice_example")
    model = build_smoke_detection_module(
        num_classes=3,
        boxes_per_image=2,
        input_channels=3,
    )
    example_inputs = _default_example_inputs()
    return XQTOptimizationSession(
        project={
            "name": "yolo_detection_practice_example",
            "artifact_dir": artifact_dir,
        },
        model=model,
        model_config={
            "params": {
                "num_classes": 3,
                "boxes_per_image": 2,
                "input_channels": 3,
            },
            "dtype": "float32",
            "device": "cpu",
        },
        task={
            "type": "detection",
            "class_names": ["class_0", "class_1", "class_2"],
            "detection_postprocess": {
                "format": "yolo_raw",
                "box_format": "xyxy",
                "score_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 100,
                "score_activation": "sigmoid",
                "has_objectness": False,
                "class_agnostic_nms": False,
            },
            "params": {"imgsz": 64},
        },
        example_inputs=example_inputs,
        calibration_inputs=[example_inputs],
    )


def _run_default_stages(session: XQTOptimizationSession) -> None:
    session.benchmark(
        name="baseline_latency",
        warmup=1,
        iterations=2,
    )
    session.export(
        name="export_fp32_onnx",
        format="onnx",
        output_path=_artifact_path(
            "yolo_practice_example",
            "export_fp32_onnx",
            "smoke_detection_fp32.onnx",
        ),
        opset=18,
        target_params={
            "input_names": ["images"],
            "output_names": ["predictions"],
            "dynamo": False,
            "runtime_diff": False,
        },
    )
    session.quant(
        name="quant_qdq",
        save_model=False,
        backend="onnxruntime_qdq",
        strategy="static_int8",
        policy={
            "source_name": "smoke_detection_source.onnx",
            "output_path": _artifact_path(
                "yolo_practice_example",
                "quant_qdq",
                "smoke_detection_qdq.onnx",
            ),
            "input_names": ["images"],
            "output_names": ["predictions"],
            "dynamo": False,
            "sample_limit": 2,
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "op_types_to_quantize": ["Conv"],
            "extra_options": {"ActivationSymmetric": False},
        },
    )
    session.prune(
        name="prune_sparse",
        from_stage="initial",
        method="global_l1_unstructured",
        target_sparsity=0.2,
    )
    session.benchmark(
        name="prune_latency",
        compare_to="baseline_latency",
        warmup=1,
        iterations=2,
    )
    session.prune(
        name="structured_prune_guard",
        from_stage="initial",
        save_model=False,
        method="structured",
        granularity="channel",
        target_sparsity=0.5,
        accept={"min_speedup": 1.01},
    )


def _run_default_session() -> OptimizedModelResult:
    session = _build_default_session()
    _run_default_stages(session)
    return session.result()


def _configured_workflow_path() -> str | None:
    value = os.environ.get(CONFIG_ENV)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return str(path)


def _get_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _format_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return "-"


def _stage_metric_summary(metrics: Mapping[str, Any]) -> str:
    if "quant" in metrics:
        report = metrics["quant"]
        return (
            f"backend={report.get('backend', '-')}, "
            f"artifact={report.get('artifact', report.get('output_path', '-'))}"
        )
    if "prune" in metrics:
        report = metrics["prune"]
        state = report.get("execution_state", "applied")
        return (
            f"method={report.get('method', '-')}, "
            f"state={state}, sparsity={_format_float(report.get('sparsity'))}, "
            f"speedup_claimed={report.get('speedup_claimed', '-')}"
        )
    if "benchmark" in metrics:
        report = metrics["benchmark"]
        return f"latency_ms={_format_float(report.get('mean_ms'))}"
    if "export" in metrics:
        artifacts = metrics["export"].get("artifacts", [])
        return f"artifacts={len(artifacts)}"
    return "-"


def _print_result(result: OptimizedModelResult) -> None:
    print(f"project: {result.context.config.project.name}")
    print(f"baseline_stage: {result.baseline_stage or '-'}")
    print(f"best_stage: {result.best_stage or '-'}")
    print("stages:")
    for stage in result.stages:
        status = "accepted" if stage.accepted else "rejected"
        summary = _stage_metric_summary(stage.metrics)
        print(f"  - {stage.name} [{stage.kind}] {status}: {summary}")

    workflow_manifest = result.context.artifacts.get("workflow_manifest")
    if workflow_manifest is not None:
        print(f"workflow_manifest: {workflow_manifest}")
    workflow_result = result.context.artifacts.get("workflow_result")
    if workflow_result is not None:
        print(f"workflow_result: {workflow_result}")


def main() -> int:
    workflow_path = _configured_workflow_path()
    if workflow_path is not None:
        print(f"workflow_config: {workflow_path}")
        result = optimize_model(workflow_path)
    else:
        print("workflow_config: pythonic interactive detection practice")
        result = _run_default_session()
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
