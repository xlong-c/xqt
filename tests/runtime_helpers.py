from __future__ import annotations

import copy
from typing import Any

from omegaconf import OmegaConf

from xqt.core.schema import BenchmarkConfig, OperatorOptimizationConfig
from xqt.core.types import XQTContext


def operator_config_from_dict(config_dict: dict[str, Any]) -> OperatorOptimizationConfig:
    merged = OmegaConf.merge(
        OmegaConf.structured(OperatorOptimizationConfig),
        config_dict["operator_optimization"],
    )
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def operator_runtime_context(
    config_dict: dict[str, Any],
    *,
    model: Any = None,
    example_inputs: Any = None,
) -> XQTContext:
    model_config = config_dict["model"]
    project = config_dict["project"]
    return XQTContext(
        model=model,
        reference_model=copy.deepcopy(model) if model is not None else None,
        example_inputs=example_inputs,
        device=str(model_config["device"]),
        artifact_dir=str(project["artifact_dir"]),
        project_name=str(project["name"]),
        task_type="classification",
        model_target=str(model_config["target"]),
        model_params=dict(model_config.get("params", {})),
        operator_config=operator_config_from_dict(config_dict),
        benchmark_config=BenchmarkConfig(**config_dict["benchmark"]),
    )


__all__ = ["operator_config_from_dict", "operator_runtime_context"]
