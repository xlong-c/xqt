"""Shared runtime types for XQT passes."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .artifact import ArtifactManifest
from .schema import (
    AnalysisConfig,
    BenchmarkConfig,
    ExportTargetConfig,
    OperatorOptimizationConfig,
    OutputDiffConfig,
    PruneConfig,
    QuantConfig,
)


@dataclass
class XQTContext:
    """Mutable context passed between XQT pipeline passes."""

    model: Any = None
    reference_model: Any = None
    example_inputs: Any = None
    calibration_inputs: Any = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    device: str = ""
    artifact_dir: str = ""
    project_name: str = ""
    task_type: str = ""
    compression_axes: list[str] | None = None
    model_target: str | None = None
    model_params: Dict[str, Any] | None = None
    quant_config: QuantConfig | None = None
    prune_config: PruneConfig | None = None
    analysis_config: AnalysisConfig | None = None
    benchmark_config: BenchmarkConfig | None = None
    operator_config: OperatorOptimizationConfig | None = None
    output_diff_config: OutputDiffConfig | None = None
    export_targets: list[ExportTargetConfig] | None = None
    manifest: Optional[ArtifactManifest] = None

    def __post_init__(self) -> None:
        if self.compression_axes is None:
            self.compression_axes = []
        if self.model_params is None:
            self.model_params = {}
        if self.export_targets is None:
            self.export_targets = []

    def require_model(self) -> Any:
        """Return the current model or raise a clear error."""

        if self.model is None:
            raise ValueError("XQTContext.model is required")
        return self.model


__all__ = ["XQTContext"]
