"""ViT + torchao FP8 experiment built on XQT helpers.

This module intentionally stays a thin experiment entrypoint. Reusable
quantization, sensitivity analysis and latency benchmarking live in
``xqt.quant`` and ``xqt.benchmark``.
"""

from __future__ import annotations

import copy
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from xqt.benchmark import LatencyReport, benchmark_callable
from xqt.core.artifact import ArtifactManifest, ArtifactRecord, MetricRecord
from xqt.core.errors import XQTBackendError
from xqt.eval.report import write_csv_report
from xqt.quant import (
    LayerAnalysisRecord,
    QuantizationPolicy,
    analyze_layer_errors,
    quantize_with_torchao,
    recommend_high_precision_modules,
)

DEFAULT_MODEL_NAME = "vit_small_patch16_224"
DEFAULT_ARTIFACT_DIR = Path("artifacts/xqt/torchao_vit")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_ANALYSIS_SAMPLE_LIMIT = 64
DEFAULT_BENCHMARK_ITERATIONS = 100
DEFAULT_BENCHMARK_WARMUP = 10


@dataclass
class TorchAOViTExperimentConfig:
    """Configuration for the standalone ViT torchao experiment."""

    model_name: str = DEFAULT_MODEL_NAME
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    image_root: Path | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    analysis_sample_limit: int = DEFAULT_ANALYSIS_SAMPLE_LIMIT
    benchmark_warmup: int = DEFAULT_BENCHMARK_WARMUP
    benchmark_iterations: int = DEFAULT_BENCHMARK_ITERATIONS
    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    strategy: str = "fp8_dynamic"
    compile_quantized: bool = True
    pretrained: bool = True
    seed: int = 11
    atol: float = 1e-3
    rtol: float = 1e-3
    policy: QuantizationPolicy = field(
        default_factory=lambda: QuantizationPolicy(
            dtype="fp8",
            scheme="dynamic_activation_weight",
            include_module_types=("Linear",),
            exclude_name_patterns=("head", "classifier"),
        )
    )


@dataclass
class TorchAOViTModels:
    """Reference and quantized model pair."""

    reference: nn.Module
    quantized: nn.Module
    quantized_modules: list[str]


@dataclass
class TorchAOViTExperimentResult:
    """Artifacts and metrics returned by the experiment runner."""

    layer_records: list[LayerAnalysisRecord]
    recommended_high_precision_modules: list[str]
    baseline_latency: LatencyReport
    quantized_latency: LatencyReport
    speedup: float
    manifest_path: Path
    analysis_csv_path: Path


def config_from_env(env: Mapping[str, str] | None = None) -> TorchAOViTExperimentConfig:
    """Build the experiment config from explicit environment variables."""

    values = env or os.environ
    image_root = values.get("XQT_TORCHAO_VIT_IMAGE_ROOT")
    artifact_dir = values.get("XQT_TORCHAO_VIT_ARTIFACT_DIR")
    model_name = values.get("XQT_TORCHAO_VIT_MODEL", DEFAULT_MODEL_NAME)

    return TorchAOViTExperimentConfig(
        model_name=model_name,
        artifact_dir=Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR,
        image_root=Path(image_root).expanduser() if image_root else None,
        batch_size=_env_int(values, "XQT_TORCHAO_VIT_BATCH_SIZE", DEFAULT_BATCH_SIZE),
        analysis_sample_limit=_env_int(
            values,
            "XQT_TORCHAO_VIT_ANALYSIS_SAMPLES",
            DEFAULT_ANALYSIS_SAMPLE_LIMIT,
        ),
        benchmark_warmup=_env_int(
            values,
            "XQT_TORCHAO_VIT_WARMUP",
            DEFAULT_BENCHMARK_WARMUP,
        ),
        benchmark_iterations=_env_int(
            values,
            "XQT_TORCHAO_VIT_ITERATIONS",
            DEFAULT_BENCHMARK_ITERATIONS,
        ),
        compile_quantized=_env_flag(
            values,
            "XQT_TORCHAO_VIT_COMPILE",
            default=True,
        ),
    )


def get_autocast(
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> AbstractContextManager[None]:
    """Return an autocast context for the selected device."""

    torch_device = torch.device(device)
    return torch.autocast(device_type=torch_device.type, dtype=dtype)


def load_models(config: TorchAOViTExperimentConfig) -> TorchAOViTModels:
    """Load a timm ViT and quantize a copy through the XQT torchao adapter."""

    try:
        import timm  # type: ignore[import-untyped]
    except ImportError as exc:
        raise XQTBackendError(
            "timm is required for xqt/torchao_vit.py. Install timm in the "
            "experiment environment."
        ) from exc

    _require_cuda_for_fp8(config)
    reference = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
    )
    reference = reference.to(config.device).to(config.dtype)
    reference.eval()

    quantized = copy.deepcopy(reference)
    quantized.eval()
    result = quantize_with_torchao(
        quantized,
        policy=config.policy,
        strategy=config.strategy,
        inplace=True,
    )
    return TorchAOViTModels(
        reference=reference,
        quantized=result.model,
        quantized_modules=list(result.quantized_modules),
    )


def build_analysis_loader(config: TorchAOViTExperimentConfig) -> DataLoader:
    """Build the analysis dataloader from ImageFolder or synthetic images."""

    if config.image_root is None:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        inputs = torch.randn(
            config.analysis_sample_limit,
            3,
            config.image_size,
            config.image_size,
            generator=generator,
        )
        targets = torch.zeros(config.analysis_sample_limit, dtype=torch.long)
        return DataLoader(
            TensorDataset(inputs, targets),
            batch_size=config.batch_size,
            shuffle=False,
        )

    try:
        from torchvision.datasets import ImageFolder  # type: ignore[import-untyped]
        from torchvision.transforms import (  # type: ignore[import-untyped]
            CenterCrop,
            Compose,
            Normalize,
            Resize,
            ToTensor,
        )
    except ImportError as exc:
        raise XQTBackendError(
            "torchvision is required when XQT_TORCHAO_VIT_IMAGE_ROOT is set."
        ) from exc

    transform = Compose(
        [
            Resize(256),
            CenterCrop(config.image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(str(config.image_root), transform=transform)
    if len(dataset) > config.analysis_sample_limit:
        dataset = Subset(dataset, range(config.analysis_sample_limit))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )


def analyze_vit_layer_errors(
    models: TorchAOViTModels,
    batch: torch.Tensor,
    config: TorchAOViTExperimentConfig,
) -> list[LayerAnalysisRecord]:
    """Run XQT per-layer error analysis for the model pair."""

    inputs = batch.to(device=config.device, dtype=config.dtype)
    with get_autocast(config.device, config.dtype):
        return analyze_layer_errors(
            models.reference,
            models.quantized,
            inputs,
            policy=config.policy,
            atol=config.atol,
            rtol=config.rtol,
            include_weight_diff=True,
        )


def analysis_records_to_rows(
    records: Sequence[LayerAnalysisRecord],
) -> list[dict[str, object]]:
    """Convert XQT analysis records to the legacy ViT CSV column shape."""

    rows: list[dict[str, object]] = []
    for record in records:
        weight_diff = record.weight_diff
        rows.append(
            {
                "layer_name": record.name,
                "layer_type": record.module_type,
                "parameter_count": record.parameter_count,
                "act_mae": record.diff.mean_abs,
                "act_mse": record.diff.mean_squared,
                "act_max_abs": record.diff.max_abs,
                "act_cos_sim": record.diff.cosine_similarity,
                "weight_mae": weight_diff.mean_abs if weight_diff is not None else None,
                "weight_mse": (
                    weight_diff.mean_squared if weight_diff is not None else None
                ),
                "weight_max_abs": (
                    weight_diff.max_abs if weight_diff is not None else None
                ),
                "weight_cos_sim": (
                    weight_diff.cosine_similarity if weight_diff is not None else None
                ),
                "recommendation": record.recommendation,
                "tags": ",".join(record.tags),
                "sort_key": record.diff.mean_abs,
            }
        )
    return sorted(rows, key=lambda row: float(row["sort_key"]), reverse=True)


def benchmark_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    config: TorchAOViTExperimentConfig,
) -> LatencyReport:
    """Benchmark a model using the shared XQT latency helper."""

    inputs = sample_input.to(device=config.device, dtype=config.dtype)

    def forward_once() -> object:
        with get_autocast(config.device, config.dtype):
            return model(inputs)

    return benchmark_callable(
        forward_once,
        warmup=config.benchmark_warmup,
        iterations=config.benchmark_iterations,
        sync_cuda=True,
        device=config.device,
    )


def run_experiment(
    config: TorchAOViTExperimentConfig | None = None,
) -> TorchAOViTExperimentResult:
    """Run the standalone ViT torchao experiment and write XQT artifacts."""

    config = config or config_from_env()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(config)
    loader = build_analysis_loader(config)
    batch, _targets = next(iter(loader))

    records = analyze_vit_layer_errors(models, batch, config)
    rows = analysis_records_to_rows(records)
    analysis_csv_path = write_csv_report(
        rows,
        config.artifact_dir / "layer_error_analysis.csv",
    )
    recommended_modules = recommend_high_precision_modules(records, top_k=10)

    baseline_latency = benchmark_model(models.reference, batch, config)
    benchmark_target = models.quantized
    if config.compile_quantized:
        benchmark_target = torch.compile(benchmark_target)
    quantized_latency = benchmark_model(benchmark_target, batch, config)
    speedup = (
        baseline_latency.mean_ms / quantized_latency.mean_ms
        if quantized_latency.mean_ms > 0.0
        else 0.0
    )

    manifest_path = _write_manifest(
        config=config,
        quantized_modules=models.quantized_modules,
        records=records,
        recommended_modules=recommended_modules,
        baseline_latency=baseline_latency,
        quantized_latency=quantized_latency,
        speedup=speedup,
        analysis_csv_path=analysis_csv_path,
    )
    return TorchAOViTExperimentResult(
        layer_records=records,
        recommended_high_precision_modules=recommended_modules,
        baseline_latency=baseline_latency,
        quantized_latency=quantized_latency,
        speedup=speedup,
        manifest_path=manifest_path,
        analysis_csv_path=analysis_csv_path,
    )


def print_summary(result: TorchAOViTExperimentResult, *, top_k: int = 10) -> None:
    """Print a compact console summary for manual runs."""

    print("\n==================== TOP high-error layers ====================")
    print(
        f"{'rank':<6} {'layer':<54} {'type':<20} "
        f"{'act_mae':<12} {'weight_mae':<12} {'act_cos':<12}"
    )
    print("-" * 128)
    for index, record in enumerate(result.layer_records[:top_k], start=1):
        weight_mae = (
            record.weight_diff.mean_abs if record.weight_diff is not None else None
        )
        print(
            f"{index:<6} {record.name:<54} {record.module_type:<20} "
            f"{record.diff.mean_abs:<12.6g} {str(weight_mae):<12} "
            f"{str(record.diff.cosine_similarity):<12}"
        )

    print("\n==================== Latency ====================")
    print(f"baseline mean: {result.baseline_latency.mean_ms:.3f} ms")
    print(f"quantized mean: {result.quantized_latency.mean_ms:.3f} ms")
    print(f"speedup: {result.speedup:.3f}x")
    print(f"analysis CSV: {result.analysis_csv_path}")
    print(f"manifest: {result.manifest_path}")
    print("\nRecommended high-precision modules:")
    for name in result.recommended_high_precision_modules:
        print(f"- {name}")


def _write_manifest(
    *,
    config: TorchAOViTExperimentConfig,
    quantized_modules: Sequence[str],
    records: Sequence[LayerAnalysisRecord],
    recommended_modules: Sequence[str],
    baseline_latency: LatencyReport,
    quantized_latency: LatencyReport,
    speedup: float,
    analysis_csv_path: Path,
) -> Path:
    manifest = ArtifactManifest(
        project_name="torchao_vit",
        compression_axes=["precision"],
        config_snapshot={
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "analysis_sample_limit": config.analysis_sample_limit,
            "benchmark_warmup": config.benchmark_warmup,
            "benchmark_iterations": config.benchmark_iterations,
            "image_size": config.image_size,
            "device": config.device,
            "dtype": str(config.dtype),
            "strategy": config.strategy,
            "compile_quantized": config.compile_quantized,
            "pretrained": config.pretrained,
        },
    )
    manifest.add_artifact(
        ArtifactRecord(
            path=str(analysis_csv_path),
            format="csv",
            runtime="xqt.quant.sensitivity",
            metadata={"record_count": len(records)},
        )
    )
    manifest.add_metric(
        MetricRecord(
            name="quantized_module_count",
            value=len(quantized_modules),
            metadata={"modules": list(quantized_modules)},
        )
    )
    manifest.add_metric(
        MetricRecord(
            name="analysis.record_count",
            value=len(records),
            metadata={
                "recommended_high_precision_modules": list(recommended_modules)
            },
        )
    )
    manifest.add_metric(
        MetricRecord(name="latency.baseline_mean_ms", value=baseline_latency.mean_ms)
    )
    manifest.add_metric(
        MetricRecord(name="latency.quantized_mean_ms", value=quantized_latency.mean_ms)
    )
    manifest.add_metric(MetricRecord(name="latency.speedup", value=speedup))
    manifest_path = config.artifact_dir / "manifest.json"
    return manifest.write_json(manifest_path)


def _env_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_flag(values: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _require_cuda_for_fp8(config: TorchAOViTExperimentConfig) -> None:
    if "fp8" not in config.strategy and "float8" not in config.strategy:
        return
    if torch.device(config.device).type != "cuda" or not torch.cuda.is_available():
        raise XQTBackendError(
            "The torchao FP8 ViT experiment requires CUDA-capable hardware. "
            "Use xqt/recipes/image_vit_torchao_fp8.yaml on a supported NVIDIA GPU, "
            "or set a non-FP8 strategy for local CPU smoke work."
        )


def main() -> None:
    """Run the experiment using environment-based configuration."""

    result = run_experiment(config_from_env())
    print_summary(result)


if __name__ == "__main__":
    main()
