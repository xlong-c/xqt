"""Typed stage parameter specs for XQT optimization workflows."""

from __future__ import annotations

from typing import Any, Mapping, cast

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from xqt.contracts.inference import InferenceContract
from xqt.core.errors import XQTConfigError
from xqt.core.schema import (
    ExportTargetConfig,
    MNNExportConfig,
    NCNNExportConfig,
    ONNXExportConfig,
    OpenVINOExportConfig,
    OutputDiffConfig,
    PruneConfig,
    QNNExportConfig,
    QuantComponentPolicyConfig,
    QuantConfig,
    TensorRTExportConfig,
    TorchScriptExportConfig,
)
from xqt.core.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    DeployRuntimeHandleSpec,
    DeployStageSpec,
    ExportStageSpec,
    ONNXRuntimeHandleConfig,
    OperatorStageSpec,
    PruneStageSpec,
    QuantStageSpec,
    StageSpec,
    TensorRTRuntimeHandleConfig,
    stage_spec_to_config,
    stage_spec_to_params,
)


_LEGACY_ONNX_TARGET_PARAM_KEYS = frozenset(
    {
        "dynamo",
        "input_names",
        "onnx_optimization",
        "optimize",
        "output_names",
        "pre_export_fusion",
        "pre_export_lowering",
        "runtime_diff",
        "validate",
    }
)

_LEGACY_TENSORRT_TARGET_PARAM_KEYS = frozenset(
    {
        "backend",
        "builder_optimization_level",
        "dry_run",
        "extra_args",
        "log_level",
        "onnx_path",
        "performance_thresholds",
        "plugin_libraries",
        "runtime_benchmark",
        "serialize_plugin_libraries",
        "timeout",
        "timing_cache_path",
        "trtexec_path",
        "validate_plugin_libraries_loadable",
        "workspace_mib",
    }
)

_LEGACY_OPENVINO_TARGET_PARAM_KEYS = frozenset(
    {
        "benchmark",
        "device",
        "dry_run",
        "input_shape",
        "onnx_path",
        "runtime_diff",
    }
)

_LEGACY_TORCH_EXPORT_TARGET_PARAM_KEYS = frozenset(
    {
        "runtime_diff",
        "strict",
        "validate",
    }
)

_LEGACY_TORCHSCRIPT_TARGET_PARAM_KEYS = frozenset(
    {
        "check_trace",
        "method",
        "runtime_diff",
    }
)

_LEGACY_EXECUTORCH_TARGET_PARAM_KEYS = frozenset({"dry_run"})

_LEGACY_NCNN_TARGET_PARAM_KEYS = frozenset(
    {
        "bin_path",
        "converter",
        "dry_run",
        "extra_args",
        "onnx2ncnn_path",
        "onnx_path",
        "pnnx_path",
        "timeout",
    }
)

_LEGACY_MNN_TARGET_PARAM_KEYS = frozenset(
    {
        "converter_path",
        "dry_run",
        "extra_args",
        "framework",
        "onnx_path",
        "timeout",
    }
)

_LEGACY_QNN_TARGET_PARAM_KEYS = frozenset(
    {
        "converter_path",
        "dry_run",
        "extra_args",
        "onnx_path",
        "timeout",
    }
)


_STAGE_SPEC_TYPES = {
    "quant": QuantStageSpec,
    "prune": PruneStageSpec,
    "operator": OperatorStageSpec,
    "export": ExportStageSpec,
    "deploy": DeployStageSpec,
    "analyze": AnalyzeStageSpec,
    "benchmark": BenchmarkStageSpec,
}


def build_stage_spec(kind: str, params: Mapping[str, Any] | None = None) -> StageSpec:
    """Build the typed spec for one workflow stage kind."""

    spec_type = _STAGE_SPEC_TYPES.get(kind)
    if spec_type is None:
        allowed = ", ".join(sorted(_STAGE_SPEC_TYPES))
        raise XQTConfigError(f"unsupported stage kind {kind}. Allowed: {allowed}")
    if kind == "quant" and (not params or not params.get("backend")):
        raise XQTConfigError("quant.params.backend is required")
    _reject_legacy_stage_params(kind, params)
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(spec_type),
            OmegaConf.create(dict(params or {})),
        )
        spec = cast(StageSpec, OmegaConf.to_object(merged))
    except OmegaConfBaseException as exc:
        raise XQTConfigError(f"failed to load {kind} stage params: {exc}") from exc
    except Exception as exc:
        raise XQTConfigError(f"failed to load {kind} stage params: {exc}") from exc
    _validate_stage_spec(kind, spec)
    return spec


def _reject_legacy_stage_params(
    kind: str,
    params: Mapping[str, Any] | None,
) -> None:
    if kind != "deploy" or not params:
        return
    raw_handle = params.get("runtime_handle")
    if isinstance(raw_handle, Mapping) and "params" in raw_handle:
        raise XQTConfigError(
            "deploy.params.runtime_handle.params is removed. Use "
            "runtime_handle.onnxruntime or runtime_handle.tensorrt."
        )


def ensure_stage_spec(stage: Any, *, rebuild: bool = False) -> StageSpec:
    """Return ``stage.spec``, building it from ``stage.params`` when needed."""

    spec = getattr(stage, "spec", None)
    if spec is None or rebuild:
        spec = build_stage_spec(str(stage.kind), stage.params)
        setattr(stage, "spec", spec)
    return cast(StageSpec, spec)


def stage_params(stage: Any, *, drop_none: bool = True) -> dict[str, Any]:
    """Return normalized stage parameters from the typed spec."""

    return stage_spec_to_params(ensure_stage_spec(stage), drop_none=drop_none)


def _validate_stage_spec(kind: str, spec: StageSpec) -> None:
    if isinstance(spec, QuantStageSpec):
        _validate_quant_stage_spec(spec)
    elif isinstance(spec, PruneStageSpec):
        _validate_prune_stage_spec(spec)
    elif isinstance(spec, OperatorStageSpec):
        _validate_operator_stage_spec(spec)
    elif isinstance(spec, ExportStageSpec):
        _validate_export_stage_spec(spec, location=f"{kind}.params.validate")
    elif isinstance(spec, DeployStageSpec):
        _validate_deploy_stage_spec(spec, location=f"{kind}.params")
    elif isinstance(spec, AnalyzeStageSpec):
        _validate_analyze_stage_spec(spec)
    elif isinstance(spec, BenchmarkStageSpec):
        _validate_benchmark_stage_spec(spec, location=f"{kind}.params")


def _validate_quant_stage_spec(spec: QuantStageSpec) -> None:
    from xqt.quant.capability import describe_quant_backend_capability

    spec.component_policies = [
        _quant_component_policy_config(component)
        for component in spec.component_policies
    ]
    try:
        quant = stage_spec_to_config(
            spec,
            QuantConfig,
            overrides={"enabled": True},
        )
    except Exception as exc:
        raise XQTConfigError(f"invalid quant.params: {exc}") from exc
    quant.component_policies = [
        _quant_component_policy_config(component)
        for component in quant.component_policies
    ]
    if not quant.backend:
        raise XQTConfigError("quant.params.backend is required")
    has_selector = (
        quant.method is not None
        or quant.strategy is not None
        or bool(quant.policy)
        or bool(quant.component_policies)
    )
    if not has_selector:
        raise XQTConfigError(
            "quant.params must specify method, strategy, policy, or component_policies"
        )
    try:
        if quant.component_policies:
            for component in quant.component_policies:
                if not component.enabled:
                    continue
                policy = dict(quant.policy)
                policy.update(component.policy)
                describe_quant_backend_capability(
                    component.backend or quant.backend,
                    method=component.method or quant.method,
                    strategy=component.strategy or quant.strategy,
                    policy=policy,
                )
        else:
            describe_quant_backend_capability(
                quant.backend,
                method=quant.method,
                strategy=quant.strategy,
                policy=quant.policy,
            )
    except ValueError as exc:
        raise XQTConfigError(f"invalid quant.params: {exc}") from exc


def _quant_component_policy_config(value: Any) -> QuantComponentPolicyConfig:
    if isinstance(value, QuantComponentPolicyConfig):
        return value
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(QuantComponentPolicyConfig),
            OmegaConf.create(dict(value)),
        )
        return cast(QuantComponentPolicyConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise XQTConfigError(f"failed to load quant component policy: {exc}") from exc


def _validate_prune_stage_spec(spec: PruneStageSpec) -> None:
    from xqt.prune import describe_prune_method

    try:
        describe_prune_method(spec.method)
    except ValueError as exc:
        raise XQTConfigError(str(exc)) from exc
    stage_spec_to_config(
        spec,
        PruneConfig,
        overrides={"enabled": True},
    )


def _validate_operator_stage_spec(spec: OperatorStageSpec) -> None:
    for index, target in enumerate(spec.targets):
        location = f"operator.params.targets.{index}"
        if target.candidate_kind not in {"single_kernel", "block_kernel"}:
            raise XQTConfigError(
                f"{location}.candidate_kind must be single_kernel or block_kernel"
            )
        if target.candidate_kind == "single_kernel" and target.block_kernel is not None:
            raise XQTConfigError(
                f"{location}.block_kernel requires candidate_kind=block_kernel"
            )
        if (
            target.candidate_kind == "single_kernel"
            and target.block_kernel_engine is not None
        ):
            raise XQTConfigError(
                f"{location}.block_kernel_engine requires candidate_kind=block_kernel"
            )
        benchmark_target = (
            target.benchmark_target
            if target.benchmark_target is not None
            else target.target
        )
        if (
            target.candidate_kind == "block_kernel"
            and benchmark_target != target.target
        ):
            raise XQTConfigError(
                f"{location}.benchmark_target must equal target for block_kernel"
            )
        engine = target.engine or spec.default_engine
        if (
            target.candidate_kind == "block_kernel"
            and engine != "torch_compile"
            and target.block_kernel is None
        ):
            raise XQTConfigError(
                f"{location}.block_kernel is required for manual block_kernel targets"
            )
        if target.block_kernel_engine is not None and not (
            target.candidate_kind == "block_kernel"
            and engine == "torch_compile"
            and target.block_kernel is not None
        ):
            raise XQTConfigError(
                f"{location}.block_kernel_engine is only valid for torch_compile "
                "block_kernel targets with a manual block_kernel fallback"
            )
        if (
            target.candidate_kind == "block_kernel"
            and engine == "torch_compile"
            and target.block_kernel is not None
            and target.block_kernel_engine is None
        ):
            raise XQTConfigError(
                f"{location}.block_kernel_engine is required when block_kernel "
                "is used as a torch_compile fallback"
            )
    if spec.benchmark is not None:
        _validate_benchmark_stage_spec(
            spec.benchmark,
            location="operator.params.benchmark",
        )


def _validate_export_stage_spec(spec: ExportStageSpec, *, location: str) -> None:
    _validate_export_targets(
        spec.targets, location=f"{location.rsplit('.', 1)[0]}.targets"
    )
    if spec.validate is not None:
        _validate_output_diff_config(spec.validate, location=location)


def _validate_deploy_stage_spec(spec: DeployStageSpec, *, location: str) -> None:
    _validate_export_targets(spec.targets, location=f"{location}.targets")
    if spec.validate is not None:
        _validate_output_diff_config(spec.validate, location=f"{location}.validate")
    if spec.runtime_handle is not None and spec.runtime_handle.materialize:
        if not spec.runtime_handle.runtime:
            raise XQTConfigError(
                f"{location}.runtime_handle.runtime is required when materialize=true"
            )
        _validate_runtime_handle_spec(
            spec.runtime_handle,
            location=f"{location}.runtime_handle",
            targets=spec.targets,
        )


def _validate_runtime_handle_spec(
    spec: DeployRuntimeHandleSpec,
    *,
    location: str,
    targets: list[ExportTargetConfig],
) -> None:
    if spec.runtime not in {"onnxruntime", "tensorrt"}:
        raise XQTConfigError(
            f"{location}.runtime must be onnxruntime or tensorrt when materialize=true"
        )
    if spec.runtime == "onnxruntime":
        onnx_targets = [target for target in targets if target.format == "onnx"]
        if len(onnx_targets) != 1:
            raise XQTConfigError(
                f"{location} requires exactly one ONNX target when runtime=onnxruntime and materialize=true"
            )
        if any(not provider for provider in spec.onnxruntime.providers):
            raise XQTConfigError(
                f"{location}.onnxruntime.providers must not contain empty values"
            )
        return
    tensorrt_targets = [target for target in targets if target.format == "tensorrt"]
    if len(tensorrt_targets) != 1:
        raise XQTConfigError(
            f"{location} requires exactly one TensorRT target when runtime=tensorrt and materialize=true"
        )
    if any(not path for path in spec.tensorrt.plugin_libraries):
        raise XQTConfigError(
            f"{location}.tensorrt.plugin_libraries must not contain empty paths"
        )


def _validate_export_targets(
    targets: list[ExportTargetConfig],
    *,
    location: str,
) -> None:
    if not targets:
        raise XQTConfigError(f"{location} must declare at least one export target")
    for index, target in enumerate(targets):
        target_location = f"{location}.{index}"
        _validate_inference_contract(
            target,
            location=f"{target_location}.inference",
        )
        if target.format == "onnx":
            _validate_onnx_target(target, location=target_location)
        elif target.format == "tensorrt":
            _validate_tensorrt_target(target, location=target_location)
        elif target.format == "openvino":
            _validate_openvino_target(target, location=target_location)
        elif target.format == "torch_export":
            _validate_torch_export_target(target, location=target_location)
        elif target.format == "torchscript":
            _validate_torchscript_target(target, location=target_location)
        elif target.format == "executorch":
            _validate_executorch_target(target, location=target_location)
        elif target.format == "ncnn":
            _validate_ncnn_target(target, location=target_location)
        elif target.format == "mnn":
            _validate_mnn_target(target, location=target_location)
        elif target.format == "qnn":
            _validate_qnn_target(target, location=target_location)


def _validate_inference_contract(
    target: ExportTargetConfig,
    *,
    location: str,
) -> None:
    if target.inference is None:
        return
    try:
        InferenceContract.from_dict(target.inference)
    except XQTConfigError as exc:
        raise XQTConfigError(f"{location} is invalid: {exc}") from exc


def _validate_onnx_target(target: ExportTargetConfig, *, location: str) -> None:
    legacy_keys = sorted(_LEGACY_ONNX_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy ONNX keys: {keys}. "
            f"Move them to {location}.onnx."
        )
    _validate_onnx_export_config(target.onnx, location=f"{location}.onnx")


def _validate_tensorrt_target(target: ExportTargetConfig, *, location: str) -> None:
    legacy_keys = sorted(_LEGACY_TENSORRT_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy TensorRT keys: {keys}. "
            f"Move them to {location}.tensorrt."
        )
    _validate_tensorrt_export_config(target.tensorrt, location=f"{location}.tensorrt")


def _validate_tensorrt_export_config(
    config: TensorRTExportConfig,
    *,
    location: str,
) -> None:
    if config.backend not in {"trtexec", "python_api"}:
        raise XQTConfigError(f"{location}.backend must be trtexec or python_api")
    if config.workspace_mib <= 0:
        raise XQTConfigError(f"{location}.workspace_mib must be positive")
    if any(not path for path in config.plugin_libraries):
        raise XQTConfigError(
            f"{location}.plugin_libraries must not contain empty paths"
        )
    benchmark = config.runtime_benchmark
    if benchmark.enabled:
        if not benchmark.input_shapes:
            raise XQTConfigError(
                f"{location}.runtime_benchmark.input_shapes is required when enabled=true"
            )
        if benchmark.warmup < 0:
            raise XQTConfigError(
                f"{location}.runtime_benchmark.warmup must be non-negative"
            )
        if benchmark.iterations <= 0:
            raise XQTConfigError(
                f"{location}.runtime_benchmark.iterations must be positive"
            )


def _validate_openvino_target(target: ExportTargetConfig, *, location: str) -> None:
    legacy_keys = sorted(_LEGACY_OPENVINO_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy OpenVINO keys: {keys}. "
            f"Move them to {location}.openvino."
        )
    _validate_openvino_export_config(target.openvino, location=f"{location}.openvino")


def _validate_openvino_export_config(
    config: OpenVINOExportConfig,
    *,
    location: str,
) -> None:
    if config.onnx_path == "":
        raise XQTConfigError(f"{location}.onnx_path must not be empty")
    if config.input_shape is not None and any(
        dimension <= 0 for dimension in config.input_shape
    ):
        raise XQTConfigError(f"{location}.input_shape dimensions must be positive")
    if not config.device:
        raise XQTConfigError(f"{location}.device must not be empty")


def _validate_torch_export_target(
    target: ExportTargetConfig,
    *,
    location: str,
) -> None:
    legacy_keys = sorted(_LEGACY_TORCH_EXPORT_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy TorchExport keys: {keys}. "
            f"Move them to {location}.torch_export."
        )


def _validate_torchscript_target(
    target: ExportTargetConfig,
    *,
    location: str,
) -> None:
    legacy_keys = sorted(_LEGACY_TORCHSCRIPT_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy TorchScript keys: {keys}. "
            f"Move them to {location}.torchscript."
        )
    _validate_torchscript_export_config(
        target.torchscript,
        location=f"{location}.torchscript",
    )


def _validate_torchscript_export_config(
    config: TorchScriptExportConfig,
    *,
    location: str,
) -> None:
    if config.method not in {"trace", "script"}:
        raise XQTConfigError(f"{location}.method must be trace or script")


def _validate_executorch_target(
    target: ExportTargetConfig,
    *,
    location: str,
) -> None:
    legacy_keys = sorted(_LEGACY_EXECUTORCH_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy ExecuTorch keys: {keys}. "
            f"Move them to {location}.executorch."
        )


def _validate_ncnn_target(target: ExportTargetConfig, *, location: str) -> None:
    legacy_keys = sorted(_LEGACY_NCNN_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy ncnn keys: {keys}. "
            f"Move them to {location}.ncnn."
        )
    _validate_ncnn_export_config(target.ncnn, location=f"{location}.ncnn")


def _validate_ncnn_export_config(
    config: NCNNExportConfig,
    *,
    location: str,
) -> None:
    if config.source_path == "":
        raise XQTConfigError(f"{location}.source_path must not be empty")
    if config.converter not in {"onnx2ncnn", "pnnx"}:
        raise XQTConfigError(f"{location}.converter must be onnx2ncnn or pnnx")
    if not config.onnx2ncnn_path:
        raise XQTConfigError(f"{location}.onnx2ncnn_path must not be empty")
    if not config.pnnx_path:
        raise XQTConfigError(f"{location}.pnnx_path must not be empty")
    if config.bin_path == "":
        raise XQTConfigError(f"{location}.bin_path must not be empty")
    if config.timeout is not None and config.timeout <= 0:
        raise XQTConfigError(f"{location}.timeout must be positive")


def _validate_mnn_target(target: ExportTargetConfig, *, location: str) -> None:
    legacy_keys = sorted(_LEGACY_MNN_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy MNN keys: {keys}. "
            f"Move them to {location}.mnn."
        )
    _validate_mnn_export_config(target.mnn, location=f"{location}.mnn")


def _validate_mnn_export_config(
    config: MNNExportConfig,
    *,
    location: str,
) -> None:
    if config.source_path == "":
        raise XQTConfigError(f"{location}.source_path must not be empty")
    if not config.converter_path:
        raise XQTConfigError(f"{location}.converter_path must not be empty")
    if not config.framework:
        raise XQTConfigError(f"{location}.framework must not be empty")
    if config.timeout is not None and config.timeout <= 0:
        raise XQTConfigError(f"{location}.timeout must be positive")


def _validate_qnn_target(target: ExportTargetConfig, *, location: str) -> None:
    legacy_keys = sorted(_LEGACY_QNN_TARGET_PARAM_KEYS & set(target.params))
    if legacy_keys:
        keys = ", ".join(legacy_keys)
        raise XQTConfigError(
            f"{location}.params contains legacy QNN keys: {keys}. "
            f"Move them to {location}.qnn."
        )
    _validate_qnn_export_config(target.qnn, location=f"{location}.qnn")


def _validate_qnn_export_config(
    config: QNNExportConfig,
    *,
    location: str,
) -> None:
    if config.source_path == "":
        raise XQTConfigError(f"{location}.source_path must not be empty")
    if not config.converter_path:
        raise XQTConfigError(f"{location}.converter_path must not be empty")
    if config.timeout is not None and config.timeout <= 0:
        raise XQTConfigError(f"{location}.timeout must be positive")


def _validate_onnx_export_config(
    config: ONNXExportConfig,
    *,
    location: str,
) -> None:
    fusion = config.pre_export_fusion
    if fusion.enabled:
        if fusion.mode not in {"eager", "fx"}:
            raise XQTConfigError(
                f"{location}.pre_export_fusion.mode must be eager or fx"
            )
        if fusion.mode == "eager":
            if not fusion.modules_to_fuse:
                raise XQTConfigError(
                    f"{location}.pre_export_fusion.modules_to_fuse must not be empty "
                    "when mode=eager and enabled=true"
                )
            if any(len(group) < 2 for group in fusion.modules_to_fuse):
                raise XQTConfigError(
                    f"{location}.pre_export_fusion.modules_to_fuse entries must "
                    "contain at least two module names"
                )
    lowering = config.pre_export_lowering
    if lowering.enabled and lowering.mode != "fp4_weight_only_to_dense_linear":
        raise XQTConfigError(
            f"{location}.pre_export_lowering.mode must be "
            "fp4_weight_only_to_dense_linear"
        )
    optimization = config.optimization
    if optimization.enabled:
        if optimization.backend != "onnxruntime":
            raise XQTConfigError(f"{location}.optimization.backend must be onnxruntime")
        if not optimization.providers:
            raise XQTConfigError(
                f"{location}.optimization.providers must not be empty when enabled=true"
            )


def _validate_analyze_stage_spec(spec: AnalyzeStageSpec) -> None:
    if not spec.metrics:
        raise XQTConfigError("analyze.params.metrics must not be empty")
    if spec.top_k is not None and spec.top_k <= 0:
        raise XQTConfigError("analyze.params.top_k must be positive when provided")


def _validate_benchmark_stage_spec(
    spec: BenchmarkStageSpec,
    *,
    location: str,
) -> None:
    if spec.warmup is not None and spec.warmup < 0:
        raise XQTConfigError(f"{location}.warmup must be non-negative")
    if spec.iterations is not None and spec.iterations <= 0:
        raise XQTConfigError(f"{location}.iterations must be positive")


def _validate_output_diff_config(
    spec: OutputDiffConfig,
    *,
    location: str,
) -> None:
    if spec.atol < 0:
        raise XQTConfigError(f"{location}.atol must be non-negative")
    if spec.rtol < 0:
        raise XQTConfigError(f"{location}.rtol must be non-negative")


__all__ = [
    "AnalyzeStageSpec",
    "BenchmarkStageSpec",
    "DeployRuntimeHandleSpec",
    "DeployStageSpec",
    "ExportStageSpec",
    "ONNXRuntimeHandleConfig",
    "OperatorStageSpec",
    "PruneStageSpec",
    "QuantStageSpec",
    "StageSpec",
    "TensorRTRuntimeHandleConfig",
    "build_stage_spec",
    "ensure_stage_spec",
    "stage_params",
    "stage_spec_to_config",
    "stage_spec_to_params",
]
