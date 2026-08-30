"""Model-side quantizer algorithms."""

from .base import (
    Quantizer,
    QuantizerOptions,
    QuantizerResult,
    QuantizerTemplate,
    component_route_handler,
)
from .fake_qdq import FakeQDQSurrogateResult, build_fake_qdq_surrogate
from .fp4_weight_only import (
    FP4QuantizationResult,
    FP4WeightOnlyLinear,
    quantize_with_awq_fp4,
    quantize_with_fp4_weight_only,
    quantize_with_gptq_fp4,
)
from .awq_gptq_weight_only import (
    AWQGPTQWeightOnlyLinear,
    AWQGPTQWeightOnlyQuantizationResult,
    quantize_with_awq_weight_only,
    quantize_with_gptq_weight_only,
)
from .int8_mma import (
    Int8MmaLinear,
    Int8MmaQuantizationResult,
    quantize_with_int8_mma,
)
from .w4_storage_int8_mma import (
    W4StorageInt8MmaLinear,
    W4StorageInt8MmaQuantizationResult,
    quantize_with_w4_storage_int8_mma,
)
from .convrot_4bit import (
    ConvRot4BitQuantizationResult,
    ConvRotMixedPrecisionLinear,
    build_regular_hadamard_matrix,
    execute_convrot_4bit_component,
    quantize_with_convrot_4bit,
)
from .convrot_int8 import (
    ConvRotInt8Linear,
    ConvRotNormInt8Linear,
    ConvRotInt8QuantizationResult,
    execute_convrot_int8_component,
    quantize_with_convrot_int8,
)
from .mxfp_weight_only import (
    MXFPQuantizationResult,
    MXFPWeightOnlyLinear,
    quantize_with_mxfp_weight_only,
)
from .fp4_dynamic import (
    FP4DynamicLinear,
    FP4DynamicQuantizationResult,
    quantize_with_dynamic_fp4,
    quantize_with_mxfp4_dynamic,
    quantize_with_nvfp4_dynamic,
)
from .svd import (
    SVDQuantResult,
    quantize_with_svd,
)
from .turboquant import (
    TurboQuantCodec,
    TurboQuantEncoding,
    TurboQuantQuantizationResult,
    TurboQuantWeightOnlyLinear,
    execute_turboquant_component,
    quantize_with_turboquant,
)

__all__ = [
    "FP4QuantizationResult",
    "AWQGPTQWeightOnlyLinear",
    "AWQGPTQWeightOnlyQuantizationResult",
    "ConvRot4BitQuantizationResult",
    "ConvRotInt8Linear",
    "ConvRotNormInt8Linear",
    "ConvRotInt8QuantizationResult",
    "ConvRotMixedPrecisionLinear",
    "FakeQDQSurrogateResult",
    "FP4DynamicLinear",
    "FP4DynamicQuantizationResult",
    "Int8MmaLinear",
    "Int8MmaQuantizationResult",
    "MXFPQuantizationResult",
    "Quantizer",
    "QuantizerOptions",
    "QuantizerResult",
    "QuantizerTemplate",
    "FP4WeightOnlyLinear",
    "MXFPWeightOnlyLinear",
    "SVDQuantResult",
    "TurboQuantCodec",
    "TurboQuantEncoding",
    "TurboQuantQuantizationResult",
    "TurboQuantWeightOnlyLinear",
    "execute_turboquant_component",
    "quantize_with_turboquant",
    "W4StorageInt8MmaLinear",
    "W4StorageInt8MmaQuantizationResult",
    "build_regular_hadamard_matrix",
    "build_fake_qdq_surrogate",
    "execute_convrot_4bit_component",
    "execute_convrot_int8_component",
    "execute_kv_scale_component",
    "execute_moe_weight_only_component",
    "quantize_with_awq_fp4",
    "quantize_with_awq_weight_only",
    "quantize_with_convrot_4bit",
    "quantize_with_convrot_int8",
    "quantize_with_dynamic_fp4",
    "quantize_with_int8_mma",
    "quantize_with_mxfp4_dynamic",
    "quantize_with_mxfp_weight_only",
    "quantize_with_nvfp4_dynamic",
    "quantize_with_fp4_weight_only",
    "quantize_with_gptq_fp4",
    "quantize_with_gptq_weight_only",
    "quantize_with_svd",
    "quantize_with_w4_storage_int8_mma",
]


# ── execution route registration (C2) ──────────────────────────────────────
# Centralized route table for the quantization dispatcher: adding a new
# quantizer means adding its module plus one register_quant_route call here,
# without touching the executor. Both the executor and backend capability
# derivation read xqt.compression.quant.registry as the single routing source of truth.
from typing import Any, Mapping, Optional  # noqa: E402

from torch import nn  # noqa: E402

from xqt.core.types import XQTContext  # noqa: E402
from xqt.compression.quant.backends.onnx_qdq import (  # noqa: E402
    execute_onnx_qdq_component,
    onnx_qdq_graph_summary,
    quantize_onnx_qdq_static,
)
from xqt.compression.quant.backends.planned import (  # noqa: E402
    execute_planned_method_component,
)
from xqt.compression.quant.backends.torchao import execute_torchao_component  # noqa: E402
from xqt.compression.quant.registry import (  # noqa: E402
    QuantRouteRegistration,
    RouteQuery,
    register_quant_route,
    route_matcher,
)
from xqt.compression.quant.types import QuantizationComponentPlan, QuantizationReport  # noqa: E402

from .awq_gptq_weight_only import execute_awq_gptq_weight_only_component  # noqa: E402
from .fp4_dynamic import execute_dynamic_fp4_component  # noqa: E402
from .fp4_weight_only import execute_fp4_weight_only_component  # noqa: E402
from .int8_mma import execute_int8_mma_component  # noqa: E402
from .kv_scale import execute_kv_scale_component  # noqa: E402
from .moe_weight_only import execute_moe_weight_only_component  # noqa: E402
from .mxfp_weight_only import execute_mxfp_weight_only_component  # noqa: E402
from .svd import execute_svdquant_component  # noqa: E402
from .w4_storage_int8_mma import execute_w4_storage_int8_mma_component  # noqa: E402

_RouteResult = tuple[Optional[nn.Module], QuantizationReport, dict[str, Any]]


def _torchao_route_handler(
    context: XQTContext,
    model: Optional[nn.Module],
    component: QuantizationComponentPlan,
    *,
    runtime: Optional[Mapping[str, Any]] = None,
) -> _RouteResult:
    runtime = runtime or {}
    quantize_fn = runtime.get("quantize_with_torchao_fn")
    if quantize_fn is None:
        current_model, report = execute_torchao_component(context, model, component)
    else:
        current_model, report = execute_torchao_component(
            context, model, component, quantize_fn=quantize_fn
        )
    return current_model, report, {}


_svd_route_handler = component_route_handler(execute_svdquant_component)
_convrot_int8_route_handler = component_route_handler(execute_convrot_int8_component)
_convrot_4bit_route_handler = component_route_handler(execute_convrot_4bit_component)
_turboquant_route_handler = component_route_handler(execute_turboquant_component)
_fp4_weight_only_route_handler = component_route_handler(
    execute_fp4_weight_only_component
)
_awq_gptq_route_handler = component_route_handler(
    execute_awq_gptq_weight_only_component
)
_mxfp_route_handler = component_route_handler(execute_mxfp_weight_only_component)
_fp4_dynamic_nvfp4_route_handler = component_route_handler(
    execute_dynamic_fp4_component,
    executor_kwargs={
        "fp4_format": "nvfp4",
        "quantize_fn": quantize_with_nvfp4_dynamic,
    },
)
_fp4_dynamic_mxfp4_route_handler = component_route_handler(
    execute_dynamic_fp4_component,
    executor_kwargs={
        "fp4_format": "mxfp4",
        "quantize_fn": quantize_with_mxfp4_dynamic,
    },
)
_int8_mma_route_handler = component_route_handler(execute_int8_mma_component)
_w4_storage_route_handler = component_route_handler(
    execute_w4_storage_int8_mma_component
)
_kv_scale_route_handler = component_route_handler(
    execute_kv_scale_component,
    require_model=True,
    missing_model_message="kv_scale route requires a PyTorch model",
)
_moe_weight_only_route_handler = component_route_handler(
    execute_moe_weight_only_component,
    require_model=True,
    missing_model_message="moe_weight_only route requires a PyTorch model",
)


def _onnx_qdq_route_handler(
    context: XQTContext,
    model: Optional[nn.Module],
    component: QuantizationComponentPlan,
    *,
    runtime: Optional[Mapping[str, Any]] = None,
) -> _RouteResult:
    from xqt.export import export_onnx

    runtime = runtime or {}
    return execute_onnx_qdq_component(
        context,
        model,
        component,
        export_onnx_fn=runtime.get("export_onnx_fn", export_onnx),
        quantize_onnx_qdq_static_fn=runtime.get(
            "quantize_onnx_qdq_static_fn", quantize_onnx_qdq_static
        ),
        graph_summary_fn=runtime.get("onnx_qdq_graph_summary_fn", onnx_qdq_graph_summary),
    )


def _planned_route_handler(
    context: XQTContext,
    model: Optional[nn.Module],
    component: QuantizationComponentPlan,
    *,
    runtime: Optional[Mapping[str, Any]] = None,
) -> _RouteResult:
    del context, runtime
    return execute_planned_method_component(model, component)


def _convrot_int8_matcher(query: RouteQuery) -> bool:
    return (
        query.backend == "pytorch"
        and query.method == "convrot"
        and (
            query.strategy in {"w8a8_int8", "convrot_w8a8"}
            or query.compute == "w8a8_int8_mma"
        )
    )


register_quant_route(
    QuantRouteRegistration(
        name="torchao",
        backend="torchao",
        handler=_torchao_route_handler,
        matcher=route_matcher(backend="torchao"),
        priority=10,
        maturity="executable",
        methods=(
            "none",
            "w8a8_int8",
            "w8a8_fp8_e4m3",
            "w8a8_fp8_e5m2",
            "w8a16_fp8_e4m3",
            "w8a16_fp8_e5m2",
            "w4a16_int4",
            "w8a16_int8",
        ),
        primary_kernel='torchao',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="svd_int8_mma",
        backend="pytorch",
        handler=_svd_route_handler,
        matcher=route_matcher(methods=("svd",), computes=("w8a8_int8_mma",)),
        priority=20,
        maturity="executable",
        methods=("svd",),
        notes=(
            "SVDQuant keeps packed 4-bit residual storage and executes residual "
            "through W8A8 INT8 MMA; low-rank branch stays source precision.",
        ),
        primary_kernel='w8a8_int8_mma',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="svd_reference",
        backend="pytorch",
        handler=_svd_route_handler,
        matcher=route_matcher(methods=("svd",)),
        priority=21,
        maturity="reference_guarded",
        methods=("svd",),
        notes=(
            "SVDQuant reference path: low-rank source-precision branch + "
            "quantized residual dequantized before GEMM.",
        ),
        primary_kernel='torch',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="convrot_int8",
        backend="pytorch",
        handler=_convrot_int8_route_handler,
        matcher=_convrot_int8_matcher,
        priority=30,
        maturity="executable",
        methods=("convrot",),
        strategies=("w8a8_int8",),
        primary_kernel='w8a8_int8_mma',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="convrot_4bit",
        backend="pytorch",
        handler=_convrot_4bit_route_handler,
        matcher=route_matcher(methods=("convrot",)),
        priority=31,
        maturity="executable",
        methods=("convrot",),
        strategies=("w4a4_int4", "w4a4_fp4", "w4a16_int4", "w4a16_fp4"),
        primary_kernel='dequant_fp16',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="turboquant",
        backend="pytorch",
        handler=_turboquant_route_handler,
        matcher=route_matcher(methods=("turboquant",)),
        priority=40,
        maturity="executable",
        methods=("turboquant",),
        primary_kernel='dequant_fp16',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="fp4_weight_only",
        backend="pytorch",
        handler=_fp4_weight_only_route_handler,
        matcher=route_matcher(strategies=("w4a16_fp4",)),
        priority=50,
        maturity="executable",
        methods=("none", "awq", "gptq"),
        strategies=("w4a16_fp4",),
        primary_kernel='dequant_fp16',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="awq_gptq_weight_only",
        backend="pytorch",
        handler=_awq_gptq_route_handler,
        matcher=route_matcher(
            methods=("awq", "gptq"),
            strategies=("w4a16_int4", "w8a16_int8"),
        ),
        priority=60,
        maturity="executable",
        methods=("awq", "gptq"),
        strategies=("w4a16_int4", "w8a16_int8"),
        primary_kernel='dequant_fp16',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="mxfp_weight_only",
        backend="pytorch",
        handler=_mxfp_route_handler,
        matcher=route_matcher(strategies=("w4a16_mxfp4", "w8a16_mxfp8")),
        priority=70,
        maturity="executable",
        strategies=("w4a16_mxfp4", "w8a16_mxfp8"),
        primary_kernel='dequant_fp16',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="fp4_dynamic_nvfp4",
        backend="pytorch",
        handler=_fp4_dynamic_nvfp4_route_handler,
        matcher=route_matcher(strategies=("w4a4_nvfp4",)),
        priority=80,
        maturity="executable",
        strategies=("w4a4_nvfp4",),
        primary_kernel='tilelang_fp4',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="fp4_dynamic_mxfp4",
        backend="pytorch",
        handler=_fp4_dynamic_mxfp4_route_handler,
        matcher=route_matcher(strategies=("w4a4_mxfp4",)),
        priority=81,
        maturity="executable",
        strategies=("w4a4_mxfp4",),
        primary_kernel='tilelang_fp4',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="int8_mma",
        backend="pytorch",
        handler=_int8_mma_route_handler,
        matcher=route_matcher(
            strategies=("w8a8_int8",),
            computes=("w8a8_int8_mma",),
        ),
        priority=90,
        maturity="executable",
        strategies=("w8a8_int8",),
        primary_kernel='w8a8_int8_mma',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="w4_storage_int8_mma",
        backend="pytorch",
        handler=_w4_storage_route_handler,
        matcher=route_matcher(
            methods=("none",),
            strategies=("w4a16_int4", "w4a16_fp4"),
            computes=("w8a8_int8_mma",),
        ),
        priority=91,
        maturity="executable",
        methods=("none",),
        strategies=("w4a16_int4", "w4a16_fp4"),
        primary_kernel='w8a8_int8_mma',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="onnx_qdq",
        backend="onnxruntime_qdq",
        handler=_onnx_qdq_route_handler,
        matcher=route_matcher(backend="onnxruntime_qdq"),
        priority=100,
        maturity="executable",
        methods=("none", "w8a8_int8"),
        primary_kernel='onnxruntime_qdq',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="kv_scale",
        backend="pytorch",
        handler=_kv_scale_route_handler,
        matcher=route_matcher(methods=("kv_scale",), strategies=("kv_scale",)),
        priority=95,
        maturity="executable",
        methods=("kv_scale",),
        strategies=("kv_scale",),
        notes=("Model-side K/V scale artifacts only; no cache runtime.",),
        primary_kernel='metadata_only',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="moe_weight_only",
        backend="pytorch",
        handler=_moe_weight_only_route_handler,
        matcher=route_matcher(methods=("moe_weight_only", "moe")),
        priority=96,
        maturity="executable",
        methods=("moe_weight_only", "moe"),
        strategies=("w4a16_int4", "w4a16_fp4"),
        notes=("Expert weight-only; router kept high precision. Smoke-oriented.",),
        primary_kernel='dequant_fp16',
        reference_kernel='torch'
    )
)
register_quant_route(
    QuantRouteRegistration(
        name="planned_awq_gptq",
        backend="pytorch",
        handler=_planned_route_handler,
        matcher=route_matcher(methods=("awq", "gptq")),
        priority=900,
        maturity="planned",
        status="planned",
        methods=("awq", "gptq"),
        notes=(
            "AWQ/GPTQ with this strategy has no executable algorithm in XQT; "
            "the report is a capability advertisement only.",
        ),
        primary_kernel=None,
        reference_kernel=None
    )
)
