"""Tests for the quantization route registry (C2)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import nn

from xqt.core.types import XQTContext
from xqt.compression.quant import registry as quant_registry
from xqt.compression.quant.capability import describe_quant_backend_capability
from xqt.compression.quant.execution import execute_quantization_plan
from xqt.compression.quant.registry import (
    QuantRouteRegistration,
    RouteQuery,
    iter_quant_routes,
    register_quant_route,
    resolve_quant_route,
    route_matcher,
)
from xqt.compression.quant.types import (
    QuantizationComponentPlan,
    QuantizationExecutionPlan,
    QuantizationReport,
)
from xqt.compression.quant.quantizers.base import component_route_handler


def _query(
    backend: str = "pytorch",
    method: str = "none",
    strategy: str = "",
    compute: str = "dequant_fp16",
) -> RouteQuery:
    return RouteQuery(
        backend=backend,
        method=method,
        strategy=strategy,
        compute=compute,
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (_query(backend="torchao", strategy="w8a8_int8"), "torchao"),
        (_query(method="svd", strategy="w4a16_int4", compute="w8a8_int8_mma"), "svd_int8_mma"),
        (_query(method="svd", strategy="w4a16_int4"), "svd_reference"),
        (_query(method="convrot", strategy="w8a8_int8"), "convrot_int8"),
        (_query(method="convrot", strategy="w4a16_int4", compute="w8a8_int8_mma"), "convrot_int8"),
        (_query(method="convrot", strategy="w4a4_int4"), "convrot_4bit"),
        (_query(method="convrot", strategy="w4a16_fp4"), "convrot_4bit"),
        (_query(method="turboquant", strategy="w4a16_int4"), "turboquant"),
        (_query(method="awq", strategy="w4a16_fp4"), "fp4_weight_only"),
        (_query(strategy="w4a16_fp4"), "fp4_weight_only"),
        (_query(method="awq", strategy="w4a16_int4"), "awq_gptq_weight_only"),
        (_query(method="gptq", strategy="w8a16_int8"), "awq_gptq_weight_only"),
        (_query(strategy="w4a16_mxfp4"), "mxfp_weight_only"),
        (_query(strategy="w8a16_mxfp8"), "mxfp_weight_only"),
        (_query(strategy="w4a4_nvfp4"), "fp4_dynamic_nvfp4"),
        (_query(strategy="w4a4_mxfp4"), "fp4_dynamic_mxfp4"),
        (_query(strategy="w8a8_int8", compute="w8a8_int8_mma"), "int8_mma"),
        (_query(strategy="w4a16_int4", compute="w8a8_int8_mma"), "w4_storage_int8_mma"),
        (_query(backend="onnxruntime_qdq", strategy="w8a8_int8"), "onnx_qdq"),
        (_query(method="awq", strategy="w8a8_int8"), "planned_awq_gptq"),
    ],
)
def test_route_resolution(query: RouteQuery, expected: str) -> None:
    registration = resolve_quant_route(query)
    assert registration is not None, query
    assert registration.name == expected


def test_route_resolution_miss_returns_none() -> None:
    assert resolve_quant_route(_query(strategy="w4a4_int4")) is None
    assert resolve_quant_route(_query(strategy="w8a8_fp8_e4m3")) is None


def test_executable_primary_kernel_maps_to_engine_or_torch() -> None:
    """U4: every executable route primary_kernel is engine-resolvable or mapped."""

    from xqt.kernels.engine_resolve import (
        get_engine_registration,
        map_primary_kernel_to_engine,
    )

    for registration in iter_quant_routes():
        if registration.maturity != "executable":
            continue
        assert registration.primary_kernel
        engine_name = map_primary_kernel_to_engine(registration.primary_kernel)
        assert engine_name is not None, registration.name
        reg = get_engine_registration(engine_name)
        assert reg is not None, (
            f"route {registration.name!r} primary_kernel="
            f"{registration.primary_kernel!r} maps to unknown engine {engine_name!r}"
        )


def test_scheme_kernel_matrix_lists_executable_routes() -> None:
    """V5: scheme_kernel_matrix is derived from the route table."""

    from xqt.compression.quant.registry import scheme_kernel_matrix

    rows = scheme_kernel_matrix()
    assert rows
    names = {row["name"] for row in rows}
    assert "int8_mma" in names or "awq_gptq_weight_only" in names
    for row in rows:
        assert row["maturity"] == "executable"
        assert row["primary_kernel"]
        assert row["reference_kernel"]


def test_every_executable_route_is_reachable() -> None:
    for registration in iter_quant_routes():
        if registration.status != "available":
            continue
        strategies = registration.strategies or ("",)
        methods = registration.methods or ("none",)
        hit = False
        for method in methods:
            for strategy in strategies:
                computes = ("dequant_fp16", "w8a8_int8_mma", "dequant_gemm")
                for compute in computes:
                    query = _query(
                        backend=registration.backend,
                        method=method,
                        strategy=strategy,
                        compute=compute,
                    )
                    if resolve_quant_route(query) is registration:
                        hit = True
        assert hit, f"route {registration.name!r} is unreachable"


def test_capability_maturity_derives_from_route_table() -> None:
    cases = [
        ("awq", "w4a16_fp4", "dequant_fp16"),
        ("gptq", "w8a16_int8", "dequant_fp16"),
        ("svd", "w4a16_int4", "dequant_fp16"),
        ("svd", "w4a16_int4", "w8a8_int8_mma"),
        ("convrot", "w4a4_int4", "dequant_fp16"),
        ("convrot", "w8a8_int8", "dequant_fp16"),
        ("turboquant", "w4a16_int4", "dequant_fp16"),
        (None, "w4a4_nvfp4", "dequant_gemm"),
        (None, "w4a4_mxfp4", "dequant_gemm"),
        (None, "w8a8_int8", "w8a8_int8_mma"),
        (None, "w4a16_int4", "w8a8_int8_mma"),
        ("awq", "w8a8_int8", "dequant_fp16"),
    ]
    for method, strategy, compute in cases:
        capability = describe_quant_backend_capability(
            "pytorch",
            method=method,
            strategy=strategy,
            compute=compute,
        )
        registration = resolve_quant_route(
            _query(
                method=method or "none",
                strategy=strategy,
                compute=compute,
            )
        )
        assert registration is not None, (method, strategy, compute)
        assert capability.maturity == registration.maturity, (
            method,
            strategy,
            compute,
        )


def _context(model: nn.Module) -> XQTContext:
    return XQTContext(
        model=model,
        reference_model=copy.deepcopy(model),
        calibration_inputs=None,
        device="cpu",
        artifact_dir="artifacts/xqt/tests/quant_registry",
        project_name="quant_registry",
        quant_config=None,
    )


def _component(strategy: str, compute: str = "dequant_fp16") -> QuantizationComponentPlan:
    return QuantizationComponentPlan(
        name="model",
        backend="pytorch",
        method=None,
        strategy=strategy,
        compute=compute,
    )


def test_component_route_handler_applies_static_algorithm_kwargs() -> None:
    def _executor(
        context: XQTContext,
        model: nn.Module | None,
        component: QuantizationComponentPlan,
        *,
        execution_state: str,
    ) -> tuple[nn.Module | None, QuantizationReport]:
        del context
        return model, QuantizationReport(
            component_name=component.name,
            backend=component.backend,
            strategy=component.strategy,
            algorithm_executable=True,
            metadata={"execution_state": execution_state},
        )

    handler = component_route_handler(
        _executor,
        executor_kwargs={"execution_state": "factory"},
    )
    model = nn.Identity()
    returned_model, report, artifacts = handler(
        _context(model),
        model,
        _component("w4a16_fp4"),
        runtime={"ignored": True},
    )

    assert returned_model is model
    assert report.metadata["execution_state"] == "factory"
    assert artifacts == {}


def test_executor_falls_back_to_planned_report_for_unregistered_route() -> None:
    plan = QuantizationExecutionPlan(components=[_component("w8a8_fp8_e4m3")])
    execution = execute_quantization_plan(_context(torch.nn.Linear(4, 4)), plan)

    assert len(execution.reports) == 1
    report = execution.reports[0]
    assert report.algorithm_executable is False
    assert report.method_semantics == "capability_report_only_no_executable_algorithm"
    assert report.metadata["execution_state"] == "planned"


def test_new_quantizer_registers_without_executor_changes() -> None:
    marker = "test_route_registration_without_executor_edit"

    def _handler(
        context: XQTContext,
        model: nn.Module | None,
        component: QuantizationComponentPlan,
        *,
        runtime: Any = None,
    ) -> tuple[nn.Module | None, QuantizationReport, dict[str, Any]]:
        del context, runtime
        report = QuantizationReport(
            component_name=component.name,
            backend=component.backend,
            method=component.method,
            strategy=component.strategy,
            algorithm_executable=True,
            method_semantics=marker,
            quantized_modules=["proof"],
        )
        return model, report, {}

    registration = QuantRouteRegistration(
        name="test_route",
        backend="pytorch",
        handler=_handler,
        matcher=route_matcher(strategies=("w8a8_fp8_e4m3",)),
        priority=50,
        maturity="executable",
        strategies=("w8a8_fp8_e4m3",),
        primary_kernel="torch",
        reference_kernel="torch",
    )
    register_quant_route(registration)
    try:
        plan = QuantizationExecutionPlan(components=[_component("w8a8_fp8_e4m3")])
        execution = execute_quantization_plan(_context(torch.nn.Linear(4, 4)), plan)
        report = execution.reports[0]
        assert report.method_semantics == marker
        assert report.algorithm_executable is True
        assert report.quantized_modules == ["proof"]
    finally:
        quant_registry._ROUTE_REGISTRY.remove(registration)


def test_executable_routes_declare_kernel_pair() -> None:
    for registration in iter_quant_routes():
        if registration.maturity != "executable":
            continue
        assert registration.primary_kernel, registration.name
        assert registration.reference_kernel, registration.name


def test_register_executable_without_reference_rejected() -> None:
    def _handler(context, model, component, *, runtime=None):  # type: ignore[no-untyped-def]
        raise AssertionError("unreachable")

    with pytest.raises(ValueError, match="reference_kernel"):
        register_quant_route(
            QuantRouteRegistration(
                name="__test_missing_ref__",
                backend="pytorch",
                handler=_handler,
                matcher=route_matcher(methods=("__test_missing_ref__",)),
                maturity="executable",
                primary_kernel="torch",
            )
        )
