"""Thin model-side runner over quantized artifacts (not a serving engine)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from torch import nn

from xqt.benchmark.phase_latency import PhaseLatencyReport, benchmark_prefill_decode
from xqt.contracts.contract_consume import (
    ContractConsumeReport,
    consume_runtime_quant_contract,
)
from xqt.contracts.layout_kernel_report import LayoutKernelReport
from xqt.contracts.quantized import QuantizedModel
from xqt.contracts.runtime_manifest import build_runtime_manifest
from xqt.contracts.runtime_quant import RuntimeQuantContract
from xqt.contracts.runtime_features import (
    RUNTIME_FEATURES_KEY,
    runtime_feature_report,
)


@dataclass(frozen=True, slots=True)
class ModelRunnerReport:
    contract_ok: bool
    selected_kernel: str | None
    fallback_reason: str | None
    storage_layout: str | None
    quantized_modules: tuple[str, ...]
    contract_errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    phase_latency: PhaseLatencyReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_ok": self.contract_ok,
            "selected_kernel": self.selected_kernel,
            "fallback_reason": self.fallback_reason,
            "storage_layout": self.storage_layout,
            "quantized_modules": list(self.quantized_modules),
            "contract_errors": list(self.contract_errors),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if self.phase_latency is not None:
            payload["phase_latency"] = self.phase_latency.to_dict()
        return payload


def _layout_from_metadata(metadata: Mapping[str, Any]) -> LayoutKernelReport | None:
    raw = metadata.get("layout_kernel")
    if raw is None:
        return None
    if isinstance(raw, LayoutKernelReport):
        return raw
    if isinstance(raw, Mapping):
        return LayoutKernelReport.from_dict(raw)
    return None


class ModelRunner:
    """Eager forward + contract/layout diagnostics for self-owned quantized models."""

    def __init__(
        self,
        model: nn.Module,
        *,
        metadata: Mapping[str, Any] | None = None,
        quantized_modules: list[str] | tuple[str, ...] | None = None,
        contract: RuntimeQuantContract | None = None,
        require_contract: bool = True,
    ) -> None:
        self.model = model
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.quantized_modules = tuple(str(x) for x in (quantized_modules or ()))
        self._contract = contract
        self._require_contract = bool(require_contract)
        self._consume = self._validate_contract()

    @classmethod
    def from_quantized(
        cls,
        quantized: QuantizedModel,
        *,
        require_contract: bool = True,
    ) -> "ModelRunner":
        contract = quantized.resolve_runtime_quant_contract()
        return cls(
            quantized.model,
            metadata=dict(quantized.metadata),
            quantized_modules=list(quantized.quantized_modules),
            contract=contract,
            require_contract=require_contract,
        )

    def _validate_contract(self) -> ContractConsumeReport:
        source: Any
        if self._contract is not None:
            source = self._contract
        else:
            source = self.metadata
        report = consume_runtime_quant_contract(
            source,
            require_kernels=self._require_contract,
            require_shapes=self._require_contract,
        )
        if self._require_contract and not report.ok:
            raise ValueError(
                "ModelRunner requires a valid RuntimeQuantContract; "
                f"errors={list(report.errors)}"
            )
        return report

    @property
    def contract(self) -> RuntimeQuantContract | None:
        return self._consume.contract

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def report(
        self,
        *,
        prefill_fn: Callable[[], object] | None = None,
        decode_fn: Callable[[], object] | None = None,
        phase_warmup: int = 0,
        phase_iterations: int = 2,
    ) -> ModelRunnerReport:
        layout = _layout_from_metadata(self.metadata)
        selected = None if layout is None else layout.selected_kernel
        fallback = None if layout is None else layout.fallback_reason
        storage = None
        if self.contract is not None:
            storage = self.contract.storage_layout
        elif layout is not None:
            storage = layout.storage_layout

        notes = list(self._consume.notes)
        if layout is None:
            notes.append("layout_kernel_absent_in_metadata")
        if not self.quantized_modules:
            notes.append("quantized_modules_empty")

        phase: PhaseLatencyReport | None = None
        if prefill_fn is not None and decode_fn is not None:
            phase = benchmark_prefill_decode(
                prefill_fn,
                decode_fn,
                warmup=phase_warmup,
                iterations=phase_iterations,
                sync_cuda=False,
            )

        meta: dict[str, Any] = {}
        manifest = build_runtime_manifest(self.metadata)
        meta["manifest_kernels"] = list(manifest.selected_kernels)
        raw_features = self.metadata.get(RUNTIME_FEATURES_KEY)
        if raw_features is not None:
            meta["runtime_features"] = runtime_feature_report(raw_features)
        if selected is None and manifest.selected_kernels:
            selected = manifest.selected_kernels[0]
        if fallback is None and manifest.fallback_kernels:
            fallback = manifest.fallback_kernels[0]

        return ModelRunnerReport(
            contract_ok=self._consume.ok,
            selected_kernel=selected,
            fallback_reason=fallback,
            storage_layout=storage,
            quantized_modules=self.quantized_modules,
            contract_errors=self._consume.errors,
            notes=tuple(notes),
            phase_latency=phase,
            metadata=meta,
        )


__all__ = [
    "ModelRunner",
    "ModelRunnerReport",
]
