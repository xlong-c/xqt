"""Validate RuntimeQuantContract fields for self-owned runtime consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import nn

from xqt.contracts.runtime_quant import (
    RUNTIME_QUANT_CONTRACT_KEY,
    RuntimeQuantContract,
    extract_runtime_quant_contract,
)
from xqt.core.errors import XQTConfigError


@dataclass(frozen=True, slots=True)
class ContractConsumeReport:
    ok: bool
    contract: RuntimeQuantContract | None
    errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "contract": None if self.contract is None else self.contract.to_dict(),
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


def consume_runtime_quant_contract(
    source: RuntimeQuantContract | Mapping[str, Any] | nn.Module | object,
    *,
    require_kernels: bool = True,
    require_shapes: bool = True,
) -> ContractConsumeReport:
    contract: RuntimeQuantContract | None
    notes: list[str] = []
    if isinstance(source, RuntimeQuantContract):
        contract = source
    elif isinstance(source, Mapping):
        if RUNTIME_QUANT_CONTRACT_KEY in source and isinstance(
            source.get(RUNTIME_QUANT_CONTRACT_KEY), (Mapping, RuntimeQuantContract)
        ):
            raw = source[RUNTIME_QUANT_CONTRACT_KEY]
            contract = (
                raw
                if isinstance(raw, RuntimeQuantContract)
                else RuntimeQuantContract.from_dict(raw)
            )
        elif "quant_spec" in source and "storage_layout" in source:
            contract = RuntimeQuantContract.from_dict(source)
        else:
            contract = extract_runtime_quant_contract(source)
    else:
        meta = getattr(source, "metadata", None)
        if isinstance(meta, Mapping):
            contract = extract_runtime_quant_contract(meta)
            if contract is None and hasattr(source, "resolve_runtime_quant_contract"):
                contract = source.resolve_runtime_quant_contract()
        elif hasattr(source, "resolve_runtime_quant_contract"):
            contract = source.resolve_runtime_quant_contract()
        else:
            contract = None

    if contract is None:
        return ContractConsumeReport(
            ok=False,
            contract=None,
            errors=("runtime_quant_contract_missing",),
            notes=("consumer requires RuntimeQuantContract before apply",),
        )

    errors: list[str] = []
    try:
        if not str(contract.storage_layout).strip():
            errors.append("empty_storage_layout")
        if require_kernels and not contract.required_kernels:
            errors.append("required_kernels_empty")
        if require_shapes:
            if not contract.global_shape or not contract.local_shape:
                errors.append("shapes_incomplete")
        if contract.quant_spec is None:
            errors.append("quant_spec_missing")
    except XQTConfigError as exc:
        errors.append(f"invalid_contract:{exc}")

    notes.append(f"storage_layout={contract.storage_layout}")
    notes.append(f"kernels={list(contract.required_kernels)}")
    return ContractConsumeReport(
        ok=not errors,
        contract=contract,
        errors=tuple(errors),
        notes=tuple(notes),
    )


__all__ = [
    "ContractConsumeReport",
    "consume_runtime_quant_contract",
]
