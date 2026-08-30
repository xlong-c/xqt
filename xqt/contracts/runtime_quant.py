"""Self-owned runtime quant contract (internal fact source, not HF adapter).

Consumers (Linear/MoE/Attention modules, quant pair sidecars) should read this
object instead of parsing external ``quantization_config`` dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from xqt.contracts.quant_scheme import QuantScheme
from xqt.core.base import XQTConfigError

RUNTIME_QUANT_CONTRACT_KEY = "runtime_quant_contract"
RUNTIME_QUANT_CONTRACT_SCHEMA_VERSION = 1


def _require_mapping(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise XQTConfigError(f"RuntimeQuantContract missing required field: {key}")
    return payload[key]


def _as_shape(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        raise XQTConfigError(f"RuntimeQuantContract.{field_name} is required")
    if not isinstance(value, (list, tuple)):
        raise XQTConfigError(
            f"RuntimeQuantContract.{field_name} must be a sequence of ints; "
            f"got {type(value).__name__}"
        )
    shape: list[int] = []
    for item in value:
        try:
            shape.append(int(item))
        except (TypeError, ValueError) as exc:
            raise XQTConfigError(
                f"RuntimeQuantContract.{field_name} entries must be int; "
                f"got {item!r}"
            ) from exc
    return tuple(shape)


def _as_kernels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise XQTConfigError(
            "RuntimeQuantContract.required_kernels must be a sequence of str"
        )
    return tuple(str(item) for item in value)


def _as_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise XQTConfigError(
        f"RuntimeQuantContract.{field_name} must be bool; got {type(value).__name__}"
    )


def _quant_scheme_from_mapping(raw: Any) -> QuantScheme:
    if isinstance(raw, QuantScheme):
        return raw
    if not isinstance(raw, Mapping):
        raise XQTConfigError(
            "RuntimeQuantContract.quant_spec must be a mapping or QuantScheme"
        )
    try:
        group_size = raw.get("group_size")
        return QuantScheme(
            weight_dtype=str(raw["weight_dtype"]),
            weight_granularity=str(raw["weight_granularity"]),
            group_size=None if group_size is None else int(group_size),
            activation_dtype=(
                None
                if raw.get("activation_dtype") is None
                else str(raw.get("activation_dtype"))
            ),
            activation_mode=str(raw.get("activation_mode", "none")),
            sym=bool(raw.get("sym", True)),
        )
    except KeyError as exc:
        raise XQTConfigError(
            f"RuntimeQuantContract.quant_spec missing field: {exc.args[0]}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise XQTConfigError(f"RuntimeQuantContract.quant_spec invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RuntimeQuantContract:
    """Internal contract from quantizer artifacts to self-owned runtime modules.

    This is not an HF / vLLM / SGLang config dict. Optional external adapters may
    *derive* from it; they must not define its field semantics.
    """

    quant_spec: QuantScheme
    storage_layout: str
    required_kernels: tuple[str, ...]
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    prefill_supported: bool
    decode_supported: bool
    repack_version: str | None = None
    shard_axis: int | None = None
    kv_cache_dtype: str | None = None
    schema_version: int = RUNTIME_QUANT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.storage_layout).strip():
            raise XQTConfigError(
                "RuntimeQuantContract.storage_layout must be a non-empty str"
            )
        if self.shard_axis is not None and int(self.shard_axis) < 0:
            raise XQTConfigError(
                "RuntimeQuantContract.shard_axis must be >= 0 when set"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "quant_spec": self.quant_spec.to_dict(),
            "storage_layout": str(self.storage_layout),
            "repack_version": self.repack_version,
            "required_kernels": list(self.required_kernels),
            "global_shape": list(self.global_shape),
            "local_shape": list(self.local_shape),
            "shard_axis": self.shard_axis,
            "prefill_supported": bool(self.prefill_supported),
            "decode_supported": bool(self.decode_supported),
            "kv_cache_dtype": self.kv_cache_dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeQuantContract:
        if not isinstance(payload, Mapping):
            raise XQTConfigError(
                "RuntimeQuantContract.from_dict expects a mapping; "
                f"got {type(payload).__name__}"
            )
        storage_layout = _require_mapping(payload, "storage_layout")
        if storage_layout is None or not str(storage_layout).strip():
            raise XQTConfigError(
                "RuntimeQuantContract missing required field: storage_layout"
            )
        quant_spec = _quant_scheme_from_mapping(_require_mapping(payload, "quant_spec"))
        kernels = _as_kernels(payload.get("required_kernels", ()))
        global_shape = _as_shape(
            _require_mapping(payload, "global_shape"), field_name="global_shape"
        )
        local_shape = _as_shape(
            _require_mapping(payload, "local_shape"), field_name="local_shape"
        )
        prefill = _as_bool(
            _require_mapping(payload, "prefill_supported"),
            field_name="prefill_supported",
        )
        decode = _as_bool(
            _require_mapping(payload, "decode_supported"),
            field_name="decode_supported",
        )
        repack = payload.get("repack_version")
        shard = payload.get("shard_axis")
        kv = payload.get("kv_cache_dtype")
        version = payload.get("schema_version", RUNTIME_QUANT_CONTRACT_SCHEMA_VERSION)
        try:
            version_i = int(version)
        except (TypeError, ValueError) as exc:
            raise XQTConfigError(
                f"RuntimeQuantContract.schema_version must be int; got {version!r}"
            ) from exc
        return cls(
            quant_spec=quant_spec,
            storage_layout=str(storage_layout),
            required_kernels=kernels,
            global_shape=global_shape,
            local_shape=local_shape,
            prefill_supported=prefill,
            decode_supported=decode,
            repack_version=None if repack is None else str(repack),
            shard_axis=None if shard is None else int(shard),
            kv_cache_dtype=None if kv is None else str(kv),
            schema_version=version_i,
        )


def attach_runtime_quant_contract(
    metadata: dict[str, Any],
    contract: RuntimeQuantContract,
) -> dict[str, Any]:
    """Return a copy of ``metadata`` with the contract under the canonical key."""

    out = dict(metadata)
    out[RUNTIME_QUANT_CONTRACT_KEY] = contract.to_dict()
    return out


def extract_runtime_quant_contract(
    metadata: Mapping[str, Any] | None,
) -> RuntimeQuantContract | None:
    """Parse contract from metadata if present; otherwise None."""

    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(RUNTIME_QUANT_CONTRACT_KEY)
    if raw is None:
        return None
    if isinstance(raw, RuntimeQuantContract):
        return raw
    if isinstance(raw, Mapping):
        return RuntimeQuantContract.from_dict(raw)
    raise XQTConfigError(
        f"metadata[{RUNTIME_QUANT_CONTRACT_KEY!r}] must be a mapping or "
        f"RuntimeQuantContract; got {type(raw).__name__}"
    )


def build_runtime_quant_contract(
    *,
    quant_spec: QuantScheme | Mapping[str, Any],
    storage_layout: str,
    required_kernels: tuple[str, ...] | list[str] = (),
    global_shape: tuple[int, ...] = (),
    local_shape: tuple[int, ...] | None = None,
    prefill_supported: bool = True,
    decode_supported: bool = True,
    repack_version: str | None = None,
    shard_axis: int | None = None,
    kv_cache_dtype: str | None = None,
) -> RuntimeQuantContract:
    """Construct a RuntimeQuantContract from scheme + layout + shapes (U8)."""

    scheme = _quant_scheme_from_mapping(quant_spec)
    local = global_shape if local_shape is None else local_shape
    return RuntimeQuantContract(
        quant_spec=scheme,
        storage_layout=storage_layout,
        required_kernels=tuple(required_kernels),
        global_shape=tuple(int(x) for x in global_shape),
        local_shape=tuple(int(x) for x in local),
        prefill_supported=prefill_supported,
        decode_supported=decode_supported,
        repack_version=repack_version,
        shard_axis=shard_axis,
        kv_cache_dtype=kv_cache_dtype,
    )


def first_linear_shapes(model: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (out, in) from the first Linear-like quantized or nn.Linear module."""

    try:
        from torch import nn
    except ImportError:
        return (), ()
    for module in getattr(model, "modules", lambda: ())():
        out_f = getattr(module, "output_features", None)
        in_f = getattr(module, "input_features", None)
        if out_f is not None and in_f is not None:
            return (int(out_f), int(in_f)), (int(out_f), int(in_f))
        if isinstance(module, nn.Linear):
            return (
                (int(module.out_features), int(module.in_features)),
                (int(module.out_features), int(module.in_features)),
            )
    return (), ()


__all__ = [
    "RUNTIME_QUANT_CONTRACT_KEY",
    "RUNTIME_QUANT_CONTRACT_SCHEMA_VERSION",
    "RuntimeQuantContract",
    "attach_runtime_quant_contract",
    "build_runtime_quant_contract",
    "extract_runtime_quant_contract",
    "first_linear_shapes",
]
