"""Inference-only AWQ W4A16 Linear backed by the SM89 decode kernel."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    PackedWeight,
    PackedWeightMetadata,
    QuantSpec,
    prepare_sm89_awq_w4a16_decode_parameters,
    prepack_sm89_awq_w4a16_decode,
    sm89_awq_w4a16_metadata,
)


def _signed_groupwise_storage_parts(
    storage: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    """Validate one signed group-64 W4 storage and return decode-ready parts.

    Returns ``(remapped_qweight, scales_2d, zero_points, bias, input_features,
    groups)`` where every tensor is CPU/CUDA-agnostic and contiguous. Storage
    nibbles are ``u = s mod 16``; the uint4 GEMM contract needs ``u' = s + 8``,
    which is exactly XOR 0x88 per byte.
    """

    qweight = getattr(storage, "quantized_weight", None)
    scales = getattr(storage, "weight_scale", None)
    bits = int(getattr(storage, "bits", 0) or 0)
    group_size = int(getattr(storage, "group_size", 0) or 0)
    padded_k = int(getattr(storage, "padded_input_features", 0) or 0)
    input_features = int(getattr(storage, "input_features", 0) or 0)
    output_features = int(getattr(storage, "output_features", 0) or 0)
    if bits != 4 or group_size != 64:
        raise XQTBackendError(
            "signed groupwise storage conversion requires 4-bit group-64 weights"
        )
    if (
        not isinstance(qweight, torch.Tensor)
        or qweight.dtype != torch.uint8
        or tuple(qweight.shape) != (output_features, padded_k // 2)
    ):
        raise XQTBackendError(
            "signed groupwise storage conversion expects canonical uint8 "
            "qweight [N, K/2]"
        )
    if padded_k != input_features:
        raise XQTBackendError(
            "signed groupwise storage conversion does not accept K padding"
        )
    if output_features % 8 != 0 or input_features % 64 != 0:
        raise XQTBackendError("SM89 AWQ decode requires N % 8 == 0 and K % 64 == 0")
    if not isinstance(scales, torch.Tensor):
        raise XQTBackendError(
            "signed groupwise storage conversion expects tensor weight_scale"
        )
    groups = input_features // group_size
    scale_2d = scales.reshape(output_features, groups).to(torch.float32).contiguous()
    zero_points = torch.full_like(scale_2d, 8.0)
    bias = getattr(storage, "bias", None)
    remapped_qweight = (qweight.detach() ^ 0x88).contiguous()
    return remapped_qweight, scale_2d, zero_points, bias, input_features, groups


class AWQW4A16Linear(nn.Module):
    """Run asymmetric group-64 INT4 projections for SM89 inference decode."""

    def __init__(
        self,
        weight: PackedWeight,
        *,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if isinstance(weight, PackedWeight) and bias is not None:
            output_features = int(weight.metadata.logical_shape[0])
            if bias.numel() != output_features:
                raise ValueError("AWQ W4A16 bias size must equal output_features")
        prepacked = prepack_sm89_awq_w4a16_decode(weight)
        self.input_features = int(prepacked.metadata.logical_shape[1])
        self.output_features = int(prepacked.metadata.logical_shape[0])
        self.group_size = int(prepacked.metadata.group_size or 0)
        self.register_buffer("qweight", prepacked.qweight)
        self.register_buffer("canonical_qweight", prepacked.canonical_qweight)
        self.register_buffer("weight_scale", prepacked.scales.to(torch.float32))
        self.register_buffer(
            "weight_zero_point", prepacked.zero_points.to(torch.float32)
        )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().reshape(-1).to(torch.float32))
        self._has_bias = bias is not None
        self._last_execution: dict[str, Any] = {
            "implementation": "not_run",
            "native": False,
        }
        self._prepared_parameters: dict[
            torch.dtype, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._spec_cache: dict[tuple[int, torch.dtype, str], GemmSpec] = {}
        self._bound_runners: dict[
            tuple[int, torch.dtype, str], Callable[[torch.Tensor], torch.Tensor]
        ] = {}
        self._residual_runners: dict[
            tuple[int, torch.dtype, str],
            Callable[[torch.Tensor, torch.Tensor], None],
        ] = {}
        self._w4a8_runners: dict[
            tuple[int, torch.dtype, str],
            Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor],
        ] = {}
        self._w4a8_residual_runners: dict[
            tuple[int, torch.dtype, str],
            Callable[[torch.Tensor, torch.Tensor | float, torch.Tensor], None],
        ] = {}
        self._native_only = False
        self._native_only_device: torch.device | None = None
        self._native_only_dtype: torch.dtype | None = None
        self._native_only_rows: frozenset[int] = frozenset()
        self._native_only_tensors: tuple[torch.Tensor, ...] = ()
        self._native_only_state_signature: tuple[Any, ...] | None = None
        self._native_only_bound_runners: dict[
            int, Callable[[torch.Tensor], torch.Tensor]
        ] = {}
        self._native_only_released_bytes = 0
        self._runtime_state_signature = self._state_signature()

    @classmethod
    def from_signed_groupwise_storage(
        cls,
        storage: Any,
    ) -> "AWQW4A16Linear":
        """Materialize a runtime decode view from signed groupwise W4 storage.

        ``AWQGPTQWeightOnlyLinear`` stores signed nibbles (code = s + 8, low
        nibble = even feature). The SM89 decode kernel consumes unsigned
        uint4 with an explicit zero point: ``w = u * scale + scaled_zero``.
        Reinterpreting the same bytes with ``zero_points = 8`` reproduces
        ``w = s * scale`` exactly, so no weight repacking is needed.
        """

        return cls.from_signed_groupwise_storages([storage])

    @classmethod
    def from_signed_groupwise_storages(
        cls,
        storages: Sequence[Any],
    ) -> "AWQW4A16Linear":
        """Concatenate several signed group-64 storages along the output dim.

        All storages must share the same input features and group size. The
        result is one decode module whose output rows are the concatenation of
        the inputs' rows, which lets callers fuse q/k/v or gate/up projections
        into a single GEMV.
        """

        from xqt.kernels.ops.gemm import PackedWeight, PackedWeightMetadata

        if not storages:
            raise XQTBackendError("at least one storage is required")
        parts = [_signed_groupwise_storage_parts(storage) for storage in storages]
        input_features = parts[0][4]
        groups = parts[0][5]
        for index, (_, _, _, _, features, part_groups) in enumerate(parts):
            if features != input_features or part_groups != groups:
                raise XQTBackendError(
                    "fused AWQ decode requires matching input features and "
                    f"group layout, storage {index} has K={features}, groups={part_groups}"
                )
        biases = [part[3] for part in parts]
        has_bias = any(bias is not None for bias in biases)
        bias: torch.Tensor | None = None
        if has_bias:
            bias = torch.cat(
                [
                    part[3]
                    if part[3] is not None
                    else torch.zeros(int(part[0].shape[0]), dtype=torch.float32)
                    for part in parts
                ]
            ).to(torch.float32)
        qweight = torch.cat([part[0] for part in parts], dim=0).contiguous()
        scales = torch.cat([part[1] for part in parts], dim=0).contiguous()
        zero_points = torch.cat([part[2] for part in parts], dim=0).contiguous()
        output_features = int(qweight.shape[0])
        packed = PackedWeight(
            qweight=qweight,
            scales=scales,
            zero_points=zero_points,
            metadata=PackedWeightMetadata(
                logical_shape=(output_features, input_features),
                storage_layout="xqt_int4_nk_v1",
                pack_version="xqt-w4a16-awq-v1",
                weight_dtype="int4",
                padded_k=input_features,
                group_size=64,
                packed_bits=4,
                nibble_order="low_high",
                nibble_signed=False,
            ),
        )
        return cls(packed, bias=bias)

    def _state_signature(self) -> tuple[int, ...]:
        tensors = (
            self.qweight,
            self.canonical_qweight,
            self.weight_scale,
            self.weight_zero_point,
            self.bias,
        )
        signature: list[int] = []
        for tensor in tensors:
            if tensor is None:
                signature.extend((0, 0))
            else:
                signature.extend((id(tensor), int(tensor._version)))
        return tuple(signature)

    def _refresh_runtime_state(self) -> None:
        if self._native_only:
            self._validate_native_only_state()
            return
        signature = self._state_signature()
        if signature == self._runtime_state_signature:
            return
        self._prepared_parameters.clear()
        self._bound_runners.clear()
        self._residual_runners.clear()
        self._runtime_state_signature = signature

    def _apply(self, fn: Any) -> "AWQW4A16Linear":
        if self._native_only:
            raise RuntimeError(
                "native-only AWQW4A16Linear cannot be moved or cast; "
                "freeze a canonical module again for the target device and dtype"
            )
        super()._apply(fn)
        self._prepared_parameters.clear()
        self._spec_cache.clear()
        self._bound_runners.clear()
        self._residual_runners.clear()
        self._runtime_state_signature = self._state_signature()
        self._last_execution = {"implementation": "not_run", "native": False}
        return self

    def _packed_weight(self) -> PackedWeight:
        if self._native_only:
            raise RuntimeError(
                "native-only AWQW4A16Linear released its canonical packed weight"
            )
        return PackedWeight(
            qweight=self.qweight,
            scales=self.weight_scale,
            zero_points=self.weight_zero_point,
            metadata=PackedWeightMetadata(
                logical_shape=(self.output_features, self.input_features),
                storage_layout="sm89_awq_w4a16_interleaved_v1",
                pack_version="xqt-sm89-awq-w4a16-v1",
                weight_dtype="int4",
                padded_k=self.input_features,
                group_size=self.group_size,
                packed_bits=4,
                nibble_order="low_high",
                nibble_signed=False,
            ),
            canonical_qweight=self.canonical_qweight,
        )

    def _spec(self, *, rows: int, dtype: torch.dtype, device: torch.device) -> GemmSpec:
        if self._native_only:
            raise RuntimeError(
                "native-only AWQW4A16Linear cannot materialize a new GEMM spec"
            )
        key = (int(rows), dtype, str(device))
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        dtype_name = "fp16" if dtype == torch.float16 else "bf16"
        spec = GemmSpec(
            problem=GemmProblem(
                m=rows,
                n=self.output_features,
                k=self.input_features,
                phase="decode",
                sm=89,
                device=str(device),
            ),
            quant=QuantSpec(
                weight_dtype="int4",
                activation_dtype=dtype_name,
                output_dtype=dtype_name,
                weight_granularity="groupwise",
                group_size=64,
                symmetric=False,
                weight_zero_point=True,
                weight_scale_source="weight_offline",
                storage_layout="xqt_int4_nk_v1",
                pack_version="xqt-w4a16-awq-v1",
            ),
            epilogue=EpilogueSpec(
                has_bias=self.bias is not None,
                output_dtype=dtype_name,
            ),
        )
        self._spec_cache[key] = spec
        return spec

    def _prepared(
        self,
        *,
        dtype: torch.dtype,
        spec: GemmSpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._native_only:
            raise RuntimeError(
                "native-only AWQW4A16Linear cannot prepare canonical parameters"
            )
        cached = self._prepared_parameters.get(dtype)
        if cached is not None:
            return cached
        prepared = prepare_sm89_awq_w4a16_decode_parameters(
            self._packed_weight(),
            spec=spec,
            dtype=dtype,
        )
        self._prepared_parameters[dtype] = prepared
        return prepared

    @property
    def native_only(self) -> bool:
        """Return whether only immutable native execution tensors remain."""

        return bool(self.__dict__.get("_native_only", False))

    @staticmethod
    def _normalize_native_device(device: torch.device | str) -> torch.device:
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("native-only AWQW4A16Linear requires a CUDA device")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        return target

    @staticmethod
    def _tensor_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
        seen: set[tuple[str, int]] = set()
        total = 0
        for tensor in tensors:
            key = (str(tensor.device), int(tensor.data_ptr()))
            if key in seen:
                continue
            seen.add(key)
            total += int(tensor.numel()) * int(tensor.element_size())
        return total

    def _canonical_storage_bytes(self) -> int:
        tensors = (
            self._buffers.get("canonical_qweight"),
            self._buffers.get("weight_scale"),
            self._buffers.get("weight_zero_point"),
            self._buffers.get("bias"),
        )
        return self._tensor_storage_bytes(
            tensor for tensor in tensors if isinstance(tensor, torch.Tensor)
        )

    @staticmethod
    def _execution_state_signature(
        tensors: Iterable[torch.Tensor],
    ) -> tuple[Any, ...]:
        signature: list[Any] = ["native_only_awq_w4a16"]
        for tensor in tensors:
            signature.append(
                (
                    str(tensor.device),
                    str(tensor.dtype),
                    int(tensor.data_ptr()),
                    int(getattr(tensor, "_version", 0)),
                    tuple(int(dim) for dim in tensor.shape),
                )
            )
        return tuple(signature)

    def _validate_native_only_state(self) -> None:
        signature = self._execution_state_signature(self._native_only_tensors)
        if (
            self._native_only_state_signature is None
            or signature != self._native_only_state_signature
        ):
            raise RuntimeError("native-only AWQ W4A16 execution state was mutated")

    def freeze_native_inference(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        rows: Iterable[int] | int = (1,),
    ) -> int:
        """Irreversibly retain only native AWQ decode execution tensors.

        Save the canonical checkpoint before calling this method. The frozen
        module cannot be moved, cast, serialized, re-prepared, or used with a
        row count, dtype, or device outside the explicit frozen contract.

        Returns the number of canonical storage bytes released.
        """

        if self._native_only:
            raise RuntimeError("AWQW4A16Linear is already frozen for native inference")
        if self.training:
            raise RuntimeError("freeze_native_inference requires eval mode")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("native-only AWQW4A16Linear requires float16 or bfloat16")
        target_device = self._normalize_native_device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("freeze_native_inference requires CUDA")
        major, minor = torch.cuda.get_device_capability(target_device)
        if (major, minor) != (8, 9):
            raise RuntimeError(
                "native-only AWQW4A16Linear currently targets "
                f"sm_89, got sm_{major}{minor}"
            )
        raw_rows = (rows,) if isinstance(rows, int) else tuple(rows)
        normalized_rows = frozenset(int(value) for value in raw_rows)
        if not normalized_rows:
            raise ValueError("freeze_native_inference requires at least one row count")
        if any(value < 1 or value > 8 for value in normalized_rows):
            raise ValueError("native-only AWQ W4A16 rows must be between 1 and 8")
        for name in (
            "qweight",
            "canonical_qweight",
            "weight_scale",
            "weight_zero_point",
            "bias",
        ):
            tensor = self._buffers.get(name)
            if isinstance(tensor, torch.Tensor) and tensor.device != target_device:
                raise RuntimeError(
                    f"canonical {name} must already be on the target device"
                )

        from xqt.kernels.ops.gemm import (
            bind_awq_w4a16_decode,
            native_awq_w4a16_available,
        )

        if not native_awq_w4a16_available(build=False):
            raise RuntimeError("native AWQ W4A16 backend is unavailable")

        self._prepared_parameters.clear()
        self._bound_runners.clear()
        with torch.inference_mode(False), torch.no_grad():
            first_rows = min(normalized_rows)
            spec = self._spec(rows=first_rows, dtype=dtype, device=target_device)
            qweight, scales, scaled_zeros = self._prepared(dtype=dtype, spec=spec)
            qweight = qweight.contiguous()
            scales = scales.contiguous()
            scaled_zeros = scaled_zeros.contiguous()
            bias = self.bias
            bias_exec = (
                None
                if bias is None
                else bias.to(device=target_device, dtype=dtype).contiguous()
            )
            execution_tensors = (qweight, scales, scaled_zeros) + (
                () if bias_exec is None else (bias_exec,)
            )
            bound_runners: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
            for active_rows in sorted(normalized_rows):
                native_bound = bind_awq_w4a16_decode(
                    qweight,
                    scales,
                    scaled_zeros,
                    rows=active_rows,
                    input_features=self.input_features,
                    output_features=self.output_features,
                    dtype=dtype,
                    device=target_device,
                    bias=bias_exec,
                )

                def bound(
                    inputs: torch.Tensor,
                    *,
                    _native_bound: Callable[
                        [torch.Tensor], torch.Tensor
                    ] = native_bound,
                    _rows: int = active_rows,
                ) -> torch.Tensor:
                    self._validate_native_only_state()
                    if (
                        inputs.ndim != 2
                        or tuple(inputs.shape) != (_rows, self.input_features)
                        or inputs.dtype != dtype
                        or inputs.device != target_device
                    ):
                        raise XQTBackendError(
                            "bound AWQ W4A16 input disagrees with the static "
                            "shape/dtype/device"
                        )
                    return _native_bound(inputs)

                bound_runners[active_rows] = bound
        torch.cuda.synchronize(target_device)

        released_bytes = self._canonical_storage_bytes()
        state_signature = self._execution_state_signature(execution_tensors)
        self._prepared_parameters.clear()
        self._spec_cache.clear()
        self._bound_runners.clear()
        self._native_only_device = target_device
        self._native_only_dtype = dtype
        self._native_only_rows = normalized_rows
        self._native_only_tensors = execution_tensors
        self._native_only_state_signature = state_signature
        self._native_only_bound_runners = bound_runners
        self._native_only_released_bytes = released_bytes
        self._buffers["qweight"] = None
        self._buffers["canonical_qweight"] = None
        self._buffers["weight_scale"] = None
        self._buffers["weight_zero_point"] = None
        self._buffers["bias"] = None
        self._runtime_state_signature = ()
        self._native_only = True
        self._last_execution = {"implementation": "not_run", "native": False}
        return released_bytes

    def train(self, mode: bool = True) -> "AWQW4A16Linear":
        if self.native_only and mode:
            raise RuntimeError("native-only AWQW4A16Linear is inference-only")
        return super().train(mode)

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        if self._native_only:
            raise RuntimeError(
                "native-only AWQW4A16Linear cannot be serialized; "
                "save the canonical checkpoint before freezing"
            )
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self._native_only:
            raise RuntimeError(
                "native-only AWQW4A16Linear cannot load state; "
                "rebuild it from a canonical checkpoint"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def warmup(self, *, rows: int, dtype: torch.dtype) -> None:
        """Materialize immutable execution tensors before a timed inference run."""

        if rows < 1 or rows > 8:
            raise ValueError("AWQ W4A16 warmup rows must be between 1 and 8")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("AWQ W4A16 warmup dtype must be float16 or bfloat16")
        if self._native_only:
            self._validate_native_only_state()
            if dtype != self._native_only_dtype:
                raise RuntimeError(
                    "native-only AWQ W4A16 dtype does not match frozen state"
                )
            if rows not in self._native_only_rows:
                raise RuntimeError(f"native-only AWQ W4A16 rows={rows} were not frozen")
            return
        self._refresh_runtime_state()
        device = self.qweight.device
        if device.type != "cuda":
            raise XQTBackendError("AWQ W4A16 warmup requires CUDA-resident weights")
        spec = self._spec(rows=rows, dtype=dtype, device=device)
        self._prepared(dtype=dtype, spec=spec)

    def bind(
        self,
        *,
        rows: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Bind a same-shape hot callable with static weights and epilogue state."""

        if self._native_only:
            if dtype != self._native_only_dtype:
                raise RuntimeError(
                    "native-only AWQ W4A16 dtype does not match frozen state"
                )
            bound = self._native_only_bound_runners.get(int(rows))
            if bound is None:
                raise RuntimeError(f"native-only AWQ W4A16 rows={rows} were not frozen")
            return bound
        self._refresh_runtime_state()
        device = self.qweight.device
        key = (int(rows), dtype, str(device))
        cached = self._bound_runners.get(key)
        if cached is not None:
            return cached
        self.warmup(rows=rows, dtype=dtype)
        spec = self._spec(rows=rows, dtype=dtype, device=device)
        qweight, scales, scaled_zeros = self._prepared(dtype=dtype, spec=spec)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype).contiguous()
        from xqt.kernels.ops.gemm import (
            bind_awq_w4a16_decode,
        )

        native_bound = bind_awq_w4a16_decode(
            qweight,
            scales,
            scaled_zeros,
            rows=rows,
            input_features=self.input_features,
            output_features=self.output_features,
            dtype=dtype,
            device=device,
            bias=bias,
        )

        def bound(inputs: torch.Tensor) -> torch.Tensor:
            if (
                inputs.ndim != 2
                or tuple(inputs.shape) != (rows, self.input_features)
                or inputs.dtype != dtype
                or inputs.device != device
            ):
                raise XQTBackendError(
                    "bound AWQ W4A16 input disagrees with the static shape/dtype/device"
                )
            return native_bound(inputs)

        self._bound_runners[key] = bound
        return bound

    def bind_residual(
        self,
        *,
        rows: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor, torch.Tensor], None]:
        """Bind a decode GEMV whose epilogue adds into a residual row in place.

        The native kernel rounds the accumulator to the output dtype before the
        epilogue add, so ``residual += projection(inputs)`` keeps exactly the
        numerics of the unfused ``residual + projection(inputs)`` expression
        while removing one elementwise launch. ``residual`` is a
        ``[rows, output_features]`` row that is both the epilogue operand and
        the destination: every thread reads its own element before overwriting
        it.
        """

        if self._native_only:
            raise RuntimeError(
                "native-only AWQW4A16Linear cannot bind a residual epilogue"
            )
        if not 1 <= int(rows) <= 8:
            raise XQTBackendError("bound AWQ W4A16 residual requires 1 <= rows <= 8")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise XQTBackendError("bound AWQ W4A16 residual requires FP16 or BF16")
        self._refresh_runtime_state()
        device = self.qweight.device
        key = (int(rows), dtype, str(device))
        cached = self._residual_runners.get(key)
        if cached is not None:
            return cached
        self.warmup(rows=rows, dtype=dtype)
        spec = self._spec(rows=rows, dtype=dtype, device=device)
        qweight, scales, scaled_zeros = self._prepared(dtype=dtype, spec=spec)
        if self.bias is not None:
            raise XQTBackendError("bound AWQ W4A16 residual cannot fold a module bias")
        from xqt.kernels.ops._impl.cuda.awq_w4a16_sm89 import _load_extension

        decode_bias_out = _load_extension().decode_bias_out

        def bound(inputs: torch.Tensor, residual: torch.Tensor) -> None:
            if (
                inputs.ndim != 2
                or tuple(inputs.shape) != (rows, self.input_features)
                or inputs.dtype != dtype
                or inputs.device != device
            ):
                raise XQTBackendError(
                    "bound AWQ W4A16 residual input disagrees with the static "
                    "shape/dtype/device"
                )
            if (
                residual.ndim != 2
                or tuple(residual.shape) != (rows, self.output_features)
                or residual.dtype != dtype
                or residual.device != device
                or not residual.is_contiguous()
            ):
                raise XQTBackendError(
                    "bound AWQ W4A16 residual must be a contiguous "
                    f"[{rows},{self.output_features}] {dtype} row"
                )
            decode_bias_out(
                inputs,
                qweight,
                scales,
                scaled_zeros,
                residual.reshape(-1),
                residual,
            )

        self._residual_runners[key] = bound
        return bound

    def bind_w4a8(
        self,
        *,
        rows: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor]:
        """Bind W4A8 decode GEMV with INT8 activation input and output dtype."""
        if not 1 <= int(rows) <= 8:
            raise XQTBackendError("bound AWQ W4A8 requires 1 <= rows <= 8")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise XQTBackendError("bound AWQ W4A8 requires FP16 or BF16")
        self._refresh_runtime_state()
        device = self.qweight.device
        key = (int(rows), dtype, str(device))
        cached = self._w4a8_runners.get(key)
        if cached is not None:
            return cached
        self.warmup(rows=rows, dtype=dtype)
        spec = self._spec(rows=rows, dtype=dtype, device=device)
        qweight, scales, _ = self._prepared(dtype=dtype, spec=spec)
        from xqt.kernels.ops._impl.cuda.awq_w4a16_sm89 import bind_awq_w4a8_decode

        runner = bind_awq_w4a8_decode(
            qweight,
            scales,
            rows=rows,
            input_features=self.input_features,
            output_features=self.output_features,
            dtype=dtype,
            device=device,
        )
        self._w4a8_runners[key] = runner
        return runner

    def bind_residual_w4a8(
        self,
        *,
        rows: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor, torch.Tensor | float, torch.Tensor], None]:
        """Bind W4A8 decode GEMV with fused in-place residual add."""
        if not 1 <= int(rows) <= 8:
            raise XQTBackendError("bound AWQ W4A8 residual requires 1 <= rows <= 8")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise XQTBackendError("bound AWQ W4A8 residual requires FP16 or BF16")
        self._refresh_runtime_state()
        device = self.qweight.device
        key = (int(rows), dtype, str(device))
        cached = self._w4a8_residual_runners.get(key)
        if cached is not None:
            return cached
        self.warmup(rows=rows, dtype=dtype)
        spec = self._spec(rows=rows, dtype=dtype, device=device)
        qweight, scales, _ = self._prepared(dtype=dtype, spec=spec)
        from xqt.kernels.ops._impl.cuda.awq_w4a16_sm89 import bind_awq_w4a8_decode_bias

        runner = bind_awq_w4a8_decode_bias(
            qweight,
            scales,
            rows=rows,
            input_features=self.input_features,
            output_features=self.output_features,
            dtype=dtype,
            device=device,
        )
        self._w4a8_residual_runners[key] = runner
        return runner

    def forward_w4a8(
        self,
        inputs: torch.Tensor,
        scale_a: torch.Tensor | float,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Run native W4A8 decode with INT8 activation and scale."""
        if inputs.ndim != 2:
            inputs = inputs.reshape(-1, self.input_features)
        rows = int(inputs.shape[0])
        runner = self.bind_w4a8(rows=rows, dtype=output_dtype)
        return runner(inputs, scale_a)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("AWQ W4A16 native path requires eval mode")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "AWQ W4A16 native path requires no_grad or inference_mode"
            )
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "AWQ W4A16 input trailing dimension does not match input_features"
            )
        if inputs.dtype not in {torch.float16, torch.bfloat16}:
            raise XQTBackendError("AWQ W4A16 inputs must be float16 or bfloat16")
        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat = inputs.reshape(-1, self.input_features).contiguous()
        rows = int(flat.shape[0])
        if self._native_only:
            if flat.device != self._native_only_device:
                raise RuntimeError(
                    "native-only AWQ W4A16 input device does not match frozen state"
                )
            if flat.dtype != self._native_only_dtype:
                raise RuntimeError(
                    "native-only AWQ W4A16 input dtype does not match frozen state"
                )
        output = self.bind(rows=rows, dtype=flat.dtype)(flat)
        if self._last_execution.get("rows") != rows or self._last_execution.get(
            "dtype"
        ) != ("fp16" if flat.dtype == torch.float16 else "bf16"):
            metadata = sm89_awq_w4a16_metadata()
            metadata.update(
                {
                    "native": True,
                    "rows": rows,
                    "input_features": self.input_features,
                    "output_features": self.output_features,
                    "dtype": "fp16" if flat.dtype == torch.float16 else "bf16",
                    "bias": self._has_bias,
                    "native_only": self._native_only,
                    "native_only_rows": sorted(self._native_only_rows),
                    "native_only_device": (
                        None
                        if self._native_only_device is None
                        else str(self._native_only_device)
                    ),
                    "native_only_dtype": (
                        None
                        if self._native_only_dtype is None
                        else str(self._native_only_dtype)
                    ),
                    "native_only_released_bytes": self._native_only_released_bytes,
                }
            )
            self._last_execution = metadata
        if inputs.ndim == 2:
            return output
        return output.reshape(*original_shape, self.output_features)

    def execution_metadata(self) -> dict[str, Any]:
        """Return metadata for the most recent invocation."""

        metadata = dict(self._last_execution)
        native_only = self.native_only
        native_only_rows = self.__dict__.get("_native_only_rows", frozenset())
        native_only_device = self.__dict__.get("_native_only_device")
        native_only_dtype = self.__dict__.get("_native_only_dtype")
        metadata.update(
            {
                "native_only": native_only,
                "native_only_rows": sorted(native_only_rows),
                "native_only_device": (
                    None if native_only_device is None else str(native_only_device)
                ),
                "native_only_dtype": (
                    None if native_only_dtype is None else str(native_only_dtype)
                ),
                "native_only_released_bytes": self.__dict__.get(
                    "_native_only_released_bytes",
                    0,
                ),
            }
        )
        return metadata


__all__ = ["AWQW4A16Linear"]
