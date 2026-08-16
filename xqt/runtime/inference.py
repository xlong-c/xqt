"""Semantic inference adapters over file-based runtime runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from xqt.contracts.inference import InferenceContract
from xqt.core.errors import XQTConfigError


class InferenceAdapter(Protocol):
    """Protocol implemented by model-side semantic adapters."""

    name: str

    def preprocess(self, request: Any, contract: InferenceContract) -> Any:
        """Convert a semantic request into runtime inputs."""

    def postprocess(
        self,
        outputs: Sequence[Any],
        contract: InferenceContract,
    ) -> Any:
        """Convert runtime outputs into a semantic result."""


def _entry_value(
    request: Any,
    entry: Mapping[str, Any] | None,
    *,
    default_key: str | None = None,
) -> Any:
    if not isinstance(request, Mapping):
        return request
    if entry is not None:
        for key in (entry.get("semantic"), entry.get("name")):
            if isinstance(key, str) and key in request:
                return request[key]
    if default_key is not None and default_key in request:
        return request[default_key]
    if len(request) == 1:
        return next(iter(request.values()))
    return request


def _physical_input_name(contract: InferenceContract) -> str | None:
    if not contract.inputs:
        return None
    raw_name = contract.inputs[0].get("name")
    return str(raw_name) if isinstance(raw_name, str) and raw_name else None


def _output_key(entry: Mapping[str, Any], index: int) -> str:
    for key in ("semantic", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return f"output_{index}"


def _named_outputs(
    outputs: Sequence[Any],
    contract: InferenceContract,
) -> dict[str, Any]:
    if not contract.outputs:
        return {}
    if len(outputs) != len(contract.outputs):
        raise ValueError(
            "inference output count does not match the contract: "
            f"expected {len(contract.outputs)}, got {len(outputs)}"
        )
    return {
        _output_key(entry, index): value
        for index, (entry, value) in enumerate(zip(contract.outputs, outputs))
    }


class TensorInferenceAdapter:
    """Pass tensor inputs through and normalize named outputs."""

    name = "tensor"

    def preprocess(self, request: Any, contract: InferenceContract) -> Any:
        if not isinstance(request, Mapping) or not contract.inputs:
            return request
        resolved: dict[str, Any] = {}
        missing: list[str] = []
        for entry in contract.inputs:
            physical_name = entry.get("name")
            if not isinstance(physical_name, str) or not physical_name:
                continue
            if physical_name in request:
                value = request[physical_name]
            else:
                semantic_name = entry.get("semantic")
                if isinstance(semantic_name, str) and semantic_name in request:
                    value = request[semantic_name]
                else:
                    missing.append(physical_name)
                    continue
            if isinstance(value, Mapping):
                missing.append(physical_name)
                continue
            resolved[physical_name] = value
        if missing:
            raise ValueError(f"inference request is missing semantic inputs: {missing}")
        return resolved or request

    def postprocess(
        self,
        outputs: Sequence[Any],
        contract: InferenceContract,
    ) -> Any:
        named = _named_outputs(outputs, contract)
        return named if named else list(outputs)


def _as_image_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, Image.Image):
        value = np.array(value)
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(np.array(value, copy=True))
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "vision.classification input must be a torch.Tensor, numpy.ndarray, "
            "or PIL.Image.Image"
        )
    tensor = value.detach()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim == 3:
        if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[1] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "vision.classification input must have shape HxW, HxWxC, CxHxW, or NCHW"
        )
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32) / 255.0
    else:
        tensor = tensor.to(torch.float32)
    if tensor.shape[1] == 4:
        tensor = tensor[:, :3]
    if tensor.shape[1] not in (1, 3):
        raise ValueError(
            "vision.classification input must have one or three channels; "
            f"got {tensor.shape[1]}"
        )
    return tensor


def _pair_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("image resize size must be positive")
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError("image resize dimensions must be positive")
        return (height, width)
    raise ValueError("image resize size must be an int or a two-item sequence")


def _float_sequence(value: Any, *, name: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of numbers")
    return [float(item) for item in value]


def _target_dtype(value: Any) -> torch.dtype:
    if value is None:
        return torch.float32
    names = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized = str(value).lower()
    if normalized not in names:
        raise ValueError(f"unsupported image input dtype: {value!r}")
    return names[normalized]


class ImageClassificationAdapter(TensorInferenceAdapter):
    """Built-in adapter for image classification tensor preprocessing."""

    name = "vision.classification"

    def preprocess(self, request: Any, contract: InferenceContract) -> Any:
        entry = contract.inputs[0] if contract.inputs else None
        image = _entry_value(request, entry, default_key="image")
        if isinstance(image, Mapping):
            raise ValueError("vision.classification request must contain one image")
        tensor = _as_image_tensor(image)
        config = dict(contract.config or {})
        resize = _pair_size(config.get("resize", config.get("size")))
        if resize is not None and tuple(tensor.shape[-2:]) != resize:
            tensor = F.interpolate(
                tensor,
                size=resize,
                mode="bilinear",
                align_corners=False,
            )
        scale = config.get("scale")
        if scale is not None:
            tensor = tensor * float(scale)
        mean = _float_sequence(config.get("mean"), name="mean")
        std = _float_sequence(config.get("std"), name="std")
        if mean is not None or std is not None:
            if mean is None or std is None or len(mean) != len(std):
                raise ValueError("mean and std must be sequences of equal length")
            if len(mean) != tensor.shape[1]:
                raise ValueError(
                    "mean/std channel count must match image channels; "
                    f"got {len(mean)} and {tensor.shape[1]}"
                )
            mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, -1, 1, 1)
            std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
            if torch.any(std_tensor == 0):
                raise ValueError("std values must be non-zero")
            tensor = (tensor - mean_tensor) / std_tensor
        layout = str(config.get("layout", entry.get("layout", "NCHW"))).upper()
        if layout == "NHWC":
            tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor.to(_target_dtype(config.get("dtype", entry.get("dtype"))))
        physical_name = _physical_input_name(contract)
        return {physical_name: tensor} if physical_name is not None else tensor

    def postprocess(
        self,
        outputs: Sequence[Any],
        contract: InferenceContract,
    ) -> Any:
        named = _named_outputs(outputs, contract)
        if not named:
            return list(outputs)
        logits = named.get("logits")
        if not isinstance(logits, (torch.Tensor, np.ndarray)):
            logits = next(
                (
                    value
                    for value in named.values()
                    if isinstance(value, (torch.Tensor, np.ndarray))
                ),
                None,
            )
        if not isinstance(logits, (torch.Tensor, np.ndarray)):
            return named
        logits_tensor = (
            logits
            if isinstance(logits, torch.Tensor)
            else torch.from_numpy(np.asarray(logits))
        )
        probabilities = torch.softmax(logits_tensor, dim=-1)
        class_ids = torch.argmax(logits_tensor, dim=-1)
        if isinstance(logits, np.ndarray):
            probabilities = probabilities.numpy()
            class_ids = class_ids.numpy()
        return {
            **named,
            "logits": logits,
            "probabilities": probabilities,
            "class_ids": class_ids,
        }


@dataclass(frozen=True, slots=True)
class _AdapterRegistration:
    factory: Callable[[], InferenceAdapter]
    version: str


_BUILTIN_ADAPTERS: dict[str, _AdapterRegistration] = {
    TensorInferenceAdapter.name: _AdapterRegistration(
        factory=TensorInferenceAdapter,
        version="1",
    ),
    ImageClassificationAdapter.name: _AdapterRegistration(
        factory=ImageClassificationAdapter,
        version="1",
    ),
}


def register_inference_adapter(
    name: str,
    factory: Callable[[], InferenceAdapter],
    *,
    version: str = "1",
    replace: bool = False,
) -> None:
    """Register one reusable model-family adapter factory."""

    normalized_name = name.strip() if isinstance(name, str) else ""
    if not normalized_name:
        raise XQTConfigError("inference adapter name must be a non-empty string")
    if not callable(factory):
        raise XQTConfigError(
            f"inference adapter factory for {normalized_name!r} must be callable"
        )
    if not isinstance(version, str) or not version:
        raise XQTConfigError(
            f"inference adapter version for {normalized_name!r} "
            "must be a non-empty string"
        )
    if normalized_name in _BUILTIN_ADAPTERS and not replace:
        raise XQTConfigError(
            f"inference adapter {normalized_name!r} is already registered"
        )
    _BUILTIN_ADAPTERS[normalized_name] = _AdapterRegistration(
        factory=factory,
        version=version,
    )


def inference_adapter_names() -> tuple[str, ...]:
    """Return the registered adapter names in stable order."""

    return tuple(sorted(_BUILTIN_ADAPTERS))


def create_inference_adapter(
    contract: InferenceContract,
    *,
    adapter: InferenceAdapter | None = None,
) -> InferenceAdapter:
    """Resolve a built-in adapter or accept a caller-owned adapter."""

    if adapter is not None:
        return adapter
    registration = _BUILTIN_ADAPTERS.get(contract.adapter)
    if registration is None:
        supported = ", ".join(inference_adapter_names())
        raise XQTConfigError(
            f"unsupported inference adapter {contract.adapter!r}; "
            f"supported adapters: {supported}"
        )
    if contract.adapter_version != registration.version:
        raise XQTConfigError(
            f"inference adapter {contract.adapter!r} does not support "
            f"contract version {contract.adapter_version!r}; "
            f"registered version is {registration.version!r}"
        )
    return registration.factory()


class InferenceSession:
    """Semantic model-package session over a low-level runtime runner."""

    def __init__(
        self,
        runner: Callable[[Any], Sequence[Any]],
        contract: InferenceContract,
        adapter: InferenceAdapter,
    ) -> None:
        self.runner = runner
        self.contract = contract
        self.adapter = adapter

    def run_tensors(self, inputs: Any) -> Sequence[Any]:
        """Run already prepared tensor inputs without semantic processing."""

        return self.runner(inputs)

    def predict(self, request: Any) -> Any:
        """Preprocess a semantic request, run it, and normalize the outputs."""

        inputs = self.adapter.preprocess(request, self.contract)
        outputs = self.runner(inputs)
        return self.adapter.postprocess(outputs, self.contract)


__all__ = [
    "ImageClassificationAdapter",
    "InferenceAdapter",
    "InferenceSession",
    "TensorInferenceAdapter",
    "create_inference_adapter",
    "inference_adapter_names",
    "register_inference_adapter",
]
