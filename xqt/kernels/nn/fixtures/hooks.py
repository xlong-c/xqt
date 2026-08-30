"""Forward hook helpers for model output analysis."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

ROOT_MODULE_NAME = "<root>"


def _resolve_module_lookup_name(name: str) -> str:
    return "" if name == ROOT_MODULE_NAME else name


def _normalize_output(output: Any, *, unwrap_tuple: bool, to_cpu: bool) -> Any:
    if unwrap_tuple and isinstance(output, (tuple, list)):
        output = output[0] if output else output
    if isinstance(output, torch.Tensor):
        tensor = output.detach()
        return tensor.cpu() if to_cpu else tensor
    return output


class ModuleOutputCapture(AbstractContextManager["ModuleOutputCapture"]):
    """Context manager that records outputs from named submodules."""

    def __init__(
        self,
        model: nn.Module,
        module_names: Sequence[str],
        *,
        to_cpu: bool = True,
        unwrap_tuple: bool = True,
    ) -> None:
        self.model = model
        self.module_names = list(module_names)
        self.to_cpu = to_cpu
        self.unwrap_tuple = unwrap_tuple
        self.outputs: Dict[str, Any] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "ModuleOutputCapture":
        modules = dict(self.model.named_modules())
        missing = [
            name
            for name in self.module_names
            if _resolve_module_lookup_name(name) not in modules
        ]
        if missing:
            raise KeyError(f"Modules not found: {missing}")

        for name in self.module_names:
            module = modules[_resolve_module_lookup_name(name)]
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self.outputs[name] = _normalize_output(
                output, unwrap_tuple=self.unwrap_tuple, to_cpu=self.to_cpu
            )

        return hook

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        while self._handles:
            handle = self._handles.pop()
            handle.remove()


def capture_module_outputs(
    model: nn.Module,
    module_names: Sequence[str],
    *forward_args: Any,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    to_cpu: bool = True,
    unwrap_tuple: bool = True,
) -> Dict[str, Any]:
    """Run a model once and capture the named module outputs."""

    kwargs = dict(forward_kwargs or {})
    with ModuleOutputCapture(
        model,
        module_names,
        to_cpu=to_cpu,
        unwrap_tuple=unwrap_tuple,
    ) as capture:
        with torch.no_grad():
            model(*forward_args, **kwargs)
    return capture.outputs


def collect_module_outputs(
    model: nn.Module,
    *forward_args: Any,
    module_names: Sequence[str],
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    to_cpu: bool = True,
    unwrap_tuple: bool = True,
) -> Dict[str, Any]:
    """Alias for `capture_module_outputs` with a keyword-only module list."""

    return capture_module_outputs(
        model,
        module_names,
        *forward_args,
        forward_kwargs=forward_kwargs,
        to_cpu=to_cpu,
        unwrap_tuple=unwrap_tuple,
    )


__all__ = [
    "ModuleOutputCapture",
    "ROOT_MODULE_NAME",
    "capture_module_outputs",
    "collect_module_outputs",
]
