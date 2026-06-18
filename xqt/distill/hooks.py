"""Forward hook helpers for teacher/student feature capture."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import AbstractContextManager
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from xqt.eval.compare import TensorDiff, compare_tensors, summarize_tensor


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


@dataclass
class FeatureAlignmentRecord:
    """Teacher/student intermediate feature alignment for one module pair."""

    teacher_name: str
    student_name: str
    teacher_module_type: str
    student_module_type: str
    diff: TensorDiff
    teacher_summary: dict[str, object]
    student_summary: dict[str, object]
    recommendation: Optional[str] = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Convert the record to a plain dictionary."""

        return {
            "teacher_name": self.teacher_name,
            "student_name": self.student_name,
            "teacher_module_type": self.teacher_module_type,
            "student_module_type": self.student_module_type,
            "diff": self.diff.to_dict(),
            "teacher_summary": dict(self.teacher_summary),
            "student_summary": dict(self.student_summary),
            "recommendation": self.recommendation,
            "tags": list(self.tags),
        }


def _resolve_module_pairs(
    teacher_model: nn.Module,
    student_model: nn.Module,
    *,
    module_names: Optional[Sequence[str]] = None,
    module_pairs: Optional[Mapping[str, str] | Sequence[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    if module_pairs is not None:
        if isinstance(module_pairs, Mapping):
            return [(str(teacher_name), str(student_name)) for teacher_name, student_name in module_pairs.items()]
        return [(str(teacher_name), str(student_name)) for teacher_name, student_name in module_pairs]
    if module_names is not None:
        return [(str(name), str(name)) for name in module_names]

    student_names = {name for name, _ in student_model.named_modules() if name}
    pairs = [
        (name, name)
        for name, _ in teacher_model.named_modules()
        if name and name in student_names
    ]
    if pairs:
        return pairs
    return [(ROOT_MODULE_NAME, ROOT_MODULE_NAME)]


def _invalid_diff(
    teacher_output: torch.Tensor,
    student_output: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    message: str,
) -> TensorDiff:
    return TensorDiff(
        max_abs=0.0,
        mean_abs=0.0,
        mean_squared=0.0,
        relative_error=None,
        cosine_similarity=None,
        correlation=None,
        argmax_mismatch_rate=None,
        allclose=False,
        atol=atol,
        rtol=rtol,
        valid=False,
        message=message,
        reference_summary=summarize_tensor(teacher_output),
        candidate_summary=summarize_tensor(student_output),
        details=None,
    )


def analyze_feature_alignment(
    teacher_model: nn.Module,
    student_model: nn.Module,
    *forward_args: Any,
    module_names: Optional[Sequence[str]] = None,
    module_pairs: Optional[Mapping[str, str] | Sequence[tuple[str, str]]] = None,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> list[FeatureAlignmentRecord]:
    """Compare teacher/student intermediate features on matched module pairs."""

    pairs = _resolve_module_pairs(
        teacher_model,
        student_model,
        module_names=module_names,
        module_pairs=module_pairs,
    )
    if not pairs:
        return []

    teacher_names = [teacher_name for teacher_name, _ in pairs]
    student_names = [student_name for _, student_name in pairs]
    teacher_outputs = collect_module_outputs(
        teacher_model,
        *forward_args,
        module_names=teacher_names,
        forward_kwargs=forward_kwargs,
    )
    student_outputs = collect_module_outputs(
        student_model,
        *forward_args,
        module_names=student_names,
        forward_kwargs=forward_kwargs,
    )

    teacher_modules = dict(teacher_model.named_modules())
    student_modules = dict(student_model.named_modules())
    records: list[FeatureAlignmentRecord] = []
    for teacher_name, student_name in pairs:
        teacher_output = teacher_outputs.get(teacher_name)
        student_output = student_outputs.get(student_name)
        if not isinstance(teacher_output, torch.Tensor) or not isinstance(student_output, torch.Tensor):
            continue

        teacher_module = teacher_modules[_resolve_module_lookup_name(teacher_name)]
        student_module = student_modules[_resolve_module_lookup_name(student_name)]
        recommendation: Optional[str] = None
        tags: list[str] = []
        if teacher_output.shape != student_output.shape:
            diff = _invalid_diff(
                teacher_output,
                student_output,
                atol=atol,
                rtol=rtol,
                message=(
                    "shape mismatch: "
                    f"{tuple(teacher_output.shape)} vs {tuple(student_output.shape)}"
                ),
            )
            recommendation = "review_layer_mapping"
            tags.append("shape_mismatch")
        else:
            diff = compare_tensors(
                teacher_output,
                student_output,
                atol=atol,
                rtol=rtol,
            )
            if diff.mean_abs > atol * 10.0:
                recommendation = "increase_feature_kd_weight"
                tags.append("high_feature_error")
            if diff.cosine_similarity is not None and diff.cosine_similarity < 0.95:
                if recommendation is None:
                    recommendation = "review_feature_alignment"
                tags.append("low_cosine")

        teacher_summary = (
            diff.reference_summary.to_dict()
            if diff.reference_summary is not None
            else summarize_tensor(teacher_output).to_dict()
        )
        student_summary = (
            diff.candidate_summary.to_dict()
            if diff.candidate_summary is not None
            else summarize_tensor(student_output).to_dict()
        )
        records.append(
            FeatureAlignmentRecord(
                teacher_name=teacher_name,
                student_name=student_name,
                teacher_module_type=type(teacher_module).__name__,
                student_module_type=type(student_module).__name__,
                diff=diff,
                teacher_summary=teacher_summary,
                student_summary=student_summary,
                recommendation=recommendation,
                tags=tuple(tags),
            )
        )

    records.sort(
        key=lambda record: (
            record.diff.valid,
            -record.diff.max_abs,
        )
    )
    return records


__all__ = [
    "FeatureAlignmentRecord",
    "ROOT_MODULE_NAME",
    "analyze_feature_alignment",
    "ModuleOutputCapture",
    "capture_module_outputs",
    "collect_module_outputs",
]
