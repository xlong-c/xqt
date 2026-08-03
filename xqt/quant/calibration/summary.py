"""Shared calibration input summary helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import torch

from xqt.core.inputs import extract_model_inputs
from xqt.export.input_utils import default_input_names, split_example_input


def _as_tensor_record(
    batch: Any,
    *,
    input_names: Optional[Sequence[str]] = None,
) -> dict[str, torch.Tensor]:
    inputs = extract_model_inputs(
        batch,
        expected_input_count=len(input_names) if input_names is not None else None,
    )
    resolved_input_names = list(input_names or default_input_names(inputs))
    normalized = split_example_input(inputs)
    if normalized.kwargs:
        missing = [name for name in resolved_input_names if name not in normalized.kwargs]
        if missing:
            raise ValueError(
                "calibration input_names are missing from mapping input: "
                f"{missing}"
            )
        values = {
            name: normalized.kwargs[name]
            for name in resolved_input_names
        }
    else:
        if len(normalized.args) != len(resolved_input_names):
            raise ValueError("calibration input arity must match input_names")
        values = dict(zip(resolved_input_names, normalized.args))
    invalid = [name for name, value in values.items() if not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(
            "calibration inputs must be torch.Tensor values: "
            f"{', '.join(sorted(invalid))}"
        )
    return {name: value.detach() for name, value in values.items()}


def build_calibration_summary(
    calibration_data: Iterable[Any],
    *,
    input_names: Optional[Sequence[str]] = None,
    sample_limit: Optional[int] = None,
    calibrator_type: str = "IterableCalibrationSummary",
    observer_type: Optional[str] = None,
) -> dict[str, Any]:
    """Summarize calibration batches into a backend-agnostic metadata payload."""

    records = [
        _as_tensor_record(batch, input_names=input_names)
        for index, batch in enumerate(calibration_data)
        if sample_limit is None or index < sample_limit
    ]
    resolved_input_names: list[str] = []
    shapes: dict[str, list[list[int]]] = {}
    dtypes: dict[str, list[str]] = {}
    if records:
        resolved_input_names = sorted(records[0].keys())
    for record in records:
        for name, value in record.items():
            shapes.setdefault(name, []).append(list(value.shape))
            dtypes.setdefault(name, []).append(str(value.dtype).removeprefix("torch."))
    return {
        "input_names": resolved_input_names,
        "sample_count": len(records),
        "batch_count": len(records),
        "shapes": {
            name: shape_list[: min(3, len(shape_list))]
            for name, shape_list in shapes.items()
        },
        "dtypes": {
            name: sorted(set(dtype_list))
            for name, dtype_list in dtypes.items()
        },
        "input_signature": {
            name: {
                "shapes": shape_list[: min(3, len(shape_list))],
                "dtypes": sorted(set(dtypes.get(name, []))),
            }
            for name, shape_list in shapes.items()
        },
        "calibrator_type": calibrator_type,
        "observer_type": observer_type or calibrator_type,
    }


__all__ = ["build_calibration_summary"]
