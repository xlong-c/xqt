"""Optional tensor visualizations for XQT analysis artifacts.

The plotting functions import matplotlib lazily so importing ``xqt.analysis``
does not require the optional visualization dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch

TensorInput: TypeAlias = torch.Tensor | Sequence[Sequence[float]]
TensorSource: TypeAlias = Literal["activation", "weight"]
ValueMode: TypeAlias = Literal["absolute", "signed"]
ReductionMode: TypeAlias = Literal["max", "mean"]


@dataclass(frozen=True)
class TensorSelection:
    """Select one activation or weight tensor from a model for plotting."""

    module_name: str
    source: TensorSource
    label: str | None = None
    row_axis: int = -2
    col_axis: int = -1
    matrix_index: int | tuple[int, ...] | None = 0
    value_mode: ValueMode = "absolute"

    def __post_init__(self) -> None:
        if not self.module_name:
            raise ValueError("module_name must not be empty")
        if self.source not in {"activation", "weight"}:
            raise ValueError(
                "source must be either 'activation' or 'weight', "
                f"got {self.source!r}"
            )

    @property
    def display_name(self) -> str:
        """Return the panel title used when no explicit label is supplied."""

        return self.label or f"{self.module_name} ({self.source})"


@dataclass(frozen=True)
class TensorPlotData:
    """Prepared matrix data and metadata used by the plotting helpers."""

    values: Any
    source_shape: tuple[int, ...]
    plotted_shape: tuple[int, int]
    reduced: bool
    value_mode: ValueMode


def _matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except ImportError as exc:
        raise ImportError(
            "3D tensor visualization requires matplotlib; "
            "install it with `pip install matplotlib` or `pip install 'xdl[xqt-viz]'`"
        ) from exc
    return matplotlib, plt, colors


def _as_matrix(
    tensor: TensorInput,
    *,
    row_axis: int = -2,
    col_axis: int = -1,
    matrix_index: int | tuple[int, ...] | None = 0,
) -> torch.Tensor:
    value = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
    if value.ndim < 2:
        raise ValueError(
            f"tensor must have at least two dimensions, got shape={tuple(value.shape)}"
        )

    normalized_row_axis = row_axis if row_axis >= 0 else value.ndim + row_axis
    normalized_col_axis = col_axis if col_axis >= 0 else value.ndim + col_axis
    if not 0 <= normalized_row_axis < value.ndim:
        raise ValueError(f"row_axis {row_axis} is out of range for ndim={value.ndim}")
    if not 0 <= normalized_col_axis < value.ndim:
        raise ValueError(f"col_axis {col_axis} is out of range for ndim={value.ndim}")
    if normalized_row_axis == normalized_col_axis:
        raise ValueError("row_axis and col_axis must be different")

    matrix = value.detach().to(dtype=torch.float32, device="cpu")
    matrix = matrix.movedim((normalized_row_axis, normalized_col_axis), (-2, -1))
    leading_shape = tuple(matrix.shape[:-2])
    if leading_shape:
        if matrix_index is None:
            matrix = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
            matrix = matrix.reshape(
                matrix.shape[0] * matrix.shape[1],
                matrix.shape[2],
            )
        else:
            if isinstance(matrix_index, int):
                index = (matrix_index,) * len(leading_shape)
            else:
                index = tuple(matrix_index)
            if len(index) != len(leading_shape):
                raise ValueError(
                    "matrix_index must contain one index per non-matrix dimension; "
                    f"expected {len(leading_shape)}, got {len(index)}"
                )
            matrix = matrix[index]
    return matrix.contiguous()


def _apply_value_mode(matrix: torch.Tensor, value_mode: ValueMode) -> torch.Tensor:
    if value_mode == "absolute":
        return matrix.abs()
    if value_mode == "signed":
        return matrix
    raise ValueError(f"unsupported value_mode: {value_mode!r}")


def _split_example_input(
    example_input: object,
) -> tuple[tuple[object, ...], Mapping[str, object] | None]:
    if isinstance(example_input, Mapping):
        return (), dict(example_input)
    if isinstance(example_input, tuple):
        return example_input, None
    if isinstance(example_input, list):
        return tuple(example_input), None
    return (example_input,), None


def _resolve_module(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    if module_name == "<root>":
        return model
    try:
        return model.get_submodule(module_name)
    except AttributeError as exc:
        raise KeyError(f"Module not found: {module_name!r}") from exc


def _module_weight_tensor(
    model: torch.nn.Module,
    module_name: str,
) -> torch.Tensor:
    module = _resolve_module(model, module_name)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.detach()
    dequantize_weight = getattr(module, "dequantize_weight", None)
    if callable(dequantize_weight):
        value = dequantize_weight()
        if isinstance(value, torch.Tensor):
            return value.detach()
    raise TypeError(
        f"Module {module_name!r} does not expose a tensor weight or "
        "dequantize_weight()"
    )


def _target_shape(rows: int, columns: int, max_bars: int) -> tuple[int, int]:
    if max_bars <= 0:
        raise ValueError(f"max_bars must be positive, got {max_bars}")
    if rows * columns <= max_bars:
        return rows, columns

    aspect = rows / columns
    target_rows = max(1, int((max_bars * aspect) ** 0.5))
    target_columns = max(1, max_bars // target_rows)
    while target_rows * target_columns > max_bars:
        if target_rows >= target_columns:
            target_rows -= 1
        else:
            target_columns -= 1
    return min(rows, target_rows), min(columns, target_columns)


def _pool_matrix(
    matrix: torch.Tensor,
    *,
    max_bars: int,
    reduction: ReductionMode,
) -> tuple[torch.Tensor, bool]:
    rows, columns = matrix.shape
    target_rows, target_columns = _target_shape(rows, columns, max_bars)
    if (target_rows, target_columns) == (rows, columns):
        return matrix, False
    if reduction not in {"max", "mean"}:
        raise ValueError(f"unsupported reduction: {reduction!r}")

    row_edges = torch.linspace(0, rows, target_rows + 1, dtype=torch.int64)
    column_edges = torch.linspace(0, columns, target_columns + 1, dtype=torch.int64)
    pooled = torch.empty((target_rows, target_columns), dtype=torch.float32)
    for row_index in range(target_rows):
        row_start = int(row_edges[row_index].item())
        row_end = max(row_start + 1, int(row_edges[row_index + 1].item()))
        for column_index in range(target_columns):
            column_start = int(column_edges[column_index].item())
            column_end = max(
                column_start + 1,
                int(column_edges[column_index + 1].item()),
            )
            block = matrix[row_start:row_end, column_start:column_end]
            if reduction == "max":
                pooled[row_index, column_index] = block.max()
            else:
                pooled[row_index, column_index] = block.mean()
    return pooled, True


def prepare_tensor_plot_data(
    tensor: TensorInput,
    *,
    row_axis: int = -2,
    col_axis: int = -1,
    matrix_index: int | tuple[int, ...] | None = 0,
    value_mode: ValueMode = "absolute",
    max_bars: int = 20_000,
    reduction: ReductionMode = "max",
) -> TensorPlotData:
    """Convert a tensor into finite, bounded 2D data for plotting.

    ``matrix_index`` selects non-matrix dimensions, for example the first
    batch item from an activation with shape ``[batch, tokens, hidden]``.
    Passing ``None`` concatenates all non-matrix dimensions into the row axis.
    Large matrices are reduced with deterministic 2D pooling.
    """

    matrix = _as_matrix(
        tensor,
        row_axis=row_axis,
        col_axis=col_axis,
        matrix_index=matrix_index,
    )
    matrix = _apply_value_mode(matrix, value_mode)
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    plotted, reduced = _pool_matrix(
        matrix,
        max_bars=max_bars,
        reduction=reduction,
    )
    return TensorPlotData(
        values=plotted.numpy(),
        source_shape=tuple(matrix.shape),
        plotted_shape=tuple(plotted.shape),
        reduced=reduced,
        value_mode=value_mode,
    )


def _draw_tensor_bar3d(
    axis: Any,
    values: Any,
    *,
    title: str | None,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    colormap: str,
    alpha: float,
    bar_gap: float,
    colorbar: bool,
    elev: float,
    azim: float,
    colors_module: Any,
    pyplot: Any,
) -> None:
    import numpy as np

    rows, columns = values.shape
    row_grid, column_grid = np.meshgrid(
        np.arange(rows, dtype=np.float32),
        np.arange(columns, dtype=np.float32),
        indexing="ij",
    )
    heights = np.asarray(values, dtype=np.float32)
    minimum = float(np.min(heights)) if heights.size else 0.0
    maximum = float(np.max(heights)) if heights.size else 0.0
    if minimum == maximum:
        maximum = minimum + 1.0
    normalization = colors_module.Normalize(vmin=minimum, vmax=maximum)
    colormap_instance = pyplot.get_cmap(colormap)
    bar_colors = colormap_instance(normalization(heights.ravel()))
    axis.bar3d(
        column_grid.ravel(),
        row_grid.ravel(),
        np.zeros(heights.size, dtype=np.float32),
        bar_gap,
        bar_gap,
        heights.ravel(),
        color=bar_colors,
        alpha=alpha,
        shade=True,
        linewidth=0.0,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_zlabel(zlabel)
    axis.view_init(elev=elev, azim=azim)
    axis.set_xlim(0, max(1, columns))
    axis.set_ylim(0, max(1, rows))
    if title:
        axis.set_title(title)
    if colorbar:
        scalar_map = pyplot.cm.ScalarMappable(
            norm=normalization,
            cmap=colormap_instance,
        )
        scalar_map.set_array(heights)
        pyplot.colorbar(scalar_map, ax=axis, pad=0.1, shrink=0.7)


def _render_tensor_panels(
    prepared: Mapping[str, TensorPlotData],
    output_path: str | Path,
    *,
    axis_labels: Mapping[str, tuple[str, str]] | None,
    zlabel: str,
    colormap: str,
    alpha: float,
    bar_gap: float,
    colorbar: bool,
    elev: float,
    azim: float,
    figsize: tuple[float, float] | None,
    dpi: int,
) -> Path:
    _, pyplot, colors = _matplotlib()
    panel_count = len(prepared)
    if figsize is None:
        figsize = (6.4 * panel_count, 5.4)
    figure, axes = pyplot.subplots(
        1,
        panel_count,
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    for axis, (name, plot_data) in zip(axes[0], prepared.items()):
        xlabel, ylabel = (axis_labels or {}).get(name, ("Column", "Row"))
        _draw_tensor_bar3d(
            axis,
            plot_data.values,
            title=name,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            colormap=colormap,
            alpha=alpha,
            bar_gap=bar_gap,
            colorbar=colorbar,
            elev=elev,
            azim=azim,
            colors_module=colors,
            pyplot=pyplot,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    pyplot.close(figure)
    return output


def plot_tensor_bar3d(
    tensor: TensorInput,
    output_path: str | Path,
    *,
    title: str | None = None,
    xlabel: str = "Column",
    ylabel: str = "Row",
    zlabel: str = "Value",
    row_axis: int = -2,
    col_axis: int = -1,
    matrix_index: int | tuple[int, ...] | None = 0,
    value_mode: ValueMode = "absolute",
    max_bars: int = 20_000,
    reduction: ReductionMode = "max",
    colormap: str = "viridis",
    alpha: float = 0.85,
    bar_gap: float = 0.8,
    colorbar: bool = True,
    elev: float = 28.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (7.0, 6.0),
    dpi: int = 160,
) -> Path:
    """Render one tensor as a 3D bar chart and save it as a raster image."""

    _, pyplot, colors = _matplotlib()
    plot_data = prepare_tensor_plot_data(
        tensor,
        row_axis=row_axis,
        col_axis=col_axis,
        matrix_index=matrix_index,
        value_mode=value_mode,
        max_bars=max_bars,
        reduction=reduction,
    )
    figure = pyplot.figure(figsize=figsize, dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    _draw_tensor_bar3d(
        axis,
        plot_data.values,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        colormap=colormap,
        alpha=alpha,
        bar_gap=bar_gap,
        colorbar=colorbar,
        elev=elev,
        azim=azim,
        colors_module=colors,
        pyplot=pyplot,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    pyplot.close(figure)
    return output


def plot_tensor_bar3d_panels(
    tensors: Mapping[str, TensorInput],
    output_path: str | Path,
    *,
    axis_labels: Mapping[str, tuple[str, str]] | None = None,
    zlabel: str = "Value",
    row_axis: int = -2,
    col_axis: int = -1,
    matrix_index: int | tuple[int, ...] | None = 0,
    value_mode: ValueMode = "absolute",
    max_bars: int = 20_000,
    reduction: ReductionMode = "max",
    colormap: str = "viridis",
    alpha: float = 0.85,
    bar_gap: float = 0.8,
    colorbar: bool = True,
    elev: float = 28.0,
    azim: float = -60.0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 160,
) -> Path:
    """Render named tensors as side-by-side 3D bar-chart panels."""

    if not tensors:
        raise ValueError("tensors must contain at least one named tensor")
    prepared = {
        name: prepare_tensor_plot_data(
            tensor,
            row_axis=row_axis,
            col_axis=col_axis,
            matrix_index=matrix_index,
            value_mode=value_mode,
            max_bars=max_bars,
            reduction=reduction,
        )
        for name, tensor in tensors.items()
    }
    return _render_tensor_panels(
        prepared,
        output_path,
        axis_labels=axis_labels,
        zlabel=zlabel,
        colormap=colormap,
        alpha=alpha,
        bar_gap=bar_gap,
        colorbar=colorbar,
        elev=elev,
        azim=azim,
        figsize=figsize,
        dpi=dpi,
    )


def plot_model_tensor_selections_bar3d(
    model: torch.nn.Module,
    example_input: object,
    selections: Sequence[TensorSelection],
    output_path: str | Path,
    *,
    axis_labels: Mapping[str, tuple[str, str]] | None = None,
    zlabel: str = "Value",
    max_bars: int = 20_000,
    reduction: ReductionMode = "max",
    colormap: str = "viridis",
    alpha: float = 0.85,
    bar_gap: float = 0.8,
    colorbar: bool = True,
    elev: float = 28.0,
    azim: float = -60.0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 160,
) -> Path:
    """Capture selected model activations and weights as 3D panels.

    Each selection names a module and chooses either its forward ``activation``
    or its ``weight``. The model family and panel semantics stay outside XQT.
    """

    if not selections:
        raise ValueError("selections must contain at least one TensorSelection")

    activation_names = list(
        dict.fromkeys(
            selection.module_name
            for selection in selections
            if selection.source == "activation"
        )
    )
    activation_outputs: Mapping[str, Any] = {}
    if activation_names:
        from xqt.kernels.nn.fixtures.hooks import collect_module_outputs

        forward_args, forward_kwargs = _split_example_input(example_input)
        activation_outputs = collect_module_outputs(
            model,
            *forward_args,
            module_names=activation_names,
            forward_kwargs=forward_kwargs,
        )

    prepared: dict[str, TensorPlotData] = {}
    for selection in selections:
        if selection.display_name in prepared:
            raise ValueError(
                f"duplicate tensor selection label: {selection.display_name!r}"
            )
        if selection.source == "activation":
            tensor = activation_outputs.get(selection.module_name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Module {selection.module_name!r} did not produce a tensor "
                    f"activation; got {type(tensor).__name__}"
                )
        else:
            tensor = _module_weight_tensor(model, selection.module_name)
        prepared[selection.display_name] = prepare_tensor_plot_data(
            tensor,
            row_axis=selection.row_axis,
            col_axis=selection.col_axis,
            matrix_index=selection.matrix_index,
            value_mode=selection.value_mode,
            max_bars=max_bars,
            reduction=reduction,
        )

    return _render_tensor_panels(
        prepared,
        output_path,
        axis_labels=axis_labels,
        zlabel=zlabel,
        colormap=colormap,
        alpha=alpha,
        bar_gap=bar_gap,
        colorbar=colorbar,
        elev=elev,
        azim=azim,
        figsize=figsize,
        dpi=dpi,
    )


__all__ = [
    "TensorSelection",
    "TensorPlotData",
    "plot_model_tensor_selections_bar3d",
    "plot_tensor_bar3d",
    "plot_tensor_bar3d_panels",
    "prepare_tensor_plot_data",
]
