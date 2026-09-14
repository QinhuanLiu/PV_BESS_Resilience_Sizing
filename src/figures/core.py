from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import math
import os

import numpy as np
import pandas as pd


RESULT_EXPORT_FORMATS = ("png", "svg", "pdf")
NATURE_PALETTE = {
    "blue": "#0F4D92",
    "blue_light": "#7AA6C2",
    "teal": "#42949E",
    "teal_light": "#9FC7C9",
    "red": "#B64342",
    "red_light": "#D9938D",
    "gold": "#D6A21E",
    "purple": "#8C6BB1",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}
PROBABILITY_UNIT_TYPES = {"probability", "fraction", "lolp", "ccp", "ratio"}
INTEGER_UNIT_TYPES = {"cost", "capacity", "energy", "ue", "money", "power", "duration"}


class FigureLayoutError(ValueError):
    """Raised when a figure fails publication layout quality checks."""


class FigureDependencyError(RuntimeError):
    """Raised when an optional figure dependency is required but unavailable."""


def matplotlib_context(out: Path | str | None = None) -> tuple[Any, Any]:
    """Return a non-interactive Matplotlib context configured for file export."""
    cache_dir = Path(out).parent / ".matplotlib_cache" if out else Path("/private/tmp/pv_bess_resilience_matplotlib_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return mpl, plt


def apply_nature_style(mpl: Any) -> None:
    """Apply the shared Nature-style plotting defaults used by this project."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
        }
    )


def save_publication_figure(
    fig: Any,
    base_path: Path | str,
    formats: Sequence[str] = RESULT_EXPORT_FORMATS,
    dpi: int = 600,
) -> dict[str, Path]:
    """Validate and export a publication figure in one or more formats."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    assert_publication_layout(fig)
    paths: dict[str, Path] = {}
    for suffix in formats:
        normalized = str(suffix).lower().lstrip(".")
        if normalized not in RESULT_EXPORT_FORMATS:
            raise ValueError(f"Unsupported publication export format: {suffix}")
        path = base.with_suffix(f".{normalized}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
        if normalized == "png":
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(path, **save_kwargs)
        if not path.exists() or path.stat().st_size == 0:
            raise FigureLayoutError(f"Figure export failed or produced an empty file: {path}")
        paths[f"{base.name}_{normalized}"] = path
    return paths


def write_source_data(base: Path | str, df: pd.DataFrame) -> Path:
    path = Path(base).with_name(Path(base).name + "_Source_Data.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _matplotlib(out: Path | str | None = None) -> tuple[Any, Any]:
    return matplotlib_context(out)


def _style(mpl: Any) -> None:
    apply_nature_style(mpl)


def _save_all(fig: Any, base: Path | str) -> dict[str, Path]:
    """Legacy-compatible export helper used by existing publication modules."""
    base_path = Path(base)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_legacy_publication_layout(fig)
    paths = {
        "png": base_path.with_suffix(".png"),
        "svg": base_path.with_suffix(".svg"),
        "pdf": base_path.with_suffix(".pdf"),
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.08)
    for path in paths.values():
        if not path.exists() or path.stat().st_size == 0:
            raise FigureLayoutError(f"Figure export failed or produced an empty file: {path}")
    return {f"{base_path.name}_{suffix}": path for suffix, path in paths.items()}


def _write_source_data(base: Path | str, df: pd.DataFrame) -> Path:
    return write_source_data(base, df)


def assert_publication_layout(fig: Any, tolerance: float = 2.0) -> None:
    """Fail fast when axes, labels, legends, or colorbars are clipped or overlapping."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    for index, ax in enumerate(fig.axes, start=1):
        box = ax.get_tightbbox(renderer)
        if box is None or box.width <= 0 or box.height <= 0:
            raise FigureLayoutError(f"Axes {index} produced an empty layout box")
        _assert_box_inside(fig_box, box, f"axes {index}", tolerance)
        _assert_text_artist_inside(fig_box, ax.title, renderer, f"axes {index} title", tolerance)
        _assert_text_artist_inside(fig_box, ax.xaxis.label, renderer, f"axes {index} x-label", tolerance)
        _assert_text_artist_inside(fig_box, ax.yaxis.label, renderer, f"axes {index} y-label", tolerance)
        legend = ax.get_legend()
        if legend is not None:
            _assert_legend_layout(fig_box, ax, legend, renderer, index, tolerance)
    for index, legend in enumerate(getattr(fig, "legends", []), start=1):
        legend_box = legend.get_window_extent(renderer)
        if legend_box is None or legend_box.width <= 0 or legend_box.height <= 0:
            raise FigureLayoutError(f"Figure legend {index} produced an empty layout box")
        _assert_box_inside(fig_box, legend_box, f"figure legend {index}", tolerance)


def _assert_legacy_publication_layout(fig: Any) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    tolerance = 2.0
    for index, ax in enumerate(fig.axes, start=1):
        box = ax.get_tightbbox(renderer)
        if box is None or box.width <= 0 or box.height <= 0:
            raise FigureLayoutError(f"Axes {index} produced an empty layout box")
        _assert_box_inside(fig_box, box, f"axes {index}", tolerance)
        _assert_text_artist_inside(fig_box, ax.title, renderer, f"axes {index} title", tolerance)
        _assert_text_artist_inside(fig_box, ax.xaxis.label, renderer, f"axes {index} x-label", tolerance)
        _assert_text_artist_inside(fig_box, ax.yaxis.label, renderer, f"axes {index} y-label", tolerance)
        legend = ax.get_legend()
        if legend is not None:
            legend_box = legend.get_window_extent(renderer)
            if legend_box is None or legend_box.width <= 0 or legend_box.height <= 0:
                raise FigureLayoutError(f"Axes {index} legend produced an empty layout box")
            _assert_box_inside(fig_box, legend_box, f"axes {index} legend", tolerance)


def nice_display_ticks(values: Iterable[float], unit_type: str = "generic", target_count: int = 5) -> list[float]:
    """Return human-readable ticks for colorbars and quantitative legends."""
    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("nice_display_ticks requires at least one finite value")
    unit = unit_type.strip().lower()
    if unit in PROBABILITY_UNIT_TYPES:
        return _probability_ticks(float(array.min()), float(array.max()))
    lower = float(array.min())
    upper = float(array.max())
    if math.isclose(lower, upper):
        return [round(lower, 6)]
    integerish = unit in INTEGER_UNIT_TYPES or max(abs(lower), abs(upper)) >= 10
    return _nice_numeric_ticks(lower, upper, max(2, int(target_count)), integerish=integerish)


def format_colorbar(
    colorbar: Any,
    values: Iterable[float],
    label: str,
    unit_type: str = "generic",
    target_count: int = 5,
) -> Any:
    """Apply project tick and label rules to a Matplotlib colorbar."""
    ticks = nice_display_ticks(values, unit_type=unit_type, target_count=target_count)
    colorbar.set_ticks(ticks)
    colorbar.set_label(label, fontsize=6.2)
    if unit_type.strip().lower() in PROBABILITY_UNIT_TYPES:
        colorbar.set_ticklabels([f"{tick * 100:.0f}%" for tick in ticks])
    else:
        colorbar.set_ticklabels([_format_tick(tick) for tick in ticks])
    colorbar.ax.tick_params(labelsize=6.0, length=2.2, width=0.6)
    return colorbar


def plot_smooth_line(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    samples: int = 200,
    show_points: bool = False,
    marker: str | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot a shape-preserving smooth trend line without exposing raw points by default."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("plot_smooth_line requires x and y with the same length")
    if x_values.size < 3:
        raise ValueError("plot_smooth_line requires at least three points for smoothing")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size < 3:
        raise ValueError("plot_smooth_line requires at least three finite points for smoothing")
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("plot_smooth_line requires strictly increasing x values")
    try:
        from scipy.interpolate import PchipInterpolator
    except ModuleNotFoundError as exc:
        raise FigureDependencyError(
            "plot_smooth_line requires scipy.interpolate.PchipInterpolator. "
            "Install scipy to draw shape-preserving publication trend lines."
        ) from exc
    interpolator = PchipInterpolator(x_values, y_values)
    sample_count = max(int(samples), x_values.size * 4)
    x_smooth = np.linspace(float(x_values.min()), float(x_values.max()), sample_count)
    y_smooth = interpolator(x_smooth)
    line_kwargs = dict(plot_kwargs)
    line_kwargs.setdefault("marker", None if not show_points else marker)
    (line,) = ax.plot(x_smooth, y_smooth, **line_kwargs)
    return line


def _assert_text_artist_inside(fig_box: Any, artist: Any, renderer: Any, label: str, tolerance: float) -> None:
    text = artist.get_text() if hasattr(artist, "get_text") else ""
    if not str(text).strip():
        return
    box = artist.get_window_extent(renderer)
    if box is None or box.width <= 0 or box.height <= 0:
        raise FigureLayoutError(f"{label} produced an empty text layout box")
    _assert_box_inside(fig_box, box, label, tolerance)


def _assert_legend_layout(fig_box: Any, ax: Any, legend: Any, renderer: Any, index: int, tolerance: float) -> None:
    legend_box = legend.get_window_extent(renderer)
    if legend_box is None or legend_box.width <= 0 or legend_box.height <= 0:
        raise FigureLayoutError(f"Axes {index} legend produced an empty layout box")
    _assert_box_inside(fig_box, legend_box, f"axes {index} legend", tolerance)
    if ax.has_data():
        data_box = ax.get_window_extent(renderer)
        if legend_box.overlaps(data_box):
            raise FigureLayoutError(
                f"Axes {index} legend overlaps the data area. "
                "Place legends outside the plotting area or in a dedicated legend axis."
            )


def _assert_box_inside(fig_box: Any, box: Any, label: str, tolerance: float) -> None:
    if (
        box.x0 < fig_box.x0 - tolerance
        or box.y0 < fig_box.y0 - tolerance
        or box.x1 > fig_box.x1 + tolerance
        or box.y1 > fig_box.y1 + tolerance
    ):
        raise FigureLayoutError(f"{label} extends beyond the figure canvas")


def _probability_ticks(lower: float, upper: float) -> list[float]:
    clipped_lower = max(0.0, min(1.0, lower))
    clipped_upper = max(0.0, min(1.0, upper))
    canonical = [0.0, 0.25, 0.5, 0.75, 1.0]
    ticks = [tick for tick in canonical if clipped_lower - 1e-12 <= tick <= clipped_upper + 1e-12]
    if not ticks:
        ticks = [clipped_lower, clipped_upper]
    if ticks[0] > clipped_lower + 1e-12:
        ticks.insert(0, clipped_lower)
    if ticks[-1] < clipped_upper - 1e-12:
        ticks.append(clipped_upper)
    return [round(tick, 6) for tick in ticks]


def _nice_numeric_ticks(lower: float, upper: float, target_count: int, integerish: bool) -> list[float]:
    span = upper - lower
    raw_step = span / max(target_count - 1, 1)
    step = _nice_step(raw_step)
    start = math.floor(lower / step) * step
    end = math.ceil(upper / step) * step
    ticks: list[float] = []
    value = start
    while value <= end + step * 1e-9:
        if value >= lower - step * 0.25 and value <= upper + step * 0.25:
            ticks.append(value)
        value += step
    if integerish:
        ticks = [round(tick) for tick in ticks]
        ticks = sorted(set(ticks))
    return [float(tick) for tick in ticks]


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiplier * magnitude
        if step >= raw_step - 1e-12:
            return float(step)
    return float(10.0 * magnitude)


def _format_tick(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{int(round(value))}"
    return f"{value:g}"
