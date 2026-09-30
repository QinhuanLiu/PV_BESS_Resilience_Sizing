"""Minimal unchanged plotting helpers from the pre-simplification snapshot.

Only the required typography, contours, and colorbar functions are retained.
The independent sweep entrypoint supplies data and standalone panel layout.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REFERENCE_REVISION = "2429818112354c88428fbe452fff0e51917f49db"
REFERENCE_TAG = "snapshot/pre-simplification-2026-09-14"
DISPLAY_PV_CAPACITY = "_display_pv_capacity"
DISPLAY_BATTERY_ENERGY = "_display_battery_energy"
SINGLE_OUTAGE_BASE_FONT_SIZE = 12.8
SINGLE_OUTAGE_TICK_FONT_SIZE = 12.0
SINGLE_OUTAGE_TITLE_FONT_SIZE = 14.0
SINGLE_OUTAGE_COLORBAR_LABEL_SIZE = 11.6
SINGLE_OUTAGE_COLORBAR_TICK_SIZE = 11.2



def _apply_single_outage_readable_font_style(mpl: Any) -> None:
    """Apply larger manuscript-readable typography only for the single-outage figures."""
    mpl.rcParams.update(
        {
            "font.size": SINGLE_OUTAGE_BASE_FONT_SIZE,
            "axes.labelsize": SINGLE_OUTAGE_BASE_FONT_SIZE,
            "xtick.labelsize": SINGLE_OUTAGE_TICK_FONT_SIZE,
            "ytick.labelsize": SINGLE_OUTAGE_TICK_FONT_SIZE,
            "legend.fontsize": SINGLE_OUTAGE_TICK_FONT_SIZE,
            "axes.titlesize": SINGLE_OUTAGE_TITLE_FONT_SIZE,
            "lines.markersize": 3.6,
        }
    )


def _draw_contour_panel(mpl: Any, ax: Any, source: pd.DataFrame, spec: dict[str, str], levels: np.ndarray) -> Any:
    points = (
        source.groupby([DISPLAY_PV_CAPACITY, DISPLAY_BATTERY_ENERGY], as_index=False)[spec["column"]]
        .mean()
        .sort_values([DISPLAY_BATTERY_ENERGY, DISPLAY_PV_CAPACITY])
    )
    x = points[DISPLAY_BATTERY_ENERGY].astype(float).to_numpy()
    y = points[DISPLAY_PV_CAPACITY].astype(float).to_numpy()
    z = points[spec["column"]].astype(float).to_numpy()
    pivot = points.pivot_table(
        index=DISPLAY_PV_CAPACITY,
        columns=DISPLAY_BATTERY_ENERGY,
        values=spec["column"],
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2 and not pivot.isna().any().any():
        xx, yy = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
        zz = pivot.to_numpy(dtype=float)
        cmap = _contour_colormap(mpl, spec)
        artist = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.88)
        ax.contour(xx, yy, zz, levels=_sparse_contour_lines(levels), colors="white", linewidths=0.32, alpha=0.65)
    elif len(points) >= 3 and np.ptp(x) > 0.0 and np.ptp(y) > 0.0:
        cmap = _contour_colormap(mpl, spec)
        artist = ax.tricontourf(x, y, z, levels=levels, cmap=cmap, alpha=0.88)
        ax.tricontour(x, y, z, levels=_sparse_contour_lines(levels), colors="white", linewidths=0.32, alpha=0.65)
    else:
        cmap = _contour_colormap(mpl, spec)
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        artist = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=28, edgecolor="white", linewidth=0.35)
    ax.set_xlim(float(source[DISPLAY_BATTERY_ENERGY].min()), float(source[DISPLAY_BATTERY_ENERGY].max()))
    ax.set_ylim(float(source[DISPLAY_PV_CAPACITY].min()), float(source[DISPLAY_PV_CAPACITY].max()))
    return artist


def _contour_colormap(mpl: Any, spec: dict[str, str]) -> Any:
    cmap_name = str(spec["cmap"])
    if spec["key"] != "ue":
        return mpl.colormaps[cmap_name]
    base = mpl.colormaps[cmap_name]
    colors = base(np.linspace(0.10, 0.94, 256))
    return mpl.colors.LinearSegmentedColormap.from_list("single_outage_ue_reds", colors)


def _metric_levels(values: np.ndarray, unit_type: str, *, bound_to_data: bool = False) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot build contour levels with no finite values.")
    if unit_type == "probability":
        percent_levels = nice_levels(np.clip(finite, 0.0, 1.0) * 100.0, target_intervals=12)
        levels = np.unique(np.clip(percent_levels / 100.0, 0.0, 1.0))
        if levels.size < 3:
            center = float(np.clip(finite[0], 0.0, 1.0))
            levels = np.unique(np.clip(np.array([center - 0.02, center, center + 0.02]), 0.0, 1.0))
        return levels
    levels = nice_levels(finite, target_intervals=14)
    if not bound_to_data:
        return levels
    data_min = float(finite.min())
    data_max = float(finite.max())
    if np.isclose(data_min, data_max):
        span = max(abs(data_min) * 0.05, 1e-9)
        return np.array([data_min - span, data_min, data_min + span], dtype=float)
    bounded = levels[(levels > data_min) & (levels < data_max)]
    return np.unique(np.concatenate(([data_min], bounded, [data_max])).astype(float))


def _format_metric_colorbar(
    colorbar: Any,
    levels: np.ndarray,
    spec: dict[str, str],
    *,
    orientation: str = "vertical",
    tick_position: str = "left",
) -> None:
    ticks = sparse_ticks_from_levels(levels, target_count=6)
    colorbar.set_ticks(ticks)
    colorbar.set_label(
        str(spec["label"]),
        fontsize=SINGLE_OUTAGE_COLORBAR_LABEL_SIZE,
        labelpad=2.4 if orientation == "vertical" else 1.4,
    )
    if spec["unit_type"] == "probability":
        colorbar.set_ticklabels([_format_percent_tick(float(tick)) for tick in ticks])
    else:
        colorbar.set_ticklabels([_format_numeric_tick(float(tick)) for tick in ticks])
    if orientation == "horizontal":
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.tick_params(axis="x", labelsize=SINGLE_OUTAGE_COLORBAR_TICK_SIZE, length=2.0, width=0.55, pad=1.0)
    elif tick_position == "right":
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.yaxis.set_label_position("right")
        colorbar.ax.tick_params(labelsize=SINGLE_OUTAGE_COLORBAR_TICK_SIZE, length=2.2, width=0.60, pad=1.0)
    else:
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(labelsize=SINGLE_OUTAGE_COLORBAR_TICK_SIZE, length=2.2, width=0.60, pad=1.0)


def _sparse_contour_lines(levels: np.ndarray) -> np.ndarray:
    ticks = sparse_ticks_from_levels(levels, target_count=5)
    return ticks[1:-1] if ticks.size > 2 else levels[1:-1]


def _format_numeric_tick(value: float) -> str:
    if np.isclose(value, round(value), atol=1e-9):
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_percent_tick(value: float) -> str:
    percent = 100.0 * float(value)
    if np.isclose(percent, round(percent), atol=1e-9):
        return f"{int(round(percent))}%"
    return f"{percent:.1f}%"


def nice_step(raw_step: float) -> float:
    if raw_step <= 0.0 or not np.isfinite(raw_step):
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = float(multiplier * magnitude)
        if step >= raw_step - 1e-12:
            return step
    return float(10.0 * magnitude)


def nice_levels(values: np.ndarray, *, target_intervals: int = 22) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot build display levels with no finite values.")
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        pad = max(abs(low) * 0.05, 1.0)
        return np.linspace(low - pad, high + pad, 3)
    step = nice_step((high - low) / max(int(target_intervals), 1))
    lower = np.floor(low / step) * step
    upper = np.ceil(high / step) * step
    if np.isclose(lower, upper):
        upper = lower + step
    return np.arange(lower, upper + step * 0.5, step, dtype=float)


def sparse_ticks_from_levels(levels: np.ndarray, *, target_count: int = 6) -> np.ndarray:
    finite = np.asarray(levels, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= target_count:
        return finite
    step = nice_step((float(finite[-1]) - float(finite[0])) / max(target_count - 1, 1))
    lower = np.ceil(float(finite[0]) / step) * step
    upper = np.floor(float(finite[-1]) / step) * step
    ticks = np.arange(lower, upper + step * 0.5, step, dtype=float)
    if ticks.size < 2:
        return np.linspace(float(finite[0]), float(finite[-1]), min(target_count, finite.size))
    return ticks
