"""Export one-dimensional Brisbane PV--BESS sensitivity line figures.

Run with ``python -m src.figures.sensitivity_grid_line_plots``.  The module
only reads a completed, validated 15-year sensitivity grid; it never reruns
optimisation or dispatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import _capacity_contour_style as contour_style
from .core import NATURE_PALETTE, apply_nature_style, matplotlib_context, plot_smooth_line, save_publication_figure, write_source_data
from .sensitivity_grid_contours import (
    DEFAULT_GRID_DIR,
    GRID_GROUPS,
    METRICS,
    display_values,
    file_sha256,
    load_completed_grids,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Sensitivity_Line_Figures_15Y"

PARAMETER_LABELS = {
    "pv_capex_sgd_per_kw": "PV CAPEX (SGD kW$^{-1}$)",
    "battery_capex_sgd_per_kwh": "Battery energy CAPEX (SGD kWh$^{-1}$)",
    "saidi_min_per_year": "SAIDI (min yr$^{-1}$)",
    "voll_sgd_per_kwh": "VoLL (SGD kWh$^{-1}$)",
}
PARAMETER_UNITS = {
    "pv_capex_sgd_per_kw": "SGD/kW",
    "battery_capex_sgd_per_kwh": "SGD/kWh",
    "saidi_min_per_year": "min/year",
    "voll_sgd_per_kwh": "SGD/kWh",
}
PARAMETER_COLUMNS = tuple(PARAMETER_LABELS)
METRIC_COLUMNS = tuple(str(metric["column"]) for metric in METRICS)
SMOOTH_SAMPLES = 200

# Core colours establish the manuscript vocabulary; the remaining two colours
# are sampled from their existing contour families to preserve metric identity.
CORE_METRIC_COLORS = {
    "annual_system_cost": NATURE_PALETTE["blue"],
    "optimal_pv_capacity": NATURE_PALETTE["teal"],
    "optimal_battery_energy": NATURE_PALETTE["purple"],
    "eue": NATURE_PALETTE["red"],
    "scr": NATURE_PALETTE["gold"],
}

LINE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "pv_capex",
        "directory": "PV_CAPEX",
        "title": "PV CAPEX sensitivity",
        "source_grid": "cost_capex",
        "x_parameter": "pv_capex_sgd_per_kw",
        "x_source": "row_value",
        "selector": {"column_value": 400.0},
        "fixed_conditions": {
            "battery_capex_sgd_per_kwh": 400.0,
            "saidi_min_per_year": 90.0,
            "voll_sgd_per_kwh": 35.0,
        },
    },
    {
        "key": "battery_capex",
        "directory": "Battery_CAPEX",
        "title": "Battery energy CAPEX sensitivity",
        "source_grid": "cost_capex",
        "x_parameter": "battery_capex_sgd_per_kwh",
        "x_source": "column_value",
        "selector": {"row_value": 1200.0},
        "fixed_conditions": {
            "pv_capex_sgd_per_kw": 1200.0,
            "saidi_min_per_year": 90.0,
            "voll_sgd_per_kwh": 35.0,
        },
    },
    {
        "key": "saidi",
        "directory": "SAIDI",
        "title": "SAIDI sensitivity",
        "source_grid": "saidi_voll",
        "x_parameter": "saidi_min_per_year",
        "x_source": "row_value",
        "selector": {"column_value": 30.0},
        "fixed_conditions": {
            "pv_capex_sgd_per_kw": 1200.0,
            "battery_capex_sgd_per_kwh": 400.0,
            "voll_sgd_per_kwh": 30.0,
        },
    },
    {
        "key": "voll",
        "directory": "VoLL",
        "title": "VoLL sensitivity",
        "source_grid": "saidi_voll",
        "x_parameter": "voll_sgd_per_kwh",
        "x_source": "column_value",
        "selector": {"row_value": 90.0},
        "fixed_conditions": {
            "pv_capex_sgd_per_kw": 1200.0,
            "battery_capex_sgd_per_kwh": 400.0,
            "saidi_min_per_year": 90.0,
        },
    },
)


def metric_line_color(mpl: Any, metric: Mapping[str, str]) -> str:
    """Return the stable line colour for one metric."""
    key = str(metric["key"])
    if key in CORE_METRIC_COLORS:
        return CORE_METRIC_COLORS[key]
    rgba = contour_style._contour_colormap(mpl, dict(metric))(0.72)
    return str(mpl.colors.to_hex(rgba))


def extract_line(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    case_key: str | None = None,
) -> pd.DataFrame:
    """Return one exact, sorted grid cross-section with explicit parameter values."""
    grid_key = str(spec["source_grid"])
    group = GRID_GROUPS[grid_key]
    source = frame.copy()
    if not source["row_parameter"].astype(str).eq(group["row_parameter"]).all():
        raise ValueError(f"{spec['key']} line uses an incompatible row parameter")
    if not source["column_parameter"].astype(str).eq(group["column_parameter"]).all():
        raise ValueError(f"{spec['key']} line uses an incompatible column parameter")

    for column, expected in dict(spec["selector"]).items():
        values = pd.to_numeric(source[column], errors="coerce")
        source = source.loc[np.isclose(values, float(expected), rtol=0.0, atol=1e-9)].copy()
    if source.empty:
        selection = ", ".join(f"{key}={value:g}" for key, value in dict(spec["selector"]).items())
        raise ValueError(f"Missing exact sensitivity cross-section for {spec['key']}: {selection}")

    x_column = str(spec["x_source"])
    x_values = pd.to_numeric(source[x_column], errors="coerce")
    if x_values.isna().any() or not np.isfinite(x_values.to_numpy(dtype=float)).all():
        raise ValueError(f"{spec['key']} line has non-finite x values")
    source = source.assign(_x_value=x_values.astype(float)).sort_values("_x_value").reset_index(drop=True)
    x = source["_x_value"].to_numpy(dtype=float)
    if len(x) < 2:
        raise ValueError(f"{spec['key']} line requires at least two samples")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{spec['key']} line requires unique, strictly increasing x values")

    _validate_fixed_conditions(source, spec, manifest=manifest, case_key=case_key)
    return _annotate_line_parameters(source, spec)


def _validate_fixed_conditions(
    source: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None,
    case_key: str | None,
) -> None:
    """Check the direct grid selectors and manifest-backed fixed conditions."""
    group = GRID_GROUPS[str(spec["source_grid"])]
    fixed = dict(spec["fixed_conditions"])
    dimension_values = {
        str(group["row_parameter"]): source["row_value"].to_numpy(dtype=float),
        str(group["column_parameter"]): source["column_value"].to_numpy(dtype=float),
    }
    for parameter, expected in fixed.items():
        if parameter in dimension_values:
            if not np.isclose(dimension_values[parameter], float(expected), rtol=0.0, atol=1e-9).all():
                raise ValueError(f"{spec['key']} line does not keep {parameter} fixed at {expected:g}")
            continue
        actual = _manifest_parameter(manifest, case_key, parameter)
        if actual is None:
            raise ValueError(f"{spec['key']} line cannot verify fixed {parameter} from the grid manifest")
        if not np.isclose(actual, float(expected), rtol=0.0, atol=1e-9):
            raise ValueError(f"{spec['key']} line manifest does not fix {parameter} at {expected:g}")


def _manifest_parameter(manifest: Mapping[str, Any] | None, case_key: str | None, parameter: str) -> float | None:
    if not manifest or not case_key:
        return None
    config = manifest.get("effective_config")
    if not isinstance(config, Mapping):
        return None
    if parameter == "pv_capex_sgd_per_kw":
        value = config.get("pv", {}).get("capex_sgd_per_kw") if isinstance(config.get("pv"), Mapping) else None
    elif parameter == "battery_capex_sgd_per_kwh":
        value = config.get("economics", {}).get("battery_capex_sgd_per_kwh") if isinstance(config.get("economics"), Mapping) else None
    else:
        scenarios = config.get("scenarios")
        scenario = scenarios.get(case_key) if isinstance(scenarios, Mapping) else None
        field = "saidi_min_per_year" if parameter == "saidi_min_per_year" else "voll_sgd_per_kwh"
        value = scenario.get(field) if isinstance(scenario, Mapping) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _annotate_line_parameters(source: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    group = GRID_GROUPS[str(spec["source_grid"])]
    annotated = source.copy()
    dimension_map = {
        str(group["row_parameter"]): "row_value",
        str(group["column_parameter"]): "column_value",
    }
    fixed = dict(spec["fixed_conditions"])
    for parameter in PARAMETER_COLUMNS:
        if parameter in dimension_map:
            annotated[parameter] = annotated[dimension_map[parameter]].astype(float)
        elif parameter in fixed:
            annotated[parameter] = float(fixed[parameter])
        else:
            raise ValueError(f"{spec['key']} line lacks a value for {parameter}")
    annotated.insert(0, "sweep_parameter", str(spec["x_parameter"]))
    annotated.insert(1, "sweep_value", annotated["_x_value"].astype(float))
    annotated.insert(2, "sweep_unit", PARAMETER_UNITS[str(spec["x_parameter"])])
    annotated.insert(3, "source_grid", str(spec["source_grid"]))
    annotated["annual_system_cost_ksgd_per_year"] = annotated["annual_cost_sgd_per_year"].astype(float) / 1000.0
    return annotated


def _plot_scr_domain(x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, str | None]:
    defined = np.isfinite(values)
    if not defined.any():
        raise ValueError("SCR line contains no defined values")
    indices = np.flatnonzero(defined)
    first, last = int(indices.min()), int(indices.max())
    if not defined[first : last + 1].all():
        raise ValueError("Undefined SCR must be confined to contiguous outer line boundaries")
    note = None
    if not defined.all():
        note = "SCR undefined outside the defined PV domain.\nNo zero-fill or extrapolation."
    return x[first : last + 1], values[first : last + 1], note


def _zero_based_y_scale(values: np.ndarray, unit_type: str) -> tuple[tuple[float, float], np.ndarray]:
    """Return a zero-baseline range and 4--6 readable major ticks."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot set y limits without finite values")
    maximum = float(finite.max())
    display_maximum = maximum * 100.0 if unit_type == "probability" else maximum
    step = contour_style.nice_step(display_maximum / 5.0) if display_maximum > 0.0 else 1.0
    upper = max(step, float(np.ceil(display_maximum / step) * step))
    if unit_type == "probability":
        upper = min(100.0, upper)
        ticks = np.arange(0.0, upper + step * 0.5, step, dtype=float) / 100.0
        return (0.0, upper / 100.0), ticks
    ticks = np.arange(0.0, upper + step * 0.5, step, dtype=float)
    return (0.0, upper), ticks


def _draw_smoothed_line(ax: Any, x: np.ndarray, y: np.ndarray, *, color: str) -> tuple[Any, str]:
    """Draw a marker-free curve, preserving the two-point SCR edge case."""
    if x.size >= 3:
        return (
            plot_smooth_line(
                ax,
                x,
                y,
                samples=SMOOTH_SAMPLES,
                show_points=False,
                color=color,
                linewidth=1.45,
                clip_on=True,
            ),
            "PCHIP",
        )
    if x.size == 2:
        smooth_x = np.linspace(float(x.min()), float(x.max()), SMOOTH_SAMPLES)
        smooth_y = np.interp(smooth_x, x, y)
        (line,) = ax.plot(smooth_x, smooth_y, color=color, linewidth=1.45, marker=None, clip_on=True)
        return line, "linear two-point fallback"
    raise ValueError("A sensitivity curve requires at least two defined values")


def build_figure(
    source: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    city_label: str = "Brisbane",
    out: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render one seven-panel, single-column sensitivity figure."""
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    contour_style._apply_single_outage_readable_font_style(mpl)
    # A tall single-column figure retains the established contour typography
    # while keeping all seven vertical metric labels inside the canvas.
    mpl.rcParams.update(
        {
            "font.size": 9.4,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "axes.titlesize": 10.2,
            "lines.markersize": 4.2,
        }
    )

    fig = plt.figure(figsize=(7.25, 14.8))
    grid = fig.add_gridspec(7, 1, hspace=0.45)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(fig.add_subplot(grid[row, 0], sharex=axes[0]) for row in range(1, len(METRICS)))
    x = source["_x_value"].to_numpy(dtype=float)
    fixed_text = _fixed_conditions_text(spec)
    title = f"{city_label}: {spec['title']}"
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.052, top=0.91)
    fig.suptitle(title, x=0.02, ha="left", y=0.982, fontsize=14.0, fontweight="bold")
    fig.text(0.02, 0.952, f"Fixed: {fixed_text}", ha="left", va="top", fontsize=10.2, color=NATURE_PALETTE["neutral_dark"])

    panels: dict[str, Any] = {}
    for index, (ax, metric) in enumerate(zip(axes, METRICS)):
        values = display_values(source, metric).to_numpy(dtype=float)
        plot_x, plot_y, note = x, values, None
        if metric["key"] == "scr":
            plot_x, plot_y, note = _plot_scr_domain(x, values)
        color = metric_line_color(mpl, metric)
        line, curve_method = _draw_smoothed_line(ax, plot_x, plot_y, color=color)
        ax.set_title(f"{chr(97 + index)}  {metric['title']}", loc="left", pad=4)
        ax.set_ylabel(str(metric["label"]))
        ax.set_xlim(float(x.min()), float(x.max()))
        y_limits, y_ticks = _zero_based_y_scale(plot_y, str(metric["unit_type"]))
        ax.set_ylim(*y_limits)
        ax.set_yticks(y_ticks)
        if metric["unit_type"] == "probability":
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=None))
        ax.tick_params(direction="out", length=2.8, width=0.65, pad=2.0)
        if index < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(PARAMETER_LABELS[str(spec["x_parameter"])])
        if note:
            ax.text(
                0.015,
                0.04,
                note,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.5,
                color=NATURE_PALETTE["neutral_dark"],
            )
        fig.canvas.draw()
        panels[str(metric["key"])] = {
            "title": ax.get_title(),
            "y_axis": ax.get_ylabel(),
            "color": color,
            "x_values": [float(value) for value in x],
            "y_values": [None if not np.isfinite(value) else float(value) for value in values],
            "y_limits": [float(value) for value in ax.get_ylim()],
            "y_tick_labels": [label.get_text() for label in ax.get_yticklabels()],
            "undefined_value_count": int((~np.isfinite(values)).sum()),
            "note": note,
            "curve_method": curve_method,
            "rendered_sample_count": int(len(line.get_xdata())),
            "marker": str(line.get_marker()),
        }

    metadata = {
        "title": title,
        "x_axis": PARAMETER_LABELS[str(spec["x_parameter"])],
        "fixed_conditions": {key: float(value) for key, value in dict(spec["fixed_conditions"]).items()},
        "panel_count": len(panels),
        "metrics": panels,
        "layout": "seven-row single-column shared x-axis",
        "shared_x_limits": [float(value) for value in axes[0].get_xlim()],
        "smoothing": "PCHIP shape-preserving interpolation (200 samples; linear only for two-point SCR boundary segments)",
        "interpolation": True,
        "show_data_points": False,
        "y_scale_policy": "zero baseline with a per-panel rounded upper bound",
        "dpi": 600,
    }
    return fig, metadata


def _fixed_conditions_text(spec: Mapping[str, Any]) -> str:
    labels = {
        "pv_capex_sgd_per_kw": "PV CAPEX",
        "battery_capex_sgd_per_kwh": "Battery CAPEX",
        "saidi_min_per_year": "SAIDI",
        "voll_sgd_per_kwh": "VoLL",
    }
    units = {
        "pv_capex_sgd_per_kw": "SGD/kW",
        "battery_capex_sgd_per_kwh": "SGD/kWh",
        "saidi_min_per_year": "min/yr",
        "voll_sgd_per_kwh": "SGD/kWh",
    }
    return "; ".join(f"{labels[key]} = {float(value):g} {units[key]}" for key, value in dict(spec["fixed_conditions"]).items())


def _source_data(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sweep_parameter",
        "sweep_value",
        "sweep_unit",
        "source_grid",
        *PARAMETER_COLUMNS,
        "grid_type",
        "row_parameter",
        "row_value",
        "row_unit",
        "column_parameter",
        "column_value",
        "column_unit",
        "case_key",
        "city_key",
        "building_key",
        "analysis_years",
        "annual_cost_sgd_per_year",
        "annual_system_cost_ksgd_per_year",
        *METRIC_COLUMNS[1:],
        "SCR_defined",
        "SCR_status",
    ]
    return frame.loc[:, columns].copy()


def run(grid_dir: Path = DEFAULT_GRID_DIR, output_dir: Path | None = None) -> dict[str, Path]:
    """Export four validated, one-dimensional sensitivity-curve figures."""
    grid_dir = Path(grid_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    frames, provenance = load_completed_grids(grid_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_key = str(next(iter(frames.values()))["case_key"].iloc[0])
    city_label = str(next(iter(frames.values()))["city_key"].iloc[0])

    lines = {
        str(spec["key"]): extract_line(
            frames[str(spec["source_grid"])],
            spec,
            manifest=provenance["manifest"],
            case_key=case_key,
        )
        for spec in LINE_SPECS
    }
    paths: dict[str, Path] = {}
    figures: dict[str, Any] = {}
    for spec in LINE_SPECS:
        key = str(spec["key"])
        directory = output_dir / str(spec["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        source_path = write_source_data(directory / f"{case_key}_{spec['directory']}_Sensitivity_Line", _source_data(lines[key]))
        paths[f"{key}_source_data"] = source_path
        figure, metadata = build_figure(lines[key], spec, city_label=city_label, out=output_dir)
        base = directory / f"{case_key}_{spec['directory']}_Sensitivity_Line"
        _, plt = matplotlib_context(output_dir)
        try:
            exports = save_publication_figure(figure, base)
        finally:
            plt.close(figure)
        for suffix in ("png", "svg", "pdf"):
            paths[f"{key}_{suffix}"] = exports[f"{base.name}_{suffix}"]
        figures[key] = {
            "directory": str(spec["directory"]),
            "source_data": source_path.name,
            "sample_count": int(len(lines[key])),
            "fixed_conditions": {name: float(value) for name, value in dict(spec["fixed_conditions"]).items()},
            "figure": metadata,
        }

    for filename, digest in provenance["input_sha256"].items():
        if file_sha256(grid_dir / filename) != digest:
            raise RuntimeError(f"Sensitivity-grid input changed while plotting: {filename}")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme_text(case_key, city_label), encoding="utf-8")
    paths["readme"] = readme_path
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key,
        "city_key": city_label,
        "analysis_years": float(next(iter(frames.values()))["analysis_years"].iloc[0]),
        "input_sha256": provenance["input_sha256"],
        "input_run_validation": provenance["validation"],
        "line_specs": [
            {
                "key": str(spec["key"]),
                "source_grid": str(spec["source_grid"]),
                "x_parameter": str(spec["x_parameter"]),
                "x_label": PARAMETER_LABELS[str(spec["x_parameter"])],
                "selector": {name: float(value) for name, value in dict(spec["selector"]).items()},
                "fixed_conditions": {name: float(value) for name, value in dict(spec["fixed_conditions"]).items()},
            }
            for spec in LINE_SPECS
        ],
        "figures": figures,
        "SCR_policy": "Undefined values are permitted only at contiguous outer line boundaries; they remain blank without zero-fill or extrapolation.",
        "exports": {key: str(path.relative_to(output_dir)) for key, path in paths.items()},
        "output_sha256": {path.relative_to(output_dir).as_posix(): file_sha256(path) for path in paths.values()},
    }
    validation_path = output_dir / "sensitivity_line_validation.json"
    validation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = validation_path
    return paths


def _readme_text(case_key: str, city_label: str) -> str:
    return f"""# {city_label} PV--BESS 一维敏感性曲线

本目录包含从已完成、已验证的 15 年敏感性网格直接截取的一维 x--y 曲线；未重新运行优化或调度。
每张图包含七项指标：年化系统总成本、最优 PV 容量、最优电池能量容量、EUE、LOLP、SSR 和 SCR。
年化系统总成本包括年化 PV/BESS 投资、运维、净购电和停电损失；成本按 kSGD/yr 显示，LOLP、SSR 和 SCR 按百分比显示。

## 横截面与固定条件

- `PV_CAPEX`：横轴为 PV CAPEX；固定电池 CAPEX=400 SGD/kWh、SAIDI=90 min/yr、VoLL=35 SGD/kWh。
- `Battery_CAPEX`：横轴为电池能量 CAPEX；固定 PV CAPEX=1200 SGD/kW、SAIDI=90 min/yr、VoLL=35 SGD/kWh。
- `SAIDI`：横轴为 SAIDI；固定 PV CAPEX=1200 SGD/kW、电池 CAPEX=400 SGD/kWh、VoLL=30 SGD/kWh。
- `VoLL`：横轴为 VoLL；固定 PV CAPEX=1200 SGD/kW、电池 CAPEX=400 SGD/kWh、SAIDI=90 min/yr。

SAIDI 图使用现有网格中的 VoLL=30 SGD/kWh 截面；它不同于原始情景基准 VoLL=35 SGD/kWh，已在图内和验证文件中明确记录。
每张图按单列排列七个面板，所有面板共享同一敏感性变量横轴及其范围。曲线采用 200 个 PCHIP 保形样本平滑绘制，未显示原始数据点，也不在原始横轴范围外外推；若 SCR 定义区间只剩两个点，则保留无标记的直线退化形式。全部纵轴从 0 开始，上限向上取整至可读刻度，避免对微小变化进行局部放大。
若 SCR 因 PV 为零而无定义，只允许横轴边界缺失：该区域保持空白并标注，绝不补零或跨越外推。

每个参数目录包含对应图件、原始绘图数据 CSV；`sensitivity_line_validation.json` 记录输入校验和、切片条件、面板元数据和产物校验和。

复现：

```bash
.venv/bin/python -m src.figures.sensitivity_grid_line_plots --grid-dir outputs/Brisbane/Brisbane_SingleFamily_Sensitivity_Grid_15Y
```

Case: `{case_key}`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR, help="Completed sensitivity-grid output directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for sensitivity line figures")
    args = parser.parse_args()
    paths = run(args.grid_dir, args.output_dir)
    figure_count = sum(path.suffix == ".png" for path in paths.values())
    print(f"Exported {figure_count} sensitivity line figures to {paths['validation'].parent}")


if __name__ == "__main__":
    main()
