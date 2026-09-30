"""Export 16:9 single-panel Brisbane sensitivity-line figures.

The module only reads a completed, validated 15-year sensitivity grid.  It
splits the existing four seven-panel sensitivity figures into 28 independent
figures and never reruns optimisation or dispatch.
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
from .core import NATURE_PALETTE, apply_nature_style, matplotlib_context, save_publication_figure, write_source_data
from .sensitivity_grid_contours import DEFAULT_GRID_DIR, METRICS, display_values, file_sha256, load_completed_grids
from .sensitivity_grid_line_plots import (
    LINE_SPECS,
    PARAMETER_LABELS,
    SMOOTH_SAMPLES,
    _draw_smoothed_line,
    _fixed_conditions_text,
    _plot_scr_domain,
    _source_data,
    _zero_based_y_scale,
    extract_line,
    metric_line_color,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Sensitivity_Split_Line_Figures_15Y"
FIGURE_SIZE_INCHES = (7.2, 5.1)
AXES_ASPECT_RATIO = 16.0 / 9.0
AXES_LEFT = 0.145
AXES_RIGHT = 0.980
AXES_BOTTOM = 0.160
AXES_WIDTH_FRACTION = AXES_RIGHT - AXES_LEFT
AXES_HEIGHT_FRACTION = AXES_WIDTH_FRACTION * FIGURE_SIZE_INCHES[0] / AXES_ASPECT_RATIO / FIGURE_SIZE_INCHES[1]
AXES_TOP = AXES_BOTTOM + AXES_HEIGHT_FRACTION


def _parameter_title(spec: Mapping[str, Any]) -> str:
    return str(spec["title"]).removesuffix(" sensitivity")


def _metric_filename(metric: Mapping[str, str]) -> str:
    return str(metric["filename"])


def build_metric_figure(
    source: pd.DataFrame,
    spec: Mapping[str, Any],
    metric: Mapping[str, str],
    *,
    city_label: str = "Brisbane",
    out: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render one real-data sensitivity metric with a 16:9 plotting region."""
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    contour_style._apply_single_outage_readable_font_style(mpl)
    mpl.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.6,
            "axes.titlesize": 9.4,
        }
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    parameter_title = _parameter_title(spec)
    title = f"{city_label}: {metric['title']} vs {parameter_title}"
    fig.suptitle(title, x=0.02, ha="left", y=0.977, fontsize=10.2, fontweight="bold")
    fig.text(
        0.02,
        0.916,
        f"Fixed: {_fixed_conditions_text(spec)}",
        ha="left",
        va="top",
        fontsize=7.6,
        color=NATURE_PALETTE["neutral_dark"],
    )

    x = source["_x_value"].to_numpy(dtype=float)
    values = display_values(source, metric).to_numpy(dtype=float)
    plot_x, plot_y, note = x, values, None
    if metric["key"] == "scr":
        plot_x, plot_y, note = _plot_scr_domain(x, values)
    color = metric_line_color(mpl, metric)
    line, curve_method = _draw_smoothed_line(ax, plot_x, plot_y, color=color)
    y_limits, y_ticks = _zero_based_y_scale(plot_y, str(metric["unit_type"]))

    ax.set_xlabel(PARAMETER_LABELS[str(spec["x_parameter"])])
    ax.set_ylabel(str(metric["label"]))
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    if metric["unit_type"] == "probability":
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=None))
    ax.tick_params(direction="out", length=3.0, width=0.65, pad=2.2)
    if note:
        ax.text(
            0.015,
            0.04,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color=NATURE_PALETTE["neutral_dark"],
        )
    fig.canvas.draw()
    axes_position = ax.get_position()
    axes_width_inches = float(axes_position.width * FIGURE_SIZE_INCHES[0])
    axes_height_inches = float(axes_position.height * FIGURE_SIZE_INCHES[1])
    axes_aspect_ratio = axes_width_inches / axes_height_inches

    metadata = {
        "title": title,
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "layout": "single-panel 16:9 sensitivity line plot",
        "metric": str(metric["key"]),
        "x_axis": ax.get_xlabel(),
        "y_axis": ax.get_ylabel(),
        "fixed_conditions": {key: float(value) for key, value in dict(spec["fixed_conditions"]).items()},
        "x_values": [float(value) for value in x],
        "x_limits": [float(value) for value in ax.get_xlim()],
        "y_values": [None if not np.isfinite(value) else float(value) for value in values],
        "y_limits": [float(value) for value in ax.get_ylim()],
        "y_tick_labels": [label.get_text() for label in ax.get_yticklabels()],
        "color": color,
        "undefined_value_count": int((~np.isfinite(values)).sum()),
        "note": note,
        "curve_method": curve_method,
        "rendered_sample_count": int(len(line.get_xdata())),
        "marker": str(line.get_marker()),
        "axes_box_inches": {"width": axes_width_inches, "height": axes_height_inches},
        "axes_aspect_ratio": axes_aspect_ratio,
        "target_axes_aspect_ratio": AXES_ASPECT_RATIO,
        "smoothing": "PCHIP shape-preserving interpolation (200 samples; linear only for two-point SCR boundary segments)",
        "interpolation": True,
        "show_data_points": False,
        "y_scale_policy": "zero baseline with a per-metric rounded upper bound",
        "dpi": 600,
    }
    return fig, metadata


def _readme_text(case_key: str, city_label: str) -> str:
    return f"""# {city_label} PV--BESS 16:9 单图一维敏感性分析

本目录将已验证的 15 年布里斯班一维敏感性分析拆分为 28 张独立图：4 个敏感性变量（PV CAPEX、电池能量 CAPEX、SAIDI、VoLL）各包含 7 项指标（年化系统总成本、最优 PV 容量、最优电池能量容量、EUE、LOLP、SSR、SCR）。所有数值均直接来自已完成、已验证的敏感性网格；未重新运行优化或调度，也没有引入虚构、代理或外推数据。

每个坐标轴内部的绘图区严格为宽:高 = 16:9。曲线使用 200 个 PCHIP 保形样本、隐藏原始数据点，并且不在原始横轴范围以外外推。纵轴从 0 开始；LOLP、SSR 和 SCR 以百分比显示。SCR 只在其定义域绘制：允许连续的横轴边界留白，但绝不补零或跨越未定义区域连接。

每个参数目录含七张 PNG（600 dpi）、SVG、PDF 图件及一份真实原始截面 CSV。`sensitivity_split_line_validation.json` 记录输入校验和、固定条件、绘图区比例、每张图的元数据与产物校验和。

复现：

```bash
.venv/bin/python -m src.figures.sensitivity_grid_split_line_plots \\
  --grid-dir outputs/Brisbane/Brisbane_SingleFamily_Sensitivity_Grid_15Y
```

Case: `{case_key}`.
"""


def run(grid_dir: Path = DEFAULT_GRID_DIR, output_dir: Path | None = None) -> dict[str, Path]:
    """Export 28 real-data, single-panel 16:9 sensitivity figures."""
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
        parameter_key = str(spec["key"])
        directory = output_dir / str(spec["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        source_path = write_source_data(
            directory / f"{case_key}_{spec['directory']}_Split_Sensitivity_Line",
            _source_data(lines[parameter_key]),
        )
        paths[f"{parameter_key}_source_data"] = source_path
        metric_metadata: dict[str, Any] = {}
        for metric in METRICS:
            metric_key = str(metric["key"])
            base = directory / f"{case_key}_{spec['directory']}_{_metric_filename(metric)}_Split_Sensitivity_Line"
            figure, metadata = build_metric_figure(lines[parameter_key], spec, metric, city_label=city_label, out=output_dir)
            _, plt = matplotlib_context(output_dir)
            try:
                exports = save_publication_figure(figure, base)
            finally:
                plt.close(figure)
            for suffix in ("png", "svg", "pdf"):
                paths[f"{parameter_key}_{metric_key}_{suffix}"] = exports[f"{base.name}_{suffix}"]
            metric_metadata[metric_key] = {"base_filename": base.name, "figure": metadata}
        figures[parameter_key] = {
            "directory": str(spec["directory"]),
            "source_data": source_path.name,
            "sample_count": int(len(lines[parameter_key])),
            "fixed_conditions": {name: float(value) for name, value in dict(spec["fixed_conditions"]).items()},
            "metrics": metric_metadata,
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
        "data_policy": "All 28 figures use only completed, validated sensitivity-grid samples; no proxy, synthetic or extrapolated values are used.",
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
        "metrics": [str(metric["key"]) for metric in METRICS],
        "figure_count": len(LINE_SPECS) * len(METRICS),
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "target_axes_aspect_ratio": AXES_ASPECT_RATIO,
        "figures": figures,
        "SCR_policy": "Undefined values are permitted only at contiguous outer line boundaries; they remain blank without zero-fill or extrapolation.",
        "exports": {key: str(path.relative_to(output_dir)) for key, path in paths.items()},
        "output_sha256": {path.relative_to(output_dir).as_posix(): file_sha256(path) for path in paths.values()},
    }
    validation_path = output_dir / "sensitivity_split_line_validation.json"
    validation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = validation_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR, help="Completed sensitivity-grid output directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for split sensitivity figures")
    args = parser.parse_args()
    paths = run(args.grid_dir, args.output_dir)
    figure_count = sum(path.suffix == ".png" for path in paths.values())
    print(f"Exported {figure_count} split 16:9 sensitivity figures to {paths['validation'].parent}")


if __name__ == "__main__":
    main()
