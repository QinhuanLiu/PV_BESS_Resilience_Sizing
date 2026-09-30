"""Plot resilience and optional mean cost/SCR/SSR contours from completed sweeps.

Run with ``python -m src.figures.capacity_sweep_contours``. No dispatch is rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import _capacity_contour_style as style
from .core import apply_nature_style, matplotlib_context, save_publication_figure, write_source_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios"
CAPACITY_KEYS = ["pv_capacity_kw", "battery_energy_kwh"]
CCP_LOLP_THRESHOLD = 0.20
SCR_DISPLAY_COLUMN = "SCR_display_mean"
SCR_DISPLAY_REFERENCE_PV_KW = 1.0
METRICS = (
    {"key": "ue", "column": "EUE_kwh_per_year_mean", "title": "Mean EUE",
     "label": "Mean EUE (kWh yr$^{-1}$)", "unit_type": "ue", "cmap": "Reds",
     "filename": "Mean_EUE_Contour"},
    {"key": "lolp", "column": "LOLP_mean", "title": "Mean LOLP",
     "label": "Mean LOLP (%)", "unit_type": "probability", "cmap": "YlGnBu",
     "filename": "Mean_LOLP_Contour"},
    {"key": "ccp", "column": "CCP_LOLP_lt20", "title": "CCP (LOLP < 20%)",
     "label": "CCP (%)", "unit_type": "probability", "cmap": "Purples",
     "filename": "CCP_LOLP_LT20_Contour"},
)
ECONOMIC_COLUMNS = ("annual_cost_sgd_per_year", "SCR", "SSR")
ECONOMIC_METRICS = (
    {"key": "cost", "column": "annual_cost_sgd_per_year_mean", "title": "Mean annual cost",
     "label": "Mean cost (kSGD yr$^{-1}$)", "unit_type": "cost", "cmap": "Blues",
     "filename": "Mean_Annual_Cost_Contour"},
    {"key": "scr", "column": "SCR_mean", "title": "Mean SCR",
     "label": "Mean SCR (%)", "unit_type": "probability", "cmap": "Oranges",
     "filename": "Mean_SCR_Contour", "display_column": SCR_DISPLAY_COLUMN},
    {"key": "ssr", "column": "SSR_mean", "title": "Mean SSR",
     "label": "Mean SSR (%)", "unit_type": "probability", "cmap": "Greens",
     "filename": "Mean_SSR_Contour"},
)


def metrics_for_source(source: pd.DataFrame) -> tuple[dict[str, str], ...]:
    return METRICS + ECONOMIC_METRICS if "annual_cost_sgd_per_year_mean" in source else METRICS


def validate_economic_fields(records: pd.DataFrame) -> bool:
    """Allow only explicitly documented zero-PV SCR nulls in enriched sweeps."""
    required = set(ECONOMIC_COLUMNS) | {"SCR_defined", "SCR_status"}
    if not required.intersection(records.columns):
        return False
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Incomplete economic metric fields: {sorted(missing)}")
    if not records.SCR_defined.isin([True, False]).all():
        raise ValueError("SCR_defined must be a non-missing boolean")
    zero_pv = records.pv_capacity_kw.eq(0)
    defined = records.SCR_defined.astype(bool)
    if not np.array_equal(defined, ~zero_pv) or not np.array_equal(records.SCR.isna(), zero_pv):
        raise ValueError("SCR can be undefined only for zero PV, and must remain blank there")
    expected_status = np.where(zero_pv, "not_applicable_no_pv", "defined")
    if not np.array_equal(records.SCR_status, expected_status):
        raise ValueError("SCR_status is inconsistent with PV capacity and SCR_defined")
    if not np.isfinite(records[["annual_cost_sgd_per_year", "SSR"]].to_numpy(dtype=float)).all():
        raise ValueError("Annual cost and SSR require finite values for every scenario")
    if not np.isfinite(records.loc[defined, "SCR"].to_numpy(dtype=float)).all():
        raise ValueError("Defined SCR values must be finite")
    if not records.SSR.between(0, 1).all() or not records.loc[defined, "SCR"].between(0, 1).all():
        raise ValueError("SCR and SSR must be fractions between zero and one")
    if not records.loc[zero_pv, "SSR"].eq(0).all():
        raise ValueError("SSR must be zero when PV capacity is zero")
    return True


def with_scr_display_values(source: pd.DataFrame) -> pd.DataFrame:
    """Copy the PV=1 kW SCR row to PV=0 kW for display only.

    ``SCR_mean`` deliberately remains null at zero PV because self-consumption
    is not defined without PV generation. The added display column is used
    solely to make the requested full-domain contour visually continuous.
    """
    if "SCR_mean" not in source:
        return source.copy()
    display = source.copy()
    display[SCR_DISPLAY_COLUMN] = display["SCR_mean"]
    zero_pv = np.isclose(display.pv_capacity_kw.to_numpy(dtype=float), 0.0)
    if not zero_pv.any():
        return display
    reference = display.loc[
        np.isclose(display.pv_capacity_kw.to_numpy(dtype=float), SCR_DISPLAY_REFERENCE_PV_KW),
        ["battery_energy_kwh", "SCR_mean"],
    ]
    if reference.empty or reference.battery_energy_kwh.duplicated().any() or reference.SCR_mean.isna().any():
        raise ValueError("SCR display fill requires one finite PV=1 kW value for every battery capacity")
    fill_by_battery = reference.set_index("battery_energy_kwh")["SCR_mean"]
    fill_values = display.loc[zero_pv, "battery_energy_kwh"].map(fill_by_battery)
    if fill_values.isna().any():
        raise ValueError("SCR display fill is missing a PV=1 kW reference for one or more battery capacities")
    display.loc[zero_pv, SCR_DISPLAY_COLUMN] = fill_values.to_numpy(dtype=float)
    if display[SCR_DISPLAY_COLUMN].isna().any():
        raise ValueError("SCR display values must be finite over the complete contour grid")
    return display


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_scenarios(records: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Validate the paired grid, then calculate scenario means and strict CCP counts."""
    required = CAPACITY_KEYS + [
        "case_key", "scenario_id", "outage_seed", "outage_profile_sha256", "analysis_years",
        "EUE_kwh_total", "EUE_kwh_per_year", "outage_load_kwh_total", "LOLP", "no_outage_load",
    ]
    missing = sorted(set(required) - set(records.columns))
    if missing:
        raise ValueError(f"Missing scenario columns: {missing}")
    enriched = validate_economic_fields(records)
    nonnullable = records.drop(columns=["SCR"]) if enriched else records
    if records.empty or nonnullable.isna().any().any():
        raise ValueError("Scenario records are empty or contain missing values")
    numeric = CAPACITY_KEYS + [
        "scenario_id", "outage_seed", "analysis_years", "EUE_kwh_total",
        "EUE_kwh_per_year", "outage_load_kwh_total", "LOLP",
    ]
    if not np.isfinite(records[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Scenario records contain non-finite values")
    if (records[numeric] < 0).any().any() or records.LOLP.gt(1).any():
        raise ValueError("Scenario capacities/metrics are outside their physical ranges")
    if records.duplicated(CAPACITY_KEYS + ["scenario_id"]).any():
        raise ValueError("Duplicate capacity/scenario keys")
    if not records.case_key.eq(spec["case_key"]).all():
        raise ValueError("Case key differs from the completed sweep manifest")
    if spec["analysis_years"] <= 0 or not records.analysis_years.eq(spec["analysis_years"]).all():
        raise ValueError("Analysis horizon differs from the completed sweep manifest")

    pv_values, battery_values = spec["pv_capacities"], spec["battery_capacities"]
    if len(set(pv_values)) < 2 or len(set(battery_values)) < 2:
        raise ValueError("Contour maps require at least two values on each capacity axis")
    expected = pd.MultiIndex.from_product(
        [pv_values, battery_values, range(1, spec["scenario_count"] + 1)],
        names=CAPACITY_KEYS + ["scenario_id"],
    ).sort_values()
    actual = pd.MultiIndex.from_frame(records[CAPACITY_KEYS + ["scenario_id"]]).sort_values()
    if not actual.equals(expected):
        raise ValueError("Incomplete or unexpected capacity/scenario grid")
    if not records.outage_seed.eq(spec["seed_start"] + records.scenario_id - 1).all():
        raise ValueError("Scenario seeds differ from the manifest")
    shared = records.groupby("scenario_id")[["outage_profile_sha256", "outage_load_kwh_total"]].nunique()
    if not shared.eq(1).all().all():
        raise ValueError("Capacity pairs do not share identical outage scenarios and demand")

    if not np.allclose(records.EUE_kwh_per_year * spec["analysis_years"], records.EUE_kwh_total,
                       rtol=1e-10, atol=1e-12):
        raise ValueError("Annual EUE does not reconcile with horizon-total EUE")
    demand = records.outage_load_kwh_total.to_numpy(dtype=float)
    unmet = records.EUE_kwh_total.to_numpy(dtype=float)
    if np.any(unmet > demand + 1e-10):
        raise ValueError("Unserved energy exceeds outage demand")
    ratio = np.divide(unmet, demand, out=np.zeros_like(unmet), where=demand > 0)
    if not np.allclose(records.LOLP, ratio, rtol=1e-10, atol=1e-12):
        raise ValueError("LOLP does not match each scenario's energy-based outage ratio")
    if not records.no_outage_load.isin([True, False]).all() or not np.array_equal(
        records.no_outage_load.to_numpy(dtype=bool), demand == 0
    ):
        raise ValueError("Zero-outage-demand flags are inconsistent")

    source = records.groupby(CAPACITY_KEYS, as_index=False, sort=True).agg(
        scenario_count=("scenario_id", "size"),
        EUE_kwh_per_year_mean=("EUE_kwh_per_year", "mean"),
        LOLP_mean=("LOLP", "mean"),
    )
    source.insert(3, "analysis_years", spec["analysis_years"])
    passing = records.assign(_ccp_pass=records.LOLP.lt(CCP_LOLP_THRESHOLD)).groupby(
        CAPACITY_KEYS, sort=True
    )["_ccp_pass"].sum()
    source["LOLP_threshold"] = CCP_LOLP_THRESHOLD
    source["LOLP_lt20_scenario_count"] = passing.reindex(
        pd.MultiIndex.from_frame(source[CAPACITY_KEYS])
    ).to_numpy(dtype=int)
    source["CCP_LOLP_lt20"] = source.LOLP_lt20_scenario_count / source.scenario_count
    if enriched:
        economic = records.groupby(CAPACITY_KEYS, as_index=False, sort=True).agg(
            annual_cost_sgd_per_year_mean=("annual_cost_sgd_per_year", "mean"),
            SCR_mean=("SCR", "mean"), SSR_mean=("SSR", "mean"),
            SCR_defined_scenario_count=("SCR", "count"),
        )
        source = source.merge(economic, on=CAPACITY_KEYS, validate="one_to_one")
        source = with_scr_display_values(source)
    return source


def reconcile_summary(source: pd.DataFrame, summary: pd.DataFrame) -> dict[str, float]:
    columns = ["EUE_kwh_per_year_mean", "LOLP_mean"]
    count_columns = ["scenario_count"]
    if "annual_cost_sgd_per_year_mean" in source:
        columns.extend(f"{column}_mean" for column in ECONOMIC_COLUMNS)
        count_columns.append("SCR_defined_scenario_count")
    required = CAPACITY_KEYS + count_columns + columns
    nonnullable = [column for column in required if column != "SCR_mean"]
    if not set(required).issubset(summary.columns) or summary[nonnullable].isna().any().any():
        raise ValueError("Capacity summary lacks required fields or contains missing values")
    if summary.duplicated(CAPACITY_KEYS).any():
        raise ValueError("Duplicate capacity keys in the existing summary")
    expected = summary.set_index(CAPACITY_KEYS).sort_index()
    actual = source.set_index(CAPACITY_KEYS).sort_index()
    if not actual.index.equals(expected.index):
        raise ValueError("Capacity summary grid differs from the raw records")
    for column in count_columns:
        if not actual[column].equals(expected[column]):
            raise ValueError(f"Capacity summary {column} differs from the raw records")
    differences = {}
    for column in columns:
        if not np.allclose(actual[column], expected[column], rtol=1e-10, atol=1e-12,
                           equal_nan=column == "SCR_mean"):
            raise ValueError(f"Recomputed {column} differs from the saved capacity summary")
        finite_differences = (actual[column] - expected[column]).abs().dropna()
        differences[column] = float(finite_differences.max()) if len(finite_differences) else 0.0
    return differences


def build_figure(
    source: pd.DataFrame, metric: dict[str, str], *, city_label: str = "Brisbane",
    out: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Keep the original Brisbane panel/colorbar geometry; export crops unused canvas."""
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    style._apply_single_outage_readable_font_style(mpl)
    display = source.assign(**{
        style.DISPLAY_PV_CAPACITY: source.pv_capacity_kw,
        style.DISPLAY_BATTERY_ENERGY: source.battery_energy_kwh,
    })
    if metric["key"] == "scr":
        display = with_scr_display_values(display)
    if metric["key"] == "cost":
        display[metric["column"]] = display[metric["column"]] / 1000.0
    display_column = metric.get("display_column", metric["column"])
    plot_metric = {**metric, "column": display_column}
    plot_source = display
    if metric["key"] == "scr":
        if plot_source.pv_capacity_kw.nunique() < 2 or plot_source.battery_energy_kwh.nunique() < 2:
            raise ValueError("SCR contours require at least two PV and battery capacities")
    # Preserve the physical panel sizes and spacing of the original five-metric,
    # four-city layout. Only the requested Brisbane panel and its colorbar exist.
    fig = plt.figure(figsize=(11.2, 11.4), constrained_layout=False)
    grid = fig.add_gridspec(
        5, 5, left=0.075, right=0.945, bottom=0.075, top=0.935,
        wspace=0.24, hspace=0.22, width_ratios=[1.0] * 4 + [0.055],
    )
    row = 1 if metric["key"] == "ue" else 2
    ax = fig.add_subplot(grid[row, 3])
    cax = fig.add_subplot(grid[row, 4])
    level_values = (np.array([0.0, 1.0]) if metric["key"] in {"ccp", "scr", "ssr"}
                    else display[display_column].to_numpy(dtype=float))
    levels = style._metric_levels(level_values, metric["unit_type"])
    artist = style._draw_contour_panel(mpl, ax, plot_source, plot_metric, levels)
    ax.set_title(city_label, loc="left", pad=5)
    ax.set_ylabel(f"{metric['title']}\nPV capacity (kW)")
    ax.set_xlabel("BESS energy (kWh)")
    ax.set_box_aspect(1.0)
    colorbar = fig.colorbar(artist, cax=cax)
    style._format_metric_colorbar(colorbar, levels, metric)
    fig.canvas.draw()
    # Standalone percentage labels can extend farther left than the original
    # inter-column gap. Translate the whole colorbar only; retain its dimensions,
    # typography and ticks, and keep all text clear of the data panel.
    renderer = fig.canvas.get_renderer()
    gap_pixels = 0.025 * fig.dpi
    shift_pixels = max(0.0, ax.bbox.x1 + gap_pixels - cax.get_tightbbox(renderer).x0)
    if shift_pixels:
        position = cax.get_position()
        cax.set_position([position.x0 + shift_pixels / fig.bbox.width,
                          position.y0, position.width, position.height])
        fig.canvas.draw()
    panel = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bar = cax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    metadata = {
        "levels": levels.tolist(), "white_line_levels": style._sparse_contour_lines(levels).tolist(),
        "colorbar_ticks": colorbar.get_ticks().tolist(),
        "colorbar_tick_labels": [label.get_text() for label in cax.get_yticklabels()],
        "panel_inches": [panel.width, panel.height], "colorbar_inches": [bar.width, bar.height],
        "standalone_colorbar_shift_inches": shift_pixels / fig.dpi,
        "x_axis": ax.get_xlabel(), "y_axis": ax.get_ylabel(),
        "x_limits": list(ax.get_xlim()), "y_limits": list(ax.get_ylim()),
        "colormap": artist.cmap.name, "fill_alpha": artist.get_alpha(),
        "white_line_width": 0.32, "white_line_alpha": 0.65,
        "grid_shape": [source.pv_capacity_kw.nunique(), source.battery_energy_kwh.nunique()],
        "plotted_grid_shape": [plot_source.pv_capacity_kw.nunique(), plot_source.battery_energy_kwh.nunique()],
        "display_divisor": 1000.0 if metric["key"] == "cost" else 1.0,
        "data_pv_limits": [float(plot_source.pv_capacity_kw.min()), float(plot_source.pv_capacity_kw.max())],
        "undefined_capacity_count": int(source[metric["column"]].isna().sum()),
        "scr_display_policy": (
            "PV=1 kW SCR values copied to the PV=0 kW display row; raw SCR remains undefined"
            if metric["key"] == "scr" else None
        ),
        "display_filled_capacity_count": (
            int(source[metric["column"]].isna().sum()) if metric["key"] == "scr" else 0
        ),
        "design_marker": None,
        "smoothing": False, "dpi": 600,
    }
    return fig, metadata


def run(results_dir: Path = DEFAULT_RESULTS, output_dir: Path | None = None) -> dict[str, Path]:
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else results_dir / "figures"
    manifest = json.loads((results_dir / "manifest.json").read_text())
    status = json.loads((results_dir / "run_status.json").read_text())
    validation = json.loads((results_dir / "validation.json").read_text())
    if status.get("state") != "complete" or validation.get("passed") is not True:
        raise ValueError("Only a completed, validated sweep can be plotted")
    input_hashes = {}
    for name in ("scenario_results.csv", "capacity_summary.csv"):
        input_hashes[name] = file_sha256(results_dir / name)
        if input_hashes[name] != validation.get("file_sha256", {}).get(name):
            raise ValueError(f"Completed sweep checksum does not match: {name}")
    records = pd.read_csv(results_dir / "scenario_results.csv")
    source = aggregate_scenarios(records, manifest["spec"])
    differences = reconcile_summary(source, pd.read_csv(results_dir / "capacity_summary.csv"))
    metrics = metrics_for_source(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    case_key = manifest["spec"]["case_key"]
    city_label = case_key.split("_")[0]
    source_path = write_source_data(output_dir / f"{case_key}_Mean_Resilience_Contours", source)
    paths = {"source_data": source_path}
    figures = {}
    _, plt = matplotlib_context(output_dir)
    for metric in metrics:
        figure, metadata = build_figure(source, metric, city_label=city_label, out=output_dir)
        base = output_dir / f"{case_key}_{metric['filename']}"
        try:
            paths.update(save_publication_figure(figure, base))
        finally:
            plt.close(figure)
        figures[metric["key"]] = metadata

    for name, digest in input_hashes.items():
        if file_sha256(results_dir / name) != digest:
            raise RuntimeError(f"Input changed while plotting: {name}")
    report = {
        "passed": True, "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key, "scenario_rows": len(records), "capacity_combinations": len(source),
        "scenarios_per_capacity": manifest["spec"]["scenario_count"],
        "analysis_years": manifest["spec"]["analysis_years"],
        "raw_checksums_match_completed_run": True, "input_files_unchanged": True,
        "complete_grid_verified": True, "shared_scenarios_verified": True,
        "summary_max_absolute_differences": differences,
        "aggregation": "Equal-weight arithmetic mean of each defined per-scenario metric; CCP uses strict scenario counts",
        "LOLP_definition": "Per-scenario horizon EUE divided by that scenario's outage demand; zero for zero demand",
        "CCP_definition": "Count of scenarios with LOLP strictly less than 0.20 divided by all scenarios at that capacity",
        "CCP_LOLP_threshold": CCP_LOLP_THRESHOLD, "CCP_comparison": "<",
        "CCP_resolution": 1.0 / manifest["spec"]["scenario_count"],
        "CCP_range": [float(source.CCP_LOLP_lt20.min()), float(source.CCP_LOLP_lt20.max())],
        "CCP_threshold_equality_rows": int(records.LOLP.eq(CCP_LOLP_THRESHOLD).sum()),
        "design_marker": None,
        "reference_tag": style.REFERENCE_TAG, "reference_revision": style.REFERENCE_REVISION,
        "plotting_helper_sha256": file_sha256(Path(style.__file__)),
        "input_sha256": input_hashes, "figures": figures,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    if len(metrics) > len(METRICS):
        report["economic_metrics"] = {
            "annual_cost_definition": "Annualized PV/battery investment + O&M + net grid purchases + outage losses",
            "annual_cost_source_unit": "SGD/year", "annual_cost_plot_unit": "kSGD/year",
            "SSR_definition": "PV-direct demand plus PV-origin battery discharge divided by total demand",
            "SCR_definition": "PV direct use plus PV battery charge divided by available PV generation",
            "SCR_not_applicable_scenarios": int(records.SCR.isna().sum()),
            "SCR_not_applicable_capacity_pairs": int(source.SCR_mean.isna().sum()),
            "SCR_defined_capacity_pairs": int(source.SCR_mean.notna().sum()),
            "SCR_defined_scenarios_per_valid_capacity": sorted(source.loc[source.SCR_mean.notna(), "SCR_defined_scenario_count"].unique().tolist()),
            "SCR_zero_PV_policy": (
                "Raw SCR remains undefined at PV=0. For the contour display only, each PV=0 row "
                "copies the corresponding battery row at PV=1 kW."
            ),
        }
    report_path = output_dir / "contour_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    paths["validation"] = report_path
    readme_path = output_dir / "README.md"
    economic_notes = ""
    if len(metrics) > len(METRICS):
        economic_notes = """
## 新增成本、SCR 与 SSR

- 平均年化成本：`annual_cost_sgd_per_year_mean`，包括 PV/电池年化投资、运维、净购电与停电损失。
  CSV 保存 SGD/年，图中仅除以 1,000 转为 kSGD/年，不再除以分析年数；蓝色 Blues。
- 平均 SCR：`SCR_mean`，逐场景自消纳率的算术平均，橙色 Oranges。
- 平均 SSR：`SSR_mean`，逐场景自给率的算术平均，绿色 Greens。
- 两种比例均先逐场景计算再平均，不用累计分子除以累计分母代替；CSV 保存 0–1，色条显示 0–100%。
- 零 PV 的 SCR 未定义，保留空值和有效场景数 0，不能视为 0%。其余组合各有完整的 100 个场景。
  源表新增 `SCR_defined_scenario_count`，成本和 SSR 仍覆盖全部 99 个容量组合。
- 零 PV 的原始 SCR 仍为未定义，且源表保留空值和有效场景数 0。仅为完整显示等高线，
  每个电池容量的 PV=0 kW 显示值复制自同一电池容量的 PV=1 kW 均值；
  `SCR_display_mean` 明确保存该显示层数据。SSR 在零 PV 时为 0。
"""
    readme_path.write_text(f"""# {city_label}：容量等高线图

本目录包含 {len(metrics)} 张独立容量等高线图，均由原始逐场景数据重新计算。
图形类型为定量容量网格。每个容量组合包含 {manifest['spec']['scenario_count']} 个等权场景，
每个场景覆盖 {manifest['spec']['analysis_years']} 个模型年；场景在各容量组合之间完全配对。

- 平均 EUE：`EUE_kwh_per_year` 的算术平均，单位 kWh/年。
- 平均 LOLP：逐场景 `LOLP` 的算术平均，图中色条显示百分比。
  保留能量型定义，不是失负荷小时概率，也不是平均 EUE 除以平均停电需求。
- CCP：每个容量组合中 `LOLP < 0.20` 的场景数除以全部场景数，等权计数。
  严格小于 20% 才达标，等于 20% 不计入；不先计算平均 LOLP。
  使用每条完整分析期场景的 LOLP，不把逐年记录当作独立场景。零停电需求场景沿用 LOLP=0，计入分母和达标数。
  CSV 以 0–1 保存，图中显示 0–100%；本次每组 {manifest['spec']['scenario_count']} 个场景，
  原始容量点的 CCP 分辨率为 {100 / manifest['spec']['scenario_count']:g}%。
  容量网格之间的连续色带由原等高线方法生成，不代表新增停电场景或更细的采样概率。
- 横轴为电池能量容量（kWh），纵轴为光伏容量（kW）。
- 原始输入为上级运行目录的 `scenario_results.csv`；`capacity_summary.csv` 仅用于独立核对。
- 配套 Source Data CSV 保存所有 {len(source)} 个容量组合、场景数、分析年数和两个均值，
  另含 `LOLP_threshold`、`LOLP_lt20_scenario_count` 和 `CCP_LOLP_lt20` 三列。
{economic_notes}

沿用多城市版本（`{style.REFERENCE_TAG}`，修订 `{style.REFERENCE_REVISION}`）的
11 个绘图辅助函数，函数原样提取。保留原布里斯班单面板、色条的物理尺寸、字体、
分级规则及线条。画布仅保留当前指标，保存时按原导出方法裁去空白。
独立图分别补齐原多城市图共享的城市标题和轴标签。EUE 使用原 Reds 0.10–0.94 色段，
LOLP 使用原 YlGnBu，CCP 使用 Purples（浅色低概率，深紫高概率）；填充透明度 0.88，白线宽 0.32、透明度 0.65。
CCP 按原概率分级函数覆盖 0–100%，色条刻度为 0%、20%、40%、60%、80%、100%。
色条从原间距开始，仅在左侧文字触及绘图区时整体右移，保留至少 0.025 英寸空白；
面板和色条的尺寸、字体与刻度均不变，平移量记录在核验文件中。
使用原规则网格 contourf，不添加平滑、加密插值、最优点或成本边界。

每张图导出 600 dpi PNG、可编辑文字 SVG 和 PDF。数据、分级、轴向和文件校验结果见
`contour_validation.json`；另附 `contour_qa.json` 记录实际数据的样式比对及文件检查结果。
本次不重新运行调度，不修改仿真输入与原始结果表。

复现：在项目目录运行 `.venv/bin/python -m src.figures.capacity_sweep_contours --results-dir {results_dir.relative_to(ROOT) if results_dir.is_relative_to(ROOT) else results_dir}`。
可通过 `--results-dir` 指定完整运行目录，通过 `--output-dir` 指定图形输出目录。
""", encoding="utf-8")
    paths["readme"] = readme_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Completed sweep directory containing scenario results and provenance")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: RESULTS/figures)")
    args = parser.parse_args()
    paths = run(args.results_dir, args.output_dir)
    figure_count = sum(path.suffix == ".png" for path in paths.values())
    print(f"Exported {figure_count} contour maps (PNG/SVG/PDF), source data and validation to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
