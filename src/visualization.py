from __future__ import annotations

from pathlib import Path
from typing import Any

import os

import numpy as np
import pandas as pd

DESIGN_CASE_ORDER = ["flexibility_only", "resilience_informed"]
DESIGN_LABELS = {
    "flexibility_only": "Flexibility-only",
    "resilience_informed": "Resilience-informed",
}
DESIGN_FILE_LABELS = {
    "flexibility_only": "Flexibility_Only",
    "resilience_informed": "Resilience_Informed",
}
_LEGACY_RESILIENCE_KEY = "resilience_" + "aware"
LEGACY_DESIGN_CASES = {
    _LEGACY_RESILIENCE_KEY: "resilience_informed",
    "Resilience-" + "aware": "Resilience-informed",
    "Resilience_" + "Aware": "Resilience_Informed",
}
PARAMETER_LABELS = {
    "pv_capex_sgd_per_kwp": ("PV CAPEX", "PV CAPEX (SGD/kWp)"),
    "battery_capex_sgd_per_kwh": ("Battery energy CAPEX", "Battery energy CAPEX (SGD/kWh)"),
    "battery_power_capex_sgd_per_kw": ("Battery power CAPEX", "Battery power CAPEX (SGD/kW)"),
    "saidi_min_per_year": ("SAIDI", "SAIDI (min/year)"),
    "voll_sgd_per_kwh": ("VoLL", "VoLL (SGD/kWh)"),
}


def _matplotlib(out: Path) -> tuple[Any, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pv_bess_resilience_matplotlib_cache")
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return mpl, plt


def _style(mpl: Any) -> None:
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
            "axes.linewidth": 0.65,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )


def _save_all(fig: Any, base: Path) -> dict[str, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": base.with_suffix(".png")}
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.08)
    for path in paths.values():
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Figure export failed or produced an empty file: {path}")
    return {f"{base.name}_{suffix}": path for suffix, path in paths.items()}


def _write_source_data(base: Path, df: pd.DataFrame) -> Path:
    path = base.with_name(base.name + "_Source_Data.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def normalize_design_case_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized = normalized.rename(columns=lambda column: str(column).replace(_LEGACY_RESILIENCE_KEY, "resilience_informed"))
    if "design_case" in normalized.columns:
        normalized["design_case"] = normalized["design_case"].replace(LEGACY_DESIGN_CASES)
    if "design_case_label" in normalized.columns:
        normalized["design_case_label"] = normalized["design_case_label"].replace(LEGACY_DESIGN_CASES)
    return normalized


def normalize_design_mapping(mapping: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for key, value in mapping.items():
        normalized_key = LEGACY_DESIGN_CASES.get(str(key), str(key))
        normalized[normalized_key] = normalize_design_case_names(value)
    return normalized


def _case_title(case_key: str) -> str:
    return case_key.replace("_", " ")


def _axis_values(values: list[float] | tuple[float, ...]) -> list[float]:
    return sorted({round(float(value), 6) for value in values})


def _refined_axis_values(anchor: float, window: float, step: float) -> list[float]:
    start = max(0.0, float(anchor) - float(window))
    end = max(start, float(anchor) + float(window))
    count = int(round((end - start) / float(step)))
    return _axis_values([start + i * float(step) for i in range(count + 1)] + [float(anchor)])


def _ensure_two_axis_values(values: list[float], fallback_values: list[float], anchor: float) -> list[float]:
    axis = _axis_values(values)
    if len(axis) >= 2:
        return axis
    for candidate in sorted(_axis_values(fallback_values), key=lambda value: abs(value - float(anchor))):
        axis = _axis_values(axis + [candidate])
        if len(axis) >= 2:
            break
    return axis


def _prune_to_complete_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    complete = matrix.dropna(how="all").dropna(axis=1, how="all").copy()
    while complete.isna().any().any() and complete.shape[0] >= 2 and complete.shape[1] >= 2:
        row_missing = complete.isna().sum(axis=1)
        column_missing = complete.isna().sum(axis=0)
        if int(row_missing.max()) >= int(column_missing.max()) and complete.shape[0] > 2:
            complete = complete.drop(index=row_missing.idxmax())
        elif complete.shape[1] > 2:
            complete = complete.drop(columns=column_missing.idxmax())
        else:
            break
    return complete


def _cost_matrix_for_axes(
    summary: pd.DataFrame,
    pv_values: list[float],
    battery_values: list[float],
    allow_prune: bool = False,
) -> pd.DataFrame:
    pv_axis = _axis_values(pv_values)
    battery_axis = _axis_values(battery_values)
    keyed = summary.copy()
    keyed["_pv_axis_key"] = keyed["pv_capacity_kwp"].astype(float).round(6)
    keyed["_battery_axis_key"] = keyed["battery_energy_kwh"].astype(float).round(6)
    regular = keyed[
        keyed["_pv_axis_key"].isin(pv_axis) & keyed["_battery_axis_key"].isin(battery_axis)
    ].copy()
    matrix = regular.pivot_table(
        index="_battery_axis_key",
        columns="_pv_axis_key",
        values="pv_bess_system_annual_cost_sgd",
        aggfunc="min",
    )
    matrix = matrix.reindex(index=battery_axis, columns=pv_axis).sort_index().sort_index(axis=1)
    if allow_prune and matrix.isna().any().any():
        matrix = _prune_to_complete_matrix(matrix)
    if matrix.empty or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("Cost surface requires at least two PV capacities and two battery capacities.")
    if matrix.isna().any().any():
        missing = int(matrix.isna().sum().sum())
        raise ValueError(f"Cost surface matrix is incomplete; {missing} PV-BESS candidate cells are missing.")
    return matrix


def _full_search_cost_matrix(summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if "search_stage" in summary.columns:
        scoped = summary[summary["search_stage"].astype(str) == "coarse"].copy()
        if not scoped.empty:
            iteration = scoped["search_iteration"].astype(int).max()
            scoped = scoped[scoped["search_iteration"].astype(int) == iteration].copy()
            return _cost_matrix_for_axes(
                scoped,
                scoped["pv_capacity_kwp"].astype(float).unique().tolist(),
                scoped["battery_energy_kwh"].astype(float).unique().tolist(),
            )
    return _cost_matrix_for_axes(
        summary,
        [float(value) for value in config["pv"]["capacity_kwp_candidates"]],
        [float(value) for value in config["battery"]["energy_kwh_candidates"]],
    )


def _optimal_region_cost_matrix(
    summary: pd.DataFrame,
    design_rows: pd.DataFrame,
    design_case: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    if "search_stage" in summary.columns:
        scoped = summary[summary["search_stage"].astype(str) == "optimal_region"].copy()
        if not scoped.empty:
            iteration = scoped["search_iteration"].astype(int).max()
            scoped = scoped[scoped["search_iteration"].astype(int) == iteration].copy()
            return _cost_matrix_for_axes(
                scoped,
                scoped["pv_capacity_kwp"].astype(float).unique().tolist(),
                scoped["battery_energy_kwh"].astype(float).unique().tolist(),
            )
    row = design_rows[design_rows["design_case"] == design_case].iloc[0]
    best_pv = float(row["pv_capacity_kwp"])
    best_battery = float(row["battery_energy_kwh"])
    pv_values = _refined_axis_values(
        best_pv,
        float(config["pv"]["local_refine_window_kwp"]),
        float(config["pv"]["local_refine_step_kwp"]),
    )
    battery_values = _refined_axis_values(
        best_battery,
        float(config["battery"]["local_refine_window_kwh"]),
        float(config["battery"]["local_refine_step_kwh"]),
    )
    pv_values = _ensure_two_axis_values(pv_values, config["pv"]["capacity_kwp_candidates"], best_pv)
    battery_values = _ensure_two_axis_values(battery_values, config["battery"]["energy_kwh_candidates"], best_battery)
    return _cost_matrix_for_axes(summary, pv_values, battery_values, allow_prune=True)


def _tick_positions(values: np.ndarray, max_ticks: int = 8) -> np.ndarray:
    if len(values) <= max_ticks:
        return np.arange(len(values))
    return np.unique(np.linspace(0, len(values) - 1, max_ticks, dtype=int))


def make_baseline_figures(
    baseline_dir: Path,
    case_key: str,
    design_rows: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    design_rows = normalize_design_case_names(design_rows)
    summaries = normalize_design_mapping(summaries)
    for design_case in DESIGN_CASE_ORDER:
        summary = summaries[design_case].copy()
        paths.update(_make_cost_surface_and_heatmaps(baseline_dir, case_key, design_case, summary, design_rows, config))
    paths.update(_make_stacked_cost_bar(baseline_dir, case_key, design_rows))
    paths.update(_make_optimal_sizing_bar(baseline_dir, case_key, design_rows))
    return paths


def _make_cost_surface_and_heatmaps(
    out: Path,
    case_key: str,
    design_case: str,
    summary: pd.DataFrame,
    design_rows: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Path]:
    full_matrix = _full_search_cost_matrix(summary, config)
    optimal_matrix = _optimal_region_cost_matrix(summary, design_rows, design_case, config)
    paths: dict[str, Path] = {}
    paths.update(
        _make_cost_surface_and_heatmap(
            out,
            case_key,
            design_case,
            full_matrix,
            scope_file_label="Full_Search",
            plot_scope="full_search",
        )
    )
    paths.update(
        _make_cost_surface_and_heatmap(
            out,
            case_key,
            design_case,
            optimal_matrix,
            scope_file_label="Optimal_Region",
            plot_scope="optimal_region",
        )
    )
    return paths


def _make_cost_surface_and_heatmap(
    out: Path,
    case_key: str,
    design_case: str,
    matrix: pd.DataFrame,
    scope_file_label: str,
    plot_scope: str,
) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    source = matrix.reset_index().melt(
        id_vars="_battery_axis_key",
        var_name="pv_capacity_kwp",
        value_name="pv_bess_system_annual_cost_sgd_per_year",
    )
    source = source.rename(columns={"_battery_axis_key": "battery_energy_kwh"})
    source.insert(0, "plot_scope", plot_scope)
    source["battery_energy_kwh"] = source["battery_energy_kwh"].astype(float)
    source["pv_capacity_kwp"] = source["pv_capacity_kwp"].astype(float)
    source["battery_energy_mwh"] = source["battery_energy_kwh"] / 1000.0
    source["pv_capacity_mwp"] = source["pv_capacity_kwp"].astype(float) / 1000.0
    source["pv_bess_system_annual_cost_million_sgd_per_year"] = (
        source["pv_bess_system_annual_cost_sgd_per_year"].astype(float) / 1e6
    )

    x = matrix.columns.astype(float).to_numpy() / 1000.0
    y = matrix.index.astype(float).to_numpy() / 1000.0
    x_grid, y_grid = np.meshgrid(x, y)
    z = matrix.to_numpy(dtype=float) / 1e6
    label = DESIGN_FILE_LABELS[design_case]

    surface_base = out / (
        f"{case_key}_Baseline_{scope_file_label}_PV_Capacity_Battery_Capacity_Annualized_Cost_Surface_{label}"
    )
    _write_source_data(surface_base, source)
    fig = plt.figure(figsize=(172 / 25.4, 126 / 25.4), constrained_layout=False)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x_grid,
        y_grid,
        z,
        cmap="viridis",
        linewidth=0.18,
        edgecolor=(0.25, 0.25, 0.25, 0.28),
        antialiased=True,
        alpha=0.95,
    )
    ax.set_xlabel("PV capacity (MWp)", labelpad=5)
    ax.set_ylabel("Battery energy capacity (MWh)", labelpad=7)
    ax.set_zlabel("Annualized system cost (million SGD/year)", labelpad=8)
    ax.view_init(elev=24, azim=-136)
    ax.tick_params(labelsize=6, pad=1)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.10)
    cbar.set_label("Annualized system cost (million SGD/year)", labelpad=6)
    fig.subplots_adjust(left=0.02, right=0.80, bottom=0.08, top=0.90)
    paths = _save_all(fig, surface_base)
    plt.close(fig)

    heatmap_base = out / (
        f"{case_key}_Baseline_{scope_file_label}_PV_Capacity_Battery_Capacity_Annualized_Cost_Heatmap_{label}"
    )
    _write_source_data(heatmap_base, source)
    fig, ax = plt.subplots(figsize=(136 / 25.4, 95 / 25.4), constrained_layout=True)
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("PV capacity (MWp)")
    ax.set_ylabel("Battery energy capacity (MWh)")
    x_ticks = _tick_positions(x)
    y_ticks = _tick_positions(y)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x[position]:g}" for position in x_ticks], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y[position]:g}" for position in y_ticks])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Annualized system cost (million SGD/year)", labelpad=6)
    paths.update(_save_all(fig, heatmap_base))
    plt.close(fig)
    return paths


def cost_breakdown_table(design_rows: pd.DataFrame) -> pd.DataFrame:
    design_rows = normalize_design_case_names(design_rows)
    rows: list[dict[str, float | str]] = []
    for design_case in DESIGN_CASE_ORDER:
        row = design_rows[design_rows["design_case"] == design_case].iloc[0]
        annualized_pv = float(row["annualized_pv_cost_sgd"])
        annualized_battery = float(row["annualized_battery_cost_sgd"])
        operating = float(row["pv_bess_grid_cost_sgd"]) + float(row["pv_om_cost_sgd"]) + float(row["battery_om_cost_sgd"])
        unserved_cost = float(row["unserved_energy_cost_sgd_per_year"])
        total = float(row["pv_bess_system_annual_cost_sgd"])
        rows.append(
            {
                "design_case": design_case,
                "design_case_label": DESIGN_LABELS[design_case],
                "annualized_pv_capital_cost_sgd_per_year": annualized_pv,
                "annualized_battery_capital_cost_sgd_per_year": annualized_battery,
                "annualized_capital_cost_sgd_per_year": annualized_pv + annualized_battery,
                "operating_cost_sgd_per_year": operating,
                "unserved_energy_kwh_per_year": float(row["unserved_energy_kwh_per_year"]),
                "unserved_energy_cost_sgd_per_year": unserved_cost,
                "pv_bess_private_annual_cost_sgd_per_year": float(row["pv_bess_private_annual_cost_sgd"]),
                "pv_bess_system_annual_cost_sgd_per_year": total,
                "component_sum_sgd_per_year": annualized_pv + annualized_battery + operating + unserved_cost,
                "pv_capacity_kwp": float(row["pv_capacity_kwp"]),
                "pv_capacity_mwp": float(row["pv_capacity_kwp"]) / 1000.0,
                "rooftop_area_m2": float(row.get("rooftop_area_m2", np.nan)),
                "pv_installed_area_m2": float(row.get("pv_installed_area_m2", np.nan)),
                "pv_rooftop_utilization_fraction": float(row.get("pv_rooftop_utilization_fraction", np.nan)),
                "pv_rooftop_utilization_percent": float(row.get("pv_rooftop_utilization_percent", np.nan)),
                "pv_module_power_density_kwp_per_m2": float(row.get("pv_module_power_density_kwp_per_m2", np.nan)),
                "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def optimal_sizing_table(design_rows: pd.DataFrame) -> pd.DataFrame:
    design_rows = normalize_design_case_names(design_rows)
    rows: list[dict[str, float | str]] = []
    for design_case in DESIGN_CASE_ORDER:
        row = design_rows[design_rows["design_case"] == design_case].iloc[0]
        battery_energy_kwh = float(row["battery_energy_kwh"])
        pv_capacity_kwp = float(row["pv_capacity_kwp"])
        rows.append(
            {
                "design_case": design_case,
                "design_case_label": DESIGN_LABELS[design_case],
                "battery_energy_kwh": battery_energy_kwh,
                "battery_energy_mwh": battery_energy_kwh / 1000.0,
                "battery_power_kw": float(row["battery_power_kw"]),
                "pv_capacity_kwp": pv_capacity_kwp,
                "pv_capacity_mwp": pv_capacity_kwp / 1000.0,
                "rooftop_area_m2": float(row.get("rooftop_area_m2", np.nan)),
                "pv_installed_area_m2": float(row.get("pv_installed_area_m2", np.nan)),
                "pv_rooftop_utilization_fraction": float(row.get("pv_rooftop_utilization_fraction", np.nan)),
                "pv_rooftop_utilization_percent": float(row.get("pv_rooftop_utilization_percent", np.nan)),
                "pv_module_power_density_kwp_per_m2": float(row.get("pv_module_power_density_kwp_per_m2", np.nan)),
                "pv_bess_private_annual_cost_sgd_per_year": float(row["pv_bess_private_annual_cost_sgd"]),
                "pv_bess_system_annual_cost_sgd_per_year": float(row["pv_bess_system_annual_cost_sgd"]),
            }
        )
    return pd.DataFrame(rows)


def _make_stacked_cost_bar(out: Path, case_key: str, design_rows: pd.DataFrame) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    table = cost_breakdown_table(design_rows)
    base = out / f"{case_key}_Baseline_Annualized_Cost_Stacked_Flexibility_vs_Resilience_Informed"
    _write_source_data(base, table)

    x = np.arange(len(table))
    components = [
        ("annualized_pv_capital_cost_sgd_per_year", "Annualized PV capital cost", "#9ca3af"),
        ("annualized_battery_capital_cost_sgd_per_year", "Annualized battery capital cost", "#4b5563"),
        ("operating_cost_sgd_per_year", "Operating cost", "#3b82a0"),
        ("unserved_energy_cost_sgd_per_year", "Unserved energy cost", "#c08457"),
    ]
    fig, ax = plt.subplots(figsize=(126 / 25.4, 88 / 25.4), constrained_layout=True)
    bottom = np.zeros(len(table))
    for column, label, color in components:
        values = table[column].to_numpy(dtype=float) / 1e6
        ax.bar(x, values, bottom=bottom, width=0.58, color=color, edgecolor="#374151", linewidth=0.35, label=label)
        bottom += values
    ax.set_ylabel("System annualized cost (million SGD/year)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["design_case_label"].tolist(), rotation=8, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=2, columnspacing=1.0, handlelength=1.2)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _make_optimal_sizing_bar(out: Path, case_key: str, design_rows: pd.DataFrame) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    table = optimal_sizing_table(design_rows)
    base = out / f"{case_key}_Baseline_Optimal_Battery_Capacity_PV_Capacity_Flexibility_vs_Resilience_Informed"
    _write_source_data(base, table)

    x = np.arange(len(table))
    colors = ["#6b7280", "#0f766e"]
    fig, axes = plt.subplots(1, 2, figsize=(136 / 25.4, 72 / 25.4), constrained_layout=True)

    panels = [
        (axes[0], "battery_energy_mwh", "Optimal battery energy capacity (MWh)", "{:.3g}"),
        (axes[1], "pv_capacity_mwp", "Optimal PV capacity (MWp)", "{:.3g}"),
    ]
    for ax, column, ylabel, value_format in panels:
        values = table[column].to_numpy(dtype=float)
        ax.bar(x, values, width=0.58, color=colors, edgecolor="#374151", linewidth=0.35)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(table["design_case_label"].tolist(), rotation=8, ha="right", rotation_mode="anchor")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
        ax.set_axisbelow(True)
        upper = float(np.nanmax(values)) if len(values) else 0.0
        ylim_top = upper * 1.18 if upper > 0 else 1.0
        ax.set_ylim(0.0, ylim_top)
        for xpos, value in zip(x, values, strict=True):
            label_y = value + ylim_top * 0.025 if value > 0 else ylim_top * 0.025
            ax.text(xpos, label_y, value_format.format(value), ha="center", va="bottom", fontsize=6.2, color="#111827")
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def make_sensitivity_figures(
    sensitivity_root: Path,
    case_key: str,
    baseline_rows: pd.DataFrame,
    sensitivity_rows: pd.DataFrame,
    parameter_table: pd.DataFrame,
    parameters_to_plot: list[str] | None = None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    baseline_rows = normalize_design_case_names(baseline_rows)
    sensitivity_rows = normalize_design_case_names(sensitivity_rows)
    selected = set(parameters_to_plot or PARAMETER_LABELS.keys())
    for parameter, (short_label, x_label) in PARAMETER_LABELS.items():
        if parameter not in selected:
            continue
        param_dir = sensitivity_root / f"{case_key}_Sensitive_Analysis_{parameter}"
        param_dir.mkdir(parents=True, exist_ok=True)
        for obsolete in param_dir.glob(f"{case_key}_Sensitive_Analysis_{parameter}_Annualized_Cost_Reduction_Percentage_vs_*"):
            obsolete.unlink()
        subset = sensitivity_rows[sensitivity_rows["sensitivity_parameter"] == parameter].copy()
        if subset.empty:
            skipped = pd.DataFrame(
                [{"parameter": parameter, "status": "skipped", "reason": "No sensitivity range provided in Parameter.csv"}]
            )
            paths[f"{parameter}_skipped"] = _write_source_data(param_dir / f"{case_key}_Sensitive_Analysis_{parameter}_Skipped", skipped)
            continue
        paths.update(_make_sensitivity_parameter_figures(param_dir, case_key, parameter, short_label, x_label, subset))
    return paths


def make_uncertainty_figures(
    uncertainty_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    file_prefix: str = "Uncertainty_Baseline",
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if all_results.empty:
        return paths
    all_results = normalize_design_case_names(all_results)
    paths.update(
        _make_uncertainty_distribution(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Optimal_Battery_Capacity",
            value_column="battery_energy_kwh",
            plot_column="battery_energy_mwh",
            y_label="Optimal battery energy capacity (MWh)",
            transform=lambda values: values.astype(float) / 1000.0,
            file_prefix=file_prefix,
        )
    )
    paths.update(
        _make_uncertainty_distribution(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Optimal_PV_Capacity",
            value_column="pv_capacity_kwp",
            plot_column="pv_capacity_mwp",
            y_label="Optimal PV capacity (MWp)",
            transform=lambda values: values.astype(float) / 1000.0,
            file_prefix=file_prefix,
        )
    )
    paths.update(
        _make_uncertainty_distribution(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Annualized_System_Cost",
            value_column="pv_bess_system_annual_cost_sgd",
            plot_column="system_annualized_cost_million_sgd_per_year",
            y_label="Annualized system cost (million SGD/year)",
            transform=lambda values: values.astype(float) / 1e6,
            file_prefix=file_prefix,
        )
    )
    paths.update(
        _make_uncertainty_distribution(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Unserved_Energy",
            value_column="unserved_energy_kwh_per_year",
            plot_column="unserved_energy_mwh_per_year",
            y_label="Unserved energy (MWh/year)",
            transform=lambda values: values.astype(float) / 1000.0,
            file_prefix=file_prefix,
        )
    )
    paths.update(
        _make_uncertainty_distribution(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Unserved_Energy_Cost",
            value_column="unserved_energy_cost_sgd_per_year",
            plot_column="unserved_energy_cost_million_sgd_per_year",
            y_label="Unserved energy cost (million SGD/year)",
            transform=lambda values: values.astype(float) / 1e6,
            file_prefix=file_prefix,
        )
    )
    paths.update(
        _make_uncertainty_ecdf(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Optimal_Battery_Capacity",
            value_column="battery_energy_kwh",
            plot_column="battery_energy_mwh",
            x_label="Optimal battery energy capacity (MWh)",
            transform=lambda values: values.astype(float) / 1000.0,
            file_prefix=file_prefix,
        )
    )
    paths.update(
        _make_uncertainty_ecdf(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Annualized_System_Cost",
            value_column="pv_bess_system_annual_cost_sgd",
            plot_column="system_annualized_cost_million_sgd_per_year",
            x_label="Annualized system cost (million SGD/year)",
            transform=lambda values: values.astype(float) / 1e6,
            file_prefix=file_prefix,
        )
    )
    paths.update(
        _make_uncertainty_ecdf(
            uncertainty_dir,
            case_key,
            all_results,
            file_label="Unserved_Energy",
            value_column="unserved_energy_kwh_per_year",
            plot_column="unserved_energy_mwh_per_year",
            x_label="Unserved energy (MWh/year)",
            transform=lambda values: values.astype(float) / 1000.0,
            file_prefix=file_prefix,
        )
    )
    return paths


def _make_uncertainty_distribution(
    out: Path,
    case_key: str,
    all_results: pd.DataFrame,
    file_label: str,
    value_column: str,
    plot_column: str,
    y_label: str,
    transform,
    file_prefix: str,
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    source = all_results.copy()
    source[plot_column] = transform(source[value_column])
    base = out / f"{case_key}_{file_prefix}_{file_label}_Distribution_Box_Jitter_Half_Violin"
    _write_source_data(base, source)
    fig, ax = plt.subplots(figsize=(116 / 25.4, 82 / 25.4), constrained_layout=True)
    rng = np.random.default_rng(2026)
    colors = {"flexibility_only": "#6b7280", "resilience_informed": "#0f766e"}
    positions = np.arange(len(DESIGN_CASE_ORDER), dtype=float)
    for pos, design_case in zip(positions, DESIGN_CASE_ORDER, strict=True):
        values = source[source["design_case"] == design_case][plot_column].astype(float).dropna().to_numpy()
        if values.size == 0:
            continue
        if values.size >= 2:
            violin = ax.violinplot(
                values,
                positions=[pos + 0.10],
                widths=0.42,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors[design_case])
                body.set_edgecolor(colors[design_case])
                body.set_alpha(0.16)
                vertices = body.get_paths()[0].vertices
                vertices[:, 0] = np.maximum(vertices[:, 0], pos + 0.10)
        box = ax.boxplot(
            values,
            positions=[pos - 0.10],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 0.75},
            boxprops={"facecolor": colors[design_case], "alpha": 0.35, "edgecolor": "#111827", "linewidth": 0.65},
            whiskerprops={"color": "#111827", "linewidth": 0.65},
            capprops={"color": "#111827", "linewidth": 0.65},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[design_case])
            patch.set_alpha(0.35)
        jitter_x = pos + rng.normal(0.08, 0.025, size=values.size)
        ax.scatter(
            jitter_x,
            values,
            s=9,
            color=colors[design_case],
            alpha=0.38,
            linewidths=0,
            label=DESIGN_LABELS[design_case],
        )
    ax.set_ylabel(y_label)
    ax.set_xticks(positions)
    ax.set_xticklabels([DESIGN_LABELS[case] for case in DESIGN_CASE_ORDER], rotation=8, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.margins(x=0.15, y=0.14)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    ax.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=2)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _make_uncertainty_ecdf(
    out: Path,
    case_key: str,
    all_results: pd.DataFrame,
    file_label: str,
    value_column: str,
    plot_column: str,
    x_label: str,
    transform,
    file_prefix: str,
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    source = all_results.copy()
    source[plot_column] = transform(source[value_column])
    base = out / f"{case_key}_{file_prefix}_{file_label}_ECDF"
    _write_source_data(base, source)
    fig, ax = plt.subplots(figsize=(108 / 25.4, 76 / 25.4), constrained_layout=True)
    colors = {"flexibility_only": "#6b7280", "resilience_informed": "#0f766e"}
    for design_case in DESIGN_CASE_ORDER:
        values = np.sort(source[source["design_case"] == design_case][plot_column].astype(float).dropna().to_numpy())
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        ax.step(values, y, where="post", linewidth=1.15, color=colors[design_case], label=DESIGN_LABELS[design_case])
        ax.scatter(values, y, s=8, color=colors[design_case], alpha=0.45, linewidths=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="both", color="#e5e7eb", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.margins(x=0.03)
    ax.legend(loc="lower right")
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def make_uncertainty_sensitivity_figures(
    sensitivity_root: Path,
    case_key: str,
    sensitivity_rows: pd.DataFrame,
    parameters_to_plot: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    sensitivity_rows = normalize_design_case_names(sensitivity_rows)
    for parameter in parameters_to_plot:
        if parameter not in PARAMETER_LABELS:
            continue
        short_label, x_label = PARAMETER_LABELS[parameter]
        subset = sensitivity_rows[sensitivity_rows["sensitivity_parameter"] == parameter].copy()
        if subset.empty:
            continue
        param_dir = sensitivity_root / f"{case_key}_Uncertainty_Sensitive_Analysis_{parameter}"
        param_dir.mkdir(parents=True, exist_ok=True)
        paths.update(_make_uncertainty_sensitivity_parameter_figures(param_dir, case_key, parameter, short_label, x_label, subset))
    return paths


def _make_uncertainty_sensitivity_parameter_figures(
    out: Path,
    case_key: str,
    parameter: str,
    short_label: str,
    x_label: str,
    subset: pd.DataFrame,
) -> dict[str, Path]:
    source = subset.copy()
    source["system_annualized_cost_million_sgd_per_year"] = source["pv_bess_system_annual_cost_sgd"].astype(float) / 1e6
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["pv_capacity_mwp"] = source["pv_capacity_kwp"].astype(float) / 1000.0
    paths: dict[str, Path] = {}
    paths.update(
        _uncertainty_band_by_design(
            out,
            case_key,
            f"{case_key}_Uncertainty_Sensitive_Analysis_{parameter}_Optimal_Annualized_Cost_vs_{_filename_label(short_label)}",
            x_label,
            "Optimal system annualized cost (million SGD/year)",
            source,
            "system_annualized_cost_million_sgd_per_year",
        )
    )
    paths.update(
        _uncertainty_band_by_design(
            out,
            case_key,
            f"{case_key}_Uncertainty_Sensitive_Analysis_{parameter}_Optimal_Battery_Capacity_vs_{_filename_label(short_label)}",
            x_label,
            "Optimal battery energy capacity (MWh)",
            source,
            "battery_energy_mwh",
        )
    )
    paths.update(
        _uncertainty_band_by_design(
            out,
            case_key,
            f"{case_key}_Uncertainty_Sensitive_Analysis_{parameter}_Optimal_PV_Capacity_vs_{_filename_label(short_label)}",
            x_label,
            "Optimal PV capacity (MWp)",
            source,
            "pv_capacity_mwp",
        )
    )
    return paths


def _uncertainty_band_by_design(
    out: Path,
    case_key: str,
    base_name: str,
    x_label: str,
    y_label: str,
    source: pd.DataFrame,
    y_column: str,
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    source = normalize_design_case_names(source)
    grouped = (
        source.groupby(["sensitivity_parameter", "sensitivity_value", "design_case", "design_case_label"], as_index=False)
        .agg(
            count=(y_column, "count"),
            median=(y_column, "median"),
            p10=(y_column, lambda values: float(pd.Series(values).quantile(0.10))),
            p90=(y_column, lambda values: float(pd.Series(values).quantile(0.90))),
        )
        .sort_values(["sensitivity_value", "design_case"])
    )
    colors = {"flexibility_only": "#6b7280", "resilience_informed": "#0f766e"}
    paths: dict[str, Path] = {}
    for design_case in DESIGN_CASE_ORDER:
        part = grouped[grouped["design_case"] == design_case].sort_values("sensitivity_value")
        if part.empty:
            continue
        base = out / f"{base_name}_{DESIGN_FILE_LABELS[design_case]}"
        _write_source_data(base, part)
        fig, ax = plt.subplots(figsize=(118 / 25.4, 78 / 25.4), constrained_layout=True)
        x = part["sensitivity_value"].astype(float).to_numpy()
        median = part["median"].astype(float).to_numpy()
        p10 = part["p10"].astype(float).to_numpy()
        p90 = part["p90"].astype(float).to_numpy()
        ax.fill_between(x, p10, p90, color=colors[design_case], alpha=0.16, linewidth=0)
        ax.plot(
            x,
            median,
            marker="o",
            markersize=3,
            linewidth=1.15,
            color=colors[design_case],
            label=DESIGN_LABELS[design_case],
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.margins(x=0.03, y=0.14)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=1, columnspacing=1.3, handlelength=1.5)
        paths.update(_save_all(fig, base))
        plt.close(fig)
    return paths


def _line_by_design(
    out: Path,
    case_key: str,
    parameter: str,
    base_name: str,
    x_label: str,
    y_label: str,
    source: pd.DataFrame,
    y_column: str,
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    source = normalize_design_case_names(source)
    source = source.sort_values(["sensitivity_value", "design_case"]).copy()
    base = out / base_name
    _write_source_data(base, source)
    fig, ax = plt.subplots(figsize=(118 / 25.4, 78 / 25.4), constrained_layout=True)
    colors = {"flexibility_only": "#6b7280", "resilience_informed": "#0f766e"}
    for design_case in DESIGN_CASE_ORDER:
        part = source[source["design_case"] == design_case].sort_values("sensitivity_value")
        if part.empty:
            continue
        ax.plot(
            part["sensitivity_value"].astype(float),
            part[y_column].astype(float),
            marker="o",
            markersize=3,
            linewidth=1.15,
            color=colors[design_case],
            label=DESIGN_LABELS[design_case],
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.margins(x=0.03, y=0.14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=2, columnspacing=1.3, handlelength=1.5)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _make_sensitivity_parameter_figures(
    out: Path,
    case_key: str,
    parameter: str,
    short_label: str,
    x_label: str,
    subset: pd.DataFrame,
) -> dict[str, Path]:
    source = subset.copy()
    source["system_annualized_cost_million_sgd_per_year"] = source["pv_bess_system_annual_cost_sgd"].astype(float) / 1e6
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["pv_capacity_mwp"] = source["pv_capacity_kwp"].astype(float) / 1000.0
    paths: dict[str, Path] = {}
    paths.update(
        _line_by_design(
            out,
            case_key,
            parameter,
            f"{case_key}_Sensitive_Analysis_{parameter}_Optimal_Annualized_Cost_vs_{_filename_label(short_label)}",
            x_label,
            "Optimal system annualized cost (million SGD/year)",
            source,
            "system_annualized_cost_million_sgd_per_year",
        )
    )
    paths.update(
        _line_by_design(
            out,
            case_key,
            parameter,
            f"{case_key}_Sensitive_Analysis_{parameter}_Optimal_Battery_Capacity_vs_{_filename_label(short_label)}",
            x_label,
            "Optimal battery energy capacity (MWh)",
            source,
            "battery_energy_mwh",
        )
    )
    paths.update(
        _line_by_design(
            out,
            case_key,
            parameter,
            f"{case_key}_Sensitive_Analysis_{parameter}_Optimal_PV_Capacity_vs_{_filename_label(short_label)}",
            x_label,
            "Optimal PV capacity (MWp)",
            source,
            "pv_capacity_mwp",
        )
    )
    return paths


def _filename_label(label: str) -> str:
    return (
        label.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("%", "Percent")
    )
