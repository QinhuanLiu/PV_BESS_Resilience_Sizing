from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math

import numpy as np
import pandas as pd

from src.io.data_loader import (
    InputCase,
    discover_input_cases,
    load_case_parameters,
    read_load_csv,
    read_pv_csv,
    read_tou_csv,
)

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    plot_smooth_line,
    save_publication_figure,
)


CITY_ORDER = ("Bangkok", "Brisbane", "Jakarta", "Manila")
CITY_LABELS = {
    "Bangkok": "Bangkok",
    "Brisbane": "Brisbane",
    "Jakarta": "Jakarta",
    "Manila": "Manila",
}
CITY_COLORS = {
    "Bangkok": NATURE_PALETTE["purple"],
    "Brisbane": NATURE_PALETTE["gold"],
    "Jakarta": NATURE_PALETTE["teal"],
    "Manila": NATURE_PALETTE["red"],
}
CASE_BY_CITY = {
    "Bangkok": "Bangkok_SingleFamily",
    "Brisbane": "Brisbane_SingleFamily",
    "Jakarta": "Jakarta_SingleFamily",
    "Manila": "Manila_SingleFamily",
}
OUTPUT_DIR_NAME = "SingleFamily_Extrapolated_Cost_Distribution"
HISTOGRAM_FIGURE_NAME = "SingleFamily_Four_NonSG_Extrapolated_Cost_Histograms_LOLP_gt_0p1"
CDF_FIGURE_NAME = "SingleFamily_Four_NonSG_Extrapolated_Cost_CDF_LOLP_gt_0p1"
SOURCE_DATA_NAME = "SingleFamily_Four_NonSG_Extrapolated_Cost_Distribution_Source_Data.csv"
FIT_SUMMARY_NAME = "SingleFamily_Four_NonSG_Extrapolated_Cost_Normal_Fit_Summary.csv"
DEFAULT_BINS = 12
DEFAULT_LOLP_THRESHOLD = 0.1
MIN_NORMAL_FIT_SAMPLE_SIZE = 8


@dataclass(frozen=True)
class CityInputStats:
    case_key: str
    annual_load_kwh: float
    annual_pv_yield_kwh_per_m2: float
    mean_tou_sgd_per_kwh: float
    pv_capex_sgd_per_kw: float
    battery_capex_sgd_per_kwh: float
    saidi_min_per_year: float
    saifi_per_year: float
    voll_sgd_per_kwh: float

    @property
    def mean_event_duration_min(self) -> float:
        if self.saifi_per_year <= 0.0:
            return 0.0
        return self.saidi_min_per_year / self.saifi_per_year


def make_non_sg_extrapolated_cost_distribution_figures(
    out: Path,
    input_root: Path,
    output_root: Path,
    *,
    bins: int = DEFAULT_BINS,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
) -> dict[str, Path]:
    """Create four-city extrapolated annual-cost histogram and CDF figures without rerunning cases."""
    out.mkdir(parents=True, exist_ok=True)
    source = build_non_sg_extrapolated_cost_distribution_source(
        input_root,
        output_root,
        lolp_threshold=lolp_threshold,
    )
    fit_summary = build_normal_fit_summary(source, lolp_threshold=lolp_threshold)
    source_path = out / SOURCE_DATA_NAME
    fit_path = out / FIT_SUMMARY_NAME
    source.to_csv(source_path, index=False)
    fit_summary.to_csv(fit_path, index=False)

    paths: dict[str, Path] = {
        "non_sg_extrapolated_cost_distribution_source_data": source_path,
        "non_sg_extrapolated_cost_distribution_normal_fit_summary": fit_path,
    }
    paths.update(plot_non_sg_cost_histograms(out, source, fit_summary, bins=bins, lolp_threshold=lolp_threshold))
    paths.update(plot_non_sg_cost_cdfs(out, source, lolp_threshold=lolp_threshold))
    return paths


def build_non_sg_extrapolated_cost_distribution_source(
    input_root: Path,
    output_root: Path,
    *,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
) -> pd.DataFrame:
    cases = {case.case_key: case for case in discover_input_cases(input_root)}
    missing_cases = [case for case in CASE_BY_CITY.values() if case not in cases]
    if missing_cases:
        raise ValueError("Missing required non-Singapore SingleFamily input case(s): " + ", ".join(missing_cases))

    existing = [
        _existing_city_rows(
            "Brisbane",
            output_root
            / "Brisbane"
            / "Brisbane_SingleFamily_Uncertainty_Analysis"
            / "Brisbane_SingleFamily_Uncertainty_Baseline"
            / "Brisbane_SingleFamily_Uncertainty_Baseline_All_Results.csv",
        ),
        _existing_city_rows(
            "Jakarta",
            output_root
            / "Jakarta"
            / "Jakarta_SingleFamily_Uncertainty_Analysis"
            / "Jakarta_SingleFamily_Variable_Uncertainty"
            / "Jakarta_SingleFamily_Variable_Uncertainty_Scenario_Optima.csv",
        ),
        _existing_city_rows(
            "Manila",
            output_root
            / "Manila"
            / "Manila_SingleFamily_Uncertainty_Analysis"
            / "Manila_SingleFamily_Uncertainty_Baseline"
            / "Manila_SingleFamily_Uncertainty_Baseline_All_Results.csv",
        ),
    ]
    jakarta = existing[1]
    bangkok = _bangkok_proxy_rows(
        jakarta,
        cases[CASE_BY_CITY["Bangkok"]],
        cases[CASE_BY_CITY["Jakarta"]],
    )
    source = pd.concat([bangkok, *existing], ignore_index=True)
    source["tail_lolp_gt_0p1"] = source["scenario_LOLP"].astype(float) > float(lolp_threshold)
    source["cost_ksgd_per_year"] = source["annual_system_cost_sgd_per_year"].astype(float) / 1000.0
    source["city_order"] = source["city_key"].map({city: index for index, city in enumerate(CITY_ORDER)})
    source = source.sort_values(["city_order", "scenario_id"]).drop(columns=["city_order"]).reset_index(drop=True)
    _validate_distribution_source(source)
    return source


def build_normal_fit_summary(
    source: pd.DataFrame,
    *,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for city in CITY_ORDER:
        city_source = source[source["city_key"].astype(str) == city].copy()
        tail = city_source[city_source["scenario_LOLP"].astype(float) > float(lolp_threshold)]
        costs = tail["annual_system_cost_sgd_per_year"].astype(float).to_numpy()
        n_tail = int(costs.size)
        mean = float(np.mean(costs)) if n_tail else np.nan
        std = float(np.std(costs, ddof=1)) if n_tail > 1 else np.nan
        fit_available = bool(n_tail >= MIN_NORMAL_FIT_SAMPLE_SIZE and np.isfinite(std) and std > 0.0)
        rows.append(
            {
                "city_key": city,
                "city_label": CITY_LABELS[city],
                "lolp_threshold": float(lolp_threshold),
                "n_all": int(len(city_source)),
                "n_tail": n_tail,
                "tail_share_percent": float(n_tail / len(city_source) * 100.0) if len(city_source) else np.nan,
                "mean_cost_sgd_per_year": mean,
                "std_cost_sgd_per_year": std,
                "fit_available": fit_available,
                "fit_status": "normal_fit" if fit_available else "insufficient_tail_sample",
                "data_status": _city_data_status(city_source),
            }
        )
    return pd.DataFrame(rows)


def plot_non_sg_cost_histograms(
    out: Path,
    source: pd.DataFrame,
    fit_summary: pd.DataFrame,
    *,
    bins: int = DEFAULT_BINS,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
) -> dict[str, Path]:
    mpl, plt = matplotlib_context(out / HISTOGRAM_FIGURE_NAME)
    apply_nature_style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=False)
    for index, city in enumerate(CITY_ORDER):
        ax = axes.flatten()[index]
        _draw_city_histogram(ax, source, fit_summary, city, bins=bins, lolp_threshold=lolp_threshold)
        ax.set_title(f"{chr(97 + index)}  {CITY_LABELS[city]}", loc="left")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=2)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.88, wspace=0.30, hspace=0.42)
    paths = save_publication_figure(fig, out / HISTOGRAM_FIGURE_NAME)
    plt.close(fig)
    return paths


def plot_non_sg_cost_cdfs(
    out: Path,
    source: pd.DataFrame,
    *,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
) -> dict[str, Path]:
    mpl, plt = matplotlib_context(out / CDF_FIGURE_NAME)
    apply_nature_style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=False)
    for index, city in enumerate(CITY_ORDER):
        ax = axes.flatten()[index]
        _draw_city_cdf(ax, source, city, lolp_threshold=lolp_threshold)
        ax.set_title(f"{chr(97 + index)}  {CITY_LABELS[city]}", loc="left")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.925, wspace=0.30, hspace=0.42)
    paths = save_publication_figure(fig, out / CDF_FIGURE_NAME)
    plt.close(fig)
    return paths


def _existing_city_rows(city_key: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required existing uncertainty output not found for {city_key}: {path}")
    raw = pd.read_csv(path)
    required = ["uncertainty_scenario_id", "pv_bess_system_annual_cost_sgd", "unserved_energy_kwh_per_year"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    result = pd.DataFrame(
        {
            "city_key": city_key,
            "city_label": CITY_LABELS[city_key],
            "case_key": CASE_BY_CITY[city_key],
            "scenario_id": pd.to_numeric(raw["uncertainty_scenario_id"], errors="coerce").astype("Int64"),
            "annual_system_cost_sgd_per_year": pd.to_numeric(
                raw["pv_bess_system_annual_cost_sgd"],
                errors="coerce",
            ),
            "unserved_energy_kwh_per_year": pd.to_numeric(raw["unserved_energy_kwh_per_year"], errors="coerce"),
            "outage_weighted_load_kwh": pd.to_numeric(
                raw["outage_weighted_load_kwh"] if "outage_weighted_load_kwh" in raw.columns else np.nan,
                errors="coerce",
            ),
            "source_file": str(path),
            "data_status": "existing_result",
            "proxy_source_city": "",
            "load_scale": 1.0,
            "tariff_scale": 1.0,
            "pv_capex_scale": 1.0,
            "battery_capex_scale": 1.0,
            "pv_yield_scale": 1.0,
            "duration_scale": 1.0,
            "voll_scale": 1.0,
        }
    )
    if "scenario_LOLP" in raw.columns:
        result["scenario_LOLP"] = pd.to_numeric(raw["scenario_LOLP"], errors="coerce")
    else:
        denominator = result["outage_weighted_load_kwh"].astype(float)
        result["scenario_LOLP"] = np.where(
            denominator > 0.0,
            result["unserved_energy_kwh_per_year"].astype(float) / denominator,
            0.0,
        )
        result["scenario_LOLP"] = np.clip(result["scenario_LOLP"], 0.0, 1.0)
    return result


def _bangkok_proxy_rows(jakarta: pd.DataFrame, bangkok_case: InputCase, jakarta_case: InputCase) -> pd.DataFrame:
    bangkok_stats = _city_input_stats(bangkok_case)
    jakarta_stats = _city_input_stats(jakarta_case)
    load_scale = _safe_ratio(bangkok_stats.annual_load_kwh, jakarta_stats.annual_load_kwh)
    tariff_scale = _safe_ratio(bangkok_stats.mean_tou_sgd_per_kwh, jakarta_stats.mean_tou_sgd_per_kwh)
    pv_capex_scale = _safe_ratio(bangkok_stats.pv_capex_sgd_per_kw, jakarta_stats.pv_capex_sgd_per_kw)
    pv_yield_scale = _safe_ratio(bangkok_stats.annual_pv_yield_kwh_per_m2, jakarta_stats.annual_pv_yield_kwh_per_m2)
    battery_capex_scale = _safe_ratio(bangkok_stats.battery_capex_sgd_per_kwh, jakarta_stats.battery_capex_sgd_per_kwh)
    duration_scale = _safe_ratio(bangkok_stats.mean_event_duration_min, jakarta_stats.mean_event_duration_min)
    voll_scale = _safe_ratio(bangkok_stats.voll_sgd_per_kwh, jakarta_stats.voll_sgd_per_kwh)
    proxy = jakarta.copy()
    grid_component = proxy["annual_system_cost_sgd_per_year"].astype(float) - proxy[
        "unserved_energy_kwh_per_year"
    ].astype(float) * jakarta_stats.voll_sgd_per_kwh
    outage_component = proxy["unserved_energy_kwh_per_year"].astype(float) * jakarta_stats.voll_sgd_per_kwh
    battery_duration_scale = max(math.sqrt(max(duration_scale, 0.0)), 0.25)
    pv_capacity_scale = load_scale / max(pv_yield_scale, 1e-9)
    cost_scale = (
        0.55 * load_scale * tariff_scale
        + 0.25 * pv_capacity_scale * pv_capex_scale
        + 0.20 * load_scale * battery_capex_scale * battery_duration_scale
    )
    annual_cost = grid_component * cost_scale + outage_component * load_scale * duration_scale * voll_scale
    unserved_energy = proxy["unserved_energy_kwh_per_year"].astype(float) * load_scale * duration_scale
    outage_weighted_load = proxy["outage_weighted_load_kwh"].astype(float) * load_scale * max(duration_scale, 0.0)
    result = proxy.copy()
    result["city_key"] = "Bangkok"
    result["city_label"] = CITY_LABELS["Bangkok"]
    result["case_key"] = CASE_BY_CITY["Bangkok"]
    result["annual_system_cost_sgd_per_year"] = annual_cost
    result["unserved_energy_kwh_per_year"] = unserved_energy
    result["outage_weighted_load_kwh"] = outage_weighted_load
    result["scenario_LOLP"] = np.clip(proxy["scenario_LOLP"].astype(float) * duration_scale, 0.0, 1.0)
    result["source_file"] = "Jakarta_SingleFamily_Variable_Uncertainty_Scenario_Optima.csv"
    result["data_status"] = "extrapolated_proxy"
    result["proxy_source_city"] = "Jakarta"
    result["load_scale"] = load_scale
    result["tariff_scale"] = tariff_scale
    result["pv_capex_scale"] = pv_capex_scale
    result["battery_capex_scale"] = battery_capex_scale
    result["pv_yield_scale"] = pv_yield_scale
    result["duration_scale"] = duration_scale
    result["voll_scale"] = voll_scale
    return result


def _city_input_stats(case: InputCase) -> CityInputStats:
    params = load_case_parameters(case.parameter_csv).base
    cooling_cop = float(params.get("cooling_cop", 5.0))
    heating_cop = float(params.get("heating_cop", 3.5))
    load = read_load_csv(case.load_csv)
    pv = read_pv_csv(case.pv_csv)
    tou = read_tou_csv(case.tou_csv)
    return CityInputStats(
        case_key=case.case_key,
        annual_load_kwh=_annual_electric_equivalent_load_kwh(load, cooling_cop, heating_cop),
        annual_pv_yield_kwh_per_m2=float(pd.to_numeric(pv.iloc[:, 0], errors="coerce").fillna(0.0).sum()),
        mean_tou_sgd_per_kwh=float(pd.to_numeric(tou.iloc[:, -1], errors="coerce").dropna().mean()),
        pv_capex_sgd_per_kw=float(params["pv_capex_sgd_per_kw"]),
        battery_capex_sgd_per_kwh=float(params["battery_capex_sgd_per_kwh"]),
        saidi_min_per_year=float(params["saidi_min_per_year"]),
        saifi_per_year=float(params["saifi_per_year"]),
        voll_sgd_per_kwh=float(params["voll_sgd_per_kwh"]),
    )


def _annual_electric_equivalent_load_kwh(load: pd.DataFrame, cooling_cop: float, heating_cop: float) -> float:
    required = ["Cooling", "Heating", "Lighting", "Equipment"]
    missing = [column for column in required if column not in load.columns]
    if missing:
        raise ValueError("Load input is missing required column(s): " + ", ".join(missing))
    cooling_cop = max(float(cooling_cop), 1e-9)
    heating_cop = max(float(heating_cop), 1e-9)
    total = (
        load["Lighting"].astype(float)
        + load["Equipment"].astype(float)
        + load["Cooling"].astype(float) / cooling_cop
        + load["Heating"].astype(float) / heating_cop
    )
    return float(total.sum())


def _draw_city_histogram(
    ax: Any,
    source: pd.DataFrame,
    fit_summary: pd.DataFrame,
    city: str,
    *,
    bins: int,
    lolp_threshold: float,
) -> None:
    tail = _tail_costs(source, city, lolp_threshold=lolp_threshold)
    color = CITY_COLORS[city]
    if tail.size:
        counts, edges = np.histogram(tail, bins=int(bins))
        frequency = counts.astype(float) / float(tail.size)
        ax.bar(
            edges[:-1],
            frequency,
            width=np.diff(edges),
            align="edge",
            color=color,
            edgecolor=color,
            linewidth=0.0,
            alpha=0.72,
            label="Frequency",
            zorder=3,
        )
        fit_row = fit_summary[fit_summary["city_key"].astype(str) == city].iloc[0]
        if bool(fit_row["fit_available"]):
            x = np.linspace(float(edges[0]), float(edges[-1]), 240)
            mean = float(fit_row["mean_cost_sgd_per_year"])
            std = float(fit_row["std_cost_sgd_per_year"])
            y = _normal_pdf(x, mean, std) * float(np.mean(np.diff(edges)))
            ax.plot(x, y, color=NATURE_PALETTE["neutral_black"], linewidth=1.1, label="Normal fit", zorder=4)
        _annotate_fit(ax, fit_summary, city)
        ax.set_ylim(0.0, max(0.05, float(np.nanmax(frequency)) * 1.28))
    else:
        _draw_empty_tail_message(ax)
    ax.set_xlabel("Annual system cost (SGD/year)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.55, zorder=0)


def _draw_city_cdf(ax: Any, source: pd.DataFrame, city: str, *, lolp_threshold: float) -> None:
    tail = _tail_costs(source, city, lolp_threshold=lolp_threshold)
    color = CITY_COLORS[city]
    if tail.size:
        x, y = _empirical_cdf_points(tail)
        if x.size >= 3:
            line = plot_smooth_line(ax, x, y, samples=240, color=color, linewidth=1.4, solid_capstyle="round")
            ydata = np.clip(line.get_ydata(), 0.0, 1.0)
            ydata[0] = 0.0
            ydata[-1] = 1.0
            line.set_ydata(ydata)
        else:
            ax.plot(x, y, color=color, linewidth=1.4)
        ax.set_xlim(float(x.min()), float(x.max()))
    else:
        _draw_empty_tail_message(ax)
    ax.set_xlabel("Annual system cost (SGD/year)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.03)
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.55, zorder=0)


def _tail_costs(source: pd.DataFrame, city: str, *, lolp_threshold: float) -> np.ndarray:
    city_source = source[source["city_key"].astype(str) == city]
    tail = city_source[city_source["scenario_LOLP"].astype(float) > float(lolp_threshold)]
    return np.sort(tail["annual_system_cost_sgd_per_year"].astype(float).to_numpy())


def _empirical_cdf_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    unique, counts = np.unique(ordered, return_counts=True)
    cdf = np.cumsum(counts).astype(float) / float(ordered.size)
    span = max(float(unique.max() - unique.min()), 1e-9)
    x0 = float(unique.min() - span * 0.015)
    x = np.concatenate(([x0], unique))
    y = np.concatenate(([0.0], cdf))
    return x, y


def _annotate_fit(ax: Any, fit_summary: pd.DataFrame, city: str) -> None:
    row = fit_summary[fit_summary["city_key"].astype(str) == city].iloc[0]
    if bool(row["fit_available"]):
        text = (
            f"n={int(row['n_tail'])}\n"
            f"mean={float(row['mean_cost_sgd_per_year']):.0f}\n"
            f"sd={float(row['std_cost_sgd_per_year']):.0f}"
        )
    else:
        text = f"n={int(row['n_tail'])}\ninsufficient tail sample"
    ax.text(
        0.03,
        0.95,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color=NATURE_PALETTE["neutral_dark"],
    )


def _draw_empty_tail_message(ax: Any) -> None:
    ax.text(
        0.5,
        0.5,
        "No LOLP > 0.1 samples",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        color=NATURE_PALETTE["neutral_dark"],
    )


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return 1.0
    return float(numerator / denominator)


def _city_data_status(city_source: pd.DataFrame) -> str:
    statuses = sorted(city_source["data_status"].dropna().astype(str).unique().tolist())
    return statuses[0] if len(statuses) == 1 else ";".join(statuses)


def _validate_distribution_source(source: pd.DataFrame) -> None:
    required = (
        "city_key",
        "city_label",
        "case_key",
        "scenario_id",
        "annual_system_cost_sgd_per_year",
        "unserved_energy_kwh_per_year",
        "scenario_LOLP",
        "tail_lolp_gt_0p1",
        "data_status",
    )
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Non-SG cost distribution source is missing column(s): " + ", ".join(missing))
    if set(source["city_key"].astype(str).unique()) != set(CITY_ORDER):
        raise ValueError("Non-SG cost distribution source must contain Bangkok, Brisbane, Jakarta, and Manila only.")
    numeric = ("annual_system_cost_sgd_per_year", "unserved_energy_kwh_per_year", "scenario_LOLP")
    for column in numeric:
        values = pd.to_numeric(source[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Non-SG cost distribution source contains non-numeric {column}.")
    if (source["annual_system_cost_sgd_per_year"].astype(float) < 0.0).any():
        raise ValueError("Annual system cost must be non-negative.")
    if not source["scenario_LOLP"].astype(float).between(0.0, 1.0).all():
        raise ValueError("scenario_LOLP must be between 0 and 1.")
