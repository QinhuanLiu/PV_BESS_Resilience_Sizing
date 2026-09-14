from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..io.config import DEFAULT_CONFIG
from ..io.data_loader import CaseParameters, InputCase, build_case_frame, load_case_parameters, load_input_case
from ..optimizer import evaluate_candidate, resilience_design, scenario_config
from ..outage import simulate_outages, summarize_outages
from ..resilience_metrics import electric_equivalent_building_load, outage_weighted_load_kwh
from ..uncertainty import run_uncertainty_analysis
from .city_comparison import SINGLEFAMILY_CASE_KEY
from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    plot_smooth_line,
    save_publication_figure,
    write_source_data,
)
from .display_units import (
    DISPLAY_BATTERY_ENERGY,
    DISPLAY_UE,
    DISPLAY_PV_CAPACITY,
    DISPLAY_SYSTEM_COST,
    add_display_unit_columns,
    figure_display_units,
    format_battery_energy_from_kwh,
    format_pv_capacity_from_kw,
    nice_levels,
    sparse_ticks_from_levels,
)
from .outage_balance import (
    WINDOW_HOURS,
    _conversion_loss,
    _draw_day_boundaries,
    _palette as _outage_balance_palette,
    _plot_outage_bounds,
    _shade_outage,
    _soc_ylim_for_zero_alignment,
    _stack_negative_balance,
    _stack_positive_balance,
    _window_start_hour,
)


CITY_ORDER = ("Singapore", "Metro_Manila", "Brisbane", "Bangkok")
FIGURE_1_TO_4_CITY_ORDER = ("Bangkok", "Singapore", "Metro_Manila", "Brisbane")
SINGAPORE_BRISBANE_CITY_ORDER = ("Singapore", "Brisbane")
FIGURE_C_CITY_ORDER = ("Singapore", "Brisbane", "Jakarta", "Metro_Manila")
FIGURE_C_NIGHT_OUTAGE_CITIES = ("Singapore", "Jakarta")
FIGURE_D_CITY_ORDER = ("Singapore", "Jakarta")
FIGURE_D_SCENARIOS = ("night", "day")
FIGURE_D_DAY_WINDOW = (10.0, 14.0)
FIGURE_D_NIGHT_WINDOW = (19.0, 23.0)
CITY_LABELS = {
    "Singapore": "Singapore",
    "Metro_Manila": "Metro Manila",
    "Brisbane": "Brisbane",
    "Bangkok": "Bangkok",
    "Jakarta": "Jakarta",
}
SINGLE_OUTAGE_CASE_BY_CITY = {
    "Singapore": "SG_SingleFamily",
    "Metro_Manila": "Manila_SingleFamily",
    "Brisbane": "Brisbane_SingleFamily",
    "Bangkok": "Bangkok_SingleFamily",
}
FIGURE_C_CASE_BY_CITY = {
    "Singapore": "SG_SingleFamily",
    "Brisbane": "Brisbane_SingleFamily",
    "Jakarta": "Jakarta_SingleFamily",
    "Metro_Manila": "Manila_SingleFamily",
}
FIGURE_NUMBERS = (1, "1b", "1c", "1d", 2, "2b", "2c", "2d", 3, "3b", "3c", 4, 5, "6a", "6b", 7)
FIGURE_NAMES = {
    1: "Figure_1_Four_City_Baseline_Design_Space_Contours",
    "1b": "Figure_1b_Singapore_Brisbane_Baseline_Design_Space_Contours",
    "1c": "Figure_1c_Four_City_Affordable_LOLP_Design_Space_Contours",
    "1d": "Figure_1d_Singapore_Jakarta_Day_Night_Affordable_LOLP_Contours",
    2: "Figure_2_Four_City_Baseline_Cost_Distribution",
    "2b": "Figure_2b_Singapore_Brisbane_Baseline_Cost_Distribution",
    "2c": "Figure_2c_Four_City_Affordable_LOLP_Cost_Distribution",
    "2d": "Figure_2d_Singapore_Jakarta_Day_Night_Cost_Distribution",
    3: "Figure_3_Four_City_Baseline_Outage_Energy_Balance",
    "3b": "Figure_3b_Singapore_Brisbane_Baseline_Outage_Energy_Balance",
    "3c": "Figure_3c_Four_City_Affordable_LOLP_Outage_Energy_Balance",
    4: "Figure_4_Four_City_100_Outage_Scenario_Distribution",
    5: "Figure_5_Four_City_CVaR_Risk_Aversion_Tradeoff",
    "6a": "Figure_6a_Four_City_TOU_Profile",
    "6b": "Figure_6b_Singapore_Brisbane_TOU_Profile",
    7: "Figure_7_Brisbane_Single_Outage_Sensitivity",
}
CONTOUR_METRICS = (
    ("system_total_cost", DISPLAY_SYSTEM_COST, "Cost", "cost", "Blues"),
    ("ue", DISPLAY_UE, "UE", "ue", "Reds"),
    ("lolp", "energy_based_LOLP", "LOLP", "probability", "YlGnBu"),
    ("ssr", "self_sufficiency_rate", "SSR", "probability", "BuGn"),
    ("scr", "self_consumption_rate", "SCR", "probability", "YlOrBr"),
)
SOURCE_KEYS = {
    1: ("design_space",),
    "1b": ("design_space",),
    "1c": ("affordable_design_space", "affordable_design_space_plot_grid"),
    "1d": ("day_night_affordable_design_space", "day_night_event_summary"),
    2: ("cost_distribution",),
    "2b": ("cost_distribution",),
    "2c": ("affordable_cost_distribution",),
    "2d": ("day_night_cost_distribution", "day_night_event_summary"),
    3: ("energy_balance", "event_summary"),
    "3b": ("energy_balance", "event_summary"),
    "3c": ("affordable_energy_balance", "affordable_event_summary"),
    4: ("scenario_design_results", "reference_design_results"),
    5: ("cvar_design_curve", "cvar_candidate_summary", "cvar_candidate_scenario_results"),
    "6a": ("tou_profile",),
    "6b": ("tou_profile",),
    7: ("brisbane_sensitivity_results",),
}
OPTIONAL_SOURCE_KEYS = {
    "1c": {"affordable_design_space_plot_grid"},
}
COST_DESIGN_CASES = ("no_pv_no_bess", "same_pv_no_bess", "optimal_pv_bess")
COST_DESIGN_LABELS = {
    "no_pv_no_bess": "0 PV / 0 BESS",
    "same_pv_no_bess": "Optimal PV / 0 BESS",
    "optimal_pv_bess": "Optimal PV / optimal BESS",
}
AFFORDABLE_COST_DESIGN_CASES = ("no_pv_no_bess", "lolp_zero_min_cost")
AFFORDABLE_COST_DESIGN_LABELS = {
    "no_pv_no_bess": "0 PV / 0 BESS",
    "lolp_zero_min_cost": "LOLP=0 min-cost design",
}
DAY_NIGHT_COST_DESIGN_CASES = (
    "night_no_pv_no_bess",
    "night_lolp_zero_design",
    "day_no_pv_no_bess",
    "day_lolp_zero_design",
    "night_design_under_day_outage",
)
DAY_NIGHT_COST_DESIGN_LABELS = {
    "night_no_pv_no_bess": "Night 0 PV / 0 BESS",
    "night_lolp_zero_design": "Night LOLP=0 design",
    "day_no_pv_no_bess": "Day 0 PV / 0 BESS",
    "day_lolp_zero_design": "Day LOLP=0 design",
    "night_design_under_day_outage": "Night design under day outage",
}
AFFORDABLE_LOLP_ZERO_TOLERANCE_KWH = 1e-6
AFFORDABLE_PLOT_GRID_EXPANSION_FACTOR = 1.25
AFFORDABLE_PLOT_GRID_MAX_EXPANSIONS = 4
SCENARIO_DISTRIBUTION_COUNT = 100
SCENARIO_DISTRIBUTION_SEED_START = 1000
CVAR_ALPHA = 0.90
CVAR_LAMBDAS = tuple(round(value, 1) for value in np.linspace(0.0, 1.0, 11))
CVAR_DEFAULT_MAX_CANDIDATES_PER_CITY = 8
CVAR_WORKER_FRAME_CACHE_LIMIT = 24
CVAR_WORKER_OUTAGE_CACHE_LIMIT = 32
BRISBANE_SENSITIVITY_DRIVER_ORDER = ("tou_spread", "saidi", "voll")
BRISBANE_SENSITIVITY_TOU_K_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
SINGLE_OUTAGE_BASE_FONT_SIZE = 12.8
SINGLE_OUTAGE_TICK_FONT_SIZE = 12.0
SINGLE_OUTAGE_TITLE_FONT_SIZE = 14.0
SINGLE_OUTAGE_COLORBAR_LABEL_SIZE = 11.6
SINGLE_OUTAGE_COLORBAR_TICK_SIZE = 11.2
SINGLE_OUTAGE_COST_XTICK_SIZE = 11.6
SINGLE_OUTAGE_BAR_WIDTH = 0.30
SINGLE_OUTAGE_BAR_STEP = 0.56
CITY_COLORS = {
    "Singapore": NATURE_PALETTE["blue"],
    "Metro_Manila": NATURE_PALETTE["red"],
    "Brisbane": NATURE_PALETTE["gold"],
    "Bangkok": NATURE_PALETTE["teal"],
    "Jakarta": NATURE_PALETTE["purple"],
}

_CVAR_WORKER_CASES: dict[str, InputCase] = {}
_CVAR_WORKER_CONFIGS: dict[str, dict[str, Any]] = {}
_CVAR_WORKER_FRAMES: OrderedDict[tuple[str, float], tuple[pd.DataFrame, dict[str, float | str]]] = OrderedDict()
_CVAR_WORKER_OUTAGES: OrderedDict[tuple[str, int, int], tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]] = OrderedDict()


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


def _single_outage_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.075,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SINGLE_OUTAGE_TITLE_FONT_SIZE,
        fontweight="bold",
    )


def build_single_outage_city_baseline_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build baseline-only, single-outage source data for the four real SingleFamily cities."""
    runtime_config = deepcopy(config or DEFAULT_CONFIG)
    cases_by_city = _singlefamily_cases_by_city(Path(input_root))
    design_spaces: list[pd.DataFrame] = []
    cost_rows: list[pd.DataFrame] = []
    energy_sources: list[pd.DataFrame] = []
    event_summaries: list[pd.DataFrame] = []

    for city_key in CITY_ORDER:
        case = cases_by_city[city_key]
        design_space, cost_source, energy_balance, event_summary = _build_city_sources(
            city_key,
            case,
            runtime_config,
        )
        design_spaces.append(design_space)
        cost_rows.append(cost_source)
        energy_sources.append(energy_balance)
        event_summaries.append(event_summary)

    return {
        "design_space": pd.concat(design_spaces, ignore_index=True),
        "cost_distribution": pd.concat(cost_rows, ignore_index=True),
        "energy_balance": pd.concat(energy_sources, ignore_index=True),
        "event_summary": pd.concat(event_summaries, ignore_index=True),
    }


def build_single_outage_city_affordable_lolp_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build Figure 1c/2c/3c source data for single-scenario affordable zero-LOLP designs."""
    runtime_config = deepcopy(config or DEFAULT_CONFIG)
    cases_by_city = _singlefamily_cases_by_city_for_order(
        Path(input_root),
        FIGURE_C_CITY_ORDER,
        FIGURE_C_CASE_BY_CITY,
    )
    tasks = [
        (
            city_key,
            cases_by_city[city_key],
            runtime_config,
            _figure_c_outage_mode(city_key),
            int(index),
        )
        for index, city_key in enumerate(FIGURE_C_CITY_ORDER)
    ]
    results = _build_affordable_lolp_city_sources_parallel(tasks, cases_by_city)
    ordered = sorted(results, key=lambda item: int(item["city_order"]))
    return {
        "affordable_design_space": _validate_affordable_design_space_source(
            pd.concat([item["design_space"] for item in ordered], ignore_index=True)
        ),
        "affordable_cost_distribution": _validate_affordable_cost_distribution_source(
            pd.concat([item["cost_distribution"] for item in ordered], ignore_index=True)
        ),
        "affordable_energy_balance": _validate_energy_balance_source(
            pd.concat([item["energy_balance"] for item in ordered], ignore_index=True),
            city_order=FIGURE_C_CITY_ORDER,
        ),
        "affordable_event_summary": pd.concat([item["event_summary"] for item in ordered], ignore_index=True),
    }


def build_single_outage_city_affordable_incremental_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
    sources: dict[str, pd.DataFrame],
    allow_new_evaluations: bool = True,
) -> dict[str, pd.DataFrame]:
    """Reuse Figure 1c evaluated points and add only boundary points needed for plotting."""
    evaluated = _validate_affordable_design_space_source(sources["affordable_design_space"]).copy()
    if "point_source" not in evaluated.columns:
        evaluated["point_source"] = "existing_evaluated"
    evaluated["point_source"] = evaluated["point_source"].fillna("existing_evaluated").astype(str)
    evaluated = _refresh_affordable_selection_metadata(evaluated, city_order=FIGURE_C_CITY_ORDER)
    if allow_new_evaluations:
        evaluated = _extend_affordable_design_space_for_complete_boundary(
            evaluated,
            Path(input_root),
            deepcopy(config or DEFAULT_CONFIG),
        )
    plot_grid = _affordable_design_space_plot_grid_source(evaluated)
    return {
        "affordable_design_space": _validate_affordable_design_space_source(evaluated),
        "affordable_design_space_plot_grid": _validate_affordable_plot_grid_source(plot_grid),
    }


def build_single_outage_city_day_night_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
    affordable_sources: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build Figure 1d day/night contour source for Singapore and Jakarta."""
    runtime_config = deepcopy(config or DEFAULT_CONFIG)
    evaluated = _validate_affordable_design_space_source(affordable_sources["affordable_design_space"])
    plot_grid = _validate_affordable_plot_grid_source(
        affordable_sources.get("affordable_design_space_plot_grid", _affordable_design_space_plot_grid_source(evaluated))
    )
    frames: list[pd.DataFrame] = []
    event_rows: list[pd.DataFrame] = []
    for city_key in FIGURE_D_CITY_ORDER:
        night_grid = plot_grid[plot_grid["city_key"].astype(str) == city_key].copy()
        night_grid = _with_day_night_scenario_columns(night_grid, city_key, "night")
        frames.append(night_grid)
        event_rows.append(_day_night_event_summary_row(Path(input_root), runtime_config, city_key, "night"))

        axes = _city_plot_axes(night_grid)
        day_points = [(pv, battery) for pv in axes["pv"] for battery in axes["battery"]]
        day_rows = _evaluate_affordable_points_for_city(
            Path(input_root),
            runtime_config,
            city_key,
            "day_window_random",
            day_points,
            point_source="day_evaluated",
            window=FIGURE_D_DAY_WINDOW,
        )
        day_source = _refresh_affordable_selection_metadata(pd.DataFrame(day_rows), city_order=(city_key,))
        day_source = _with_day_night_scenario_columns(day_source, city_key, "day")
        frames.append(day_source)
        event_rows.append(_day_night_event_summary_row(Path(input_root), runtime_config, city_key, "day"))
    design_space = _validate_day_night_design_space_source(pd.concat(frames, ignore_index=True, sort=False))
    event_summary = pd.concat(event_rows, ignore_index=True, sort=False)
    return {
        "day_night_affordable_design_space": design_space,
        "day_night_event_summary": event_summary,
    }


def build_single_outage_city_day_night_cost_distribution_source(day_night_design_space: pd.DataFrame) -> pd.DataFrame:
    source = _validate_day_night_design_space_source(day_night_design_space)
    rows: list[dict[str, Any]] = []
    for city_key in FIGURE_D_CITY_ORDER:
        city_source = source[
            (source["city_key"].astype(str) == city_key)
            & (source["point_source"].astype(str) != "interpolated_for_plot")
        ].copy()
        night = city_source[city_source["scenario_key"].astype(str) == "night"].copy()
        day = city_source[city_source["scenario_key"].astype(str) == "day"].copy()
        night_zero = _design_space_row_at_capacity(night, 0.0, 0.0)
        night_selected = _selected_design_space_row(night)
        day_zero = _design_space_row_at_capacity(day, 0.0, 0.0)
        day_selected = _selected_design_space_row(day)
        night_under_day = _design_space_row_at_capacity(
            day,
            float(night_selected["pv_capacity_kw"]),
            float(night_selected["battery_energy_kwh"]),
        )
        design_rows = (
            ("night_no_pv_no_bess", night_zero),
            ("night_lolp_zero_design", night_selected),
            ("day_no_pv_no_bess", day_zero),
            ("day_lolp_zero_design", day_selected),
            ("night_design_under_day_outage", night_under_day),
        )
        for order, (design_case, row) in enumerate(design_rows, start=1):
            rows.extend(_day_night_cost_component_rows(city_key, design_case, order, row))
    return _validate_day_night_cost_distribution_source(pd.DataFrame(rows))


def build_single_outage_city_tou_profile_source(input_root: Path | str = "Input") -> pd.DataFrame:
    """Build 24 h TOU tariff source data for the real SingleFamily city cases."""
    cases_by_city = _singlefamily_cases_by_city(Path(input_root))
    rows: list[dict[str, Any]] = []
    for city_key in FIGURE_1_TO_4_CITY_ORDER:
        case = cases_by_city[city_key]
        tou = pd.read_csv(case.tou_csv)
        required = {"hour", "tou_sgd_per_kwh"}
        missing = required - set(tou.columns)
        if missing:
            raise ValueError(f"{case.tou_csv} missing TOU column(s): " + ", ".join(sorted(missing)))
        tou = tou.copy()
        tou["hour"] = pd.to_numeric(tou["hour"], errors="coerce").astype("Int64")
        tou["tou_sgd_per_kwh"] = pd.to_numeric(tou["tou_sgd_per_kwh"], errors="coerce")
        if tou[["hour", "tou_sgd_per_kwh"]].isna().any().any():
            raise ValueError(f"{case.tou_csv} contains non-numeric TOU values.")
        tou = tou.sort_values("hour").reset_index(drop=True)
        expected_hours = list(range(24))
        hours = tou["hour"].astype(int).tolist()
        if hours != expected_hours:
            raise ValueError(f"{case.tou_csv} must cover hours 0-23 exactly.")
        for row in tou.to_dict("records"):
            hour = int(row["hour"])
            rows.append(
                {
                    "city_key": city_key,
                    "city_label": CITY_LABELS[city_key],
                    "case_key": case.case_key,
                    "source_case_key": case.case_key,
                    "hour": hour,
                    "interval_start_hour": float(hour),
                    "interval_end_hour": float(hour + 1),
                    "tou_sgd_per_kwh": float(row["tou_sgd_per_kwh"]),
                }
            )
    return _validate_tou_profile_source(pd.DataFrame(rows))


def build_single_outage_city_brisbane_sensitivity_source(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build Figure 7 source data from Brisbane single-outage sensitivity optimizations."""
    runtime_config = deepcopy(config or DEFAULT_CONFIG)
    cases_by_city = _singlefamily_cases_by_city(Path(input_root))
    case = cases_by_city["Brisbane"]
    parameters = load_case_parameters(case.parameter_csv)
    tasks = _brisbane_sensitivity_tasks(parameters)
    workers = _brisbane_sensitivity_worker_count(parameters, len(tasks))
    rows = _evaluate_brisbane_sensitivity_tasks_parallel(
        tasks,
        Path(input_root),
        runtime_config,
        workers=workers,
    )
    return _validate_brisbane_sensitivity_source(pd.DataFrame(rows))


def build_single_outage_city_scenario_distribution_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
    baseline_sources: dict[str, pd.DataFrame] | None = None,
    scenario_count: int = SCENARIO_DISTRIBUTION_COUNT,
    seed_start: int = SCENARIO_DISTRIBUTION_SEED_START,
) -> dict[str, pd.DataFrame]:
    """Build Figure 4 source data from per-scenario 100-outage optimizations."""
    if int(scenario_count) != SCENARIO_DISTRIBUTION_COUNT:
        raise ValueError(f"Figure 4 requires exactly {SCENARIO_DISTRIBUTION_COUNT} scenarios per city.")
    runtime_config = deepcopy(config or DEFAULT_CONFIG)
    baseline = baseline_sources or build_single_outage_city_baseline_sources(input_root, config=runtime_config)
    reference = _reference_design_results_source(baseline["design_space"])
    cases_by_city = _singlefamily_cases_by_city(Path(input_root))
    rng = np.random.default_rng(int(seed_start) + 4703)
    scenario_sources: list[pd.DataFrame] = []
    for city_key in CITY_ORDER:
        case = cases_by_city[city_key]
        _, _, case_config, _ = load_input_case(case, runtime_config)
        cached = _cached_uncertainty_results(runtime_config, case, case_config, int(scenario_count))
        if cached is None:
            all_results, _, _ = run_uncertainty_analysis(
                case,
                case_config,
                case.case_key,
                scenario_count=int(scenario_count),
                seed_start=int(seed_start),
            )
            result_source = "fresh_optimization"
        else:
            all_results = cached
            result_source = "cached_uncertainty_baseline"
        scenario_sources.append(_city_scenario_distribution_source(city_key, case, all_results, rng, result_source))
    return {
        "scenario_design_results": _validate_scenario_distribution_source(pd.concat(scenario_sources, ignore_index=True)),
        "reference_design_results": _validate_reference_design_source(reference),
    }


def build_single_outage_city_cvar_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
    baseline_sources: dict[str, pd.DataFrame] | None = None,
    scenario_sources: dict[str, pd.DataFrame] | None = None,
    alpha: float = CVAR_ALPHA,
    lambdas: Sequence[float] = CVAR_LAMBDAS,
) -> dict[str, pd.DataFrame]:
    """Build Figure 5 source data from fixed-design evaluations over existing outage seeds."""
    runtime_config = deepcopy(config or DEFAULT_CONFIG)
    baseline = baseline_sources or build_single_outage_city_baseline_sources(input_root, config=runtime_config)
    if scenario_sources is None or "scenario_design_results" not in scenario_sources:
        raise FileNotFoundError(
            "Figure 5 requires existing Figure 4 scenario_design_results. "
            "Generate Figure 4 first, or pass Figure 4 source data explicitly."
        )
    design_space = _validate_design_space_source(baseline["design_space"])
    scenario_results = _validate_scenario_distribution_source(scenario_sources["scenario_design_results"])
    max_candidates = _cvar_max_candidates_per_city(runtime_config)
    candidates = _build_cvar_candidate_set(design_space, scenario_results, max_candidates_per_city=max_candidates)
    cases_by_city = _singlefamily_cases_by_city(Path(input_root))
    case_configs_by_city: dict[str, dict[str, Any]] = {}
    for city_key in CITY_ORDER:
        case = cases_by_city[city_key]
        _, _, case_config, _ = load_input_case(case, runtime_config)
        case_configs_by_city[city_key] = scenario_config(case_config, case.case_key)
    scenario_eval = _evaluate_cvar_candidate_scenarios_parallel(
        candidates,
        scenario_results,
        cases_by_city,
        case_configs_by_city,
        runtime_config,
    )
    summary = _summarize_cvar_candidate_results(candidates, scenario_eval, alpha=float(alpha))
    curve = _select_cvar_design_curve(summary, alpha=float(alpha), lambdas=lambdas)
    return {
        "cvar_design_curve": _validate_cvar_design_curve_source(curve),
        "cvar_candidate_summary": _validate_cvar_candidate_summary_source(summary),
        "cvar_candidate_scenario_results": _validate_cvar_candidate_scenario_source(scenario_eval),
    }


def make_single_outage_city_baseline_figures(
    out: Path | str,
    *,
    input_root: Path | str = "Input",
    config: dict[str, Any] | None = None,
    figures: Sequence[int | str] | None = None,
    sources: dict[str, pd.DataFrame] | None = None,
    case_key: str = SINGLEFAMILY_CASE_KEY,
) -> dict[str, Path]:
    """Create single-outage baseline figures and matching source data."""
    selected = _expand_figures(figures)
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_single_outage_city_sources(
        input_root,
        config=config,
        figures=selected,
        sources=sources,
        output_dir=output_dir,
        case_key=case_key,
    )
    paths: dict[str, Path] = {}
    for figure_no in selected:
        base = output_dir / f"{case_key}_{FIGURE_NAMES[figure_no]}"
        paths[f"figure_{figure_no}_source_data"] = write_source_data(base, _figure_source(prepared, figure_no))
        plotter = {
            1: plot_single_outage_city_design_space_contours,
            "1b": plot_single_outage_city_design_space_contours_singapore_brisbane,
            "1c": plot_single_outage_city_affordable_lolp_contours,
            "1d": plot_single_outage_city_day_night_affordable_lolp_contours,
            2: plot_single_outage_city_cost_distribution,
            "2b": plot_single_outage_city_cost_distribution_singapore_brisbane,
            "2c": plot_single_outage_city_affordable_lolp_cost_distribution,
            "2d": plot_single_outage_city_day_night_cost_distribution,
            3: plot_single_outage_city_energy_balance,
            "3b": plot_single_outage_city_energy_balance_singapore_brisbane,
            "3c": plot_single_outage_city_affordable_lolp_energy_balance,
            4: plot_single_outage_city_scenario_distribution,
            5: plot_single_outage_city_cvar_tradeoff,
            "6a": plot_single_outage_city_tou_profile_four_city,
            "6b": plot_single_outage_city_tou_profile_singapore_brisbane,
            7: plot_single_outage_city_brisbane_sensitivity,
        }[figure_no]
        paths.update(plotter(output_dir, case_key, prepared))
    return paths


def _prepare_single_outage_city_sources(
    input_root: Path | str,
    *,
    config: dict[str, Any] | None,
    figures: Sequence[int | str],
    sources: dict[str, pd.DataFrame] | None,
    output_dir: Path,
    case_key: str,
) -> dict[str, pd.DataFrame]:
    selected = set(figures)
    if sources is not None:
        prepared = dict(sources)
    elif selected.intersection({1, "1b", 2, "2b", 3, "3b", 4}):
        prepared = build_single_outage_city_baseline_sources(input_root, config=config)
    elif 5 in selected:
        prepared = _read_existing_figure_source(output_dir, case_key, 1)
    else:
        prepared = {}
    if selected.intersection({"6a", "6b"}) and "tou_profile" not in prepared:
        prepared["tou_profile"] = build_single_outage_city_tou_profile_source(input_root)
    selected_affordable_figures = selected.intersection({"1c", "2c", "3c", "1d", "2d"})
    required_affordable_sources = {
        key
        for figure in selected_affordable_figures
        for key in SOURCE_KEYS[figure]
    }
    if selected_affordable_figures and sources is None:
        for figure in selected_affordable_figures:
            if set(SOURCE_KEYS[figure]).issubset(prepared):
                continue
            try:
                prepared.update(_read_existing_figure_source(output_dir, case_key, figure))
            except FileNotFoundError:
                pass
        if selected.intersection({"1d", "2d"}) and "affordable_design_space" not in prepared:
            try:
                prepared.update(_read_existing_figure_source(output_dir, case_key, "1c"))
            except FileNotFoundError:
                pass
    base_affordable_required = {"affordable_design_space"}
    if selected_affordable_figures and not base_affordable_required.issubset(prepared):
        prepared.update(build_single_outage_city_affordable_lolp_sources(input_root, config=config))
    if selected_affordable_figures and "affordable_design_space" in prepared:
        prepared.update(
            build_single_outage_city_affordable_incremental_sources(
                input_root,
                config=config,
                sources=prepared,
                allow_new_evaluations=sources is None,
            )
        )
    if selected.intersection({"1d", "2d"}) and not {"day_night_affordable_design_space", "day_night_event_summary"}.issubset(prepared):
        prepared.update(
            build_single_outage_city_day_night_sources(
                input_root,
                config=config,
                affordable_sources=prepared,
            )
        )
    if "2d" in selected and "day_night_cost_distribution" not in prepared:
        prepared["day_night_cost_distribution"] = build_single_outage_city_day_night_cost_distribution_source(
            prepared["day_night_affordable_design_space"]
        )
    if selected_affordable_figures and not required_affordable_sources.issubset(prepared):
        missing = required_affordable_sources - set(prepared)
        required_without_optional = missing - OPTIONAL_SOURCE_KEYS.get("1c", set())
        if required_without_optional:
            raise ValueError("Single-outage affordable source preparation missing table(s): " + ", ".join(sorted(required_without_optional)))
    if 7 in selected and "brisbane_sensitivity_results" not in prepared:
        prepared["brisbane_sensitivity_results"] = build_single_outage_city_brisbane_sensitivity_source(
            input_root,
            config=config,
        )
    if 4 in set(figures) and not all(key in prepared for key in SOURCE_KEYS[4]):
        if "design_space" not in prepared:
            prepared.update(build_single_outage_city_baseline_sources(input_root, config=config))
        prepared.update(
            build_single_outage_city_scenario_distribution_sources(
                input_root,
                config=config,
                baseline_sources=prepared,
            )
        )
    if 5 in set(figures) and not all(key in prepared for key in SOURCE_KEYS[5]):
        if "design_space" not in prepared:
            prepared.update(build_single_outage_city_baseline_sources(input_root, config=config))
        if "scenario_design_results" not in prepared:
            prepared.update(_read_existing_figure_source(output_dir, case_key, 4))
        prepared.update(
            build_single_outage_city_cvar_sources(
                input_root,
                config=config,
                baseline_sources=prepared,
                scenario_sources=prepared,
            )
        )
    return prepared


def calculate_pv_self_use_metrics(dispatch: pd.DataFrame, annual_load_kwh: float | None = None) -> dict[str, float]:
    """Return source-tracked SSR/SCR metrics from dispatch flow columns."""
    required = {"pv_to_load_kwh", "battery_discharge_from_pv_kwh", "pv_available_kwh"}
    missing = required - set(dispatch.columns)
    if missing:
        raise ValueError("PV self-use metrics require dispatch flow column(s): " + ", ".join(sorted(missing)))
    direct_pv_to_load = float(pd.to_numeric(dispatch["pv_to_load_kwh"], errors="coerce").sum())
    pv_origin_battery_discharge = float(pd.to_numeric(dispatch["battery_discharge_from_pv_kwh"], errors="coerce").sum())
    pv_self_served = max(0.0, direct_pv_to_load + pv_origin_battery_discharge)
    pv_available = max(0.0, float(pd.to_numeric(dispatch["pv_available_kwh"], errors="coerce").sum()))
    if annual_load_kwh is None:
        if "total_load_e_equiv_kwh" not in dispatch.columns:
            raise ValueError("annual_load_kwh or total_load_e_equiv_kwh is required for SSR.")
        load = max(0.0, float(pd.to_numeric(dispatch["total_load_e_equiv_kwh"], errors="coerce").sum()))
    else:
        load = max(0.0, float(annual_load_kwh))
    return {
        "direct_pv_to_load_kwh": direct_pv_to_load,
        "pv_origin_battery_discharge_to_load_kwh": pv_origin_battery_discharge,
        "pv_self_served_load_kwh": min(pv_self_served, load) if load > 0.0 else 0.0,
        "annual_pv_available_kwh": pv_available,
        "annual_load_e_equiv_kwh": load,
        "self_sufficiency_rate": float(np.clip(pv_self_served / load, 0.0, 1.0)) if load > 0.0 else 0.0,
        "self_consumption_rate": float(np.clip(pv_self_served / pv_available, 0.0, 1.0)) if pv_available > 0.0 else 0.0,
    }


def plot_single_outage_city_design_space_contours(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_design_space_contours(
        out,
        case_key,
        sources,
        figure_no=1,
        city_order=FIGURE_1_TO_4_CITY_ORDER,
        layout="metric_rows",
    )


def plot_single_outage_city_design_space_contours_singapore_brisbane(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_design_space_contours(
        out,
        case_key,
        sources,
        figure_no="1b",
        city_order=SINGAPORE_BRISBANE_CITY_ORDER,
        layout="city_rows",
    )


def plot_single_outage_city_affordable_lolp_contours(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    source = _validate_affordable_design_space_source(sources["affordable_design_space"])
    plot_source = _validate_affordable_plot_grid_source(
        sources.get("affordable_design_space_plot_grid", _affordable_design_space_plot_grid_source(source))
    )
    _, units = add_display_unit_columns(pd.concat([source, plot_source], ignore_index=True, sort=False))
    display_source, _ = add_display_unit_columns(source, units=units)
    display_plot_source, _ = add_display_unit_columns(plot_source, units=units)
    metric_specs = tuple(
        spec
        for spec in _contour_metric_specs(units)
        if spec["key"] in {"system_total_cost", "lolp"}
    )
    city_order = _city_order(display_source, order=FIGURE_C_CITY_ORDER)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES["1c"])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(14.2, 6.7), constrained_layout=False)
    grid = fig.add_gridspec(
        len(metric_specs),
        len(city_order) + 1,
        left=0.070,
        right=0.945,
        bottom=0.135,
        top=0.895,
        wspace=0.28,
        hspace=0.34,
        width_ratios=[1.0] * len(city_order) + [0.060],
    )
    for metric_index, spec in enumerate(metric_specs):
        metric_values = display_plot_source[spec["column"]].astype(float).to_numpy()
        levels = _metric_levels(metric_values, spec["unit_type"])
        cbar_artist = None
        for city_index, city_key in enumerate(city_order):
            ax = fig.add_subplot(grid[metric_index, city_index])
            city_source = display_source[display_source["city_key"].astype(str) == city_key].copy()
            city_plot_source = display_plot_source[display_plot_source["city_key"].astype(str) == city_key].copy()
            cbar_artist = _draw_contour_panel(mpl, ax, city_plot_source, spec, levels)
            boundary = float(city_plot_source["affordable_boundary_cost_sgd_per_year"].iloc[0]) / max(float(units.cost_divisor), 1e-9)
            _draw_affordable_cost_boundary(ax, city_plot_source, boundary, DISPLAY_SYSTEM_COST)
            _mark_lolp_zero_design(ax, city_source)
            if metric_index == 0:
                ax.set_title(CITY_LABELS[city_key], loc="left", pad=5)
            if city_index == 0:
                ax.set_ylabel(f"{spec['title']}\n{units.pv_axis_label}")
            else:
                ax.set_yticklabels([])
            if metric_index == len(metric_specs) - 1:
                ax.set_xlabel(units.battery_axis_label)
            else:
                ax.set_xticklabels([])
            ax.set_box_aspect(1.0)
        cax = fig.add_subplot(grid[metric_index, -1])
        if cbar_artist is not None:
            cbar = fig.colorbar(cbar_artist, cax=cax)
            _format_metric_colorbar(cbar, levels, spec)

    legend_handles = [
        Line2D([0], [0], color=NATURE_PALETTE["neutral_black"], linewidth=1.8, label="0 PV / 0 BESS cost"),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=NATURE_PALETTE["neutral_black"],
            markeredgecolor="white",
            markeredgewidth=0.50,
            markersize=8.5,
            label="LOLP=0 min-cost design",
        ),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.52, 0.018),
        handlelength=1.65,
    )
    base = out / f"{case_key}_{FIGURE_NAMES['1c']}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_day_night_affordable_lolp_contours(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    source = _validate_day_night_design_space_source(sources["day_night_affordable_design_space"])
    _, units = add_display_unit_columns(source)
    display_source, _ = add_display_unit_columns(source, units=units)
    metric_specs = tuple(
        spec
        for spec in _contour_metric_specs(units)
        if spec["key"] in {"system_total_cost", "lolp"}
    )
    panels = [(city, scenario) for city in FIGURE_D_CITY_ORDER for scenario in FIGURE_D_SCENARIOS]
    mpl, plt = matplotlib_context(out / FIGURE_NAMES["1d"])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(14.2, 6.7), constrained_layout=False)
    grid = fig.add_gridspec(
        len(metric_specs),
        len(panels) + 1,
        left=0.070,
        right=0.945,
        bottom=0.135,
        top=0.895,
        wspace=0.28,
        hspace=0.34,
        width_ratios=[1.0] * len(panels) + [0.060],
    )
    for metric_index, spec in enumerate(metric_specs):
        metric_values = display_source[spec["column"]].astype(float).to_numpy()
        levels = _metric_levels(metric_values, spec["unit_type"])
        cbar_artist = None
        for panel_index, (city_key, scenario_key) in enumerate(panels):
            ax = fig.add_subplot(grid[metric_index, panel_index])
            panel_source = display_source[
                (display_source["city_key"].astype(str) == city_key)
                & (display_source["scenario_key"].astype(str) == scenario_key)
            ].copy()
            cbar_artist = _draw_contour_panel(mpl, ax, panel_source, spec, levels)
            boundary = float(panel_source["affordable_boundary_cost_sgd_per_year"].iloc[0]) / max(float(units.cost_divisor), 1e-9)
            _draw_affordable_cost_boundary(ax, panel_source, boundary, DISPLAY_SYSTEM_COST)
            _mark_lolp_zero_design(ax, panel_source[panel_source["point_source"].astype(str) != "interpolated_for_plot"].copy())
            if metric_index == 0:
                ax.set_title(f"{CITY_LABELS[city_key]} {scenario_key}", loc="left", pad=5)
            if panel_index == 0:
                ax.set_ylabel(f"{spec['title']}\n{units.pv_axis_label}")
            else:
                ax.set_yticklabels([])
            if metric_index == len(metric_specs) - 1:
                ax.set_xlabel(units.battery_axis_label)
            else:
                ax.set_xticklabels([])
            ax.set_box_aspect(1.0)
        cax = fig.add_subplot(grid[metric_index, -1])
        if cbar_artist is not None:
            cbar = fig.colorbar(cbar_artist, cax=cax)
            _format_metric_colorbar(cbar, levels, spec)
    legend_handles = [
        Line2D([0], [0], color=NATURE_PALETTE["neutral_black"], linewidth=1.8, label="0 PV / 0 BESS cost"),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=NATURE_PALETTE["neutral_black"],
            markeredgecolor="white",
            markeredgewidth=0.50,
            markersize=8.5,
            label="LOLP=0 min-cost design",
        ),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.52, 0.018),
        handlelength=1.65,
    )
    base = out / f"{case_key}_{FIGURE_NAMES['1d']}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_single_outage_city_design_space_contours(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
    *,
    figure_no: int | str,
    city_order: Sequence[str],
    layout: str,
) -> dict[str, Path]:
    source = _validate_design_space_source(sources["design_space"])
    display_source, units = add_display_unit_columns(source)
    metric_specs = _contour_metric_specs(units)
    city_order = _city_order(display_source, order=city_order)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES[figure_no])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    if layout == "city_rows":
        metric_specs = tuple(spec for spec in metric_specs if spec["key"] != "lolp")
        display_source = display_source[display_source["city_key"].astype(str).isin(city_order)].copy()
        fig = plt.figure(figsize=(14.0, 6.2), constrained_layout=False)
        width_ratios = [ratio for _ in metric_specs for ratio in (1.0, 0.085)]
        grid = fig.add_gridspec(
            len(city_order),
            len(metric_specs) * 2,
            left=0.065,
            right=0.950,
            bottom=0.155,
            top=0.855,
            wspace=0.34,
            hspace=0.28,
            width_ratios=width_ratios,
        )
        for metric_index, spec in enumerate(metric_specs):
            metric_values = display_source[spec["column"]].astype(float).to_numpy()
            levels = _metric_levels(metric_values, spec["unit_type"], bound_to_data=spec["key"] == "ue")
            cbar_artist = None
            panel_column = metric_index * 2
            colorbar_column = panel_column + 1
            for city_index, city_key in enumerate(city_order):
                ax = fig.add_subplot(grid[city_index, panel_column])
                city_source = display_source[display_source["city_key"].astype(str) == city_key].copy()
                cbar_artist = _draw_contour_panel(mpl, ax, city_source, spec, levels)
                _mark_selected_design(ax, city_source)
                if city_index == 0:
                    ax.set_title(spec["title"], loc="left", pad=5)
                if metric_index == 0:
                    ax.set_ylabel(f"{CITY_LABELS[city_key]}\n{units.pv_axis_label}")
                else:
                    ax.set_yticklabels([])
                if city_index == len(city_order) - 1:
                    ax.set_xlabel(units.battery_axis_label)
                else:
                    ax.set_xticklabels([])
                ax.set_box_aspect(1.0)
            cax = fig.add_subplot(grid[:, colorbar_column])
            if cbar_artist is not None:
                cbar = fig.colorbar(cbar_artist, cax=cax)
                _format_metric_colorbar(cbar, levels, spec, tick_position="right")
        base = out / f"{case_key}_{FIGURE_NAMES[figure_no]}"
        paths = save_publication_figure(fig, base)
        plt.close(fig)
        return paths

    fig = plt.figure(figsize=(11.2, 11.4), constrained_layout=False)
    grid = fig.add_gridspec(
        len(metric_specs),
        len(city_order) + 1,
        left=0.075,
        right=0.945,
        bottom=0.075,
        top=0.935,
        wspace=0.24,
        hspace=0.22,
        width_ratios=[1.0] * len(city_order) + [0.055],
    )

    for metric_index, spec in enumerate(metric_specs):
        metric_values = display_source[spec["column"]].astype(float).to_numpy()
        levels = _metric_levels(metric_values, spec["unit_type"])
        cbar_artist = None
        for city_index, city_key in enumerate(city_order):
            ax = fig.add_subplot(grid[metric_index, city_index])
            city_source = display_source[display_source["city_key"].astype(str) == city_key].copy()
            cbar_artist = _draw_contour_panel(mpl, ax, city_source, spec, levels)
            _mark_selected_design(ax, city_source)
            if metric_index == 0:
                ax.set_title(CITY_LABELS[city_key], loc="left", pad=5)
            if city_index == 0:
                ax.set_ylabel(f"{spec['title']}\n{units.pv_axis_label}")
            else:
                ax.set_yticklabels([])
            if metric_index == len(metric_specs) - 1:
                ax.set_xlabel(units.battery_axis_label)
            else:
                ax.set_xticklabels([])
            ax.set_box_aspect(1.0)
        cax = fig.add_subplot(grid[metric_index, -1])
        if cbar_artist is not None:
            cbar = fig.colorbar(cbar_artist, cax=cax)
            _format_metric_colorbar(cbar, levels, spec)

    base = out / f"{case_key}_{FIGURE_NAMES[figure_no]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_cost_distribution(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_cost_distribution(
        out,
        case_key,
        sources,
        figure_no=2,
        city_order=FIGURE_1_TO_4_CITY_ORDER,
        layout="four_city",
    )


def plot_single_outage_city_cost_distribution_singapore_brisbane(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_cost_distribution(
        out,
        case_key,
        sources,
        figure_no="2b",
        city_order=SINGAPORE_BRISBANE_CITY_ORDER,
        layout="two_city",
    )


def plot_single_outage_city_affordable_lolp_cost_distribution(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    costs = _validate_affordable_cost_distribution_source(sources["affordable_cost_distribution"])
    city_order = _city_order(costs, order=FIGURE_C_CITY_ORDER)
    units = figure_display_units(_best_design_frame_from_costs(costs))
    divisor = units.cost_divisor
    value_label = units.annual_cost_axis_label.replace(" (", "\n(", 1)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES["2c"])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.100, right=0.980, bottom=0.255, top=0.895, wspace=0.25, hspace=0.70)
    colors = _cost_component_colors()
    y_max = _cost_axis_upper(costs, divisor)
    axes_flat = axes.flat
    for index, city_key in enumerate(city_order):
        ax = axes_flat[index]
        city_costs = costs[costs["city_key"].astype(str) == city_key].copy()
        _stack_city_design_cost_bars(
            ax,
            city_costs,
            colors,
            divisor=divisor,
            units=units,
            design_case_order=AFFORDABLE_COST_DESIGN_CASES,
        )
        ax.set_title(CITY_LABELS[city_key], loc="left")
        ax.set_ylim(0.0, y_max)
        if index % 2 == 0:
            ax.set_ylabel(value_label)
        _single_outage_panel_label(ax, chr(ord("a") + index))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.030), ncol=4, handlelength=1.15)
    base = out / f"{case_key}_{FIGURE_NAMES['2c']}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_day_night_cost_distribution(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    costs = _validate_day_night_cost_distribution_source(sources["day_night_cost_distribution"])
    units = figure_display_units(_best_design_frame_from_costs(costs))
    divisor = units.cost_divisor
    value_label = units.annual_cost_axis_label.replace(" (", "\n(", 1)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES["2d"])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 4.5), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.940, bottom=0.410, top=0.805, wspace=0.24)
    colors = _cost_component_colors()
    y_max = _cost_axis_upper(costs, divisor)
    axes_flat = np.ravel(axes)
    for index, city_key in enumerate(FIGURE_D_CITY_ORDER):
        ax = axes_flat[index]
        city_costs = costs[costs["city_key"].astype(str) == city_key].copy()
        _stack_city_design_cost_bars(
            ax,
            city_costs,
            colors,
            divisor=divisor,
            units=units,
            design_case_order=DAY_NIGHT_COST_DESIGN_CASES,
        )
        for tick in ax.get_xticklabels():
            tick.set_ha("center")
            tick.set_rotation(18)
        ax.set_title(CITY_LABELS[city_key], loc="left")
        ax.set_ylim(0.0, y_max)
        if index == 0:
            ax.set_ylabel(value_label)
        _single_outage_panel_label(ax, chr(ord("a") + index))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.035), ncol=4, handlelength=1.15)
    base = out / f"{case_key}_{FIGURE_NAMES['2d']}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_single_outage_city_cost_distribution(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
    *,
    figure_no: int | str,
    city_order: Sequence[str],
    layout: str,
) -> dict[str, Path]:
    costs = _validate_cost_distribution_source(sources["cost_distribution"])
    city_order = _city_order(costs, order=city_order)
    units = figure_display_units(_best_design_frame_from_costs(costs))
    divisor = units.cost_divisor
    value_label = units.annual_cost_axis_label.replace(" (", "\n(", 1)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES[figure_no])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    if layout == "two_city":
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.95), sharey=True, constrained_layout=False)
        fig.subplots_adjust(left=0.090, right=0.985, bottom=0.360, top=0.820, wspace=0.25)
        axes_flat = np.ravel(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharey=True, constrained_layout=False)
        fig.subplots_adjust(left=0.085, right=0.982, bottom=0.240, top=0.895, wspace=0.22, hspace=0.68)
        axes_flat = axes.flat
    colors = _cost_component_colors()
    y_max = _cost_axis_upper(costs, divisor)

    for index, city_key in enumerate(city_order):
        ax = axes_flat[index]
        city_costs = costs[costs["city_key"].astype(str) == city_key].copy()
        _stack_city_design_cost_bars(ax, city_costs, colors, divisor=divisor, units=units)
        ax.set_title(CITY_LABELS[city_key], loc="left")
        ax.set_ylim(0.0, y_max)
        if index % 2 == 0:
            ax.set_ylabel(value_label)
        _single_outage_panel_label(ax, chr(ord("a") + index))

    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_y = 0.035 if layout == "two_city" else 0.030
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, legend_y), ncol=4, handlelength=1.15)
    base = out / f"{case_key}_{FIGURE_NAMES[figure_no]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_tou_profile_four_city(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_tou_profile(
        out,
        case_key,
        sources,
        figure_no="6a",
        city_order=FIGURE_1_TO_4_CITY_ORDER,
    )


def plot_single_outage_city_tou_profile_singapore_brisbane(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_tou_profile(
        out,
        case_key,
        sources,
        figure_no="6b",
        city_order=SINGAPORE_BRISBANE_CITY_ORDER,
    )


def _plot_single_outage_city_tou_profile(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
    *,
    figure_no: str,
    city_order: Sequence[str],
) -> dict[str, Path]:
    source = _validate_tou_profile_source(sources["tou_profile"])
    city_order = _city_order(source, order=city_order)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES[figure_no])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    fig_width = 7.4 if figure_no == "6a" else 6.1
    fig, ax = plt.subplots(figsize=(fig_width, 3.80), constrained_layout=False)
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.230, top=0.745)

    for city_key in city_order:
        group = source[source["city_key"].astype(str) == city_key].sort_values("hour")
        x, y = _tou_step_xy(group)
        ax.step(
            x,
            y,
            where="post",
            color=CITY_COLORS[city_key],
            linewidth=1.35,
            label=CITY_LABELS[city_key],
        )

    ax.set_xlim(0.0, 24.0)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("TOU tariff (SGD kWh$^{-1}$)")
    y_max = max(float(source["tou_sgd_per_kwh"].max()) * 1.12, 0.05)
    ax.set_ylim(0.0, y_max)
    ax.grid(False)
    _single_outage_panel_label(ax, "a" if figure_no == "6a" else "b")
    legend_ncol = len(city_order)
    fig.legend(
        loc="upper center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.55, 0.950),
        handlelength=1.8,
    )
    base = out / f"{case_key}_{FIGURE_NAMES[figure_no]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_energy_balance(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_energy_balance(
        out,
        case_key,
        sources,
        figure_no=3,
        city_order=FIGURE_1_TO_4_CITY_ORDER,
        layout="four_city",
    )


def plot_single_outage_city_energy_balance_singapore_brisbane(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_energy_balance(
        out,
        case_key,
        sources,
        figure_no="3b",
        city_order=SINGAPORE_BRISBANE_CITY_ORDER,
        layout="two_city",
    )


def plot_single_outage_city_affordable_lolp_energy_balance(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    return _plot_single_outage_city_energy_balance(
        out,
        case_key,
        sources,
        figure_no="3c",
        city_order=FIGURE_C_CITY_ORDER,
        layout="four_city",
        source_key="affordable_energy_balance",
    )


def _plot_single_outage_city_energy_balance(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
    *,
    figure_no: int | str,
    city_order: Sequence[str],
    layout: str,
    source_key: str = "energy_balance",
) -> dict[str, Path]:
    raw_source = sources[source_key]
    if "city_key" in raw_source.columns:
        raw_source = raw_source[raw_source["city_key"].astype(str).isin(city_order)].copy()
    source = _validate_energy_balance_source(
        raw_source,
        city_order=city_order,
    )
    city_order = _city_order(source, order=city_order)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES[figure_no])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    palette = _single_outage_energy_balance_palette()
    shared_ylim = _shared_energy_balance_ylim(source)
    if layout == "two_city":
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.55), sharex=True, sharey=True, constrained_layout=False)
        fig.subplots_adjust(left=0.120, right=0.895, bottom=0.410, top=0.790, wspace=0.30)
        axes_flat = np.ravel(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.35), sharex=True, sharey=True, constrained_layout=False)
        fig.subplots_adjust(left=0.105, right=0.900, bottom=0.290, top=0.850, wspace=0.30, hspace=0.46)
        axes_flat = axes.flat
    for index, city_key in enumerate(city_order):
        ax = axes_flat[index]
        group = source[source["city_key"].astype(str) == city_key].sort_values("plot_hour")
        _draw_energy_balance_panel(
            ax,
            group,
            palette,
            shared_ylim,
            show_ylabel=index % 2 == 0,
            show_xlabel=layout == "two_city" or index >= 2,
            show_soc_ylabel=(layout == "two_city" and index == len(city_order) - 1) or (layout != "two_city" and index % 2 == 1),
        )
        _single_outage_panel_label(ax, chr(ord("a") + index))

    legend_handles = _energy_balance_legend_handles(Patch, Line2D, palette)
    legend_ncol = 4 if layout == "two_city" else 5
    legend_y = 0.035 if layout == "two_city" else 0.035
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.52, legend_y),
        handlelength=1.35,
    )
    base = out / f"{case_key}_{FIGURE_NAMES[figure_no]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_scenario_distribution(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    scenario_results = _validate_scenario_distribution_source(sources["scenario_design_results"])
    reference = _validate_reference_design_source(sources["reference_design_results"])
    scenario_display, reference_display, units = _scenario_distribution_display_sources(scenario_results, reference)
    city_order = _city_order(scenario_display, order=FIGURE_1_TO_4_CITY_ORDER)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES[4])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.65), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.370, top=0.780, wspace=0.58)
    metrics = [
        (DISPLAY_PV_CAPACITY, units.pv_axis_label, "PV size", DISPLAY_PV_CAPACITY),
        (DISPLAY_BATTERY_ENERGY, units.battery_axis_label, "Battery size", DISPLAY_BATTERY_ENERGY),
        (DISPLAY_UE, units.ue_axis_label, "Unserved energy", DISPLAY_UE),
        ("energy_based_lolp_percent", "Energy-based LOLP (%)", "Energy-based LOLP", "energy_based_lolp_percent"),
    ]
    for label, ax, (scenario_column, ylabel, title, reference_column) in zip("abcd", axes, metrics, strict=True):
        _plot_scenario_distribution_metric(ax, scenario_display, reference_display, city_order, scenario_column, reference_column)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        _single_outage_panel_label(ax, label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=NATURE_PALETTE["neutral_mid"],
            markeredgewidth=0.0,
            markersize=4.0,
            alpha=0.70,
            label="100 scenarios",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=NATURE_PALETTE["neutral_black"],
            markeredgewidth=0.8,
            markersize=4.2,
            label="Baseline reference",
        ),
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=2, bbox_to_anchor=(0.55, 0.98))
    base = out / f"{case_key}_{FIGURE_NAMES[4]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_cvar_tradeoff(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    curve = _validate_cvar_design_curve_source(sources["cvar_design_curve"])
    display_base = curve.copy()
    display_base["system_total_cost_sgd_per_year"] = display_base["expected_annual_cost_sgd_per_year"]
    display_base["system_total_cost_million_sgd_per_year"] = display_base["system_total_cost_sgd_per_year"] / 1e6
    display, units = add_display_unit_columns(display_base)
    display["_display_expected_annual_cost"] = display["expected_annual_cost_sgd_per_year"].astype(float) / units.cost_divisor
    display["_display_cvar90_outage_loss"] = display["cvar90_outage_loss_sgd_per_year"].astype(float) / units.cost_divisor
    city_order = _city_order(display)
    mpl, plt = matplotlib_context(out / FIGURE_NAMES[5])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D

    cost_label = "kSGD yr$^{-1}$" if units.use_ksgd else "million SGD yr$^{-1}$"
    metrics = [
        (DISPLAY_PV_CAPACITY, units.pv_axis_label, "Optimal PV capacity"),
        (DISPLAY_BATTERY_ENERGY, units.battery_axis_label, "Optimal BESS energy"),
        ("_display_expected_annual_cost", f"Expected annual cost ({cost_label})", "Expected annual cost"),
        ("_display_cvar90_outage_loss", f"CVaR90 outage loss ({cost_label})", "Tail outage loss (CVaR90)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.160, top=0.835, wspace=0.33, hspace=0.40)
    for index, (ax, (column, ylabel, title)) in enumerate(zip(axes.flat, metrics, strict=True)):
        for city_key in city_order:
            group = display[display["city_key"].astype(str) == city_key].sort_values("risk_aversion_lambda")
            ax.plot(
                group["risk_aversion_lambda"].astype(float),
                group[column].astype(float),
                color=CITY_COLORS[city_key],
                marker="o",
                markersize=3.2,
                markeredgewidth=0.0,
                linewidth=1.25,
                label=CITY_LABELS[city_key],
            )
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.60)
        _single_outage_panel_label(ax, chr(ord("a") + index))
        if index >= 2:
            ax.set_xlabel("Risk-aversion coefficient lambda")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CITY_COLORS[city_key],
            marker="o",
            markeredgewidth=0.0,
            markersize=3.4,
            linewidth=1.25,
            label=CITY_LABELS[city_key],
        )
        for city_key in city_order
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.55, 0.970),
        handlelength=1.45,
    )
    base = out / f"{case_key}_{FIGURE_NAMES[5]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_single_outage_city_brisbane_sensitivity(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    source = _validate_brisbane_sensitivity_source(sources["brisbane_sensitivity_results"])
    mpl, plt = matplotlib_context(out / FIGURE_NAMES[7])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.7), constrained_layout=False)
    fig.subplots_adjust(left=0.080, right=0.890, bottom=0.240, top=0.775, wspace=0.52)
    driver_specs = [
        ("tou_spread", "TOU spread", "TOU spread coefficient k"),
        ("saidi", "SAIDI", "SAIDI (min yr$^{-1}$)"),
        ("voll", "VoLL", "VoLL (SGD kWh$^{-1}$)"),
    ]
    capacity_max = max(
        float(source["pv_capacity_kw"].max()),
        float(source["battery_energy_kwh"].max()),
        1.0,
    )
    cost_ksgd = source["system_total_cost_sgd_per_year"].astype(float) / 1000.0
    cost_max = max(float(cost_ksgd.max()), 1.0)
    capacity_ylim = (0.0, capacity_max * 1.15)
    cost_ylim = (0.0, cost_max * 1.12)

    for index, (ax, (driver, title, xlabel)) in enumerate(zip(axes, driver_specs, strict=True)):
        group = source[source["sensitivity_driver"].astype(str) == driver].sort_values("x_value")
        x_values = group["x_value"].astype(float)
        cost_axis = ax.twinx()
        cost_axis.spines["right"].set_visible(True)
        plot_smooth_line(
            ax,
            x_values,
            group["pv_capacity_kw"].astype(float),
            color=NATURE_PALETTE["blue"],
            linewidth=1.35,
            linestyle="-",
            show_points=False,
        )
        plot_smooth_line(
            ax,
            x_values,
            group["battery_energy_kwh"].astype(float),
            color=NATURE_PALETTE["gold"],
            linewidth=1.35,
            linestyle="--",
            show_points=False,
        )
        plot_smooth_line(
            cost_axis,
            x_values,
            group["system_total_cost_sgd_per_year"].astype(float) / 1000.0,
            color=NATURE_PALETTE["neutral_dark"],
            linewidth=1.25,
            linestyle="-.",
            show_points=False,
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylim(*capacity_ylim)
        cost_axis.set_ylim(*cost_ylim)
        ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
        if index == 0:
            ax.set_ylabel("Optimal capacity (kW or kWh)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        if index == len(driver_specs) - 1:
            cost_axis.set_ylabel("Annualized cost (kSGD yr$^{-1}$)")
        else:
            cost_axis.set_ylabel("")
            cost_axis.tick_params(axis="y", labelright=False)
        _single_outage_panel_label(ax, chr(ord("a") + index))

    legend_handles = [
        Line2D([0], [0], color=NATURE_PALETTE["blue"], linewidth=1.35, linestyle="-", label="Optimal PV"),
        Line2D([0], [0], color=NATURE_PALETTE["gold"], linewidth=1.35, linestyle="--", label="Optimal BESS"),
        Line2D([0], [0], color=NATURE_PALETTE["neutral_dark"], linewidth=1.25, linestyle="-.", label="Annualized cost"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.50, 0.970),
        handlelength=2.0,
    )
    base = out / f"{case_key}_{FIGURE_NAMES[7]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _energy_balance_legend_handles(patch_cls: Any, line_cls: Any, palette: dict[str, str]) -> list[Any]:
    return [
        patch_cls(facecolor=palette["pv"], edgecolor="#3A3A3A", label="PV used"),
        patch_cls(facecolor=palette["grid"], edgecolor="#3A3A3A", label="Grid import"),
        patch_cls(facecolor=palette["battery_discharge"], edgecolor="#3A3A3A", label="BESS discharge"),
        patch_cls(facecolor=palette["load"], edgecolor="#3A3A3A", label="Served energy"),
        patch_cls(facecolor=palette["battery_charge"], edgecolor="#3A3A3A", label="BESS charge"),
        patch_cls(facecolor=palette["curtailment"], edgecolor="#3A3A3A", label="PV curtailed"),
        patch_cls(facecolor=palette["loss"], edgecolor="#3A3A3A", label="Conversion loss"),
        patch_cls(facecolor=palette["unserved"], edgecolor="#3A3A3A", label="Unserved energy"),
        line_cls([0], [0], color=palette["soc"], linestyle="-.", linewidth=1.1, label="BESS SOC"),
    ]


def _build_city_sources(
    city_key: str,
    case: InputCase,
    global_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, _, case_config, _ = load_input_case(case, global_config)
    preview_frame, _ = build_case_frame(case, case_config, max(case_config["pv"]["capacity_kw_candidates"]))
    outage_profile, outage_events = _baseline_outage(case_config, case.case_key, len(preview_frame))
    outage_summary = summarize_outages(outage_profile, outage_events)
    outage_weighted_total = _outage_weighted_load_total(preview_frame, outage_profile, case_config)

    def pv_capacity_frame_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, case_config, pv_capacity_kw)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    best_rows, best_dispatch, candidate_summary, candidate_dispatches = resilience_design(
        pv_capacity_frame_builder,
        case_config,
        case.case_key,
    )
    aware_cfg = scenario_config(case_config, case.case_key)
    annual_load = float(case_config["capacity_search"]["annual_load_e_equiv_kwh"])
    design_space = _city_design_space_source(
        city_key,
        case,
        candidate_summary,
        candidate_dispatches,
        annual_load,
        outage_weighted_total,
        outage_summary,
    )
    best = best_rows.iloc[0]
    cost_source = _city_cost_distribution_source(
        city_key,
        case,
        aware_cfg,
        candidate_summary,
        pv_capacity_frame_builder,
        best,
    )
    selected_frame, _ = build_case_frame(case, case_config, float(best["pv_capacity_kw"]))
    selected_frame = selected_frame.merge(outage_profile, on="hour_index", how="left")
    energy_balance, event_summary = _city_energy_balance_source(
        city_key,
        case,
        case_config,
        selected_frame,
        best_dispatch,
        outage_events,
        best,
    )
    return design_space, cost_source, energy_balance, event_summary


def _figure_c_outage_mode(city_key: str) -> str:
    return "night_window_random" if city_key in FIGURE_C_NIGHT_OUTAGE_CITIES else "random"


def _build_affordable_lolp_city_sources_parallel(
    tasks: Sequence[tuple[str, InputCase, dict[str, Any], str, int]],
    cases_by_city: dict[str, InputCase],
) -> list[dict[str, Any]]:
    if not tasks:
        raise ValueError("Figure 1c/2c/3c source builder requires at least one city task.")
    workers = _figure_c_parallel_worker_count(cases_by_city, len(tasks))
    if workers <= 1:
        return [_build_affordable_lolp_city_sources_task(task) for task in tasks]
    chunksize = max(1, min(4, len(tasks) // max(workers, 1) or 1))
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_build_affordable_lolp_city_sources_task, tasks, chunksize=chunksize))
    except OSError as exc:
        raise RuntimeError(
            "Figure 1c/2c/3c source building requires process-level parallel execution, "
            "but ProcessPoolExecutor could not start."
        ) from exc


def _figure_c_parallel_worker_count(cases_by_city: dict[str, InputCase], total_tasks: int) -> int:
    configured_workers: list[int] = []
    for city_key in FIGURE_C_CITY_ORDER:
        case = cases_by_city.get(city_key)
        if case is None:
            continue
        try:
            value = load_case_parameters(case.parameter_csv).base.get("parallel_workers")
        except (FileNotFoundError, ValueError):
            value = None
        if value is not None:
            configured_workers.append(max(1, int(float(value))))
    base_workers = max(configured_workers) if configured_workers else (os.cpu_count() or 1)
    return min(max(1, int(base_workers)), max(1, int(total_tasks)))


def _extend_affordable_design_space_for_complete_boundary(
    evaluated: pd.DataFrame,
    input_root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    source = _validate_affordable_design_space_source(evaluated).copy()
    for city_key in FIGURE_C_CITY_ORDER:
        for _ in range(AFFORDABLE_PLOT_GRID_MAX_EXPANSIONS):
            city_source = source[source["city_key"].astype(str) == city_key].copy()
            plot_grid = _affordable_design_space_plot_grid_source(city_source)
            flags = _affordable_boundary_edge_touch_flags(plot_grid)
            if not flags["top"] and not flags["right"]:
                break
            new_points = _outer_ring_capacity_points(city_source, expand_pv=flags["top"], expand_battery=flags["right"])
            if not new_points:
                break
            new_rows = _evaluate_affordable_points_for_city(
                input_root,
                config,
                city_key,
                _figure_c_outage_mode(city_key),
                new_points,
                point_source="new_evaluated",
                zero_zero_cost=float(city_source["zero_pv_zero_bess_cost_sgd_per_year"].iloc[0]),
            )
            source = pd.concat([source, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
            source = _refresh_affordable_selection_metadata(source, city_order=FIGURE_C_CITY_ORDER)
    return _validate_affordable_design_space_source(source)


def _affordable_boundary_edge_touch_flags(plot_grid: pd.DataFrame) -> dict[str, bool]:
    source = _validate_affordable_plot_grid_source(plot_grid)
    boundary = float(source["affordable_boundary_cost_sgd_per_year"].iloc[0])
    pivot = source.pivot_table(
        index="pv_capacity_kw",
        columns="battery_energy_kwh",
        values="system_total_cost_sgd_per_year",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.empty:
        return {"top": False, "right": False}
    top_values = pivot.iloc[-1].to_numpy(dtype=float)
    right_values = pivot.iloc[:, -1].to_numpy(dtype=float)
    return {
        "top": _edge_contains_level(top_values, boundary),
        "right": _edge_contains_level(right_values, boundary),
    }


def _edge_contains_level(values: np.ndarray, level: float) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return float(finite.min()) <= float(level) <= float(finite.max())


def _outer_ring_capacity_points(
    city_source: pd.DataFrame,
    *,
    expand_pv: bool,
    expand_battery: bool,
) -> list[tuple[float, float]]:
    pv_axis = sorted({round(float(value), 6) for value in city_source["pv_capacity_kw"].astype(float)})
    battery_axis = sorted({round(float(value), 6) for value in city_source["battery_energy_kwh"].astype(float)})
    if not pv_axis or not battery_axis:
        return []
    new_pv_axis = list(pv_axis)
    new_battery_axis = list(battery_axis)
    new_pv: float | None = None
    new_battery: float | None = None
    if expand_pv:
        max_pv = max(pv_axis)
        step = _positive_axis_step(pv_axis)
        new_pv = round(max(max_pv * AFFORDABLE_PLOT_GRID_EXPANSION_FACTOR, max_pv + step), 6)
        new_pv_axis.append(new_pv)
    if expand_battery:
        max_battery = max(battery_axis)
        step = _positive_axis_step(battery_axis)
        new_battery = round(max(max_battery * AFFORDABLE_PLOT_GRID_EXPANSION_FACTOR, max_battery + step), 6)
        new_battery_axis.append(new_battery)
    existing = {
        (round(float(row["pv_capacity_kw"]), 6), round(float(row["battery_energy_kwh"]), 6))
        for row in city_source.to_dict("records")
    }
    points = []
    for pv in new_pv_axis:
        for battery in new_battery_axis:
            if (new_pv is not None and np.isclose(pv, new_pv, atol=1e-6)) or (
                new_battery is not None and np.isclose(battery, new_battery, atol=1e-6)
            ):
                key = (round(float(pv), 6), round(float(battery), 6))
                if key not in existing:
                    points.append(key)
    return sorted(set(points))


def _positive_axis_step(axis: Sequence[float]) -> float:
    values = sorted({float(value) for value in axis})
    diffs = [b - a for a, b in zip(values, values[1:], strict=False) if b - a > 1e-9]
    return float(min(diffs)) if diffs else max(float(values[-1]) * 0.25, 1.0)


def _evaluate_affordable_points_for_city(
    input_root: Path,
    config: dict[str, Any],
    city_key: str,
    outage_mode: str,
    points: Sequence[tuple[float, float]],
    *,
    point_source: str,
    zero_zero_cost: float | None = None,
    window: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    if not points:
        return []
    context = _city_outage_context(input_root, config, city_key, outage_mode, window=window)
    case = context["case"]
    case_config = context["case_config"]
    aware_cfg = context["aware_config"]
    annual_load = float(case_config["capacity_search"]["annual_load_e_equiv_kwh"])
    outage_weighted_total = float(context["outage_weighted_total_kwh"])
    outage_summary = context["outage_summary"]
    outage_profile = context["outage_profile"]
    by_pv: dict[float, list[float]] = {}
    for pv, battery in points:
        by_pv.setdefault(round(float(pv), 6), []).append(round(float(battery), 6))
    if zero_zero_cost is None:
        by_pv.setdefault(0.0, [])
        if 0.0 not in by_pv[0.0]:
            by_pv[0.0].append(0.0)
    rows: list[dict[str, Any]] = []
    for pv_capacity in sorted(by_pv):
        frame, pv_meta = build_case_frame(case, case_config, pv_capacity)
        scenario_frame = frame.merge(outage_profile, on="hour_index", how="left")
        zero_result, zero_dispatch = evaluate_candidate(scenario_frame, 0.0, pv_meta, aware_cfg)
        zero_reference = _no_bess_reference_from_result_record(zero_result)
        if zero_zero_cost is None and np.isclose(pv_capacity, 0.0, atol=1e-6):
            zero_zero_cost = float(zero_result["pv_bess_system_annual_cost_sgd"])
        for battery_energy in sorted(set(by_pv[pv_capacity])):
            if np.isclose(battery_energy, 0.0, atol=1e-6):
                result = zero_result
                dispatch = zero_dispatch
            else:
                result, dispatch = evaluate_candidate(
                    scenario_frame,
                    battery_energy,
                    pv_meta,
                    aware_cfg,
                    no_bess_reference=zero_reference,
                )
            if zero_zero_cost is None:
                raise ValueError(f"0 PV / 0 BESS cost is required for {CITY_LABELS[city_key]}.")
            rows.append(
                _affordable_design_row_from_evaluation(
                    city_key,
                    case,
                    result,
                    dispatch,
                    annual_load,
                    outage_weighted_total,
                    outage_summary,
                    outage_mode,
                    int(case_config["outage"].get("seed", 42)),
                    float(zero_zero_cost),
                    point_source=point_source,
                )
            )
    requested = {(round(float(pv), 6), round(float(battery), 6)) for pv, battery in points}
    return [
        row
        for row in rows
        if (round(float(row["pv_capacity_kw"]), 6), round(float(row["battery_energy_kwh"]), 6)) in requested
    ]


def _city_outage_context(
    input_root: Path,
    config: dict[str, Any],
    city_key: str,
    outage_mode: str,
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    cases_by_city = _singlefamily_cases_by_city_for_order(input_root, FIGURE_C_CITY_ORDER, FIGURE_C_CASE_BY_CITY)
    case = cases_by_city[city_key]
    _, _, case_config, _ = load_input_case(case, deepcopy(config))
    case_config = deepcopy(case_config)
    outage = case_config.setdefault("outage", {})
    if outage_mode in {"day_window_random", "night_window_random"}:
        outage["forced_event_mode"] = "night_window_random"
        start, end = window or (FIGURE_D_NIGHT_WINDOW if outage_mode == "night_window_random" else FIGURE_D_DAY_WINDOW)
        outage["night_window_start_hour"] = float(start)
        outage["night_window_end_hour"] = float(end)
    else:
        outage["forced_event_mode"] = str(outage_mode)
    preview_frame, _ = build_case_frame(case, case_config, max(case_config["pv"]["capacity_kw_candidates"]))
    outage_profile, outage_events = _baseline_outage(case_config, case.case_key, len(preview_frame))
    return {
        "case": case,
        "case_config": case_config,
        "aware_config": scenario_config(case_config, case.case_key),
        "outage_profile": outage_profile,
        "outage_events": outage_events,
        "outage_summary": summarize_outages(outage_profile, outage_events),
        "outage_weighted_total_kwh": _outage_weighted_load_total(preview_frame, outage_profile, case_config),
    }


def _affordable_design_row_from_evaluation(
    city_key: str,
    case: InputCase,
    result: dict[str, Any],
    dispatch: pd.DataFrame,
    annual_load_kwh: float,
    outage_weighted_total_kwh: float,
    outage_summary: dict[str, float],
    outage_mode: str,
    outage_seed: int,
    zero_zero_cost: float,
    *,
    point_source: str,
) -> dict[str, Any]:
    solar = calculate_pv_self_use_metrics(dispatch, annual_load_kwh)
    ue = float(result["unserved_energy_kwh_per_year"])
    lolp = ue / outage_weighted_total_kwh if outage_weighted_total_kwh > 0.0 else 0.0
    total_cost = float(result["pv_bess_system_annual_cost_sgd"])
    pv_capacity = float(result["pv_capacity_kw"])
    battery_energy = float(result["battery_energy_kwh"])
    return {
        "city_key": city_key,
        "city_label": CITY_LABELS[city_key],
        "case_key": case.case_key,
        "source_case_key": case.case_key,
        "search_stage": str(point_source),
        "search_iteration": 0,
        "pv_capacity_kw": pv_capacity,
        "pv_capacity_mw": pv_capacity / 1000.0,
        "battery_energy_kwh": battery_energy,
        "battery_energy_mwh": battery_energy / 1000.0,
        "system_total_cost_sgd_per_year": total_cost,
        "system_total_cost_million_sgd_per_year": total_cost / 1e6,
        "grid_cost_sgd_per_year": float(result["pv_bess_grid_cost_sgd"]),
        "pv_annual_cost_sgd_per_year": float(result["annualized_pv_cost_sgd"]) + float(result["pv_om_cost_sgd"]),
        "battery_annual_cost_sgd_per_year": float(result["annualized_battery_cost_sgd"]) + float(result["battery_om_cost_sgd"]),
        "lost_load_cost_sgd_per_year": float(result["unserved_energy_cost_sgd_per_year"]),
        "UE_kwh_per_year": ue,
        "UE_mwh_per_year": ue / 1000.0,
        "outage_weighted_load_kwh": outage_weighted_total_kwh,
        "energy_based_LOLP": float(np.clip(lolp, 0.0, 1.0)),
        "energy_based_lolp_percent": float(np.clip(lolp, 0.0, 1.0)) * 100.0,
        "self_sufficiency_rate": solar["self_sufficiency_rate"],
        "self_consumption_rate": solar["self_consumption_rate"],
        "pv_self_served_load_kwh": solar["pv_self_served_load_kwh"],
        "direct_pv_to_load_kwh": solar["direct_pv_to_load_kwh"],
        "pv_origin_battery_discharge_to_load_kwh": solar["pv_origin_battery_discharge_to_load_kwh"],
        "annual_pv_available_kwh": solar["annual_pv_available_kwh"],
        "annual_load_e_equiv_kwh": solar["annual_load_e_equiv_kwh"],
        "simulated_saidi_min_per_year": outage_summary.get("simulated_saidi_min_per_year", np.nan),
        "simulated_saifi_per_year": outage_summary.get("simulated_saifi_per_year", np.nan),
        "outage_mode": str(outage_mode),
        "outage_seed": int(outage_seed),
        "zero_pv_zero_bess_cost_sgd_per_year": float(zero_zero_cost),
        "affordable_boundary_cost_sgd_per_year": float(zero_zero_cost),
        "cost_delta_vs_no_pv_no_bess_sgd_per_year": total_cost - float(zero_zero_cost),
        "point_source": str(point_source),
    }


def _refresh_affordable_selection_metadata(
    source: pd.DataFrame,
    *,
    city_order: Sequence[str],
) -> pd.DataFrame:
    refreshed_frames: list[pd.DataFrame] = []
    for city_key in city_order:
        group = source[source["city_key"].astype(str) == city_key].copy()
        if group.empty:
            continue
        zero_zero = _design_space_row_at_capacity(group, 0.0, 0.0)
        zero_zero_cost = float(zero_zero["system_total_cost_sgd_per_year"])
        group["zero_pv_zero_bess_cost_sgd_per_year"] = zero_zero_cost
        group["affordable_boundary_cost_sgd_per_year"] = zero_zero_cost
        group["cost_delta_vs_no_pv_no_bess_sgd_per_year"] = group["system_total_cost_sgd_per_year"].astype(float) - zero_zero_cost
        group["is_minimum_cost_design"] = False
        group.loc[group["system_total_cost_sgd_per_year"].astype(float).idxmin(), "is_minimum_cost_design"] = True
        group["is_no_pv_no_bess_reference"] = (
            np.isclose(group["pv_capacity_kw"].astype(float), 0.0, atol=1e-6)
            & np.isclose(group["battery_energy_kwh"].astype(float), 0.0, atol=1e-6)
        )
        group["is_lolp_zero_min_cost_design"] = False
        eligible = group[group["UE_kwh_per_year"].astype(float) <= AFFORDABLE_LOLP_ZERO_TOLERANCE_KWH].copy()
        if eligible.empty:
            raise ValueError(f"Figure 1c requires at least one LOLP=0 candidate for {CITY_LABELS[city_key]}.")
        selected_index = eligible.sort_values(
            ["system_total_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"]
        ).index[0]
        group.loc[selected_index, "is_lolp_zero_min_cost_design"] = True
        selected = group.loc[selected_index]
        group["selected_lolp_zero_pv_capacity_kw"] = float(selected["pv_capacity_kw"])
        group["selected_lolp_zero_battery_energy_kwh"] = float(selected["battery_energy_kwh"])
        group["selected_lolp_zero_cost_sgd_per_year"] = float(selected["system_total_cost_sgd_per_year"])
        group["selected_lolp_zero_UE_kwh_per_year"] = float(selected["UE_kwh_per_year"])
        group["selected_lolp_zero_energy_based_LOLP"] = float(selected["energy_based_LOLP"])
        refreshed_frames.append(group)
    return pd.concat(refreshed_frames, ignore_index=True, sort=False)


def _affordable_design_space_plot_grid_source(source: pd.DataFrame) -> pd.DataFrame:
    evaluated = source.copy()
    if "point_source" not in evaluated.columns:
        evaluated["point_source"] = "existing_evaluated"
    rows: list[dict[str, Any]] = []
    for city_key, group in evaluated.groupby("city_key", sort=False):
        city = str(city_key)
        group = group.copy().reset_index(drop=True)
        pv_axis = sorted({round(float(value), 6) for value in group["pv_capacity_kw"].astype(float)})
        battery_axis = sorted({round(float(value), 6) for value in group["battery_energy_kwh"].astype(float)})
        exact: dict[tuple[float, float], dict[str, Any]] = {}
        for key, candidates in group.groupby(["pv_capacity_kw", "battery_energy_kwh"], sort=False):
            row = candidates.sort_values("system_total_cost_sgd_per_year").iloc[0].to_dict()
            exact[(round(float(key[0]), 6), round(float(key[1]), 6))] = row
        xy = group[["pv_capacity_kw", "battery_energy_kwh"]].astype(float).to_numpy()
        pv_scale = max(max(pv_axis) - min(pv_axis), 1.0)
        battery_scale = max(max(battery_axis) - min(battery_axis), 1.0)
        for pv in pv_axis:
            for battery in battery_axis:
                key = (round(float(pv), 6), round(float(battery), 6))
                if key in exact:
                    row = dict(exact[key])
                    row["point_source"] = str(row.get("point_source", "existing_evaluated"))
                else:
                    deltas = ((xy[:, 0] - float(pv)) / pv_scale) ** 2 + ((xy[:, 1] - float(battery)) / battery_scale) ** 2
                    row = dict(group.iloc[int(np.argmin(deltas))].to_dict())
                    row["point_source"] = "interpolated_for_plot"
                    row["is_minimum_cost_design"] = False
                    row["is_no_pv_no_bess_reference"] = False
                    row["is_lolp_zero_min_cost_design"] = False
                row["city_key"] = city
                row["pv_capacity_kw"] = float(pv)
                row["pv_capacity_mw"] = float(pv) / 1000.0
                row["battery_energy_kwh"] = float(battery)
                row["battery_energy_mwh"] = float(battery) / 1000.0
                row["cost_delta_vs_no_pv_no_bess_sgd_per_year"] = (
                    float(row["system_total_cost_sgd_per_year"]) - float(row["zero_pv_zero_bess_cost_sgd_per_year"])
                )
                row["energy_based_lolp_percent"] = 100.0 * float(row["energy_based_LOLP"])
                rows.append(row)
    return pd.DataFrame(rows)


def _city_plot_axes(source: pd.DataFrame) -> dict[str, list[float]]:
    return {
        "pv": sorted({round(float(value), 6) for value in source["pv_capacity_kw"].astype(float)}),
        "battery": sorted({round(float(value), 6) for value in source["battery_energy_kwh"].astype(float)}),
    }


def _with_day_night_scenario_columns(source: pd.DataFrame, city_key: str, scenario_key: str) -> pd.DataFrame:
    frame = source.copy()
    start, end = FIGURE_D_NIGHT_WINDOW if scenario_key == "night" else FIGURE_D_DAY_WINDOW
    scenario_order = 0 if scenario_key == "night" else 1
    frame["scenario_key"] = scenario_key
    frame["scenario_order"] = scenario_order
    frame["scenario_label"] = f"{CITY_LABELS[city_key]} {scenario_key}"
    frame["outage_window_start_hour"] = float(start)
    frame["outage_window_end_hour"] = float(end)
    frame["day_window_start_hour"] = float(FIGURE_D_DAY_WINDOW[0])
    frame["day_window_end_hour"] = float(FIGURE_D_DAY_WINDOW[1])
    frame["axis_pv_max_kw"] = float(frame["pv_capacity_kw"].astype(float).max())
    frame["axis_battery_max_kwh"] = float(frame["battery_energy_kwh"].astype(float).max())
    return frame


def _day_night_event_summary_row(
    input_root: Path,
    config: dict[str, Any],
    city_key: str,
    scenario_key: str,
) -> pd.DataFrame:
    outage_mode = "night_window_random" if scenario_key == "night" else "day_window_random"
    window = FIGURE_D_NIGHT_WINDOW if scenario_key == "night" else FIGURE_D_DAY_WINDOW
    context = _city_outage_context(input_root, config, city_key, outage_mode, window=window)
    events = context["outage_events"].copy()
    event = events.sort_values("start_hour").iloc[0]
    return pd.DataFrame(
        [
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": context["case"].case_key,
                "source_case_key": context["case"].case_key,
                "scenario_key": scenario_key,
                "scenario_order": 0 if scenario_key == "night" else 1,
                "scenario_label": f"{CITY_LABELS[city_key]} {scenario_key}",
                "outage_mode": outage_mode,
                "outage_seed": int(context["case_config"]["outage"].get("seed", 42)),
                "event_start_hour": float(event["start_hour"]),
                "event_end_hour": float(event["end_hour"]),
                "event_start_hour_of_day": float(event["start_hour"]) % 24.0,
                "event_duration_minutes": float(event["duration_minutes"]),
                "outage_window_start_hour": float(window[0]),
                "outage_window_end_hour": float(window[1]),
                "day_window_start_hour": float(FIGURE_D_DAY_WINDOW[0]),
                "day_window_end_hour": float(FIGURE_D_DAY_WINDOW[1]),
            }
        ]
    )


def _build_affordable_lolp_city_sources_task(
    args: tuple[str, InputCase, dict[str, Any], str, int],
) -> dict[str, Any]:
    city_key, case, global_config, outage_mode, city_order = args
    _, _, case_config, _ = load_input_case(case, deepcopy(global_config))
    case_config = deepcopy(case_config)
    case_config.setdefault("outage", {})["forced_event_mode"] = str(outage_mode)
    preview_frame, _ = build_case_frame(case, case_config, max(case_config["pv"]["capacity_kw_candidates"]))
    outage_profile, outage_events = _baseline_outage(case_config, case.case_key, len(preview_frame))
    outage_summary = summarize_outages(outage_profile, outage_events)
    outage_weighted_total = _outage_weighted_load_total(preview_frame, outage_profile, case_config)

    def pv_capacity_frame_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, case_config, pv_capacity_kw)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    _, _, candidate_summary, candidate_dispatches = resilience_design(
        pv_capacity_frame_builder,
        case_config,
        case.case_key,
    )
    aware_cfg = scenario_config(case_config, case.case_key)
    annual_load = float(case_config["capacity_search"]["annual_load_e_equiv_kwh"])
    design_space = _affordable_lolp_design_space_source(
        city_key,
        case,
        candidate_summary,
        candidate_dispatches,
        annual_load,
        outage_weighted_total,
        outage_summary,
        outage_mode,
        int(case_config["outage"].get("seed", 42)),
    )
    selected = _select_lolp_zero_min_cost_design(design_space, city_order=(city_key,))
    selected_result = _cost_design_row(
        candidate_summary,
        pv_capacity_frame_builder,
        aware_cfg,
        float(selected["pv_capacity_kw"]),
        float(selected["battery_energy_kwh"]),
    )
    selected_dispatch = _candidate_dispatch(selected.to_dict(), candidate_dispatches)
    cost_source = _affordable_lolp_cost_distribution_source(
        city_key,
        case,
        aware_cfg,
        candidate_summary,
        pv_capacity_frame_builder,
        selected_result,
        design_space,
    )
    selected_frame, _ = build_case_frame(case, case_config, float(selected_result["pv_capacity_kw"]))
    selected_frame = selected_frame.merge(outage_profile, on="hour_index", how="left")
    energy_balance, event_summary = _city_energy_balance_source(
        city_key,
        case,
        case_config,
        selected_frame,
        selected_dispatch,
        outage_events,
        selected_result,
        city_order=FIGURE_C_CITY_ORDER,
    )
    for frame in (energy_balance, event_summary):
        frame["design_case"] = "lolp_zero_min_cost"
        frame["design_case_label"] = AFFORDABLE_COST_DESIGN_LABELS["lolp_zero_min_cost"]
        frame["outage_mode"] = str(outage_mode)
        frame["outage_seed"] = int(case_config["outage"].get("seed", 42))
        frame["zero_pv_zero_bess_cost_sgd_per_year"] = float(selected["zero_pv_zero_bess_cost_sgd_per_year"])
        frame["affordable_boundary_cost_sgd_per_year"] = float(selected["affordable_boundary_cost_sgd_per_year"])
        frame["selected_lolp_zero_cost_sgd_per_year"] = float(selected["system_total_cost_sgd_per_year"])
        frame["cost_delta_vs_no_pv_no_bess_sgd_per_year"] = float(selected["cost_delta_vs_no_pv_no_bess_sgd_per_year"])
    return {
        "city_order": int(city_order),
        "design_space": design_space,
        "cost_distribution": cost_source,
        "energy_balance": energy_balance,
        "event_summary": event_summary,
    }


def _brisbane_sensitivity_tasks(parameters: CaseParameters) -> list[dict[str, Any]]:
    base_saidi = float(parameters.base["saidi_min_per_year"])
    base_voll = float(parameters.base["voll_sgd_per_kwh"])
    tasks: list[dict[str, Any]] = []
    for value in BRISBANE_SENSITIVITY_TOU_K_VALUES:
        tasks.append(
            {
                "driver_order": 0,
                "sensitivity_driver": "tou_spread",
                "x_value": float(value),
                "x_label": f"k={float(value):g}",
                "tou_spread_k": float(value),
                "saidi_min_per_year": base_saidi,
                "voll_sgd_per_kwh": base_voll,
            }
        )
    for value in sorted(float(item) for item in parameters.sensitivity.get("saidi_min_per_year", [])):
        tasks.append(
            {
                "driver_order": 1,
                "sensitivity_driver": "saidi",
                "x_value": float(value),
                "x_label": f"{float(value):g} min yr$^{{-1}}$",
                "tou_spread_k": 1.0,
                "saidi_min_per_year": float(value),
                "voll_sgd_per_kwh": base_voll,
            }
        )
    for value in sorted(float(item) for item in parameters.sensitivity.get("voll_sgd_per_kwh", [])):
        tasks.append(
            {
                "driver_order": 2,
                "sensitivity_driver": "voll",
                "x_value": float(value),
                "x_label": f"{float(value):g} SGD kWh$^{{-1}}$",
                "tou_spread_k": 1.0,
                "saidi_min_per_year": base_saidi,
                "voll_sgd_per_kwh": float(value),
            }
        )
    if not tasks:
        raise ValueError("Brisbane sensitivity analysis has no parameter points.")
    return sorted(tasks, key=lambda item: (int(item["driver_order"]), float(item["x_value"])))


def _brisbane_sensitivity_worker_count(parameters: CaseParameters, total_tasks: int) -> int:
    configured = parameters.base.get("parallel_workers")
    if configured is None or pd.isna(configured):
        base_workers = os.cpu_count() or 1
    else:
        base_workers = max(1, int(float(configured)))
    return min(max(1, base_workers), max(1, int(total_tasks)))


def _evaluate_brisbane_sensitivity_tasks_parallel(
    tasks: Sequence[dict[str, Any]],
    input_root: Path,
    config: dict[str, Any],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    ordered_tasks = sorted(tasks, key=lambda item: (int(item["driver_order"]), float(item["x_value"])))
    payloads = [
        {
            **dict(task),
            "input_root": str(input_root),
            "config": deepcopy(config),
        }
        for task in ordered_tasks
    ]
    if workers <= 1:
        rows = [_evaluate_brisbane_sensitivity_task(payload) for payload in payloads]
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                rows = list(executor.map(_evaluate_brisbane_sensitivity_task, payloads, chunksize=1))
        except OSError:
            rows = [_evaluate_brisbane_sensitivity_task(payload) for payload in payloads]
    return sorted(rows, key=lambda item: (int(item["driver_order"]), float(item["x_value"])))


def _evaluate_brisbane_sensitivity_task(payload: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(str(payload["input_root"]))
    runtime_config = deepcopy(payload["config"])
    case = _input_case_for_exact_case_key(input_root, SINGLE_OUTAGE_CASE_BY_CITY["Brisbane"])
    parameter_overrides = {
        "saidi_min_per_year": float(payload["saidi_min_per_year"]),
        "voll_sgd_per_kwh": float(payload["voll_sgd_per_kwh"]),
    }
    _, _, case_config, _ = load_input_case(case, runtime_config, parameter_overrides=parameter_overrides)
    tou_profile = _mean_preserving_tou_profile(pd.read_csv(case.tou_csv), float(payload["tou_spread_k"]))
    preview_frame, _ = build_case_frame(case, case_config, max(case_config["pv"]["capacity_kw_candidates"]))
    preview_frame = _apply_24h_tou_profile(preview_frame, tou_profile)
    outage_profile, outage_events = _baseline_outage(case_config, case.case_key, len(preview_frame))
    outage_summary = summarize_outages(outage_profile, outage_events)
    outage_weighted_total = _outage_weighted_load_total(preview_frame, outage_profile, case_config)

    def pv_capacity_frame_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, case_config, pv_capacity_kw)
        frame = _apply_24h_tou_profile(frame, tou_profile)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    best_rows, _, _, _ = resilience_design(
        pv_capacity_frame_builder,
        case_config,
        case.case_key,
    )
    best = best_rows.iloc[0]
    ue = float(best["unserved_energy_kwh_per_year"])
    lolp = ue / outage_weighted_total if outage_weighted_total > 0.0 else 0.0
    pv_capacity = float(best["pv_capacity_kw"])
    battery_energy = float(best["battery_energy_kwh"])
    system_cost = float(best["pv_bess_system_annual_cost_sgd"])
    tou_values = tou_profile["tou_sgd_per_kwh"].astype(float)
    return {
        "driver_order": int(payload["driver_order"]),
        "sensitivity_driver": str(payload["sensitivity_driver"]),
        "x_value": float(payload["x_value"]),
        "x_label": str(payload["x_label"]),
        "city_key": "Brisbane",
        "city_label": CITY_LABELS["Brisbane"],
        "case_key": case.case_key,
        "source_case_key": case.case_key,
        "pv_capacity_kw": pv_capacity,
        "pv_capacity_mw": pv_capacity / 1000.0,
        "battery_energy_kwh": battery_energy,
        "battery_energy_mwh": battery_energy / 1000.0,
        "system_total_cost_sgd_per_year": system_cost,
        "system_total_cost_million_sgd_per_year": system_cost / 1e6,
        "UE_kwh_per_year": ue,
        "UE_mwh_per_year": ue / 1000.0,
        "energy_based_LOLP": float(np.clip(lolp, 0.0, 1.0)),
        "energy_based_lolp": float(np.clip(lolp, 0.0, 1.0)),
        "energy_based_lolp_percent": 100.0 * float(np.clip(lolp, 0.0, 1.0)),
        "outage_weighted_load_kwh": outage_weighted_total,
        "saidi_min_per_year": float(payload["saidi_min_per_year"]),
        "voll_sgd_per_kwh": float(payload["voll_sgd_per_kwh"]),
        "tou_spread_k": float(payload["tou_spread_k"]),
        "tou_mean_sgd_per_kwh": float(tou_values.mean()),
        "tou_min_sgd_per_kwh": float(tou_values.min()),
        "tou_max_sgd_per_kwh": float(tou_values.max()),
        "simulated_saidi_min_per_year": outage_summary.get("simulated_saidi_min_per_year", np.nan),
        "simulated_saifi_per_year": outage_summary.get("simulated_saifi_per_year", np.nan),
    }


def _mean_preserving_tou_profile(tou: pd.DataFrame, spread_k: float) -> pd.DataFrame:
    required = {"hour", "tou_sgd_per_kwh"}
    missing = required - set(tou.columns)
    if missing:
        raise ValueError("TOU spread sensitivity requires column(s): " + ", ".join(sorted(missing)))
    profile = tou[list(required)].copy()
    profile["hour"] = pd.to_numeric(profile["hour"], errors="coerce").astype("Int64")
    profile["tou_sgd_per_kwh"] = pd.to_numeric(profile["tou_sgd_per_kwh"], errors="coerce")
    if profile[["hour", "tou_sgd_per_kwh"]].isna().any().any():
        raise ValueError("TOU spread sensitivity received non-numeric TOU values.")
    profile = profile.sort_values("hour").reset_index(drop=True)
    if profile["hour"].astype(int).tolist() != list(range(24)):
        raise ValueError("TOU spread sensitivity requires hours 0-23 exactly.")
    values = profile["tou_sgd_per_kwh"].astype(float)
    mean_value = float(values.mean())
    adjusted = mean_value + float(spread_k) * (values - mean_value)
    if float(adjusted.min()) < -1e-9:
        raise ValueError(f"TOU spread k={spread_k:g} creates a negative tariff.")
    profile["tou_sgd_per_kwh"] = np.maximum(adjusted, 0.0)
    return profile[["hour", "tou_sgd_per_kwh"]]


def _apply_24h_tou_profile(frame: pd.DataFrame, tou_profile: pd.DataFrame) -> pd.DataFrame:
    adjusted = frame.copy()
    mapping = dict(zip(tou_profile["hour"].astype(int), tou_profile["tou_sgd_per_kwh"].astype(float)))
    if "timestamp" in adjusted.columns:
        hours = pd.to_datetime(adjusted["timestamp"]).dt.hour
    else:
        hours = adjusted["hour_index"].astype(int) % 24
    adjusted["tou_sgd_per_kwh"] = hours.map(mapping).astype(float)
    if adjusted["tou_sgd_per_kwh"].isna().any():
        raise ValueError("Adjusted TOU profile did not cover every frame hour.")
    return adjusted


def _baseline_outage(config: dict[str, Any], case_key: str, hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = config["scenarios"][case_key]
    return simulate_outages(
        saifi_per_year=float(scenario["saifi_per_year"]),
        saidi_min_per_year=float(scenario["saidi_min_per_year"]),
        hours=int(hours),
        seed=int(config["outage"].get("seed", 42)),
        forced_event_mode=str(config["outage"].get("forced_event_mode", "match_saidi")),
        night_window_start_hour=float(config["outage"].get("night_window_start_hour", 19.0)),
        night_window_end_hour=float(config["outage"].get("night_window_end_hour", 23.0)),
    )


def _city_design_space_source(
    city_key: str,
    case: InputCase,
    candidate_summary: pd.DataFrame,
    candidate_dispatches: dict[tuple[str, int, float, float], pd.DataFrame],
    annual_load_kwh: float,
    outage_weighted_total_kwh: float,
    outage_summary: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in candidate_summary.to_dict("records"):
        dispatch = _candidate_dispatch(row, candidate_dispatches)
        solar = calculate_pv_self_use_metrics(dispatch, annual_load_kwh)
        ue = float(row["unserved_energy_kwh_per_year"])
        lolp = ue / outage_weighted_total_kwh if outage_weighted_total_kwh > 0.0 else 0.0
        rows.append(
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": case.case_key,
                "source_case_key": case.case_key,
                "search_stage": str(row["search_stage"]),
                "search_iteration": int(row["search_iteration"]),
                "pv_capacity_kw": float(row["pv_capacity_kw"]),
                "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
                "battery_energy_kwh": float(row["battery_energy_kwh"]),
                "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
                "system_total_cost_sgd_per_year": float(row["pv_bess_system_annual_cost_sgd"]),
                "system_total_cost_million_sgd_per_year": float(row["pv_bess_system_annual_cost_sgd"]) / 1e6,
                "grid_cost_sgd_per_year": float(row["pv_bess_grid_cost_sgd"]),
                "pv_annual_cost_sgd_per_year": float(row["annualized_pv_cost_sgd"]) + float(row["pv_om_cost_sgd"]),
                "battery_annual_cost_sgd_per_year": float(row["annualized_battery_cost_sgd"]) + float(row["battery_om_cost_sgd"]),
                "lost_load_cost_sgd_per_year": float(row["unserved_energy_cost_sgd_per_year"]),
                "UE_kwh_per_year": ue,
                "UE_mwh_per_year": ue / 1000.0,
                "outage_weighted_load_kwh": outage_weighted_total_kwh,
                "energy_based_LOLP": float(np.clip(lolp, 0.0, 1.0)),
                "self_sufficiency_rate": solar["self_sufficiency_rate"],
                "self_consumption_rate": solar["self_consumption_rate"],
                "pv_self_served_load_kwh": solar["pv_self_served_load_kwh"],
                "direct_pv_to_load_kwh": solar["direct_pv_to_load_kwh"],
                "pv_origin_battery_discharge_to_load_kwh": solar["pv_origin_battery_discharge_to_load_kwh"],
                "annual_pv_available_kwh": solar["annual_pv_available_kwh"],
                "annual_load_e_equiv_kwh": solar["annual_load_e_equiv_kwh"],
                "simulated_saidi_min_per_year": outage_summary.get("simulated_saidi_min_per_year", np.nan),
                "simulated_saifi_per_year": outage_summary.get("simulated_saifi_per_year", np.nan),
            }
        )
    source = pd.DataFrame(rows)
    best_index = source["system_total_cost_sgd_per_year"].astype(float).idxmin()
    source["is_minimum_cost_design"] = False
    source.loc[best_index, "is_minimum_cost_design"] = True
    return _validate_design_space_source(source)


def _affordable_lolp_design_space_source(
    city_key: str,
    case: InputCase,
    candidate_summary: pd.DataFrame,
    candidate_dispatches: dict[tuple[str, int, float, float], pd.DataFrame],
    annual_load_kwh: float,
    outage_weighted_total_kwh: float,
    outage_summary: dict[str, float],
    outage_mode: str,
    outage_seed: int,
) -> pd.DataFrame:
    zero_zero = candidate_summary[
        np.isclose(candidate_summary["pv_capacity_kw"].astype(float), 0.0, atol=1e-6)
        & np.isclose(candidate_summary["battery_energy_kwh"].astype(float), 0.0, atol=1e-6)
    ].copy()
    if zero_zero.empty:
        raise ValueError(f"Figure 1c requires a 0 PV / 0 BESS candidate for {CITY_LABELS[city_key]}.")
    zero_zero_cost = float(zero_zero.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]["pv_bess_system_annual_cost_sgd"])
    rows: list[dict[str, Any]] = []
    for row in candidate_summary.to_dict("records"):
        dispatch = _candidate_dispatch(row, candidate_dispatches)
        solar = calculate_pv_self_use_metrics(dispatch, annual_load_kwh)
        ue = float(row["unserved_energy_kwh_per_year"])
        lolp = ue / outage_weighted_total_kwh if outage_weighted_total_kwh > 0.0 else 0.0
        total_cost = float(row["pv_bess_system_annual_cost_sgd"])
        rows.append(
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": case.case_key,
                "source_case_key": case.case_key,
                "search_stage": str(row["search_stage"]),
                "search_iteration": int(row["search_iteration"]),
                "pv_capacity_kw": float(row["pv_capacity_kw"]),
                "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
                "battery_energy_kwh": float(row["battery_energy_kwh"]),
                "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
                "system_total_cost_sgd_per_year": total_cost,
                "system_total_cost_million_sgd_per_year": total_cost / 1e6,
                "grid_cost_sgd_per_year": float(row["pv_bess_grid_cost_sgd"]),
                "pv_annual_cost_sgd_per_year": float(row["annualized_pv_cost_sgd"]) + float(row["pv_om_cost_sgd"]),
                "battery_annual_cost_sgd_per_year": float(row["annualized_battery_cost_sgd"]) + float(row["battery_om_cost_sgd"]),
                "lost_load_cost_sgd_per_year": float(row["unserved_energy_cost_sgd_per_year"]),
                "UE_kwh_per_year": ue,
                "UE_mwh_per_year": ue / 1000.0,
                "outage_weighted_load_kwh": outage_weighted_total_kwh,
                "energy_based_LOLP": float(np.clip(lolp, 0.0, 1.0)),
                "energy_based_lolp_percent": float(np.clip(lolp, 0.0, 1.0)) * 100.0,
                "self_sufficiency_rate": solar["self_sufficiency_rate"],
                "self_consumption_rate": solar["self_consumption_rate"],
                "pv_self_served_load_kwh": solar["pv_self_served_load_kwh"],
                "direct_pv_to_load_kwh": solar["direct_pv_to_load_kwh"],
                "pv_origin_battery_discharge_to_load_kwh": solar["pv_origin_battery_discharge_to_load_kwh"],
                "annual_pv_available_kwh": solar["annual_pv_available_kwh"],
                "annual_load_e_equiv_kwh": solar["annual_load_e_equiv_kwh"],
                "simulated_saidi_min_per_year": outage_summary.get("simulated_saidi_min_per_year", np.nan),
                "simulated_saifi_per_year": outage_summary.get("simulated_saifi_per_year", np.nan),
                "outage_mode": str(outage_mode),
                "outage_seed": int(outage_seed),
                "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                "cost_delta_vs_no_pv_no_bess_sgd_per_year": total_cost - zero_zero_cost,
                "point_source": "existing_evaluated",
            }
        )
    source = pd.DataFrame(rows)
    source["is_minimum_cost_design"] = False
    source.loc[source["system_total_cost_sgd_per_year"].astype(float).idxmin(), "is_minimum_cost_design"] = True
    source["is_no_pv_no_bess_reference"] = (
        np.isclose(source["pv_capacity_kw"].astype(float), 0.0, atol=1e-6)
        & np.isclose(source["battery_energy_kwh"].astype(float), 0.0, atol=1e-6)
    )
    source["is_lolp_zero_min_cost_design"] = False
    zero_lolp = source[source["UE_kwh_per_year"].astype(float) <= AFFORDABLE_LOLP_ZERO_TOLERANCE_KWH].copy()
    if zero_lolp.empty:
        raise ValueError(
            f"Figure 1c requires at least one LOLP=0 candidate for {CITY_LABELS[city_key]}; "
            "expand the PV/BESS capacity search range."
        )
    selected_index = zero_lolp.sort_values(
        ["system_total_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"]
    ).index[0]
    source.loc[selected_index, "is_lolp_zero_min_cost_design"] = True
    selected = source.loc[selected_index]
    source["selected_lolp_zero_pv_capacity_kw"] = float(selected["pv_capacity_kw"])
    source["selected_lolp_zero_battery_energy_kwh"] = float(selected["battery_energy_kwh"])
    source["selected_lolp_zero_cost_sgd_per_year"] = float(selected["system_total_cost_sgd_per_year"])
    source["selected_lolp_zero_UE_kwh_per_year"] = float(selected["UE_kwh_per_year"])
    source["selected_lolp_zero_energy_based_LOLP"] = float(selected["energy_based_LOLP"])
    return _validate_affordable_design_space_source(source, city_order=(city_key,))


def _select_lolp_zero_min_cost_design(
    design_space: pd.DataFrame,
    *,
    city_order: Sequence[str] = FIGURE_C_CITY_ORDER,
) -> pd.Series:
    source = _validate_affordable_design_space_source(design_space, city_order=city_order)
    selected = source[source["is_lolp_zero_min_cost_design"].astype(bool)]
    if selected.empty:
        city = str(source["city_label"].iloc[0]) if not source.empty else "unknown city"
        raise ValueError(f"Figure 1c source has no LOLP=0 min-cost design for {city}.")
    return selected.sort_values(["system_total_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"]).iloc[0]


def _reference_design_results_source(design_space: pd.DataFrame) -> pd.DataFrame:
    source = _validate_design_space_source(design_space)
    rows: list[dict[str, Any]] = []
    for city_key in CITY_ORDER:
        group = source[source["city_key"].astype(str) == city_key].copy()
        if group.empty:
            continue
        selected = group[group["is_minimum_cost_design"].astype(bool)]
        row = selected.iloc[0] if not selected.empty else group.sort_values("system_total_cost_sgd_per_year").iloc[0]
        ue = float(row["UE_kwh_per_year"])
        lolp = float(row["energy_based_LOLP"])
        rows.append(
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": row["case_key"],
                "source_case_key": row.get("source_case_key", row["case_key"]),
                "design_case": "baseline_minimum_cost",
                "pv_capacity_kw": float(row["pv_capacity_kw"]),
                "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
                "battery_energy_kwh": float(row["battery_energy_kwh"]),
                "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
                "UE_kwh_per_year": ue,
                "UE_mwh_per_year": ue / 1000.0,
                "ue_kwh_per_year": ue,
                "energy_based_lolp": lolp,
                "energy_based_lolp_percent": 100.0 * lolp,
                "outage_weighted_load_kwh": float(row["outage_weighted_load_kwh"]),
                "system_total_cost_sgd_per_year": float(row["system_total_cost_sgd_per_year"]),
                "system_total_cost_million_sgd_per_year": float(row["system_total_cost_sgd_per_year"]) / 1e6,
            }
        )
    return _validate_reference_design_source(pd.DataFrame(rows))


def _city_scenario_distribution_source(
    city_key: str,
    case: InputCase,
    all_results: pd.DataFrame,
    rng: np.random.Generator,
    result_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sorted_results = all_results.sort_values("uncertainty_scenario_id").copy()
    for row in sorted_results.to_dict("records"):
        ue = _record_float(row, "unserved_energy_kwh_per_year")
        outage_load = _record_float(row, "outage_weighted_load_kwh")
        lolp = 0.0 if outage_load <= 0.0 else float(np.clip(ue / outage_load, 0.0, 1.0))
        pv_capacity = _record_float(row, "pv_capacity_kw")
        battery_energy = _record_float(row, "battery_energy_kwh")
        system_cost = _record_float(row, "pv_bess_system_annual_cost_sgd")
        rows.append(
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": case.case_key,
                "source_case_key": case.case_key,
                "optimization_result_source": result_source,
                "scenario_id": int(row["uncertainty_scenario_id"]),
                "outage_seed": int(row.get("outage_seed", 0)),
                "pv_capacity_kw": pv_capacity,
                "pv_capacity_mw": pv_capacity / 1000.0,
                "battery_energy_kwh": battery_energy,
                "battery_energy_mwh": battery_energy / 1000.0,
                "UE_kwh_per_year": ue,
                "UE_mwh_per_year": ue / 1000.0,
                "ue_kwh_per_year": ue,
                "energy_based_lolp": lolp,
                "energy_based_lolp_percent": 100.0 * lolp,
                "outage_weighted_load_kwh": outage_load,
                "annual_outage_hours": _record_float(row, "simulated_saidi_min_per_year") / 60.0,
                "simulated_saidi_min_per_year": _record_float(row, "simulated_saidi_min_per_year"),
                "simulated_saifi_per_year": _record_float(row, "simulated_saifi_per_year"),
                "max_event_duration_hours": _record_float(row, "max_event_duration_min") / 60.0,
                "system_total_cost_sgd_per_year": system_cost,
                "system_total_cost_million_sgd_per_year": system_cost / 1e6,
                "plot_jitter": float(rng.uniform(-0.16, 0.16)),
            }
        )
    return pd.DataFrame(rows)


def _cached_uncertainty_results(
    config: dict[str, Any],
    case: InputCase,
    case_config: dict[str, Any],
    scenario_count: int,
) -> pd.DataFrame | None:
    output_root = Path(config.get("outputs", {}).get("directory", "outputs"))
    path = (
        output_root
        / case.city_key
        / f"{case.case_key}_Uncertainty_Analysis"
        / f"{case.case_key}_Uncertainty_Baseline"
        / f"{case.case_key}_Uncertainty_Baseline_All_Results.csv"
    )
    if not path.exists():
        return None
    cached = pd.read_csv(path)
    required = {
        "uncertainty_scenario_id",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "unserved_energy_kwh_per_year",
        "outage_weighted_load_kwh",
        "pv_bess_system_annual_cost_sgd",
    }
    if required - set(cached.columns) or len(cached) < int(scenario_count):
        return None
    if not _cached_uncertainty_parameters_match(cached, case, case_config):
        return None
    return cached.sort_values("uncertainty_scenario_id").head(int(scenario_count)).reset_index(drop=True)


def _cached_uncertainty_parameters_match(
    cached: pd.DataFrame,
    case: InputCase,
    case_config: dict[str, Any],
) -> bool:
    scenario = case_config["scenarios"][case.case_key]
    expected = {
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "saifi_per_year": float(scenario["saifi_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "battery_capex_sgd_per_kwh": float(case_config["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(case_config["economics"]["battery_power_capex_sgd_per_kw"]),
        "battery_om_fraction_per_year": float(case_config["economics"]["battery_om_fraction_per_year"]),
        "pv_capex_sgd_per_kw": float(case_config["pv"]["capex_sgd_per_kw"]),
    }
    if set(expected) - set(cached.columns):
        return False
    first = cached.iloc[0]
    for column, value in expected.items():
        if not np.isclose(float(first[column]), value, rtol=1e-7, atol=1e-9):
            return False
    return True


def _record_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _build_cvar_candidate_set(
    design_space: pd.DataFrame,
    scenario_results: pd.DataFrame,
    *,
    max_candidates_per_city: int | None = CVAR_DEFAULT_MAX_CANDIDATES_PER_CITY,
) -> pd.DataFrame:
    design_source = _validate_design_space_source(design_space)
    scenario_source = _validate_scenario_distribution_source(scenario_results)
    rows: list[dict[str, Any]] = []
    for city_key in CITY_ORDER:
        points: dict[tuple[float, float], dict[str, Any]] = {}
        city_design = design_source[design_source["city_key"].astype(str) == city_key]
        city_scenarios = scenario_source[scenario_source["city_key"].astype(str) == city_key]
        expost_mandatory = _representative_cvar_expost_keys(city_scenarios, max_candidates_per_city)
        for row in city_design.to_dict("records"):
            pv_capacity = round(float(row["pv_capacity_kw"]), 6)
            battery_energy = round(float(row["battery_energy_kwh"]), 6)
            key = (pv_capacity, battery_energy)
            if key not in points:
                points[key] = {
                    "city_key": city_key,
                    "city_label": CITY_LABELS[city_key],
                    "case_key": str(row["case_key"]),
                    "source_case_key": str(row.get("source_case_key", row["case_key"])),
                    "pv_capacity_kw": pv_capacity,
                    "pv_capacity_mw": pv_capacity / 1000.0,
                    "battery_energy_kwh": battery_energy,
                    "battery_energy_mwh": battery_energy / 1000.0,
                    "from_figure1_design_space": False,
                    "from_figure4_expost_optima": False,
                    "is_cvar_mandatory_candidate": False,
                }
            points[key]["from_figure1_design_space"] = True
            if bool(row.get("is_minimum_cost_design", False)) or (np.isclose(pv_capacity, 0.0) and np.isclose(battery_energy, 0.0)):
                points[key]["is_cvar_mandatory_candidate"] = True
        for row in city_scenarios.to_dict("records"):
            pv_capacity = round(float(row["pv_capacity_kw"]), 6)
            battery_energy = round(float(row["battery_energy_kwh"]), 6)
            key = (pv_capacity, battery_energy)
            if key not in points:
                points[key] = {
                    "city_key": city_key,
                    "city_label": CITY_LABELS[city_key],
                    "case_key": str(row["case_key"]),
                    "source_case_key": str(row.get("source_case_key", row["case_key"])),
                    "pv_capacity_kw": pv_capacity,
                    "pv_capacity_mw": pv_capacity / 1000.0,
                    "battery_energy_kwh": battery_energy,
                    "battery_energy_mwh": battery_energy / 1000.0,
                    "from_figure1_design_space": False,
                    "from_figure4_expost_optima": False,
                    "is_cvar_mandatory_candidate": False,
                }
            points[key]["from_figure4_expost_optima"] = True
            if key in expost_mandatory:
                points[key]["is_cvar_mandatory_candidate"] = True
        selected_points = _thin_cvar_candidate_points(list(points.values()), max_candidates_per_city)
        for candidate_order, row in enumerate(
            sorted(selected_points, key=lambda item: (float(item["pv_capacity_kw"]), float(item["battery_energy_kwh"]))),
            start=1,
        ):
            row = dict(row)
            row["candidate_order"] = int(candidate_order)
            row["candidate_id"] = f"{city_key}_D{candidate_order:04d}"
            sources = []
            if row["from_figure1_design_space"]:
                sources.append("Figure 1 design space")
            if row["from_figure4_expost_optima"]:
                sources.append("Figure 4 ex-post optima")
            row["candidate_source"] = "; ".join(sources)
            rows.append(row)
    if not rows:
        raise ValueError("Figure 5 CVaR candidate set is empty.")
    return pd.DataFrame(rows)


def _cvar_max_candidates_per_city(config: dict[str, Any]) -> int | None:
    section = config.get("single_outage_city_baseline", {})
    if not isinstance(section, dict):
        return CVAR_DEFAULT_MAX_CANDIDATES_PER_CITY
    value = section.get("figure5_max_candidates_per_city", CVAR_DEFAULT_MAX_CANDIDATES_PER_CITY)
    if value is None:
        return None
    result = int(value)
    return None if result <= 0 else result


def _representative_cvar_expost_keys(city_scenarios: pd.DataFrame, max_candidates: int | None) -> set[tuple[float, float]]:
    unique = city_scenarios.copy()
    unique["_pv_key"] = unique["pv_capacity_kw"].astype(float).round(6)
    unique["_battery_key"] = unique["battery_energy_kwh"].astype(float).round(6)
    grouped = (
        unique.groupby(["_pv_key", "_battery_key"], as_index=False)
        .agg(
            scenario_count=("scenario_id", "count"),
            mean_ue=("UE_kwh_per_year", "mean"),
            max_ue=("UE_kwh_per_year", "max"),
            mean_cost=("system_total_cost_sgd_per_year", "mean"),
        )
        .copy()
    )
    if grouped.empty:
        return set()
    if max_candidates is None:
        return {(float(row["_pv_key"]), float(row["_battery_key"])) for row in grouped.to_dict("records")}
    limit = max(1, min(3, int(max_candidates) // 2))
    selected: dict[tuple[float, float], None] = {}
    selectors = [
        grouped.sort_values(["scenario_count", "mean_cost"], ascending=[False, True]),
        grouped.sort_values(["max_ue", "mean_ue"], ascending=[False, False]),
        grouped.sort_values(["mean_cost", "scenario_count"], ascending=[True, False]),
    ]
    for selector in selectors:
        if len(selected) >= limit:
            break
        row = selector.iloc[0]
        selected[(float(row["_pv_key"]), float(row["_battery_key"]))] = None
    return set(selected)


def _thin_cvar_candidate_points(points: list[dict[str, Any]], max_candidates: int | None) -> list[dict[str, Any]]:
    if max_candidates is None or len(points) <= int(max_candidates):
        return points
    limit = max(1, int(max_candidates))
    selected: dict[tuple[float, float], dict[str, Any]] = {}
    for point in points:
        if bool(point.get("is_cvar_mandatory_candidate", False)):
            selected[(float(point["pv_capacity_kw"]), float(point["battery_energy_kwh"]))] = point
    if len(selected) >= limit:
        return list(selected.values())

    all_points = list(points)
    pv_values = np.array([float(point["pv_capacity_kw"]) for point in all_points], dtype=float)
    battery_values = np.array([float(point["battery_energy_kwh"]) for point in all_points], dtype=float)
    slots = max(1, limit - len(selected))
    pv_bins = max(2, int(round(np.sqrt(slots))))
    battery_bins = max(2, int(np.ceil(slots / pv_bins)))
    pv_targets = np.linspace(float(pv_values.min()), float(pv_values.max()), pv_bins)
    battery_targets = np.linspace(float(battery_values.min()), float(battery_values.max()), battery_bins)
    pv_span = max(float(np.ptp(pv_values)), 1e-9)
    battery_span = max(float(np.ptp(battery_values)), 1e-9)

    ranked: list[tuple[float, dict[str, Any]]] = []
    for pv_target in pv_targets:
        for battery_target in battery_targets:
            best_point = min(
                all_points,
                key=lambda point: (
                    ((float(point["pv_capacity_kw"]) - pv_target) / pv_span) ** 2
                    + ((float(point["battery_energy_kwh"]) - battery_target) / battery_span) ** 2,
                    float(point["pv_capacity_kw"]),
                    float(point["battery_energy_kwh"]),
                ),
            )
            key = (float(best_point["pv_capacity_kw"]), float(best_point["battery_energy_kwh"]))
            if key not in selected:
                distance = (
                    ((float(best_point["pv_capacity_kw"]) - pv_target) / pv_span) ** 2
                    + ((float(best_point["battery_energy_kwh"]) - battery_target) / battery_span) ** 2
                )
                ranked.append((float(distance), best_point))

    for _, point in sorted(ranked, key=lambda item: item[0]):
        if len(selected) >= limit:
            break
        selected[(float(point["pv_capacity_kw"]), float(point["battery_energy_kwh"]))] = point

    if len(selected) < limit:
        remaining = [
            point
            for point in sorted(all_points, key=lambda item: (float(item["pv_capacity_kw"]), float(item["battery_energy_kwh"])))
            if (float(point["pv_capacity_kw"]), float(point["battery_energy_kwh"])) not in selected
        ]
        if remaining:
            indices = np.linspace(0, len(remaining) - 1, limit - len(selected), dtype=int)
            for index in indices:
                point = remaining[int(index)]
                selected[(float(point["pv_capacity_kw"]), float(point["battery_energy_kwh"]))] = point
                if len(selected) >= limit:
                    break
    return list(selected.values())


def _evaluate_cvar_candidate_scenarios_parallel(
    candidates: pd.DataFrame,
    scenario_results: pd.DataFrame,
    cases_by_city: dict[str, InputCase],
    case_configs_by_city: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    candidate_source = candidates.copy()
    scenario_source = _validate_scenario_distribution_source(scenario_results)
    if "outage_seed" not in scenario_source.columns:
        raise ValueError("Figure 5 requires outage_seed in Figure 4 scenario_design_results.")
    tasks: list[tuple[str, str, int, int, int, float, float]] = []
    for city_key in CITY_ORDER:
        city_candidates = candidate_source[candidate_source["city_key"].astype(str) == city_key].sort_values("candidate_order")
        city_scenarios = scenario_source[scenario_source["city_key"].astype(str) == city_key].sort_values("scenario_id")
        for candidate in city_candidates.to_dict("records"):
            for scenario in city_scenarios.to_dict("records"):
                tasks.append(
                    (
                        city_key,
                        str(candidate["candidate_id"]),
                        int(candidate["candidate_order"]),
                        int(scenario["scenario_id"]),
                        int(scenario["outage_seed"]),
                        float(candidate["pv_capacity_kw"]),
                        float(candidate["battery_energy_kwh"]),
                    )
                )
    if not tasks:
        raise ValueError("Figure 5 fixed-design scenario evaluation has no tasks.")
    workers = _cvar_parallel_worker_count(config, len(tasks))
    if workers <= 1:
        _init_cvar_worker(cases_by_city, case_configs_by_city)
        rows = [_evaluate_cvar_candidate_scenario_task(task) for task in tasks]
    else:
        chunksize = max(1, min(128, len(tasks) // max(workers * 8, 1)))
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_cvar_worker,
                initargs=(cases_by_city, case_configs_by_city),
            ) as executor:
                rows = list(executor.map(_evaluate_cvar_candidate_scenario_task, tasks, chunksize=chunksize))
        except OSError as exc:
            raise RuntimeError(
                "Figure 5 CVaR fixed-design evaluation requires process-level parallel execution, "
                "but ProcessPoolExecutor could not start."
            ) from exc
    evaluated = pd.DataFrame(rows)
    merged = evaluated.merge(
        candidate_source[
            [
                "city_key",
                "candidate_id",
                "candidate_source",
                "from_figure1_design_space",
                "from_figure4_expost_optima",
            ]
        ],
        on=["city_key", "candidate_id"],
        how="left",
    )
    return _validate_cvar_candidate_scenario_source(merged)


def _cvar_parallel_worker_count(config: dict[str, Any], total_tasks: int) -> int:
    configured = (
        config.get("single_outage_city_baseline", {})
        .get("figure5_parallel_workers")
        if isinstance(config.get("single_outage_city_baseline", {}), dict)
        else None
    )
    if configured is None:
        base_workers = os.cpu_count() or 1
    else:
        base_workers = max(1, int(configured))
    return min(max(1, int(base_workers)), max(1, int(total_tasks)))


def _init_cvar_worker(cases_by_city: dict[str, InputCase], case_configs_by_city: dict[str, dict[str, Any]]) -> None:
    global _CVAR_WORKER_CASES, _CVAR_WORKER_CONFIGS, _CVAR_WORKER_FRAMES, _CVAR_WORKER_OUTAGES
    _CVAR_WORKER_CASES = cases_by_city
    _CVAR_WORKER_CONFIGS = case_configs_by_city
    _CVAR_WORKER_FRAMES = OrderedDict()
    _CVAR_WORKER_OUTAGES = OrderedDict()
    setattr(_cvar_worker_no_bess_reference, "_cache", OrderedDict())


def _evaluate_cvar_candidate_scenario_task(args: tuple[str, str, int, int, int, float, float]) -> dict[str, Any]:
    city_key, candidate_id, candidate_order, scenario_id, outage_seed, pv_capacity_kw, battery_energy_kwh = args
    case = _CVAR_WORKER_CASES[city_key]
    config = _CVAR_WORKER_CONFIGS[city_key]
    frame, pv_meta = _cvar_worker_frame(city_key, pv_capacity_kw)
    outage_profile, outage_events, outage_summary = _cvar_worker_outage(city_key, len(frame), outage_seed)
    scenario_frame = frame.merge(outage_profile, on="hour_index", how="left")
    if abs(float(battery_energy_kwh)) <= 1e-9:
        result, _ = evaluate_candidate(scenario_frame, 0.0, pv_meta, config)
        _store_cvar_worker_no_bess_reference(city_key, pv_capacity_kw, outage_seed, result)
    else:
        no_bess_reference = _cvar_worker_no_bess_reference(city_key, pv_capacity_kw, outage_seed, scenario_frame, pv_meta, config)
        result, _ = evaluate_candidate(scenario_frame, float(battery_energy_kwh), pv_meta, config, no_bess_reference=no_bess_reference)
    ue = float(result["unserved_energy_kwh_per_year"])
    outage_load = _outage_weighted_load_total(frame, outage_profile, config)
    lolp = float(np.clip(ue / outage_load, 0.0, 1.0)) if outage_load > 0.0 else 0.0
    fixed_investment = (
        float(result["annualized_pv_cost_sgd"])
        + float(result["pv_om_cost_sgd"])
        + float(result["annualized_battery_cost_sgd"])
        + float(result["battery_om_cost_sgd"])
    )
    operation_cost = float(result["pv_bess_grid_cost_sgd"])
    outage_loss = float(result["unserved_energy_cost_sgd_per_year"])
    system_cost = float(result["pv_bess_system_annual_cost_sgd"])
    max_duration = float(outage_events["duration_minutes"].max()) / 60.0 if not outage_events.empty else 0.0
    return {
        "city_key": city_key,
        "city_label": CITY_LABELS[city_key],
        "case_key": case.case_key,
        "source_case_key": case.case_key,
        "candidate_id": candidate_id,
        "candidate_order": int(candidate_order),
        "scenario_id": int(scenario_id),
        "outage_seed": int(outage_seed),
        "pv_capacity_kw": float(pv_capacity_kw),
        "pv_capacity_mw": float(pv_capacity_kw) / 1000.0,
        "battery_energy_kwh": float(battery_energy_kwh),
        "battery_energy_mwh": float(battery_energy_kwh) / 1000.0,
        "fixed_investment_cost_sgd_per_year": fixed_investment,
        "operation_cost_sgd_per_year": operation_cost,
        "outage_loss_sgd_per_year": outage_loss,
        "system_total_cost_sgd_per_year": system_cost,
        "system_total_cost_million_sgd_per_year": system_cost / 1e6,
        "UE_kwh_per_year": ue,
        "UE_mwh_per_year": ue / 1000.0,
        "ue_kwh_per_year": ue,
        "energy_based_lolp": lolp,
        "energy_based_lolp_percent": 100.0 * lolp,
        "outage_weighted_load_kwh": outage_load,
        "annual_outage_hours": float(outage_summary.get("simulated_saidi_min_per_year", 0.0)) / 60.0,
        "simulated_saidi_min_per_year": float(outage_summary.get("simulated_saidi_min_per_year", 0.0)),
        "simulated_saifi_per_year": float(outage_summary.get("simulated_saifi_per_year", 0.0)),
        "max_event_duration_hours": max_duration,
    }


def _cvar_worker_frame(city_key: str, pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
    key = (city_key, round(float(pv_capacity_kw), 6))
    cached = _CVAR_WORKER_FRAMES.get(key)
    if cached is not None:
        _CVAR_WORKER_FRAMES.move_to_end(key)
        return cached
    frame, pv_meta = build_case_frame(_CVAR_WORKER_CASES[city_key], _CVAR_WORKER_CONFIGS[city_key], float(pv_capacity_kw))
    _CVAR_WORKER_FRAMES[key] = (frame, pv_meta)
    _trim_ordered_cache(_CVAR_WORKER_FRAMES, CVAR_WORKER_FRAME_CACHE_LIMIT)
    return frame, pv_meta


def _cvar_worker_outage(city_key: str, hours: int, outage_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    key = (city_key, int(hours), int(outage_seed))
    cached = _CVAR_WORKER_OUTAGES.get(key)
    if cached is not None:
        _CVAR_WORKER_OUTAGES.move_to_end(key)
        return cached
    config = _CVAR_WORKER_CONFIGS[city_key]
    case = _CVAR_WORKER_CASES[city_key]
    scenario = config["scenarios"][case.case_key]
    outage_profile, outage_events = simulate_outages(
        saifi_per_year=float(scenario["saifi_per_year"]),
        saidi_min_per_year=float(scenario["saidi_min_per_year"]),
        hours=int(hours),
        seed=int(outage_seed),
        forced_event_mode=str(config["outage"].get("forced_event_mode", "match_saidi")),
    )
    outage_summary = summarize_outages(outage_profile, outage_events)
    _CVAR_WORKER_OUTAGES[key] = (outage_profile, outage_events, outage_summary)
    _trim_ordered_cache(_CVAR_WORKER_OUTAGES, CVAR_WORKER_OUTAGE_CACHE_LIMIT)
    return outage_profile, outage_events, outage_summary


def _cvar_worker_no_bess_reference(
    city_key: str,
    pv_capacity_kw: float,
    outage_seed: int,
    scenario_frame: pd.DataFrame,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> dict[str, Any]:
    cache = _cvar_worker_no_bess_cache()
    key = (city_key, round(float(pv_capacity_kw), 6), int(outage_seed))
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return dict(cached)
    zero_result, _ = evaluate_candidate(scenario_frame, 0.0, pv_meta, config)
    return _store_cvar_worker_no_bess_reference(city_key, pv_capacity_kw, outage_seed, zero_result)


def _cvar_worker_no_bess_cache() -> OrderedDict[Any, Any]:
    cache = getattr(_cvar_worker_no_bess_reference, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(_cvar_worker_no_bess_reference, "_cache", cache)
    return cache


def _store_cvar_worker_no_bess_reference(
    city_key: str,
    pv_capacity_kw: float,
    outage_seed: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    cache = _cvar_worker_no_bess_cache()
    key = (city_key, round(float(pv_capacity_kw), 6), int(outage_seed))
    reference = _no_bess_reference_from_result_record(result)
    cache[key] = reference
    cache.move_to_end(key)
    _trim_ordered_cache(cache, 256)
    return dict(reference)


def _no_bess_reference_from_result_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "no_bess_reference_grid_energy_kwh": float(result["no_bess_reference_grid_energy_kwh"]),
        "no_bess_reference_grid_cost_sgd": float(result["no_bess_reference_grid_cost_sgd"]),
        "no_bess_reference_private_annual_cost_sgd": float(result["no_bess_reference_private_annual_cost_sgd"]),
        "no_bess_reference_unserved_energy_kwh_per_year": float(result["no_bess_reference_unserved_energy_kwh_per_year"]),
        "no_bess_reference_unserved_energy_cost_sgd_per_year": float(result["no_bess_reference_unserved_energy_cost_sgd_per_year"]),
        "no_bess_reference_system_annual_cost_sgd": float(result["no_bess_reference_system_annual_cost_sgd"]),
    }


def _trim_ordered_cache(cache: OrderedDict[Any, Any], limit: int) -> None:
    while len(cache) > int(limit):
        cache.popitem(last=False)


def _summarize_cvar_candidate_results(candidates: pd.DataFrame, scenario_eval: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    evaluated = _validate_cvar_candidate_scenario_source(scenario_eval)
    candidate_flags = candidates[
        [
            "city_key",
            "candidate_id",
            "candidate_source",
            "from_figure1_design_space",
            "from_figure4_expost_optima",
        ]
    ].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for (city_key, candidate_id), group in evaluated.groupby(["city_key", "candidate_id"], sort=False):
        ordered = group.sort_values("scenario_id")
        losses = ordered["outage_loss_sgd_per_year"].astype(float).to_numpy()
        tail_count = _cvar_tail_count(len(losses), float(alpha))
        cvar_loss = float(np.sort(losses)[-tail_count:].mean()) if losses.size else 0.0
        ue_values = ordered["UE_kwh_per_year"].astype(float).to_numpy()
        cvar_ue = float(np.sort(ue_values)[-tail_count:].mean()) if ue_values.size else 0.0
        outage_load = float(ordered["outage_weighted_load_kwh"].astype(float).mean())
        mean_ue = float(ordered["UE_kwh_per_year"].astype(float).mean())
        mean_lolp = float(np.clip(mean_ue / outage_load, 0.0, 1.0)) if outage_load > 0.0 else 0.0
        row0 = ordered.iloc[0]
        fixed = float(ordered["fixed_investment_cost_sgd_per_year"].astype(float).mean())
        expected_operation = float(ordered["operation_cost_sgd_per_year"].astype(float).mean())
        expected_loss = float(ordered["outage_loss_sgd_per_year"].astype(float).mean())
        expected_cost = fixed + expected_operation + expected_loss
        rows.append(
            {
                "city_key": str(city_key),
                "city_label": str(row0["city_label"]),
                "case_key": str(row0["case_key"]),
                "source_case_key": str(row0.get("source_case_key", row0["case_key"])),
                "candidate_id": str(candidate_id),
                "candidate_order": int(row0["candidate_order"]),
                "pv_capacity_kw": float(row0["pv_capacity_kw"]),
                "pv_capacity_mw": float(row0["pv_capacity_mw"]),
                "battery_energy_kwh": float(row0["battery_energy_kwh"]),
                "battery_energy_mwh": float(row0["battery_energy_mwh"]),
                "scenario_count": int(len(ordered)),
                "cvar_alpha": float(alpha),
                "cvar_tail_count": int(tail_count),
                "fixed_investment_cost_sgd_per_year": fixed,
                "expected_operation_cost_sgd_per_year": expected_operation,
                "expected_outage_loss_sgd_per_year": expected_loss,
                "cvar90_outage_loss_sgd_per_year": cvar_loss,
                "expected_annual_cost_sgd_per_year": expected_cost,
                "system_total_cost_sgd_per_year": expected_cost,
                "system_total_cost_million_sgd_per_year": expected_cost / 1e6,
                "UE_kwh_per_year": mean_ue,
                "UE_mwh_per_year": mean_ue / 1000.0,
                "cvar90_UE_kwh_per_year": cvar_ue,
                "energy_based_lolp": mean_lolp,
                "energy_based_lolp_percent": 100.0 * mean_lolp,
                "outage_weighted_load_kwh": outage_load,
            }
        )
    summary = pd.DataFrame(rows).merge(candidate_flags, on=["city_key", "candidate_id"], how="left")
    return _validate_cvar_candidate_summary_source(summary)


def _select_cvar_design_curve(summary: pd.DataFrame, *, alpha: float, lambdas: Sequence[float]) -> pd.DataFrame:
    source = _validate_cvar_candidate_summary_source(summary)
    rows: list[dict[str, Any]] = []
    for city_key in CITY_ORDER:
        city_source = source[source["city_key"].astype(str) == city_key].copy()
        if city_source.empty:
            continue
        for lambda_value in lambdas:
            risk = float(lambda_value)
            scored = city_source.copy()
            scored["risk_aversion_lambda"] = risk
            scored["cvar_objective_sgd_per_year"] = (
                scored["fixed_investment_cost_sgd_per_year"].astype(float)
                + scored["expected_operation_cost_sgd_per_year"].astype(float)
                + (1.0 - risk) * scored["expected_outage_loss_sgd_per_year"].astype(float)
                + risk * scored["cvar90_outage_loss_sgd_per_year"].astype(float)
            )
            selected = scored.sort_values(
                [
                    "cvar_objective_sgd_per_year",
                    "expected_annual_cost_sgd_per_year",
                    "cvar90_outage_loss_sgd_per_year",
                    "battery_energy_kwh",
                    "pv_capacity_kw",
                ]
            ).iloc[0]
            row = selected.to_dict()
            row["risk_aversion_lambda"] = risk
            row["cvar_alpha"] = float(alpha)
            row["is_selected_design"] = True
            rows.append(row)
    return _validate_cvar_design_curve_source(pd.DataFrame(rows))


def _cvar_tail_count(scenario_count: int, alpha: float) -> int:
    count = max(1, int(scenario_count))
    tail_fraction = max(0.0, 1.0 - float(alpha))
    return max(1, min(count, int(np.ceil(tail_fraction * count - 1e-12))))


def _city_cost_distribution_source(
    city_key: str,
    case: InputCase,
    config: dict[str, Any],
    candidate_summary: pd.DataFrame,
    pv_capacity_frame_builder: Any,
    best: pd.Series,
) -> pd.DataFrame:
    design_rows = (
        ("no_pv_no_bess", _cost_design_row(candidate_summary, pv_capacity_frame_builder, config, 0.0, 0.0)),
        (
            "same_pv_no_bess",
            _cost_design_row(
                candidate_summary,
                pv_capacity_frame_builder,
                config,
                float(best["pv_capacity_kw"]),
                0.0,
            ),
        ),
        ("optimal_pv_bess", best),
    )
    rows: list[dict[str, Any]] = []
    for design_order, (design_case, row) in enumerate(design_rows, start=1):
        rows.extend(_cost_component_rows(city_key, case, design_case, design_order, row))
    return _validate_cost_distribution_source(pd.DataFrame(rows))


def _affordable_lolp_cost_distribution_source(
    city_key: str,
    case: InputCase,
    config: dict[str, Any],
    candidate_summary: pd.DataFrame,
    pv_capacity_frame_builder: Any,
    selected: pd.Series,
    design_space: pd.DataFrame,
) -> pd.DataFrame:
    zero_zero = _cost_design_row(candidate_summary, pv_capacity_frame_builder, config, 0.0, 0.0)
    design_rows = (
        ("no_pv_no_bess", zero_zero),
        ("lolp_zero_min_cost", selected),
    )
    selected_row = _select_lolp_zero_min_cost_design(design_space, city_order=(city_key,))
    rows: list[dict[str, Any]] = []
    for design_order, (design_case, row) in enumerate(design_rows, start=1):
        component_rows = _cost_component_rows(
            city_key,
            case,
            design_case,
            design_order,
            row,
            labels=AFFORDABLE_COST_DESIGN_LABELS,
        )
        for component in component_rows:
            component["outage_mode"] = str(selected_row["outage_mode"])
            component["outage_seed"] = int(selected_row["outage_seed"])
            component["zero_pv_zero_bess_cost_sgd_per_year"] = float(selected_row["zero_pv_zero_bess_cost_sgd_per_year"])
            component["affordable_boundary_cost_sgd_per_year"] = float(selected_row["affordable_boundary_cost_sgd_per_year"])
            component["selected_lolp_zero_cost_sgd_per_year"] = float(selected_row["system_total_cost_sgd_per_year"])
            component["cost_delta_vs_no_pv_no_bess_sgd_per_year"] = float(selected_row["cost_delta_vs_no_pv_no_bess_sgd_per_year"])
        rows.extend(component_rows)
    return _validate_affordable_cost_distribution_source(pd.DataFrame(rows), city_order=(city_key,))


def _cost_design_row(
    candidate_summary: pd.DataFrame,
    pv_capacity_frame_builder: Any,
    config: dict[str, Any],
    pv_capacity_kw: float,
    battery_energy_kwh: float,
) -> pd.Series:
    match = candidate_summary[
        np.isclose(candidate_summary["pv_capacity_kw"].astype(float), float(pv_capacity_kw), atol=1e-6)
        & np.isclose(candidate_summary["battery_energy_kwh"].astype(float), float(battery_energy_kwh), atol=1e-6)
    ].copy()
    if not match.empty:
        return match.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]
    frame, pv_meta = pv_capacity_frame_builder(float(pv_capacity_kw))
    result, _ = evaluate_candidate(frame, float(battery_energy_kwh), pv_meta, config)
    return pd.Series(result)


def _design_space_row_at_capacity(source: pd.DataFrame, pv_capacity_kw: float, battery_energy_kwh: float) -> pd.Series:
    match = source[
        np.isclose(source["pv_capacity_kw"].astype(float), float(pv_capacity_kw), atol=1e-6)
        & np.isclose(source["battery_energy_kwh"].astype(float), float(battery_energy_kwh), atol=1e-6)
    ].copy()
    if match.empty:
        city = str(source["city_label"].iloc[0]) if "city_label" in source.columns and not source.empty else "unknown city"
        raise ValueError(
            f"Missing evaluated design at {float(pv_capacity_kw):g} kW PV / "
            f"{float(battery_energy_kwh):g} kWh BESS for {city}."
        )
    if "point_source" in match.columns:
        exact = match[match["point_source"].astype(str) != "interpolated_for_plot"]
        if not exact.empty:
            match = exact
    return match.sort_values("system_total_cost_sgd_per_year").iloc[0]


def _selected_design_space_row(source: pd.DataFrame) -> pd.Series:
    evaluated = source
    if "point_source" in evaluated.columns:
        evaluated = evaluated[evaluated["point_source"].astype(str) != "interpolated_for_plot"].copy()
    selected = evaluated[evaluated["is_lolp_zero_min_cost_design"].astype(bool)]
    if selected.empty:
        city = str(source["city_label"].iloc[0]) if "city_label" in source.columns and not source.empty else "unknown city"
        raise ValueError(f"No LOLP=0 selected design exists for {city}.")
    return selected.sort_values(["system_total_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"]).iloc[0]


def _day_night_cost_component_rows(
    city_key: str,
    design_case: str,
    design_order: int,
    row: pd.Series,
) -> list[dict[str, Any]]:
    components = (
        ("grid_bill", "Grid bill", float(row["grid_cost_sgd_per_year"]), 1),
        ("pv_annual_cost", "PV depreciation + O&M", float(row["pv_annual_cost_sgd_per_year"]), 2),
        ("battery_annual_cost", "Battery depreciation + O&M", float(row["battery_annual_cost_sgd_per_year"]), 3),
        ("lost_load_cost", "Lost load cost", float(row["lost_load_cost_sgd_per_year"]), 4),
    )
    total = max(float(row["system_total_cost_sgd_per_year"]), 1e-9)
    rows = []
    for component_key, component_label, value, stack_order in components:
        rows.append(
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": str(row["case_key"]),
                "source_case_key": str(row.get("source_case_key", row["case_key"])),
                "scenario_key": str(row["scenario_key"]),
                "scenario_label": str(row["scenario_label"]),
                "design_case": design_case,
                "design_case_label": DAY_NIGHT_COST_DESIGN_LABELS[design_case],
                "design_case_order": int(design_order),
                "axis_label": _day_night_axis_label(design_case, row),
                "component_key": component_key,
                "component_label": component_label,
                "stack_order": int(stack_order),
                "cost_sgd": float(value),
                "cost_ksgd": float(value) / 1000.0,
                "cost_million_sgd": float(value) / 1e6,
                "share_of_total_percent": 100.0 * float(value) / total,
                "system_total_cost_sgd_per_year": float(row["system_total_cost_sgd_per_year"]),
                "pv_capacity_kw": float(row["pv_capacity_kw"]),
                "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
                "battery_energy_kwh": float(row["battery_energy_kwh"]),
                "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
                "UE_kwh_per_year": float(row["UE_kwh_per_year"]),
                "energy_based_LOLP": float(row["energy_based_LOLP"]),
                "outage_mode": str(row["outage_mode"]),
                "outage_seed": int(row["outage_seed"]),
            }
        )
    return rows


def _day_night_axis_label(design_case: str, row: pd.Series) -> str:
    prefix = {
        "night_no_pv_no_bess": "Night 0/0",
        "night_lolp_zero_design": "Night LOLP=0",
        "day_no_pv_no_bess": "Day 0/0",
        "day_lolp_zero_design": "Day LOLP=0",
        "night_design_under_day_outage": "Night design\nunder day",
    }[design_case]
    units = figure_display_units(pd.DataFrame([row]))
    return (
        f"{prefix}\n"
        f"{format_pv_capacity_from_kw(float(row['pv_capacity_kw']), units)} PV\n"
        f"{format_battery_energy_from_kwh(float(row['battery_energy_kwh']), units)} BESS"
    )


def _cost_component_rows(
    city_key: str,
    case: InputCase,
    design_case: str,
    design_order: int,
    row: pd.Series,
    *,
    labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    components = (
        ("grid_bill", "Grid bill", float(row["pv_bess_grid_cost_sgd"]), 1, NATURE_PALETTE["neutral_mid"]),
        (
            "pv_annual_cost",
            "PV depreciation + O&M",
            float(row["annualized_pv_cost_sgd"]) + float(row["pv_om_cost_sgd"]),
            2,
            "#7EBC67",
        ),
        (
            "battery_annual_cost",
            "Battery depreciation + O&M",
            float(row["annualized_battery_cost_sgd"]) + float(row["battery_om_cost_sgd"]),
            3,
            NATURE_PALETTE["blue"],
        ),
        (
            "lost_load_cost",
            "Lost load cost",
            float(row["unserved_energy_cost_sgd_per_year"]),
            4,
            NATURE_PALETTE["red"],
        ),
    )
    total = max(float(row["pv_bess_system_annual_cost_sgd"]), 1e-9)
    rows = []
    for key, label, value, order, color in components:
        rows.append(
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": case.case_key,
                "source_case_key": case.case_key,
                "design_case": design_case,
                "design_case_label": (labels or COST_DESIGN_LABELS).get(design_case, str(design_case)),
                "design_case_order": int(design_order),
                "component_key": key,
                "component_label": label,
                "stack_order": order,
                "component_color": color,
                "cost_sgd": float(value),
                "cost_ksgd": float(value) / 1000.0,
                "cost_million_sgd": float(value) / 1e6,
                "share_of_total_percent": 100.0 * float(value) / total,
                "system_total_cost_sgd_per_year": float(row["pv_bess_system_annual_cost_sgd"]),
                "pv_capacity_kw": float(row["pv_capacity_kw"]),
                "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
                "battery_energy_kwh": float(row["battery_energy_kwh"]),
                "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
                "UE_kwh_per_year": float(row["unserved_energy_kwh_per_year"]),
            }
        )
    return rows


def _city_energy_balance_source(
    city_key: str,
    case: InputCase,
    config: dict[str, Any],
    frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    outage_events: pd.DataFrame,
    best: pd.Series,
    *,
    city_order: Sequence[str] = CITY_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aware_cfg = scenario_config(config, case.case_key)
    critical_load = electric_equivalent_building_load(
        frame,
        float(aware_cfg["technical"].get("cooling_cop", aware_cfg["technical"].get("chiller_cop", 5.0))),
        float(aware_cfg["technical"].get("heating_cop", 3.5)),
    )
    catalog = _energy_balance_event_catalog(outage_events, dispatch, frame, critical_load)
    selected = _select_energy_balance_event(catalog, total_hours=len(frame))
    summary = _energy_balance_event_summary(city_key, case, selected, best, aware_cfg)
    source = _energy_balance_window_source(city_key, case, frame, dispatch, critical_load, selected, best, aware_cfg)
    summary["window_UE_kwh"] = float(source["unmet_load_kwh_equiv"].astype(float).sum())
    summary["window_start_hour"] = int(source["hour_index"].astype(int).min())
    summary["window_end_hour"] = int(source["hour_index"].astype(int).max())
    return _validate_energy_balance_source(source, city_order=city_order), summary


def _energy_balance_event_catalog(
    outage_events: pd.DataFrame,
    dispatch: pd.DataFrame,
    frame: pd.DataFrame,
    critical_load: pd.Series,
) -> pd.DataFrame:
    if outage_events.empty:
        raise ValueError("Single-outage city baseline energy balance requires at least one outage event.")
    rows: list[dict[str, Any]] = []
    for event in outage_events.sort_values("event_id").to_dict("records"):
        start = float(event["start_hour"])
        end = float(event["end_hour"])
        mask = _event_hour_mask(frame["hour_index"], start, end)
        if not bool(mask.any()):
            continue
        frame_event = frame.loc[mask].copy()
        dispatch_event = dispatch.loc[mask].copy()
        outage_fraction = frame_event["outage_fraction"].astype(float)
        event_load = critical_load.loc[mask].astype(float)
        ue = float(dispatch_event["unmet_load_kwh_equiv"].astype(float).sum())
        rows.append(
            {
                "event_id": int(event["event_id"]),
                "event_start_hour": start,
                "event_end_hour": end,
                "event_duration_hours": float(event["duration_minutes"]) / 60.0,
                "event_UE_kwh": ue,
                "event_outage_weighted_load_kwh": float(event_load.mul(outage_fraction).sum()),
                "event_pv_available_kwh": float(frame_event["pv_kwh"].astype(float).mul(outage_fraction).sum()),
                "event_net_deficit_kwh": float(
                    np.maximum(
                        event_load.to_numpy(dtype=float) - frame_event["pv_kwh"].astype(float).to_numpy(),
                        0.0,
                    ).dot(outage_fraction.to_numpy(dtype=float))
                ),
            }
        )
    if not rows:
        raise ValueError("No outage events align with the dispatch frame.")
    return pd.DataFrame(rows)


def _select_energy_balance_event(catalog: pd.DataFrame, total_hours: int) -> dict[str, Any]:
    ranked = catalog.copy()
    ranked["window_start_hour"] = ranked["event_start_hour"].astype(float).map(_window_start_hour)
    ranked["window_end_exclusive_hour"] = ranked["window_start_hour"].astype(int) + WINDOW_HOURS
    valid = ranked[
        (ranked["window_start_hour"].astype(int) >= 0)
        & (ranked["window_end_exclusive_hour"].astype(int) <= int(total_hours))
    ].copy()
    if valid.empty:
        raise ValueError("No baseline outage event can support a complete 72 h four-city window.")
    return dict(
        valid.sort_values(
            ["event_UE_kwh", "event_net_deficit_kwh", "event_outage_weighted_load_kwh", "event_duration_hours", "event_id"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
    )


def _energy_balance_event_summary(
    city_key: str,
    case: InputCase,
    selected: dict[str, Any],
    best: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    battery_energy = float(best["battery_energy_kwh"])
    minimum_soc_fraction = float(config["battery"]["soc_min_fraction"]) if battery_energy > 0.0 else 0.0
    return pd.DataFrame(
        [
            {
                "city_key": city_key,
                "city_label": CITY_LABELS[city_key],
                "case_key": case.case_key,
                "source_case_key": case.case_key,
                "selection_rule": "max_event_UE_then_net_deficit",
                "event_id": int(selected["event_id"]),
                "event_start_hour": float(selected["event_start_hour"]),
                "event_end_hour": float(selected["event_end_hour"]),
                "event_duration_hours": float(selected["event_duration_hours"]),
                "event_UE_kwh": float(selected["event_UE_kwh"]),
                "event_net_deficit_kwh": float(selected["event_net_deficit_kwh"]),
                "event_outage_weighted_load_kwh": float(selected["event_outage_weighted_load_kwh"]),
                "event_pv_available_kwh": float(selected["event_pv_available_kwh"]),
                "window_hours": WINDOW_HOURS,
                "pv_capacity_kw": float(best["pv_capacity_kw"]),
                "battery_energy_kwh": battery_energy,
                "minimum_soc_fraction": minimum_soc_fraction,
            }
        ]
    )


def _energy_balance_window_source(
    city_key: str,
    case: InputCase,
    frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    critical_load: pd.Series,
    selected: dict[str, Any],
    best: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    start = int(selected["window_start_hour"])
    end = int(selected["window_end_exclusive_hour"])
    merged = dispatch.copy()
    if "timestamp" in frame.columns:
        merged["timestamp"] = frame["timestamp"].values
    merged["critical_load_kwh"] = critical_load.astype(float).values
    merged["outage_weighted_load_kwh"] = merged["critical_load_kwh"] * frame["outage_fraction"].astype(float).values
    merged["served_load_kwh"] = np.maximum(
        0.0,
        merged["critical_load_kwh"].astype(float) - merged["unmet_load_kwh_equiv"].astype(float),
    )
    eta = float(config["battery"]["roundtrip_efficiency"]) ** 0.5
    merged["conversion_loss_kwh"] = _conversion_loss(
        merged["battery_charge_kwh"],
        merged["battery_discharge_kwh"],
        eta,
    )
    battery_energy = float(best["battery_energy_kwh"])
    previous_soc = merged["battery_soc_kwh"].astype(float).shift(1)
    if len(previous_soc) > 0:
        initial_soc = battery_energy * float(config["battery"].get("initial_soc_fraction", 0.0)) if battery_energy > 0.0 else 0.0
        previous_soc.iloc[0] = initial_soc
    merged["battery_soc_delta_kwh"] = merged["battery_soc_kwh"].astype(float) - previous_soc.fillna(0.0)
    merged["battery_soc_expected_delta_kwh"] = (
        merged["battery_charge_kwh"].astype(float) * eta
        - merged["battery_discharge_kwh"].astype(float) / max(eta, 1e-9)
    )
    merged["battery_soc_delta_residual_kwh"] = (
        merged["battery_soc_delta_kwh"].astype(float)
        - merged["battery_soc_expected_delta_kwh"].astype(float)
    )
    if "simultaneous_charge_discharge_kwh" not in merged.columns:
        merged["simultaneous_charge_discharge_kwh"] = np.minimum(
            merged["battery_charge_kwh"].astype(float),
            merged["battery_discharge_kwh"].astype(float),
        )
    window = merged[
        (merged["hour_index"].astype(int) >= start)
        & (merged["hour_index"].astype(int) < end)
    ].copy()
    if len(window) != WINDOW_HOURS:
        raise ValueError(f"Expected {WINDOW_HOURS} rows for four-city energy balance, got {len(window)}")
    minimum_soc_fraction = float(config["battery"]["soc_min_fraction"]) if battery_energy > 0.0 else 0.0
    window["city_key"] = city_key
    window["city_label"] = CITY_LABELS[city_key]
    window["case_key"] = case.case_key
    window["source_case_key"] = case.case_key
    window["plot_hour"] = window["hour_index"].astype(int) - start
    window["event_start_hour"] = float(selected["event_start_hour"])
    window["event_end_hour"] = float(selected["event_end_hour"])
    window["event_duration_hours"] = float(selected["event_duration_hours"])
    window["event_UE_kwh"] = float(selected["event_UE_kwh"])
    window["event_net_deficit_kwh"] = float(selected["event_net_deficit_kwh"])
    window["pv_capacity_kw"] = float(best["pv_capacity_kw"])
    window["battery_energy_kwh"] = battery_energy
    if battery_energy > 0.0:
        window["battery_soc_fraction"] = window["battery_soc_kwh"].astype(float) / battery_energy
    else:
        window["battery_soc_fraction"] = 0.0
    window["minimum_soc_fraction"] = minimum_soc_fraction
    columns = [
        "city_key",
        "city_label",
        "case_key",
        "source_case_key",
        "plot_hour",
        "hour_index",
        "timestamp",
        "event_start_hour",
        "event_end_hour",
        "event_duration_hours",
        "event_UE_kwh",
        "event_net_deficit_kwh",
        "grid_availability",
        "outage_fraction",
        "critical_load_kwh",
        "outage_weighted_load_kwh",
        "served_load_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "pv_curtailed_kwh",
        "conversion_loss_kwh",
        "unmet_load_kwh_equiv",
        "battery_soc_kwh",
        "battery_soc_fraction",
        "battery_soc_delta_kwh",
        "battery_soc_expected_delta_kwh",
        "battery_soc_delta_residual_kwh",
        "simultaneous_charge_discharge_kwh",
        "minimum_soc_fraction",
        "pv_capacity_kw",
        "battery_energy_kwh",
    ]
    return window[[column for column in columns if column in window.columns]].reset_index(drop=True)


def _candidate_dispatch(
    row: dict[str, Any],
    candidate_dispatches: dict[tuple[str, int, float, float], pd.DataFrame],
) -> pd.DataFrame:
    key = (
        str(row["search_stage"]),
        int(row["search_iteration"]),
        round(float(row["pv_capacity_kw"]), 6),
        round(float(row["battery_energy_kwh"]), 6),
    )
    if key in candidate_dispatches:
        return candidate_dispatches[key]
    for candidate_key, dispatch in candidate_dispatches.items():
        if (
            str(candidate_key[0]) == key[0]
            and int(candidate_key[1]) == key[1]
            and np.isclose(float(candidate_key[2]), float(row["pv_capacity_kw"]))
            and np.isclose(float(candidate_key[3]), float(row["battery_energy_kwh"]))
        ):
            return dispatch
    raise ValueError(
        "Could not find candidate dispatch for "
        f"{key[0]} iteration {key[1]}, PV {key[2]}, BESS {key[3]}."
    )


def _outage_weighted_load_total(frame: pd.DataFrame, outage_profile: pd.DataFrame, config: dict[str, Any]) -> float:
    technical = config["technical"]
    total_load = electric_equivalent_building_load(
        frame,
        float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0))),
        float(technical.get("heating_cop", 3.5)),
    )
    outage = frame[["hour_index"]].merge(
        outage_profile[["hour_index", "outage_fraction"]],
        on="hour_index",
        how="left",
    )["outage_fraction"].fillna(0.0)
    return float(outage_weighted_load_kwh(total_load, outage).sum())


def _singlefamily_cases_by_city(input_root: Path) -> dict[str, InputCase]:
    return _singlefamily_cases_by_city_for_order(input_root, CITY_ORDER, SINGLE_OUTAGE_CASE_BY_CITY)


def _singlefamily_cases_by_city_for_order(
    input_root: Path,
    city_order: Sequence[str],
    case_by_city: dict[str, str],
) -> dict[str, InputCase]:
    resolved: dict[str, InputCase] = {}
    missing = []
    for city_key in city_order:
        case_key = case_by_city[city_key]
        try:
            resolved[city_key] = _input_case_for_exact_case_key(input_root, case_key)
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"{case_key} ({exc})")
            continue
    if missing:
        raise ValueError("Missing real SingleFamily input case(s): " + ", ".join(missing))
    return resolved


def _input_case_for_exact_case_key(input_root: Path, case_key: str) -> InputCase:
    case_dir = input_root / case_key
    if not case_dir.is_dir():
        raise FileNotFoundError(f"directory not found: {case_dir}")

    load_candidates = [case_dir / f"{case_key}_Load.csv", *sorted(case_dir.glob("*_hourly_loads_kWh.csv"))]
    load_csv = next((path for path in load_candidates if path.exists()), None)
    files = {
        "PV": case_dir / f"{case_key}_PV.csv",
        "TOU": case_dir / f"{case_key}_TOU.csv",
        "Parameter": case_dir / f"{case_key}_Parameter.csv",
    }
    missing = [suffix for suffix, path in files.items() if not path.exists()]
    if load_csv is None:
        missing.insert(0, "Load")
    if missing:
        raise ValueError("missing files: " + ", ".join(missing))
    if "_" not in case_key:
        raise ValueError(f"case key must include a building name: {case_key}")
    city_key, building_key = case_key.rsplit("_", 1)
    return InputCase(
        city_key=city_key,
        building_key=building_key,
        case_key=case_key,
        city_dir=case_dir,
        load_csv=load_csv,
        pv_csv=files["PV"],
        tou_csv=files["TOU"],
        parameter_csv=files["Parameter"],
    )


def _read_existing_figure_source(output_dir: Path, case_key: str, figure_no: int | str) -> dict[str, pd.DataFrame]:
    path = output_dir / f"{case_key}_{FIGURE_NAMES[figure_no]}_Source_Data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Figure {figure_no} source data is required before redrawing from existing sources: {path}"
        )
    source = pd.read_csv(path)
    if "source_table" not in source.columns:
        raise ValueError(f"Existing Figure {figure_no} source data lacks source_table: {path}")
    frames: dict[str, pd.DataFrame] = {}
    for key in SOURCE_KEYS[figure_no]:
        subset = source[source["source_table"].astype(str) == key].drop(columns=["source_table"]).reset_index(drop=True)
        if not subset.empty:
            frames[key] = subset
    missing = set(SOURCE_KEYS[figure_no]) - set(frames)
    missing = missing - OPTIONAL_SOURCE_KEYS.get(figure_no, set())
    if missing:
        raise ValueError(f"Existing Figure {figure_no} source data missing table(s): {', '.join(sorted(missing))}")
    return frames


def _figure_source(sources: dict[str, pd.DataFrame], figure_no: int | str) -> pd.DataFrame:
    frames = []
    for key in SOURCE_KEYS[figure_no]:
        if key not in sources and key in OPTIONAL_SOURCE_KEYS.get(figure_no, set()):
            continue
        frame = sources[key].copy()
        if figure_no in {"1b", "2b", "3b", "6b"} and "city_key" in frame.columns:
            frame = frame[frame["city_key"].astype(str).isin(SINGAPORE_BRISBANE_CITY_ORDER)].copy()
        frame.insert(0, "source_table", key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _expand_figures(figures: Sequence[int | str] | None) -> tuple[int | str, ...]:
    if figures is None:
        return FIGURE_NUMBERS
    selected: list[int | str] = []
    for figure in figures:
        if isinstance(figure, str) and figure.strip().lower() == "all":
            selected.extend(FIGURE_NUMBERS)
            continue
        normalized = str(figure).strip().lower()
        value: int | str = int(normalized) if normalized.isdigit() else normalized
        if value not in FIGURE_NUMBERS:
            raise ValueError(f"Unsupported single-outage city baseline figure: {figure}")
        selected.append(value)
    return tuple(dict.fromkeys(selected))


def _contour_metric_specs(units: Any) -> tuple[dict[str, str], ...]:
    labels = {
        "system_total_cost": units.cost_axis_label,
        "ue": units.ue_axis_label,
        "lolp": "LOLP (%)",
        "ssr": "SSR (%)",
        "scr": "SCR (%)",
    }
    return tuple(
        {
            "key": key,
            "column": column,
            "title": title,
            "unit_type": unit_type,
            "cmap": cmap,
            "label": labels[key],
        }
        for key, column, title, unit_type, cmap in CONTOUR_METRICS
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


def _mark_selected_design(ax: Any, source: pd.DataFrame) -> None:
    selected = source[source["is_minimum_cost_design"].astype(bool)]
    if selected.empty:
        return
    row = selected.iloc[0]
    ax.scatter(
        [float(row[DISPLAY_BATTERY_ENERGY])],
        [float(row[DISPLAY_PV_CAPACITY])],
        marker="*",
        s=70,
        facecolor=NATURE_PALETTE["neutral_black"],
        edgecolor="white",
        linewidth=0.42,
        zorder=8,
    )


def _mark_lolp_zero_design(ax: Any, source: pd.DataFrame) -> None:
    selected = source[source["is_lolp_zero_min_cost_design"].astype(bool)]
    if selected.empty:
        return
    row = selected.iloc[0]
    ax.scatter(
        [float(row[DISPLAY_BATTERY_ENERGY])],
        [float(row[DISPLAY_PV_CAPACITY])],
        marker="*",
        s=88,
        facecolor=NATURE_PALETTE["neutral_black"],
        edgecolor="white",
        linewidth=0.50,
        zorder=9,
    )


def _draw_affordable_cost_boundary(
    ax: Any,
    source: pd.DataFrame,
    boundary_cost: float,
    cost_column: str,
) -> None:
    points = (
        source.groupby([DISPLAY_PV_CAPACITY, DISPLAY_BATTERY_ENERGY], as_index=False)[cost_column]
        .mean()
        .sort_values([DISPLAY_BATTERY_ENERGY, DISPLAY_PV_CAPACITY])
    )
    if len(points) < 3:
        return
    z_values = points[cost_column].to_numpy(dtype=float)
    finite = z_values[np.isfinite(z_values)]
    if finite.size == 0:
        return
    z_min = float(finite.min())
    z_max = float(finite.max())
    if np.isclose(z_min, z_max):
        return
    epsilon = max((z_max - z_min) * 1e-6, 1e-9)
    level = float(np.clip(float(boundary_cost), z_min + epsilon, z_max - epsilon))
    pivot = points.pivot_table(
        index=DISPLAY_PV_CAPACITY,
        columns=DISPLAY_BATTERY_ENERGY,
        values=cost_column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2 and not pivot.isna().any().any():
        values = pivot.to_numpy(dtype=float)
        xx, yy = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
        ax.contour(
            xx,
            yy,
            values,
            levels=[level],
            colors=NATURE_PALETTE["neutral_black"],
            linewidths=1.65,
            alpha=0.95,
        )
        return
    x = points[DISPLAY_BATTERY_ENERGY].to_numpy(dtype=float)
    y = points[DISPLAY_PV_CAPACITY].to_numpy(dtype=float)
    if np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return
    ax.tricontour(
        x,
        y,
        z_values,
        levels=[level],
        colors=NATURE_PALETTE["neutral_black"],
        linewidths=1.65,
        alpha=0.95,
    )


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


def _stack_city_design_cost_bars(
    ax: Any,
    costs: pd.DataFrame,
    colors: dict[str, str],
    *,
    divisor: float,
    units: Any,
    design_case_order: Sequence[str] = COST_DESIGN_CASES,
) -> None:
    design_cases = [case for case in design_case_order if case in set(costs["design_case"].astype(str))]
    x = np.arange(len(design_cases), dtype=float) * SINGLE_OUTAGE_BAR_STEP
    bottom = np.zeros(len(design_cases), dtype=float)
    ordered_components = (
        costs[["component_key", "component_label", "stack_order"]]
        .drop_duplicates()
        .sort_values("stack_order")
        .to_dict("records")
    )
    for component in ordered_components:
        key = str(component["component_key"])
        subset = costs[costs["component_key"].astype(str) == key].set_index("design_case")
        values = np.array(
            [float(subset.loc[design_case, "cost_sgd"]) / max(float(divisor), 1e-9) if design_case in subset.index else 0.0 for design_case in design_cases],
            dtype=float,
        )
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=SINGLE_OUTAGE_BAR_WIDTH,
            color=colors[key],
            edgecolor="white",
            linewidth=0.35,
            label=str(component["component_label"]),
        )
        bottom += values
    ax.set_xticks(x)
    if len(x) > 0:
        ax.set_xlim(float(x[0] - SINGLE_OUTAGE_BAR_STEP * 0.52), float(x[-1] + SINGLE_OUTAGE_BAR_STEP * 0.52))
    ax.set_xticklabels(
        _cost_design_axis_labels(costs, design_cases, units),
        rotation=20,
        ha="right",
        fontsize=SINGLE_OUTAGE_COST_XTICK_SIZE,
    )


def _cost_design_axis_labels(costs: pd.DataFrame, design_cases: Sequence[str], units: Any) -> list[str]:
    if "axis_label" in costs.columns:
        labels = costs[["design_case", "axis_label"]].drop_duplicates(subset=["design_case"]).set_index("design_case")
        return [str(labels.loc[design_case, "axis_label"]) for design_case in design_cases]
    designs = (
        costs[["design_case", "pv_capacity_kw", "battery_energy_kwh"]]
        .drop_duplicates(subset=["design_case"])
        .set_index("design_case")
    )
    labels = []
    for design_case in design_cases:
        if design_case not in designs.index:
            labels.append(str(design_case))
            continue
        row = designs.loc[design_case]
        pv_label = format_pv_capacity_from_kw(float(row["pv_capacity_kw"]), units)
        battery_label = format_battery_energy_from_kwh(float(row["battery_energy_kwh"]), units)
        labels.append(f"{pv_label} PV\n{battery_label} BESS")
    return labels


def _cost_axis_upper(costs: pd.DataFrame, divisor: float) -> float:
    totals = (
        costs[["city_key", "design_case", "system_total_cost_sgd_per_year"]]
        .drop_duplicates()
        ["system_total_cost_sgd_per_year"]
        .astype(float)
        .to_numpy()
    )
    if totals.size == 0:
        return 1.0
    return max(float(totals.max()) / max(float(divisor), 1e-9) * 1.10, 1.0)


def _draw_energy_balance_panel(
    ax: Any,
    group: pd.DataFrame,
    palette: dict[str, str],
    shared_ylim: tuple[float, float],
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    show_soc_ylabel: bool,
) -> None:
    x = group["plot_hour"].astype(float).to_numpy()
    x_mid = x + 0.5
    outage_start, outage_end = _plot_outage_bounds(group)
    _shade_outage(ax, outage_start, outage_end)
    _draw_day_boundaries(ax, palette)
    _stack_positive_balance(ax, x, group, palette)
    _stack_negative_balance(ax, x, group, palette)
    ax.axhline(0.0, color=palette["neutral_black"], linewidth=0.75)
    ax.set_xlim(0.0, float(WINDOW_HOURS))
    ax.set_ylim(shared_ylim)
    ax.set_title(str(group["city_label"].iloc[0]), loc="left", pad=4)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    if show_ylabel:
        ax.set_ylabel("Hourly energy\nbalance (kWh)")
    if show_xlabel:
        ax.set_xlabel("Hour in 72 h window")

    ax_soc = ax.twinx()
    ax_soc.plot(x_mid, group["battery_soc_fraction"].astype(float), color=palette["soc"], linestyle="-.", linewidth=1.10, zorder=6)
    if "minimum_soc_fraction" in group.columns:
        ax_soc.plot(x_mid, group["minimum_soc_fraction"].astype(float), color=palette["unserved"], linestyle=":", linewidth=0.75, zorder=5)
    ax_soc.set_ylim(_soc_ylim_for_zero_alignment(ax.get_ylim(), soc_upper=1.05))
    ax_soc.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_soc.spines["right"].set_visible(True)
    ax_soc.spines["top"].set_visible(False)
    ax_soc.spines["right"].set_color(palette["neutral_dark"])
    ax_soc.tick_params(axis="y", colors=palette["neutral_dark"], length=2.5, width=0.65)
    if show_soc_ylabel:
        ax_soc.set_ylabel("BESS SOC")
    else:
        ax_soc.set_yticklabels([])


def _shared_energy_balance_ylim(source: pd.DataFrame) -> tuple[float, float]:
    positive = (
        source["pv_used_kwh"].astype(float)
        + source["grid_import_kwh"].astype(float)
        + source["battery_discharge_kwh"].astype(float)
    )
    negative = (
        source["served_load_kwh"].astype(float)
        + source["battery_charge_kwh"].astype(float)
        + source["pv_curtailed_kwh"].astype(float)
        + source["conversion_loss_kwh"].astype(float)
        + source["unmet_load_kwh_equiv"].astype(float)
    )
    upper = max(float(positive.max()), 1.0)
    lower = -max(float(negative.max()), 1.0)
    pad = 0.14 * max(upper, abs(lower))
    return lower - pad, upper + pad


def _scenario_distribution_display_sources(
    scenario_results: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    combined = pd.concat([scenario_results, reference], ignore_index=True, sort=False)
    units = figure_display_units(combined)
    scenario_display, _ = add_display_unit_columns(scenario_results, units=units)
    reference_display, _ = add_display_unit_columns(reference, units=units)
    return scenario_display, reference_display, units


def _plot_scenario_distribution_metric(
    ax: Any,
    scenario_results: pd.DataFrame,
    reference: pd.DataFrame,
    city_order: Sequence[str],
    scenario_column: str,
    reference_column: str,
) -> None:
    positions = np.arange(1, len(city_order) + 1, dtype=float)
    data = [
        scenario_results.loc[scenario_results["city_key"].astype(str) == city, scenario_column].astype(float).to_numpy()
        for city in city_order
    ]
    box = ax.boxplot(
        data,
        positions=positions - 0.12,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": NATURE_PALETTE["neutral_black"], "linewidth": 0.8},
        whiskerprops={"color": NATURE_PALETTE["neutral_mid"], "linewidth": 0.7},
        capprops={"color": NATURE_PALETTE["neutral_mid"], "linewidth": 0.7},
    )
    for patch, city in zip(box["boxes"], city_order, strict=True):
        patch.set_facecolor(CITY_COLORS[city])
        patch.set_edgecolor("white")
        patch.set_alpha(0.62)

    for position, city, values in zip(positions, city_order, data, strict=True):
        color = CITY_COLORS[city]
        group = scenario_results[scenario_results["city_key"].astype(str) == city].sort_values("scenario_id")
        jitter = group["plot_jitter"].astype(float).to_numpy()
        ax.scatter(
            position + jitter,
            group[scenario_column].astype(float).to_numpy(),
            s=7.0,
            color=color,
            alpha=0.55,
            linewidth=0.0,
            zorder=3,
        )
        if len(values) >= 2 and np.isfinite(values).any():
            violin_center = position + 0.22
            parts = ax.violinplot([values], positions=[violin_center], widths=0.42, showmeans=False, showmedians=False, showextrema=False)
            body = parts["bodies"][0]
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], violin_center)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.34)

    reference_values = []
    for city in city_order:
        row = reference[reference["city_key"].astype(str) == city].iloc[0]
        reference_values.append(float(row[reference_column]))
    ax.scatter(
        positions - 0.34,
        reference_values,
        marker="D",
        s=22.0,
        facecolor="white",
        edgecolor=NATURE_PALETTE["neutral_black"],
        linewidth=0.8,
        zorder=5,
    )
    ax.set_xlim(0.45, len(city_order) + 0.72)
    ax.set_xticks(positions)
    ax.set_xticklabels(_city_axis_labels(scenario_results, city_order), rotation=30, ha="right")
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.60)


def _city_axis_labels(source: pd.DataFrame, city_order: Sequence[str]) -> list[str]:
    labels = source[["city_key", "city_label"]].drop_duplicates().set_index("city_key")["city_label"].to_dict()
    return [str(labels.get(city, CITY_LABELS[city])) for city in city_order]


def _tou_step_xy(tou_profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    source = _validate_tou_profile_source(tou_profile)
    ordered = source.sort_values("hour").copy()
    x = ordered["interval_start_hour"].astype(float).to_numpy()
    y = ordered["tou_sgd_per_kwh"].astype(float).to_numpy()
    return np.append(x, float(ordered["interval_end_hour"].astype(float).iloc[-1])), np.append(y, float(y[-1]))


def _cost_component_colors() -> dict[str, str]:
    return {
        "grid_bill": "#6F7E86",
        "pv_annual_cost": "#5F8A70",
        "battery_annual_cost": "#4E5F8A",
        "lost_load_cost": "#8F5B55",
    }


def _single_outage_energy_balance_palette() -> dict[str, str]:
    palette = _outage_balance_palette()
    palette.update(
        {
            "pv": "#8FCF73",
            "grid": NATURE_PALETTE["teal_light"],
            "battery_discharge": "#C7B5E8",
            "battery_charge": "#F4A7A0",
            "curtailment": "#D7D0EB",
            "loss": "#BFC3C7",
            "load": "#AAB2B8",
            "unserved": NATURE_PALETTE["red_light"],
            "soc": NATURE_PALETTE["purple"],
        }
    )
    return palette


def _best_design_frame_from_costs(costs: pd.DataFrame) -> pd.DataFrame:
    rows = (
        costs[["city_key", "pv_capacity_kw", "pv_capacity_mw", "battery_energy_kwh", "battery_energy_mwh", "system_total_cost_sgd_per_year", "UE_kwh_per_year"]]
        .drop_duplicates()
        .copy()
    )
    rows["UE_mwh_per_year"] = rows["UE_kwh_per_year"].astype(float) / 1000.0
    return rows


def _city_order(source: pd.DataFrame, order: Sequence[str] = CITY_ORDER) -> list[str]:
    present = set(source["city_key"].astype(str))
    return [city for city in order if city in present]


def _event_hour_mask(hour_index: pd.Series, start_hour: float, end_hour: float) -> pd.Series:
    hours = hour_index.astype(float)
    return (hours + 1.0 > float(start_hour)) & (hours < float(end_hour))


def _validate_design_space_source(
    source: pd.DataFrame,
    *,
    city_order: Sequence[str] = CITY_ORDER,
) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "outage_weighted_load_kwh",
        "energy_based_LOLP",
        "self_sufficiency_rate",
        "self_consumption_rate",
        "is_minimum_cost_design",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Single-outage design-space source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    numeric = [column for column in required if column not in {"city_key", "city_label", "case_key", "is_minimum_cost_design"}]
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric].isna().any().any():
        raise ValueError("Single-outage design-space source contains non-numeric values.")
    for column in ("energy_based_LOLP", "self_sufficiency_rate", "self_consumption_rate"):
        if not validated[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be between 0 and 1.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(city_order), ordered=True)
    validated = validated.sort_values(["city_key", "pv_capacity_kw", "battery_energy_kwh"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_affordable_design_space_source(
    source: pd.DataFrame,
    *,
    city_order: Sequence[str] = FIGURE_C_CITY_ORDER,
) -> pd.DataFrame:
    extra_required = {
        "outage_mode",
        "outage_seed",
        "zero_pv_zero_bess_cost_sgd_per_year",
        "affordable_boundary_cost_sgd_per_year",
        "cost_delta_vs_no_pv_no_bess_sgd_per_year",
        "is_no_pv_no_bess_reference",
        "is_lolp_zero_min_cost_design",
        "selected_lolp_zero_pv_capacity_kw",
        "selected_lolp_zero_battery_energy_kwh",
        "selected_lolp_zero_cost_sgd_per_year",
        "selected_lolp_zero_UE_kwh_per_year",
        "selected_lolp_zero_energy_based_LOLP",
    }
    missing = extra_required - set(source.columns)
    if missing:
        raise ValueError("Figure 1c design-space source missing column(s): " + ", ".join(sorted(missing)))
    validated = _validate_design_space_source(source, city_order=city_order)
    numeric = [
        "outage_seed",
        "zero_pv_zero_bess_cost_sgd_per_year",
        "affordable_boundary_cost_sgd_per_year",
        "cost_delta_vs_no_pv_no_bess_sgd_per_year",
        "selected_lolp_zero_pv_capacity_kw",
        "selected_lolp_zero_battery_energy_kwh",
        "selected_lolp_zero_cost_sgd_per_year",
        "selected_lolp_zero_UE_kwh_per_year",
        "selected_lolp_zero_energy_based_LOLP",
    ]
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric].isna().any().any():
        raise ValueError("Figure 1c design-space source contains non-numeric affordable metadata.")
    if set(validated["city_key"].astype(str)) != set(city_order):
        raise ValueError("Figure 1c design-space source must contain exactly the configured c-series cities.")
    for city_key in city_order:
        group = validated[validated["city_key"].astype(str) == city_key]
        if int(group["is_no_pv_no_bess_reference"].astype(bool).sum()) < 1:
            raise ValueError(f"Figure 1c source lacks 0 PV / 0 BESS reference for {CITY_LABELS[city_key]}.")
        if int(group["is_lolp_zero_min_cost_design"].astype(bool).sum()) != 1:
            raise ValueError(f"Figure 1c source must select exactly one LOLP=0 design for {CITY_LABELS[city_key]}.")
        selected = group[group["is_lolp_zero_min_cost_design"].astype(bool)].iloc[0]
        if float(selected["UE_kwh_per_year"]) > AFFORDABLE_LOLP_ZERO_TOLERANCE_KWH:
            raise ValueError(f"Figure 1c selected design is not zero-LOLP for {CITY_LABELS[city_key]}.")
        if not np.isclose(
            float(group["zero_pv_zero_bess_cost_sgd_per_year"].iloc[0]),
            float(group["affordable_boundary_cost_sgd_per_year"].iloc[0]),
        ):
            raise ValueError(f"Figure 1c affordable boundary must match 0 PV / 0 BESS cost for {CITY_LABELS[city_key]}.")
    return validated


def _validate_affordable_plot_grid_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "system_total_cost_sgd_per_year",
        "energy_based_LOLP",
        "affordable_boundary_cost_sgd_per_year",
        "point_source",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Figure 1c plot-grid source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    numeric = [
        "pv_capacity_kw",
        "battery_energy_kwh",
        "system_total_cost_sgd_per_year",
        "energy_based_LOLP",
        "affordable_boundary_cost_sgd_per_year",
    ]
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric].isna().any().any():
        raise ValueError("Figure 1c plot-grid source contains blank contour values.")
    if not validated["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Figure 1c plot-grid LOLP values must be between 0 and 1.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(FIGURE_C_CITY_ORDER), ordered=True)
    validated = validated.sort_values(["city_key", "pv_capacity_kw", "battery_energy_kwh"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_day_night_design_space_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "scenario_key",
        "scenario_label",
        "scenario_order",
        "point_source",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "system_total_cost_sgd_per_year",
        "grid_cost_sgd_per_year",
        "pv_annual_cost_sgd_per_year",
        "battery_annual_cost_sgd_per_year",
        "lost_load_cost_sgd_per_year",
        "UE_kwh_per_year",
        "energy_based_LOLP",
        "is_lolp_zero_min_cost_design",
        "affordable_boundary_cost_sgd_per_year",
        "outage_window_start_hour",
        "outage_window_end_hour",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Figure 1d day/night source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    numeric = [
        "scenario_order",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "system_total_cost_sgd_per_year",
        "grid_cost_sgd_per_year",
        "pv_annual_cost_sgd_per_year",
        "battery_annual_cost_sgd_per_year",
        "lost_load_cost_sgd_per_year",
        "UE_kwh_per_year",
        "energy_based_LOLP",
        "affordable_boundary_cost_sgd_per_year",
        "outage_window_start_hour",
        "outage_window_end_hour",
    ]
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric].isna().any().any():
        raise ValueError("Figure 1d day/night source contains non-numeric values.")
    if not validated["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Figure 1d day/night LOLP values must be between 0 and 1.")
    if set(validated["city_key"].astype(str)) != set(FIGURE_D_CITY_ORDER):
        raise ValueError("Figure 1d source must contain Singapore and Jakarta.")
    if set(validated["scenario_key"].astype(str)) != set(FIGURE_D_SCENARIOS):
        raise ValueError("Figure 1d source must contain night and day scenarios.")
    for city_key in FIGURE_D_CITY_ORDER:
        for scenario_key in FIGURE_D_SCENARIOS:
            group = validated[
                (validated["city_key"].astype(str) == city_key)
                & (validated["scenario_key"].astype(str) == scenario_key)
                & (validated["point_source"].astype(str) != "interpolated_for_plot")
            ]
            if int(group["is_lolp_zero_min_cost_design"].astype(bool).sum()) != 1:
                raise ValueError(f"Figure 1d requires one evaluated LOLP=0 selected design for {CITY_LABELS[city_key]} {scenario_key}.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(FIGURE_D_CITY_ORDER), ordered=True)
    validated["scenario_key"] = pd.Categorical(validated["scenario_key"].astype(str), categories=list(FIGURE_D_SCENARIOS), ordered=True)
    validated = validated.sort_values(["city_key", "scenario_key", "pv_capacity_kw", "battery_energy_kwh"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    validated["scenario_key"] = validated["scenario_key"].astype(str)
    return validated


def _validate_cost_distribution_source(
    source: pd.DataFrame,
    *,
    city_order: Sequence[str] = CITY_ORDER,
    design_cases: Sequence[str] = COST_DESIGN_CASES,
) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "design_case",
        "design_case_label",
        "design_case_order",
        "component_key",
        "component_label",
        "stack_order",
        "cost_sgd",
        "share_of_total_percent",
        "system_total_cost_sgd_per_year",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "UE_kwh_per_year",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Single-outage cost source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {"city_key", "city_label", "design_case", "design_case_label", "component_key", "component_label"}
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Single-outage cost source contains non-numeric values.")
    if (validated[["cost_sgd", "system_total_cost_sgd_per_year"]] < 0.0).any().any():
        raise ValueError("Single-outage cost source costs must be non-negative.")
    present_design_cases = set(validated["design_case"].astype(str))
    if present_design_cases != set(design_cases):
        raise ValueError("Single-outage cost source must include expected design cases: " + ", ".join(design_cases))
    counts = validated.groupby(["city_key", "design_case"]).size()
    if set(counts.astype(int).tolist()) != {4}:
        raise ValueError("Single-outage cost source must contain four cost components for each city/design case.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(city_order), ordered=True)
    validated["design_case"] = pd.Categorical(validated["design_case"].astype(str), categories=list(design_cases), ordered=True)
    validated = validated.sort_values(["city_key", "design_case", "stack_order"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    validated["design_case"] = validated["design_case"].astype(str)
    return validated


def _validate_affordable_cost_distribution_source(
    source: pd.DataFrame,
    *,
    city_order: Sequence[str] = FIGURE_C_CITY_ORDER,
) -> pd.DataFrame:
    validated = _validate_cost_distribution_source(
        source,
        city_order=city_order,
        design_cases=AFFORDABLE_COST_DESIGN_CASES,
    )
    if set(validated["city_key"].astype(str)) != set(city_order):
        raise ValueError("Figure 2c cost source must contain exactly the configured c-series cities.")
    for city_key in city_order:
        group = validated[validated["city_key"].astype(str) == city_key]
        cases = group["design_case"].drop_duplicates().astype(str).tolist()
        if cases != list(AFFORDABLE_COST_DESIGN_CASES):
            raise ValueError(f"Figure 2c design cases are not in the expected order for {CITY_LABELS[city_key]}.")
    return validated


def _validate_day_night_cost_distribution_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "design_case",
        "design_case_label",
        "design_case_order",
        "axis_label",
        "component_key",
        "component_label",
        "stack_order",
        "cost_sgd",
        "share_of_total_percent",
        "system_total_cost_sgd_per_year",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "UE_kwh_per_year",
        "energy_based_LOLP",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Figure 2d cost source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {
        "city_key",
        "city_label",
        "design_case",
        "design_case_label",
        "axis_label",
        "component_key",
        "component_label",
        "scenario_key",
        "scenario_label",
        "outage_mode",
    }
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Figure 2d cost source contains non-numeric values.")
    if set(validated["city_key"].astype(str)) != set(FIGURE_D_CITY_ORDER):
        raise ValueError("Figure 2d cost source must contain Singapore and Jakarta.")
    if set(validated["design_case"].astype(str)) != set(DAY_NIGHT_COST_DESIGN_CASES):
        raise ValueError("Figure 2d cost source must include all day/night design cases.")
    counts = validated.groupby(["city_key", "design_case"]).size().astype(int)
    if set(counts.tolist()) != {4}:
        raise ValueError("Figure 2d cost source must contain four components for each city/design case.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(FIGURE_D_CITY_ORDER), ordered=True)
    validated["design_case"] = pd.Categorical(validated["design_case"].astype(str), categories=list(DAY_NIGHT_COST_DESIGN_CASES), ordered=True)
    validated = validated.sort_values(["city_key", "design_case", "stack_order"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    validated["design_case"] = validated["design_case"].astype(str)
    return validated


def _validate_energy_balance_source(
    source: pd.DataFrame,
    *,
    city_order: Sequence[str] = CITY_ORDER,
) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "plot_hour",
        "hour_index",
        "event_start_hour",
        "event_end_hour",
        "grid_availability",
        "critical_load_kwh",
        "served_load_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "pv_curtailed_kwh",
        "conversion_loss_kwh",
        "unmet_load_kwh_equiv",
        "battery_soc_fraction",
        "battery_soc_delta_kwh",
        "battery_soc_expected_delta_kwh",
        "battery_soc_delta_residual_kwh",
        "simultaneous_charge_discharge_kwh",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Single-outage energy-balance source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    for column in required - {"city_key", "city_label"}:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - {"city_key", "city_label"})].isna().any().any():
        raise ValueError("Single-outage energy-balance source contains non-numeric values.")
    counts = validated.groupby("city_key").size().to_dict()
    if any(counts.get(city, 0) != WINDOW_HOURS for city in city_order if city in counts):
        raise ValueError(f"Single-outage energy balance requires {WINDOW_HOURS} rows per city.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(city_order), ordered=True)
    validated = validated.sort_values(["city_key", "plot_hour"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_scenario_distribution_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "scenario_id",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "ue_kwh_per_year",
        "energy_based_lolp",
        "energy_based_lolp_percent",
        "outage_weighted_load_kwh",
        "annual_outage_hours",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "plot_jitter",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Single-outage scenario distribution source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {"city_key", "city_label", "case_key", "source_case_key"}
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Single-outage scenario distribution source contains non-numeric values.")
    if not validated["energy_based_lolp"].between(0.0, 1.0).all():
        raise ValueError("energy_based_lolp must be between 0 and 1.")
    counts = validated.groupby("city_key").size().astype(int).to_dict()
    for city in CITY_ORDER:
        if counts.get(city, 0) != SCENARIO_DISTRIBUTION_COUNT:
            raise ValueError(f"Figure 4 requires {SCENARIO_DISTRIBUTION_COUNT} scenario rows for {city}.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(CITY_ORDER), ordered=True)
    validated = validated.sort_values(["city_key", "scenario_id"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_reference_design_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "design_case",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "ue_kwh_per_year",
        "energy_based_lolp",
        "energy_based_lolp_percent",
        "outage_weighted_load_kwh",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Single-outage reference design source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {"city_key", "city_label", "case_key", "source_case_key", "design_case"}
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Single-outage reference design source contains non-numeric values.")
    if not validated["energy_based_lolp"].between(0.0, 1.0).all():
        raise ValueError("reference energy_based_lolp must be between 0 and 1.")
    present = set(validated["city_key"].astype(str))
    if present != set(CITY_ORDER):
        raise ValueError("Single-outage reference design source must contain exactly the four configured cities.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(CITY_ORDER), ordered=True)
    validated = validated.sort_values("city_key").reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_tou_profile_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "hour",
        "interval_start_hour",
        "interval_end_hour",
        "tou_sgd_per_kwh",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Single-outage TOU profile source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {"city_key", "city_label", "case_key", "source_case_key"}
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Single-outage TOU profile source contains non-numeric values.")
    for city_key, group in validated.groupby(validated["city_key"].astype(str), sort=False):
        hours = sorted(group["hour"].astype(int).tolist())
        if hours != list(range(24)):
            raise ValueError(f"TOU profile requires hours 0-23 exactly for {city_key}.")
    validated["city_key"] = pd.Categorical(
        validated["city_key"].astype(str),
        categories=list(FIGURE_1_TO_4_CITY_ORDER),
        ordered=True,
    )
    validated = validated.sort_values(["city_key", "hour"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_brisbane_sensitivity_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sensitivity_driver",
        "x_value",
        "x_label",
        "city_key",
        "city_label",
        "case_key",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "energy_based_LOLP",
        "saidi_min_per_year",
        "voll_sgd_per_kwh",
        "tou_spread_k",
        "tou_mean_sgd_per_kwh",
        "tou_min_sgd_per_kwh",
        "tou_max_sgd_per_kwh",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Figure 7 Brisbane sensitivity source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    if "driver_order" not in validated.columns:
        order = {driver: index for index, driver in enumerate(BRISBANE_SENSITIVITY_DRIVER_ORDER)}
        validated["driver_order"] = validated["sensitivity_driver"].astype(str).map(order)
    text_columns = {"sensitivity_driver", "x_label", "city_key", "city_label", "case_key", "source_case_key"}
    for column in (required | {"driver_order"}) - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list((required | {"driver_order"}) - text_columns)].isna().any().any():
        raise ValueError("Figure 7 Brisbane sensitivity source contains non-numeric values.")
    if set(validated["city_key"].astype(str)) != {"Brisbane"}:
        raise ValueError("Figure 7 Brisbane sensitivity source must only contain Brisbane.")
    if set(validated["sensitivity_driver"].astype(str)) != set(BRISBANE_SENSITIVITY_DRIVER_ORDER):
        raise ValueError("Figure 7 Brisbane sensitivity source must contain TOU, SAIDI, and VoLL drivers.")
    if not validated["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Figure 7 energy_based_LOLP must be between 0 and 1.")
    if (validated[["pv_capacity_kw", "battery_energy_kwh", "system_total_cost_sgd_per_year", "UE_kwh_per_year"]] < 0.0).any().any():
        raise ValueError("Figure 7 sizing, cost, and UE values must be non-negative.")
    if (validated[["tou_min_sgd_per_kwh", "tou_mean_sgd_per_kwh", "tou_max_sgd_per_kwh"]] < -1e-9).any().any():
        raise ValueError("Figure 7 TOU tariffs must be non-negative.")
    validated["sensitivity_driver"] = pd.Categorical(
        validated["sensitivity_driver"].astype(str),
        categories=list(BRISBANE_SENSITIVITY_DRIVER_ORDER),
        ordered=True,
    )
    validated = validated.sort_values(["sensitivity_driver", "x_value"]).reset_index(drop=True)
    validated["sensitivity_driver"] = validated["sensitivity_driver"].astype(str)
    return validated


def _validate_cvar_candidate_scenario_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "candidate_id",
        "candidate_order",
        "scenario_id",
        "outage_seed",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "fixed_investment_cost_sgd_per_year",
        "operation_cost_sgd_per_year",
        "outage_loss_sgd_per_year",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "ue_kwh_per_year",
        "energy_based_lolp",
        "energy_based_lolp_percent",
        "outage_weighted_load_kwh",
        "annual_outage_hours",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Figure 5 candidate-scenario source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {
        "city_key",
        "city_label",
        "case_key",
        "source_case_key",
        "candidate_id",
        "candidate_source",
    }
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Figure 5 candidate-scenario source contains non-numeric values.")
    if not validated["energy_based_lolp"].between(0.0, 1.0).all():
        raise ValueError("Figure 5 candidate-scenario energy_based_lolp must be between 0 and 1.")
    if (validated[["fixed_investment_cost_sgd_per_year", "operation_cost_sgd_per_year", "outage_loss_sgd_per_year"]] < 0.0).any().any():
        raise ValueError("Figure 5 candidate-scenario costs must be non-negative.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(CITY_ORDER), ordered=True)
    validated = validated.sort_values(["city_key", "candidate_order", "scenario_id"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_cvar_candidate_summary_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "candidate_id",
        "candidate_order",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "scenario_count",
        "cvar_alpha",
        "cvar_tail_count",
        "fixed_investment_cost_sgd_per_year",
        "expected_operation_cost_sgd_per_year",
        "expected_outage_loss_sgd_per_year",
        "cvar90_outage_loss_sgd_per_year",
        "expected_annual_cost_sgd_per_year",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "cvar90_UE_kwh_per_year",
        "energy_based_lolp",
        "energy_based_lolp_percent",
        "outage_weighted_load_kwh",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Figure 5 candidate summary source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {
        "city_key",
        "city_label",
        "case_key",
        "source_case_key",
        "candidate_id",
        "candidate_source",
    }
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Figure 5 candidate summary source contains non-numeric values.")
    if not validated["energy_based_lolp"].between(0.0, 1.0).all():
        raise ValueError("Figure 5 candidate summary energy_based_lolp must be between 0 and 1.")
    if (validated[["expected_annual_cost_sgd_per_year", "cvar90_outage_loss_sgd_per_year"]] < 0.0).any().any():
        raise ValueError("Figure 5 candidate summary costs must be non-negative.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(CITY_ORDER), ordered=True)
    validated = validated.sort_values(["city_key", "candidate_order"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    return validated


def _validate_cvar_design_curve_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "candidate_id",
        "candidate_order",
        "risk_aversion_lambda",
        "cvar_alpha",
        "cvar_objective_sgd_per_year",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "fixed_investment_cost_sgd_per_year",
        "expected_operation_cost_sgd_per_year",
        "expected_outage_loss_sgd_per_year",
        "cvar90_outage_loss_sgd_per_year",
        "expected_annual_cost_sgd_per_year",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "energy_based_lolp",
        "energy_based_lolp_percent",
        "is_selected_design",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Figure 5 CVaR design curve source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {
        "city_key",
        "city_label",
        "case_key",
        "source_case_key",
        "candidate_id",
        "candidate_source",
        "is_selected_design",
    }
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Figure 5 CVaR design curve source contains non-numeric values.")
    if not validated["energy_based_lolp"].between(0.0, 1.0).all():
        raise ValueError("Figure 5 CVaR design curve energy_based_lolp must be between 0 and 1.")
    lambda_values = sorted(round(float(value), 10) for value in CVAR_LAMBDAS)
    counts = validated.groupby("city_key")["risk_aversion_lambda"].nunique().astype(int).to_dict()
    for city in CITY_ORDER:
        if city in set(validated["city_key"].astype(str)) and counts.get(city, 0) != len(lambda_values):
            raise ValueError(f"Figure 5 requires {len(lambda_values)} lambda values for {city}.")
    selected_counts = validated.groupby(["city_key", "risk_aversion_lambda"]).size()
    if not selected_counts.empty and set(selected_counts.astype(int).tolist()) != {1}:
        raise ValueError("Figure 5 CVaR design curve must contain one selected design per city/lambda.")
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(CITY_ORDER), ordered=True)
    validated = validated.sort_values(["city_key", "risk_aversion_lambda"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    validated["is_selected_design"] = validated["is_selected_design"].astype(bool)
    return validated


def _format_numeric_tick(value: float) -> str:
    if np.isclose(value, round(value), atol=1e-9):
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_percent_tick(value: float) -> str:
    percent = 100.0 * float(value)
    if np.isclose(percent, round(percent), atol=1e-9):
        return f"{int(round(percent))}%"
    return f"{percent:.1f}%"
