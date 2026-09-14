from __future__ import annotations

import argparse

from ..io.config import load_config
from .paths import input_dir, output_dir
from .workflow import run_selected


DISPATCH_BACKEND_CHOICES = ["auto", "greedy", "milp"]


def _resolve_analysis_selection(args: argparse.Namespace) -> tuple[str, str, list[str] | None, list[str] | None]:
    analysis = args.analysis
    uncertainty_analysis = args.uncertainty_analysis
    sensitivity_parameters = args.sensitivity_parameter
    uncertainty_sensitivity_parameters = args.uncertainty_sensitivity_parameter

    if args.run_mode:
        if args.run_mode == "all":
            analysis = "all"
        elif args.run_mode == "baseline":
            analysis = "baseline"
        elif args.run_mode == "sensitivity":
            analysis = "sensitivity"
        elif args.run_mode == "uncertainty":
            analysis = "uncertainty"
            uncertainty_analysis = "baseline"
        elif args.run_mode == "uncertainty-sensitivity":
            analysis = "uncertainty"
            uncertainty_analysis = "sensitivity"
            if uncertainty_sensitivity_parameters is None:
                uncertainty_sensitivity_parameters = sensitivity_parameters

    if args.skip_sensitivity:
        analysis = "baseline"

    return analysis, uncertainty_analysis, sensitivity_parameters, uncertainty_sensitivity_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-driven PV-BESS resilience-informed optimal sizing.")
    parser.add_argument("--config", default="config.yaml", help="Optional YAML config override.")
    parser.add_argument("--input-dir", default=None, help="Input directory; default comes from config.")
    parser.add_argument("--output-dir", default=None, help="Output directory; default comes from config.")
    parser.add_argument(
        "--dispatch-backend",
        choices=DISPATCH_BACKEND_CHOICES,
        default=None,
        help="Dispatch solver backend. Defaults to config dispatch.backend.",
    )
    parser.add_argument(
        "--save-candidate-dispatch-workbooks",
        action="store_true",
        help="Write one XLSX workbook per evaluated candidate dispatch. Disabled by default because it is I/O intensive.",
    )
    parser.add_argument("--skip-sensitivity", action="store_true", help="Deprecated alias for --analysis baseline.")
    parser.add_argument("--city", action="append", default=None, help="City key to run; can be repeated.")
    parser.add_argument("--case", action="append", default=None, help="Case key to run, such as LA_UTown; can be repeated.")
    parser.add_argument(
        "--sensitivity-parameter",
        action="append",
        default=None,
        help=(
            "Ordinary sensitivity parameter to run; can be repeated. Defaults to battery_capex_sgd_per_kwh, "
            "pv_capex_sgd_per_kw, saidi_min_per_year, saifi_per_year, and voll_sgd_per_kwh for --analysis sensitivity."
        ),
    )
    parser.add_argument(
        "--uncertainty-sensitivity-parameter",
        action="append",
        default=None,
        help="Uncertainty sensitivity parameter to run; can be repeated. Defaults to all five for uncertainty sensitivity.",
    )
    parser.add_argument(
        "--analysis",
        choices=[
            "all",
            "baseline",
            "sensitivity",
            "uncertainty",
            "frontier",
            "representative-sensitivity",
            "result-figures",
            "city-comparison",
            "single-outage-city-baseline",
            "single-family-preliminary-result",
            "lolp-zero-uncertainty",
            "lolp-zero-three-year-uncertainty",
            "variable-uncertainty",
            "variable-uncertainty-figures",
            "non-sg-extrapolated-cost-distribution-figures",
        ],
        default="all",
        help="Top-level analysis selection. Default: all.",
    )
    parser.add_argument(
        "--uncertainty-analysis",
        choices=["all", "baseline", "sensitivity"],
        default="all",
        help="Sub-analysis to run when --analysis uncertainty is selected. Default: all.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["all", "baseline", "sensitivity", "uncertainty", "uncertainty-sensitivity"],
        default=None,
        help="Deprecated alias for --analysis / --uncertainty-analysis.",
    )
    parser.add_argument(
        "--figure",
        action="append",
        type=int,
        choices=[4, 5, 6, 7, 8, 9, 10, 11, 12],
        default=None,
        help="Result figure number to draw for --analysis result-figures; can be repeated.",
    )
    parser.add_argument(
        "--city-comparison-figure",
        action="append",
        type=str,
        choices=["1", "2", "3", "4", "5", "6", "7", "7b", "8", "9", "all"],
        default=None,
        help="City-comparison figure number to draw for --analysis city-comparison; can be repeated.",
    )
    parser.add_argument(
        "--single-outage-city-figure",
        action="append",
        type=str,
        choices=["1", "1b", "1c", "1d", "2", "2b", "2c", "2d", "3", "3b", "3c", "4", "5", "6a", "6b", "7", "all"],
        default=None,
        help="Single-outage four-city baseline figure number to draw; can be repeated.",
    )
    parser.add_argument(
        "--single-family-preliminary-figure",
        action="append",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default=None,
        help="Single Family preliminary QE-report figure number to draw; can be repeated.",
    )
    parser.add_argument(
        "--city-comparison-source",
        choices=["apartment-synthetic", "singlefamily-real"],
        default="apartment-synthetic",
        help="Source data mode for --analysis city-comparison.",
    )
    parser.add_argument(
        "--reuse-source-data",
        action="store_true",
        help="For --analysis result-figures, redraw figures from existing source-data CSV files.",
    )
    parser.add_argument(
        "--uncertainty-scenario-count",
        type=int,
        default=None,
        help="Override the random outage scenario count for --analysis variable-uncertainty.",
    )
    parser.add_argument(
        "--uncertainty-seed-start",
        type=int,
        default=None,
        help="Override the first random outage seed for --analysis variable-uncertainty.",
    )
    parser.add_argument(
        "--lolp-threshold",
        type=float,
        default=None,
        help="Optional energy-based LOLP threshold for --analysis variable-uncertainty. Omit for no LOLP constraint.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dispatch_backend is not None:
        config.setdefault("dispatch", {})["backend"] = args.dispatch_backend
    if args.save_candidate_dispatch_workbooks:
        config.setdefault("outputs", {})["save_candidate_dispatch_workbooks"] = True
    input_root = input_dir(config, args.input_dir)
    out = output_dir(config, args.output_dir)
    analysis, uncertainty_analysis, sensitivity_parameters, uncertainty_sensitivity_parameters = _resolve_analysis_selection(args)
    run_baseline = analysis in {"all", "baseline"}
    run_sensitivity = analysis in {"all", "sensitivity"}
    run_uncertainty_baseline = analysis == "all" or (
        analysis == "uncertainty" and uncertainty_analysis in {"all", "baseline"}
    )
    run_uncertainty_sensitivity = analysis == "all" or (
        analysis == "uncertainty" and uncertainty_analysis in {"all", "sensitivity"}
    )
    run_frontier = analysis == "frontier"
    run_representative_sensitivity = analysis == "representative-sensitivity"
    run_result_figures = analysis == "result-figures"
    run_city_comparison = analysis == "city-comparison"
    run_single_outage_city_baseline = analysis == "single-outage-city-baseline"
    run_single_family_preliminary_result = analysis == "single-family-preliminary-result"
    run_lolp_zero_uncertainty = analysis == "lolp-zero-uncertainty"
    run_lolp_zero_three_year_uncertainty = analysis == "lolp-zero-three-year-uncertainty"
    run_variable_uncertainty = analysis == "variable-uncertainty"
    run_variable_uncertainty_figures = analysis == "variable-uncertainty-figures"
    run_non_sg_extrapolated_cost_distribution_figures = analysis == "non-sg-extrapolated-cost-distribution-figures"
    paths = run_selected(
        config,
        input_root,
        out,
        run_baseline=run_baseline,
        run_sensitivity=run_sensitivity,
        run_uncertainty_baseline=run_uncertainty_baseline,
        run_uncertainty_sensitivity=run_uncertainty_sensitivity,
        run_lolp_zero_uncertainty=run_lolp_zero_uncertainty,
        run_lolp_zero_three_year_uncertainty=run_lolp_zero_three_year_uncertainty,
        run_variable_uncertainty=run_variable_uncertainty,
        run_variable_uncertainty_figures=run_variable_uncertainty_figures,
        run_non_sg_extrapolated_cost_distribution_figures=run_non_sg_extrapolated_cost_distribution_figures,
        run_frontier=run_frontier,
        run_representative_sensitivity=run_representative_sensitivity,
        run_result_figures=run_result_figures,
        run_city_comparison=run_city_comparison,
        run_single_outage_city_baseline=run_single_outage_city_baseline,
        run_single_family_preliminary_result=run_single_family_preliminary_result,
        city_filters=args.city,
        case_filters=args.case,
        sensitivity_parameters=sensitivity_parameters,
        uncertainty_sensitivity_parameters=uncertainty_sensitivity_parameters,
        result_figures=args.figure,
        city_comparison_figures=args.city_comparison_figure,
        single_outage_city_figures=args.single_outage_city_figure,
        single_family_preliminary_figures=args.single_family_preliminary_figure,
        city_comparison_source=args.city_comparison_source,
        reuse_source_data=args.reuse_source_data,
        uncertainty_scenario_count=args.uncertainty_scenario_count,
        uncertainty_seed_start=args.uncertainty_seed_start,
        lolp_threshold=args.lolp_threshold,
    )
    print("Input-driven PV-BESS resilience sizing completed.")
    print(f"Input directory: {input_root}")
    for name, path in paths.items():
        print(f"{name}: {path}")
