from __future__ import annotations

import argparse
import os

from ..io.config import load_config
from ..sensitivity import SENSITIVITY_MODES, run_sensitivity
from .paths import input_dir, output_dir
from .workflow import run_selected


DISPATCH_BACKEND_CHOICES = ["auto", "greedy", "milp"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Core PV-BESS resilience sizing engine.")
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
        help="Write one XLSX workbook per evaluated candidate dispatch. Disabled by default.",
    )
    parser.add_argument("--city", action="append", default=None, help="City key to run; can be repeated.")
    parser.add_argument("--case", action="append", default=None, help="Case key to run; can be repeated.")
    parser.add_argument(
        "--sensitivity-parameter",
        action="append",
        default=None,
        metavar="PARAM",
        help="Numeric system parameter to vary; can be repeated. Use 'all' for every parameter with a configured range.",
    )
    parser.add_argument(
        "--sensitivity-mode",
        choices=SENSITIVITY_MODES,
        default="oat",
        help="Sensitivity experiment design: oat (default) or factorial.",
    )
    parser.add_argument(
        "--sensitivity-analysis-years",
        type=int,
        default=None,
        metavar="N",
        help="Override analysis_years in each selected case's Parameter.csv.",
    )
    parser.add_argument(
        "--sensitivity-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        metavar="N",
        help="Parallel workers for independent sensitivity chains.",
    )
    parser.add_argument(
        "--allow-large-sensitivity",
        action="store_true",
        help="Allow a factorial plan with more than 500 unique experiment points.",
    )
    parser.add_argument(
        "--disable-sensitivity-warm-start",
        action="store_true",
        help="Disable warm-start between consecutive points in each factorial sensitivity chain.",
    )
    parser.add_argument(
        "--tou-analysis",
        action="store_true",
        help="Run mean-preserving TOU spread analysis with the default k values.",
    )
    parser.add_argument(
        "--tou-spread-k",
        action="append",
        type=float,
        default=None,
        help="TOU spread k value to run; can be repeated. Providing this enables TOU analysis.",
    )
    args = parser.parse_args()

    default_workers = min(8, os.cpu_count() or 1)
    sensitivity_requested = bool(args.sensitivity_parameter)
    sensitivity_options_used = (
        args.sensitivity_analysis_years is not None
        or args.sensitivity_workers != default_workers
        or args.allow_large_sensitivity
        or args.disable_sensitivity_warm_start
        or args.sensitivity_mode != "oat"
    )
    if args.sensitivity_workers <= 0:
        parser.error("--sensitivity-workers must be a positive integer.")
    if args.sensitivity_analysis_years is not None and args.sensitivity_analysis_years <= 0:
        parser.error("--sensitivity-analysis-years must be a positive integer.")
    if sensitivity_options_used and not sensitivity_requested:
        parser.error("Sensitivity options require at least one --sensitivity-parameter.")
    if args.disable_sensitivity_warm_start and args.sensitivity_mode != "factorial":
        parser.error("--disable-sensitivity-warm-start requires --sensitivity-mode factorial.")
    if sensitivity_requested and (args.tou_analysis or args.tou_spread_k is not None):
        parser.error("Sensitivity analysis cannot be combined with TOU analysis in one command.")

    config = load_config(args.config)
    if args.dispatch_backend is not None:
        config.setdefault("dispatch", {})["backend"] = args.dispatch_backend
    if args.save_candidate_dispatch_workbooks:
        config.setdefault("outputs", {})["save_candidate_dispatch_workbooks"] = True

    input_root = input_dir(config, args.input_dir)
    out = output_dir(config, args.output_dir)
    if sensitivity_requested:
        paths = run_sensitivity(
            config=config,
            input_dir=input_root,
            output_dir=out,
            requested_parameters=args.sensitivity_parameter,
            mode=args.sensitivity_mode,
            analysis_years=args.sensitivity_analysis_years,
            workers=args.sensitivity_workers,
            allow_large=args.allow_large_sensitivity,
            warm_start_enabled=not args.disable_sensitivity_warm_start,
            city_filters=args.city,
            case_filters=args.case,
        )
        print("Unified PV-BESS sensitivity analysis completed.")
    else:
        paths = run_selected(
            config,
            input_root,
            out,
            city_filters=args.city,
            case_filters=args.case,
            tou_analysis=args.tou_analysis,
            tou_spread_k_values=args.tou_spread_k,
        )
        print("Core PV-BESS resilience sizing completed.")

    print(f"Input directory: {input_root}")
    print(f"Output directory: {out}")
    for name, path in paths.items():
        print(f"{name}: {path}")
