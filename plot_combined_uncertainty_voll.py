from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization import (
    DESIGN_CASE_ORDER,
    DESIGN_FILE_LABELS,
    DESIGN_LABELS,
    _matplotlib,
    _save_all,
    _style,
    _write_source_data,
    normalize_design_case_names,
)


DEFAULT_CASE = "SG_UTown"
DEFAULT_PARAMETER = "voll_sgd_per_kwh"

PARAMETER_FILE_LABELS = {
    "voll_sgd_per_kwh": "VoLL",
}
PARAMETER_AXIS_LABELS = {
    "voll_sgd_per_kwh": "VoLL (SGD/kWh)",
}

METRICS = [
    (
        "Optimal_Annualized_Cost_vs_{parameter_label}",
        "Optimal annualized system cost (million SGD/year)",
        "annualized_system_cost_million_sgd_per_year",
    ),
    (
        "Optimal_Battery_Capacity_vs_{parameter_label}",
        "Optimal battery energy capacity (MWh)",
        "battery_energy_capacity_mwh",
    ),
    (
        "Optimal_PV_Capacity_vs_{parameter_label}",
        "Optimal PV capacity (MWp)",
        "pv_capacity_mwp",
    ),
]

REQUIRED_COLUMNS = {
    "sensitivity_parameter",
    "sensitivity_value",
    "design_case",
    "design_case_label",
    "count",
    "median",
    "p10",
    "p90",
}

COLORS = {
    "flexibility_only": "#6b7280",
    "resilience_informed": "#0f766e",
}


def _case_city(case_key: str) -> str:
    return case_key.split("_", 1)[0]


def _parameter_label(parameter: str) -> str:
    return PARAMETER_FILE_LABELS.get(parameter, parameter)


def _parameter_axis_label(parameter: str) -> str:
    return PARAMETER_AXIS_LABELS.get(parameter, parameter)


def _uncertainty_parameter_dir(output_root: Path, case_key: str, parameter: str) -> Path:
    city_key = _case_city(case_key)
    return (
        output_root
        / city_key
        / f"{case_key}_Uncertainty_Analysis"
        / f"{case_key}_Uncertainty_Sensitive_Analysis"
        / f"{case_key}_Uncertainty_Sensitive_Analysis_{parameter}"
    )


def _read_design_source(path: Path, expected_design_case: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source-data CSV: {path}")
    frame = normalize_design_case_names(pd.read_csv(path))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    design_cases = set(frame["design_case"].astype(str))
    if design_cases != {expected_design_case}:
        raise ValueError(
            f"{path} should contain only {expected_design_case!r}, "
            f"but contains {sorted(design_cases)}"
        )
    return frame.sort_values("sensitivity_value").copy()


def _validate_matching_x_values(sources: dict[str, pd.DataFrame]) -> None:
    reference_case = DESIGN_CASE_ORDER[0]
    reference = sources[reference_case]["sensitivity_value"].astype(float).to_numpy()
    for design_case in DESIGN_CASE_ORDER[1:]:
        values = sources[design_case]["sensitivity_value"].astype(float).to_numpy()
        if len(reference) != len(values) or not np.allclose(reference, values, rtol=0.0, atol=1e-9):
            raise ValueError(
                "The two design cases do not share identical sensitivity_value points; "
                f"cannot merge {reference_case} with {design_case}."
            )


def _combined_source(sources: dict[str, pd.DataFrame], metric_name: str) -> pd.DataFrame:
    rows = []
    for design_case in DESIGN_CASE_ORDER:
        part = sources[design_case].copy()
        part["metric"] = metric_name
        rows.append(part)
    return pd.concat(rows, ignore_index=True).sort_values(["sensitivity_value", "design_case"])


def _plot_combined_band(
    out_dir: Path,
    base_name: str,
    x_label: str,
    y_label: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out_dir)
    _style(mpl)
    base = out_dir / f"{base_name}_Combined"
    _write_source_data(base, source)

    fig, ax = plt.subplots(figsize=(118 / 25.4, 78 / 25.4), constrained_layout=True)
    for design_case in DESIGN_CASE_ORDER:
        part = source[source["design_case"] == design_case].sort_values("sensitivity_value")
        if part.empty:
            continue
        x = part["sensitivity_value"].astype(float).to_numpy()
        median = part["median"].astype(float).to_numpy()
        p10 = part["p10"].astype(float).to_numpy()
        p90 = part["p90"].astype(float).to_numpy()
        ax.fill_between(x, p10, p90, color=COLORS[design_case], alpha=0.16, linewidth=0)
        ax.plot(
            x,
            median,
            marker="o",
            markersize=3,
            linewidth=1.15,
            color=COLORS[design_case],
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


def make_combined_voll_figures(output_root: Path, case_key: str, parameter: str) -> dict[str, Path]:
    out_dir = _uncertainty_parameter_dir(output_root, case_key, parameter)
    parameter_label = _parameter_label(parameter)
    x_label = _parameter_axis_label(parameter)
    prefix = f"{case_key}_Uncertainty_Sensitive_Analysis_{parameter}"
    paths: dict[str, Path] = {}

    for metric_template, y_label, metric_name in METRICS:
        metric_file_label = metric_template.format(parameter_label=parameter_label)
        sources: dict[str, pd.DataFrame] = {}
        for design_case in DESIGN_CASE_ORDER:
            source_path = out_dir / f"{prefix}_{metric_file_label}_{DESIGN_FILE_LABELS[design_case]}_Source_Data.csv"
            sources[design_case] = _read_design_source(source_path, design_case)
        _validate_matching_x_values(sources)
        source = _combined_source(sources, metric_name)
        base_name = f"{prefix}_{metric_file_label}"
        paths.update(_plot_combined_band(out_dir, base_name, x_label, y_label, source))

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine existing VoLL uncertainty sensitivity bands into one figure per metric."
    )
    parser.add_argument("--case", default=DEFAULT_CASE, help="Case key, for example SG_UTown.")
    parser.add_argument("--parameter", default=DEFAULT_PARAMETER, help="Sensitivity parameter to combine.")
    parser.add_argument("--output-root", default="outputs", help="Existing output root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = make_combined_voll_figures(Path(args.output_root), args.case, args.parameter)
    print("Combined uncertainty sensitivity figures completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
