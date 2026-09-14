from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.figures.non_sg_extrapolated_cost_distribution import (
    CITY_ORDER,
    CDF_FIGURE_NAME,
    FIT_SUMMARY_NAME,
    HISTOGRAM_FIGURE_NAME,
    SOURCE_DATA_NAME,
    build_non_sg_extrapolated_cost_distribution_source,
    build_normal_fit_summary,
    make_non_sg_extrapolated_cost_distribution_figures,
)


def _write_case_inputs(input_root: Path, case_key: str, city_name: str, *, saidi: float, saifi: float) -> None:
    case_dir = input_root / case_key
    case_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows = [
        ("pv_capex_sgd_per_kw", 900.0),
        ("battery_capex_sgd_per_kwh", 300.0),
        ("battery_power_capex_sgd_per_kw", 150.0),
        ("saidi_min_per_year", saidi),
        ("saifi_per_year", saifi),
        ("voll_sgd_per_kwh", 10.0),
        ("city_name", city_name),
        ("building_name", "SingleFamily"),
        ("cooling_cop", 5.0),
        ("heating_cop", 3.5),
    ]
    pd.DataFrame(
        [
            {
                "parameter": name,
                "base_value": value,
                "unit": "-",
                "sensitivity_start": "",
                "sensitivity_end": "",
                "sensitivity_step": "",
                "description": "",
                "中文说明": "",
            }
            for name, value in parameter_rows
        ]
    ).to_csv(case_dir / f"{case_key}_Parameter.csv", index=False)
    pd.DataFrame(
        {
            "Time": [f"01/01 {hour:02d}:00:00" for hour in range(1, 25)],
            "Cooling load (kWh_th)": np.linspace(0.8, 1.2, 24),
            "Heating load (kWh_th)": np.zeros(24),
            "Lighting load (kWh_e)": np.full(24, 0.1),
            "Equipment load (kWh_e)": np.full(24, 0.5),
        }
    ).to_csv(case_dir / f"{case_key}_Load.csv", index=False)
    pd.DataFrame({"PV_output_kWh_per_m2": np.linspace(0.0, 0.8, 24)}).to_csv(
        case_dir / f"{case_key}_PV.csv",
        index=False,
    )
    pd.DataFrame({"hour": list(range(24)), "tou_sgd_per_kwh": np.linspace(0.08, 0.16, 24)}).to_csv(
        case_dir / f"{case_key}_TOU.csv",
        index=False,
    )


def _write_existing_result(output_root: Path, city: str, case_key: str, *, include_lolp: bool) -> None:
    if city == "Jakarta":
        path = (
            output_root
            / city
            / f"{case_key}_Uncertainty_Analysis"
            / f"{case_key}_Variable_Uncertainty"
            / f"{case_key}_Variable_Uncertainty_Scenario_Optima.csv"
        )
    else:
        path = (
            output_root
            / city
            / f"{case_key}_Uncertainty_Analysis"
            / f"{case_key}_Uncertainty_Baseline"
            / f"{case_key}_Uncertainty_Baseline_All_Results.csv"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_id in range(1, 13):
        lolp = scenario_id / 20.0
        rows.append(
            {
                "uncertainty_scenario_id": scenario_id,
                "pv_bess_system_annual_cost_sgd": 1000.0 + scenario_id * 12.0 + len(city),
                "unserved_energy_kwh_per_year": lolp * 10.0,
                "outage_weighted_load_kwh": 10.0,
                **({"scenario_LOLP": lolp} if include_lolp else {}),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _fixture(root: Path) -> tuple[Path, Path]:
    input_root = root / "Input"
    output_root = root / "outputs"
    _write_case_inputs(input_root, "Bangkok_SingleFamily", "Bangkok", saidi=90.0, saifi=1.0)
    _write_case_inputs(input_root, "Brisbane_SingleFamily", "Brisbane", saidi=90.0, saifi=2.0)
    _write_case_inputs(input_root, "Jakarta_SingleFamily", "Jakarta", saidi=180.0, saifi=2.0)
    _write_case_inputs(input_root, "Manila_SingleFamily", "Manila", saidi=120.0, saifi=2.0)
    _write_existing_result(output_root, "Brisbane", "Brisbane_SingleFamily", include_lolp=False)
    _write_existing_result(output_root, "Jakarta", "Jakarta_SingleFamily", include_lolp=True)
    _write_existing_result(output_root, "Manila", "Manila_SingleFamily", include_lolp=False)
    return input_root, output_root


class NonSgExtrapolatedCostDistributionTests(unittest.TestCase):
    def test_source_data_contains_four_non_sg_cities_and_lolp_tail_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            input_root, output_root = _fixture(Path(tmp))
            source = build_non_sg_extrapolated_cost_distribution_source(input_root, output_root)

            self.assertEqual(set(source["city_key"]), set(CITY_ORDER))
            self.assertNotIn("Singapore", set(source["city_key"]))
            self.assertEqual(
                set(source.loc[source["city_key"] == "Bangkok", "data_status"]),
                {"extrapolated_proxy"},
            )
            for city in ("Brisbane", "Jakarta", "Manila"):
                self.assertEqual(set(source.loc[source["city_key"] == city, "data_status"]), {"existing_result"})
            self.assertTrue(
                (
                    source["tail_lolp_gt_0p1"].astype(bool)
                    == (source["scenario_LOLP"].astype(float) > 0.1)
                ).all()
            )
            self.assertTrue(pd.to_numeric(source["annual_system_cost_sgd_per_year"], errors="coerce").notna().all())
            self.assertTrue(source["scenario_LOLP"].astype(float).between(0.0, 1.0).all())

    def test_normal_fit_summary_only_fits_sufficient_tail_samples(self) -> None:
        with TemporaryDirectory() as tmp:
            input_root, output_root = _fixture(Path(tmp))
            source = build_non_sg_extrapolated_cost_distribution_source(input_root, output_root)
            summary = build_normal_fit_summary(source)

            self.assertEqual(set(summary["city_key"]), set(CITY_ORDER))
            self.assertTrue((summary["n_tail"].astype(int) >= 8).all())
            self.assertTrue(summary["fit_available"].astype(bool).all())
            self.assertTrue((summary["std_cost_sgd_per_year"].astype(float) > 0.0).all())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required for smooth CDF rendering")
    def test_make_figures_exports_histogram_cdf_and_source_files(self) -> None:
        with TemporaryDirectory() as tmp:
            input_root, output_root = _fixture(Path(tmp))
            out = output_root / "City_Comparison" / "SingleFamily_Extrapolated_Cost_Distribution"
            paths = make_non_sg_extrapolated_cost_distribution_figures(out, input_root, output_root)

            expected = [
                out / SOURCE_DATA_NAME,
                out / FIT_SUMMARY_NAME,
                out / f"{HISTOGRAM_FIGURE_NAME}.png",
                out / f"{HISTOGRAM_FIGURE_NAME}.svg",
                out / f"{HISTOGRAM_FIGURE_NAME}.pdf",
                out / f"{CDF_FIGURE_NAME}.png",
                out / f"{CDF_FIGURE_NAME}.svg",
                out / f"{CDF_FIGURE_NAME}.pdf",
            ]
            for path in expected:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)
            self.assertGreaterEqual(len(paths), len(expected))


if __name__ == "__main__":
    unittest.main()
