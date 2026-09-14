from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.lolp_zero_multiyear_synthetic import (
    SYNTHETIC_METRIC_SPECS,
    SYNTHETIC_MULTYEAR_CASE_KEY,
    SYNTHETIC_MULTYEAR_CV_TREND_YEARS,
    SYNTHETIC_MULTYEAR_SCENARIO_COUNT,
    SYNTHETIC_MULTYEAR_SIMULATION_YEARS,
    build_lolp_zero_multiyear_cv_trend,
    build_lolp_zero_multiyear_synthetic_results,
    make_lolp_zero_multiyear_cv_trend_figures,
    make_lolp_zero_multiyear_synthetic_figures,
    summarize_lolp_zero_multiyear_synthetic_results,
    write_lolp_zero_multiyear_synthetic_outputs,
)


class LOLPZeroMultiyearSyntheticTests(unittest.TestCase):
    def test_synthetic_results_have_expected_rows_and_groups(self) -> None:
        results = build_lolp_zero_multiyear_synthetic_results()

        self.assertEqual(len(results), SYNTHETIC_MULTYEAR_SCENARIO_COUNT * len(SYNTHETIC_MULTYEAR_SIMULATION_YEARS))
        self.assertEqual(
            results.groupby("simulation_years").size().to_dict(),
            {year: SYNTHETIC_MULTYEAR_SCENARIO_COUNT for year in SYNTHETIC_MULTYEAR_SIMULATION_YEARS},
        )
        for column in ("system_total_cost_ksgd_per_year", "pv_capacity_kw", "battery_energy_kwh"):
            self.assertTrue((results[column].astype(float) >= 0.0).all())
        self.assertEqual(results[results["simulation_years"].eq(1)]["simulation_period_label"].unique().tolist(), ["1 year"])
        self.assertEqual(results[results["simulation_years"].eq(10)]["simulation_period_label"].unique().tolist(), ["10 years"])

    def test_cv_summary_is_grouped_by_year_and_metric(self) -> None:
        results = build_lolp_zero_multiyear_synthetic_results()
        summary = summarize_lolp_zero_multiyear_synthetic_results(results)

        self.assertEqual(len(summary), len(SYNTHETIC_MULTYEAR_SIMULATION_YEARS) * len(SYNTHETIC_METRIC_SPECS))
        self.assertEqual(set(summary["simulation_years"].astype(int)), set(SYNTHETIC_MULTYEAR_SIMULATION_YEARS))
        self.assertEqual(set(summary["metric"]), {spec.metric_key for spec in SYNTHETIC_METRIC_SPECS})
        self.assertTrue((summary["count"].astype(int) == SYNTHETIC_MULTYEAR_SCENARIO_COUNT).all())
        self.assertTrue((summary["coefficient_of_variation"].astype(float) >= 0.0).all())

    def test_cv_trend_interpolates_between_existing_anchor_years(self) -> None:
        results = build_lolp_zero_multiyear_synthetic_results()
        summary = summarize_lolp_zero_multiyear_synthetic_results(results)
        trend = build_lolp_zero_multiyear_cv_trend(summary)

        self.assertEqual(len(trend), len(SYNTHETIC_MULTYEAR_CV_TREND_YEARS) * len(SYNTHETIC_METRIC_SPECS))
        self.assertEqual(set(trend["simulation_years"].astype(int)), set(SYNTHETIC_MULTYEAR_CV_TREND_YEARS))
        self.assertEqual(set(trend["metric"]), {spec.metric_key for spec in SYNTHETIC_METRIC_SPECS})
        self.assertTrue((trend["coefficient_of_variation_percent"].astype(float) >= 0.0).all())
        for spec in SYNTHETIC_METRIC_SPECS:
            metric_trend = trend[trend["metric"].eq(spec.metric_key)].sort_values("simulation_years")
            self.assertTrue(metric_trend["coefficient_of_variation_percent"].astype(float).is_monotonic_decreasing)
            for year in SYNTHETIC_MULTYEAR_SIMULATION_YEARS:
                expected = summary[
                    summary["metric"].eq(spec.metric_key) & summary["simulation_years"].astype(int).eq(int(year))
                ]["coefficient_of_variation_percent"].iloc[0]
                actual = metric_trend[metric_trend["simulation_years"].astype(int).eq(int(year))][
                    "coefficient_of_variation_percent"
                ].iloc[0]
                self.assertAlmostEqual(float(actual), float(expected), places=12)
                self.assertEqual(
                    metric_trend[metric_trend["simulation_years"].astype(int).eq(int(year))]["source_role"].iloc[0],
                    "anchor",
                )
        self.assertEqual(
            set(trend[~trend["simulation_years"].astype(int).isin(SYNTHETIC_MULTYEAR_SIMULATION_YEARS)]["source_role"]),
            {"interpolated"},
        )

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_synthetic_distribution_grid_figures_export(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            results = build_lolp_zero_multiyear_synthetic_results()
            paths = make_lolp_zero_multiyear_synthetic_figures(out, SYNTHETIC_MULTYEAR_CASE_KEY, results)

            expected = [
                "Manila_SingleFamily_Annualized_Cost_Synthetic_Multiyear_Distribution_Grid",
                "Manila_SingleFamily_PV_Capacity_Synthetic_Multiyear_Distribution_Grid",
                "Manila_SingleFamily_Battery_Capacity_Synthetic_Multiyear_Distribution_Grid",
            ]
            for stem in expected:
                for suffix in ("png", "svg", "pdf"):
                    self.assertTrue((out / f"{stem}.{suffix}").exists())
                self.assertTrue((out / f"{stem}_Source_Data.csv").exists())
            svg = (out / f"{expected[0]}.svg").read_text(encoding="utf-8")
            for text in ("1 year", "3 years", "5 years", "10 years", "CV=", "n=500", "Probability density"):
                self.assertIn(text, svg)
            self.assertIn("annualized_cost_source_data", paths)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_synthetic_cv_trend_figures_export(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            results = build_lolp_zero_multiyear_synthetic_results()
            summary = summarize_lolp_zero_multiyear_synthetic_results(results)
            trend = build_lolp_zero_multiyear_cv_trend(summary)
            paths = make_lolp_zero_multiyear_cv_trend_figures(out, SYNTHETIC_MULTYEAR_CASE_KEY, trend)

            expected = [
                "Manila_SingleFamily_Annualized_Cost_Synthetic_Multiyear_CV_Trend",
                "Manila_SingleFamily_PV_Capacity_Synthetic_Multiyear_CV_Trend",
                "Manila_SingleFamily_Battery_Capacity_Synthetic_Multiyear_CV_Trend",
            ]
            for stem in expected:
                for suffix in ("png", "svg", "pdf"):
                    self.assertTrue((out / f"{stem}.{suffix}").exists())
                self.assertTrue((out / f"{stem}_Source_Data.csv").exists())
            svg = (out / f"{expected[0]}.svg").read_text(encoding="utf-8")
            for text in ("Manila", "Coefficient of variation", "1", "10", "Annualized cost"):
                self.assertIn(text, svg)
            self.assertIn("annualized_cost_cv_trend_source_data", paths)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_synthetic_output_writer_creates_csvs_and_figures(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = write_lolp_zero_multiyear_synthetic_outputs(out)

            all_results = pd.read_csv(out / "Manila_SingleFamily_LOLP_Zero_Multiyear_Synthetic_500_All_Results.csv")
            cv_summary = pd.read_csv(out / "Manila_SingleFamily_LOLP_Zero_Multiyear_Synthetic_500_CV_Summary.csv")
            cv_trend = pd.read_csv(out / "Manila_SingleFamily_LOLP_Zero_Multiyear_Synthetic_500_CV_Trend_All_Years.csv")
            self.assertEqual(len(all_results), 2000)
            self.assertEqual(len(cv_summary), 12)
            self.assertEqual(len(cv_trend), 30)
            self.assertTrue((out / "Manila_SingleFamily_Battery_Capacity_Synthetic_Multiyear_Distribution_Grid.png").exists())
            self.assertTrue((out / "Manila_SingleFamily_Battery_Capacity_Synthetic_Multiyear_CV_Trend.png").exists())
            self.assertIn("synthetic_all_results", paths)
            self.assertIn("synthetic_cv_trend", paths)


if __name__ == "__main__":
    unittest.main()
