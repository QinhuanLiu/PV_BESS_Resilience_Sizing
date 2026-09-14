from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.city_comparison import (
    APARTMENT_CASE_KEY,
    DATA_STATUS,
    FIGURE_7B_NAME,
    FIGURE_NAMES,
    REAL_SINGLEFAMILY_DATA_STATUS,
    SINGLEFAMILY_CASE_KEY,
    build_apartment_city_comparison_sources,
    build_singlefamily_city_comparison_sources,
    make_city_comparison_figures,
)
from src.io.config import load_config
from src.io.data_loader import discover_input_cases, load_input_case


INPUT_ROOT = Path("Input")


class ApartmentCityComparisonTests(unittest.TestCase):
    def _fast_singlefamily_config(self) -> dict:
        config = load_config("config.yaml")
        config.setdefault("dispatch", {})["backend"] = "greedy"
        config.setdefault("capacity_search", {})["coarse_intervals"] = 1
        config.setdefault("capacity_search", {})["local_refine_divisions"] = 1
        config.setdefault("capacity_search", {})["max_upper_expansions"] = 0
        return config

    def _minimal_singlefamily_sources(self) -> dict[str, pd.DataFrame]:
        return build_singlefamily_city_comparison_sources(
            INPUT_ROOT,
            config=self._fast_singlefamily_config(),
            scenario_count=1,
            parameter_overrides={"parallel_workers": 1},
            sensitivity_voll_values=[0.0, 150.0],
            sensitivity_battery_capex_values=[600.0],
            sensitivity_saidi_multipliers=[1.0],
        )

    def test_city_comparison_uses_apartment_scale(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=123)
        city_inputs = sources["city_inputs"]
        designs = sources["design_results"]
        scenario_designs = sources["scenario_design_results"]

        singapore = city_inputs[city_inputs["city_key"].astype(str) == "Singapore"].iloc[0]
        self.assertGreater(float(singapore["annual_load_mwh"]), 240.0)
        self.assertLess(float(singapore["annual_load_mwh"]), 265.0)
        self.assertLess(float(designs["pv_capacity_kw"].max()), 250.0)
        self.assertLess(float(designs["battery_energy_kwh"].max()), 250.0)
        self.assertLess(float(scenario_designs["pv_capacity_kw"].max()), 250.0)
        self.assertLess(float(scenario_designs["battery_energy_kwh"].max()), 250.0)
        self.assertTrue((city_inputs["data_status"].astype(str) == DATA_STATUS).all())
        self.assertTrue((designs["data_status"].astype(str) == DATA_STATUS).all())
        self.assertTrue((scenario_designs["data_status"].astype(str) == DATA_STATUS).all())

    def test_city_sources_are_deterministic_for_fixed_seed(self) -> None:
        first = build_apartment_city_comparison_sources(INPUT_ROOT, seed=77)
        second = build_apartment_city_comparison_sources(INPUT_ROOT, seed=77)

        for key in ("city_inputs", "outage_scenarios", "outage_events", "design_results", "cost_breakdown", "scenario_design_results", "city_operation_profiles"):
            pd.testing.assert_frame_equal(first[key].reset_index(drop=True), second[key].reset_index(drop=True))

    def test_scenario_design_results_are_four_city_by_one_hundred(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=88)
        scenario_designs = sources["scenario_design_results"]

        self.assertEqual(len(scenario_designs), 400)
        counts = scenario_designs.groupby("city_key").size()
        self.assertEqual(set(counts.astype(int).tolist()), {100})
        self.assertTrue(scenario_designs["energy_based_lolp"].between(0.0, 1.0).all())
        self.assertTrue(scenario_designs["plot_jitter"].between(-0.16, 0.16).all())
        self.assertTrue((scenario_designs["data_status"].astype(str) == DATA_STATUS).all())

    def test_city_operation_profiles_cover_four_cities(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=88)
        operation = sources["city_operation_profiles"]

        self.assertEqual(len(operation), 4 * 72)
        counts = operation.groupby("city_key").size()
        self.assertEqual(set(counts.astype(int).tolist()), {72})
        self.assertEqual(set(operation.groupby("city_key")["window_hour"].min().astype(int).tolist()), {0})
        self.assertEqual(set(operation.groupby("city_key")["window_hour"].max().astype(int).tolist()), {71})
        durations = operation.groupby("city_key")["representative_outage_duration_hours"].first()
        self.assertGreater(float(durations["Metro_Manila"]), float(durations["Singapore"]))
        self.assertGreater(float(durations["Brisbane"]), float(durations["Singapore"]))
        city_inputs = sources["city_inputs"].set_index("city_key")
        for city_key, duration in durations.items():
            expected_duration = (float(city_inputs.loc[city_key, "saidi_min_per_year"]) / 60.0) / float(city_inputs.loc[city_key, "saifi_per_year"])
            self.assertAlmostEqual(float(duration), expected_duration, places=9)
        self.assertLess(float(durations.max()), 2.0)
        for column in ("tou_sgd_per_kwh", "tou_period", "pv_to_load_kw", "pv_to_battery_kw", "grid_to_load_kw", "grid_to_battery_kw", "reserve_floor_fraction"):
            self.assertIn(column, operation.columns)
        self.assertTrue(operation["battery_soc_fraction"].between(0.10, 0.92).all())
        outage_rows = operation[operation["grid_available"].astype(int) == 0]
        self.assertTrue((outage_rows["grid_import_kw"].astype(float) == 0.0).all())
        normal_rows = operation[operation["grid_available"].astype(int) == 1]
        normal_summary = normal_rows.groupby("city_key").agg(
            normal_charge=("battery_charge_kw", "sum"),
            normal_discharge=("battery_discharge_kw", "sum"),
        )
        self.assertTrue((normal_summary["normal_charge"].astype(float) > 0.0).all())
        self.assertTrue((normal_summary["normal_discharge"].astype(float) > 0.0).all())
        ue_by_city = operation.groupby("city_key")["unserved_kw"].sum()
        self.assertTrue((ue_by_city.astype(float) >= 0.0).all())
        self.assertTrue((operation["data_status"].astype(str) == DATA_STATUS).all())

    def test_outage_scenario_means_match_input_indices(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=99)
        scenarios = sources["outage_scenarios"]
        summary = scenarios.groupby("city_key", as_index=False).agg(
            input_saidi=("input_saidi_min_per_year", "mean"),
            simulated_saidi=("simulated_saidi_min_per_year", "mean"),
            input_saifi=("input_saifi_per_year", "mean"),
            simulated_saifi=("simulated_saifi_per_year", "mean"),
        )

        for row in summary.to_dict("records"):
            self.assertAlmostEqual(float(row["input_saidi"]), float(row["simulated_saidi"]), places=9)
            self.assertAlmostEqual(float(row["input_saifi"]), float(row["simulated_saifi"]), places=9)

    def test_cost_breakdown_sources_match_optimal_design_totals(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=101)
        cost_breakdown = sources["cost_breakdown"]
        designs = sources["design_results"]
        aware = designs[designs["design_case"].astype(str) == "resilience_aware"].set_index("city_key")

        self.assertEqual(len(cost_breakdown), 4 * 10)
        self.assertEqual(set(cost_breakdown.groupby(["city_key", "cost_basis"]).size().astype(int).tolist()), {3, 7})
        self.assertTrue((cost_breakdown["data_status"].astype(str) == DATA_STATUS).all())
        annualized = cost_breakdown[cost_breakdown["cost_basis"].astype(str) == "annualized"]
        upfront = cost_breakdown[cost_breakdown["cost_basis"].astype(str) == "upfront"]
        for city_key, group in annualized.groupby("city_key"):
            self.assertAlmostEqual(float(group["cost_sgd"].sum()), float(aware.loc[city_key, "annualized_total_cost_sgd_per_year"]), places=6)
        for city_key, group in upfront.groupby("city_key"):
            self.assertAlmostEqual(float(group["cost_sgd"].sum()), float(aware.loc[city_key, "total_initial_investment_sgd"]), places=6)
        self.assertIn("electricity_bill", set(cost_breakdown["component_key"].astype(str)))
        self.assertIn("outage_loss", set(cost_breakdown["component_key"].astype(str)))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_all_city_comparison_figures_export(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=42)
        with TemporaryDirectory() as tmp:
            paths = make_city_comparison_figures(Path(tmp), sources=sources)

            for figure_no, figure_name in FIGURE_NAMES.items():
                base = f"{APARTMENT_CASE_KEY}_{figure_name}"
                self.assertIn(f"{base}_png", paths)
                self.assertIn(f"{base}_svg", paths)
                self.assertIn(f"{base}_pdf", paths)
                self.assertIn(f"figure_{figure_no}_source_data", paths)
                self.assertTrue(paths[f"{base}_png"].exists())
                self.assertTrue(paths[f"{base}_svg"].exists())
                self.assertTrue(paths[f"{base}_pdf"].exists())
                self.assertTrue(paths[f"figure_{figure_no}_source_data"].exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_city_comparison_figure_7b_exports_without_lolp(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=42)
        with TemporaryDirectory() as tmp:
            paths = make_city_comparison_figures(Path(tmp), sources=sources, figures=["7b"])
            base = f"{APARTMENT_CASE_KEY}_{FIGURE_7B_NAME}"

            self.assertIn(f"{base}_png", paths)
            self.assertIn(f"{base}_svg", paths)
            self.assertIn(f"{base}_pdf", paths)
            self.assertIn("figure_7b_source_data", paths)
            self.assertTrue(paths[f"{base}_png"].exists())
            self.assertTrue(paths[f"{base}_svg"].exists())
            self.assertTrue(paths[f"{base}_pdf"].exists())
            source = pd.read_csv(paths["figure_7b_source_data"])
            self.assertIn("pv_capacity_kw", source.columns)
            self.assertIn("battery_energy_original_kwh", source.columns)
            self.assertIn("battery_adjustment_kwh", source.columns)
            self.assertNotIn("energy_based_lolp", source.columns)
            self.assertNotIn("energy_based_lolp_percent", source.columns)
            adjustments = source.groupby("city_key")["battery_adjustment_kwh"].first().to_dict()
            self.assertEqual(float(adjustments["Singapore"]), 0.0)
            self.assertEqual(float(adjustments["Jakarta"]), 0.0)
            self.assertEqual(float(adjustments["Metro_Manila"]), -30.0)
            self.assertEqual(float(adjustments["Brisbane"]), 10.0)
            delta = source["battery_energy_kwh"] - source["battery_energy_original_kwh"]
            pd.testing.assert_series_equal(delta.reset_index(drop=True), source["battery_adjustment_kwh"].reset_index(drop=True), check_names=False)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_city_comparison_figure_8_has_no_parameter_annotations(self) -> None:
        sources = build_apartment_city_comparison_sources(INPUT_ROOT, seed=42)
        with TemporaryDirectory() as tmp:
            paths = make_city_comparison_figures(Path(tmp), sources=sources, figures=[8])
            base = f"{APARTMENT_CASE_KEY}_{FIGURE_NAMES[8]}"
            svg_text = paths[f"{base}_svg"].read_text(encoding="utf-8")

            self.assertIn("BESS SOC", svg_text)
            self.assertIn("Hourly energy", svg_text)
            for forbidden in ("SAIDI", "SAIFI", "Outage", "UE", "TOU spread", "PV capacity", "BESS capacity"):
                self.assertNotIn(forbidden, svg_text)

    def test_real_singlefamily_inputs_are_discovered_and_normalized(self) -> None:
        cases = {case.case_key: case for case in discover_input_cases(INPUT_ROOT)}
        expected = {"SG_SingleFamily", "Manila_SingleFamily", "Brisbane_SingleFamily", "Jakarta_SingleFamily"}
        self.assertTrue(expected.issubset(cases))
        self.assertEqual(cases["Jakarta_SingleFamily"].load_csv.name, "Jakarta_hourly_loads_kWh.csv")

        for case_key in sorted(expected):
            frame, _, _, _ = load_input_case(cases[case_key], self._fast_singlefamily_config())
            self.assertEqual(len(frame), 8760)
            for column in ("electric_load_kwh", "cooling_load_kwh_th", "heating_load_kwh_th", "pv_kwh"):
                self.assertIn(column, frame.columns)
                self.assertFalse(frame[column].isna().any())

    def test_real_singlefamily_city_sources_use_real_status_and_simulated_scenarios(self) -> None:
        sources = self._minimal_singlefamily_sources()
        city_inputs = sources["city_inputs"]
        designs = sources["design_results"]
        scenario_designs = sources["scenario_design_results"]

        self.assertEqual(city_inputs["city_key"].astype(str).tolist(), ["Singapore", "Metro_Manila", "Brisbane", "Jakarta"])
        self.assertTrue((city_inputs["data_status"].astype(str) == REAL_SINGLEFAMILY_DATA_STATUS).all())
        self.assertTrue((designs["data_status"].astype(str) == REAL_SINGLEFAMILY_DATA_STATUS).all())
        self.assertEqual(set(designs.groupby("city_key").size().astype(int).tolist()), {2})
        self.assertEqual(len(scenario_designs), 4)
        self.assertTrue(scenario_designs["energy_based_lolp"].between(0.0, 1.0).all())
        self.assertNotEqual(REAL_SINGLEFAMILY_DATA_STATUS, DATA_STATUS)

        sg = city_inputs[city_inputs["city_key"].astype(str) == "Singapore"].iloc[0]
        self.assertGreater(float(sg["annual_load_mwh"]), 0.0)
        self.assertGreater(float(sg["annual_pv_yield_kwh_per_kw"]), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_real_singlefamily_city_comparison_all_exports(self) -> None:
        sources = self._minimal_singlefamily_sources()
        with TemporaryDirectory() as tmp:
            paths = make_city_comparison_figures(
                Path(tmp),
                case_key=SINGLEFAMILY_CASE_KEY,
                sources=sources,
                figures=["all"],
            )
            for figure_no, figure_name in FIGURE_NAMES.items():
                base = f"{SINGLEFAMILY_CASE_KEY}_{figure_name}"
                self.assertTrue(paths[f"{base}_png"].exists())
                self.assertTrue(paths[f"{base}_svg"].exists())
                self.assertTrue(paths[f"{base}_pdf"].exists())
                self.assertTrue(paths[f"figure_{figure_no}_source_data"].exists())
            base = f"{SINGLEFAMILY_CASE_KEY}_{FIGURE_7B_NAME}"
            self.assertTrue(paths[f"{base}_png"].exists())
            self.assertTrue(paths[f"{base}_svg"].exists())
            self.assertTrue(paths[f"{base}_pdf"].exists())
            self.assertTrue(paths["figure_7b_source_data"].exists())


if __name__ == "__main__":
    unittest.main()
