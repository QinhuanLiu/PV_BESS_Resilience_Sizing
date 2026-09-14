from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

import pandas as pd

from src.app import cli as cli_module
from src.variable_uncertainty import (
    aggregate_variable_uncertainty_candidates,
    optimize_variable_uncertainty_design_with_refinement,
    select_variable_uncertainty_designs,
    select_variable_uncertainty_scenario_design,
    summarize_variable_uncertainty_results,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pv_capacity_kw": 4.0,
                "battery_energy_kwh": 4.0,
                "pv_bess_system_annual_cost_sgd": 80.0,
                "scenario_LOLP": 0.20,
                "UE_kwh_per_year": 20.0,
            },
            {
                "pv_capacity_kw": 5.0,
                "battery_energy_kwh": 6.0,
                "pv_bess_system_annual_cost_sgd": 90.0,
                "scenario_LOLP": 0.05,
                "UE_kwh_per_year": 5.0,
            },
            {
                "pv_capacity_kw": 6.0,
                "battery_energy_kwh": 8.0,
                "pv_bess_system_annual_cost_sgd": 110.0,
                "scenario_LOLP": 0.0,
                "UE_kwh_per_year": 0.0,
            },
        ]
    )


def _candidate_scenario_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pv, battery, costs, lolps in (
        (4.0, 4.0, [80.0, 90.0], [0.20, 0.10]),
        (5.0, 6.0, [95.0, 105.0], [0.05, 0.04]),
    ):
        for scenario_id, (cost, lolp) in enumerate(zip(costs, lolps, strict=True), start=1):
            rows.append(
                {
                    "case_key": "Test_Case",
                    "city_key": "Test",
                    "building_key": "Case",
                    "uncertainty_scenario_id": scenario_id,
                    "outage_seed": 1000 + scenario_id,
                    "pv_capacity_kw": pv,
                    "pv_capacity_mw": pv / 1000.0,
                    "battery_energy_kwh": battery,
                    "battery_power_kw": battery / 2.0,
                    "battery_c_rate": 0.5,
                    "UE_kwh_per_year": 100.0 * lolp,
                    "scenario_LOLP": lolp,
                    "outage_load_served_ratio": 1.0 - lolp,
                    "unserved_energy_cost_sgd_per_year": 50.0 * lolp,
                    "pv_bess_grid_cost_sgd": 20.0,
                    "pv_bess_private_annual_cost_sgd": cost - 5.0,
                    "pv_bess_system_annual_cost_sgd": cost,
                }
            )
    return pd.DataFrame(rows)


class VariableUncertaintyTests(unittest.TestCase):
    def test_scenario_selection_uses_cost_without_lolp_constraint(self) -> None:
        selected = select_variable_uncertainty_scenario_design(_candidate_rows(), lolp_threshold=None)

        self.assertEqual(selected["pv_capacity_kw"], 4.0)
        self.assertEqual(selected["battery_energy_kwh"], 4.0)
        self.assertEqual(selected["pv_bess_system_annual_cost_sgd"], 80.0)
        self.assertTrue(selected["available"])

    def test_scenario_selection_uses_lolp_threshold_when_present(self) -> None:
        selected = select_variable_uncertainty_scenario_design(_candidate_rows(), lolp_threshold=0.10)

        self.assertEqual(selected["pv_capacity_kw"], 5.0)
        self.assertEqual(selected["battery_energy_kwh"], 6.0)
        self.assertEqual(selected["pv_bess_system_annual_cost_sgd"], 90.0)

    def test_unavailable_scenario_does_not_raise_after_capacity_expansion_limit(self) -> None:
        config = {
            "pv": {"capacity_kw_candidates": [0.0, 1.0], "local_refine_window_kw": 1.0, "local_refine_step_kw": 1.0},
            "battery": {
                "energy_kwh_candidates": [0.0, 1.0],
                "local_refine_window_kwh": 1.0,
                "local_refine_step_kwh": 1.0,
            },
            "capacity_search": {"max_upper_expansions": 0},
        }
        infeasible = _candidate_rows()
        infeasible["scenario_LOLP"] = 0.30
        with patch("src.variable_uncertainty._evaluate_variable_candidate_grid", return_value=infeasible):
            _, selected = optimize_variable_uncertainty_design_with_refinement(
                lambda pv_capacity_kw: (pd.DataFrame(), {"pv_capacity_kw": pv_capacity_kw}),
                deepcopy(config),
                outage_weighted_load_kwh=100.0,
                lolp_threshold=0.10,
            )

        self.assertFalse(selected["available"])
        self.assertEqual(selected["unavailable_reason"], "No candidate satisfies scenario_LOLP <= lolp_threshold.")
        self.assertAlmostEqual(selected["min_evaluated_scenario_LOLP"], 0.30)

    def test_candidate_summary_and_selected_designs_support_cross_scenario_constraint(self) -> None:
        summary = aggregate_variable_uncertainty_candidates(_candidate_scenario_rows(), lolp_threshold=0.10)
        selected = select_variable_uncertainty_designs(summary, lolp_threshold=0.10)

        constrained = selected[selected["design_type"] == "reliability_constrained_design"].iloc[0]
        self.assertEqual(int(summary["scenario_count"].iloc[0]), 2)
        self.assertAlmostEqual(float(summary.loc[summary["pv_capacity_kw"] == 4.0, "mean_scenario_LOLP"].iloc[0]), 0.15)
        self.assertAlmostEqual(float(summary.loc[summary["pv_capacity_kw"] == 5.0, "chance_constraint_probability"].iloc[0]), 1.0)
        self.assertTrue(bool(constrained["available"]))
        self.assertEqual(float(constrained["pv_capacity_kw"]), 5.0)

    def test_distribution_summary_skips_unavailable_rows(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "available": True,
                    "pv_capacity_kw": 4.0,
                    "battery_energy_kwh": 4.0,
                    "pv_bess_private_annual_cost_sgd": 70.0,
                    "pv_bess_system_annual_cost_sgd": 80.0,
                    "unserved_energy_kwh_per_year": 2.0,
                    "scenario_LOLP": 0.20,
                },
                {
                    "available": False,
                    "pv_capacity_kw": pd.NA,
                    "battery_energy_kwh": pd.NA,
                    "pv_bess_private_annual_cost_sgd": pd.NA,
                    "pv_bess_system_annual_cost_sgd": pd.NA,
                    "unserved_energy_kwh_per_year": pd.NA,
                    "scenario_LOLP": pd.NA,
                },
            ]
        )

        summary = summarize_variable_uncertainty_results(rows)

        self.assertEqual(set(summary["count"].astype(int)), {1})
        self.assertIn("scenario_LOLP", set(summary["metric"]))

    def test_cli_variable_uncertainty_arguments_reach_workflow(self) -> None:
        argv = [
            "pv_bess_sizing",
            "--analysis",
            "variable-uncertainty",
            "--case",
            "SG_UTown",
            "--uncertainty-scenario-count",
            "12",
            "--uncertainty-seed-start",
            "3000",
            "--lolp-threshold",
            "0.08",
        ]
        config = {"input": {"directory": "Input"}, "outputs": {"directory": "outputs"}}
        with (
            patch("sys.argv", argv),
            patch("src.app.cli.load_config", return_value=config),
            patch("src.app.cli.run_selected", return_value={}) as run_selected_mock,
        ):
            cli_module.main()

        kwargs = run_selected_mock.call_args.kwargs
        self.assertTrue(kwargs["run_variable_uncertainty"])
        self.assertFalse(kwargs["run_baseline"])
        self.assertEqual(kwargs["case_filters"], ["SG_UTown"])
        self.assertEqual(kwargs["uncertainty_scenario_count"], 12)
        self.assertEqual(kwargs["uncertainty_seed_start"], 3000)
        self.assertAlmostEqual(kwargs["lolp_threshold"], 0.08)

    def test_cli_variable_uncertainty_figures_flag_reaches_workflow(self) -> None:
        argv = ["pv_bess_sizing", "--analysis", "variable-uncertainty-figures", "--case", "SG_UTown"]
        config = {"input": {"directory": "Input"}, "outputs": {"directory": "outputs"}}
        with (
            patch("sys.argv", argv),
            patch("src.app.cli.load_config", return_value=config),
            patch("src.app.cli.run_selected", return_value={}) as run_selected_mock,
        ):
            cli_module.main()

        kwargs = run_selected_mock.call_args.kwargs
        self.assertTrue(kwargs["run_variable_uncertainty_figures"])
        self.assertFalse(kwargs["run_variable_uncertainty"])
        self.assertFalse(kwargs["run_baseline"])


if __name__ == "__main__":
    unittest.main()
