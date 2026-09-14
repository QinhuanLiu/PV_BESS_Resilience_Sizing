from __future__ import annotations

import unittest

import pandas as pd

from src.design_selection import (
    aggregate_reliability_by_candidate,
    available_design_row,
    select_affordable_backup_design,
    select_minimum_total_cost_design,
    select_reliability_constrained_design,
)


class DesignSelectionTests(unittest.TestCase):
    def test_aggregate_reliability_by_candidate_computes_common_metrics(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "case_key": "SG_UTown",
                    "pv_capacity_kw": 10.0,
                    "battery_energy_kwh": 20.0,
                    "scenario_LOLP": 0.05,
                    "UE_kwh_per_year": 5.0,
                    "system_cost": 100.0,
                },
                {
                    "case_key": "SG_UTown",
                    "pv_capacity_kw": 10.0,
                    "battery_energy_kwh": 20.0,
                    "scenario_LOLP": 0.15,
                    "UE_kwh_per_year": 15.0,
                    "system_cost": 120.0,
                },
            ]
        )
        aggregated = aggregate_reliability_by_candidate(
            rows,
            group_columns=["case_key", "pv_capacity_kw", "battery_energy_kwh"],
            lolp_threshold=0.10,
            mean_columns={
                "mean_UE_kwh_per_year": "UE_kwh_per_year",
                "system_total_cost_sgd_per_year": "system_cost",
            },
            scenario_count_column="scenario_count",
        )
        row = aggregated.iloc[0]
        self.assertEqual(int(row["scenario_count"]), 2)
        self.assertAlmostEqual(float(row["mean_UE_kwh_per_year"]), 10.0)
        self.assertAlmostEqual(float(row["mean_scenario_LOLP"]), 0.10)
        self.assertAlmostEqual(float(row["chance_constraint_probability"]), 0.5)

    def test_selects_minimum_and_reliability_constrained_designs(self) -> None:
        candidates = pd.DataFrame(
            [
                {"pv_capacity_kw": 10.0, "battery_energy_kwh": 5.0, "mean_scenario_LOLP": 0.20, "system_total_cost_sgd_per_year": 80.0},
                {"pv_capacity_kw": 20.0, "battery_energy_kwh": 10.0, "mean_scenario_LOLP": 0.05, "system_total_cost_sgd_per_year": 100.0},
            ]
        )
        minimum = select_minimum_total_cost_design(candidates)
        constrained = select_reliability_constrained_design(candidates, lolp_threshold=0.10)
        self.assertEqual(minimum["pv_capacity_kw"], 10.0)
        self.assertEqual(constrained["pv_capacity_kw"], 20.0)
        self.assertTrue(constrained["available"])

    def test_selects_affordable_backup_design_by_backup_not_minimum_cost(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "pv_capacity_kw": 0.0,
                    "battery_energy_kwh": 0.0,
                    "mean_outage_load_served_ratio": 0.20,
                    "system_total_cost_sgd_per_year": 80.0,
                    "is_affordable": True,
                },
                {
                    "pv_capacity_kw": 20.0,
                    "battery_energy_kwh": 10.0,
                    "mean_outage_load_served_ratio": 0.75,
                    "system_total_cost_sgd_per_year": 100.0,
                    "is_affordable": True,
                },
                {
                    "pv_capacity_kw": 30.0,
                    "battery_energy_kwh": 20.0,
                    "mean_outage_load_served_ratio": 0.90,
                    "system_total_cost_sgd_per_year": 130.0,
                    "is_affordable": False,
                },
            ]
        )
        minimum = select_minimum_total_cost_design(candidates)
        affordable = select_affordable_backup_design(candidates)
        self.assertEqual(minimum["pv_capacity_kw"], 0.0)
        self.assertEqual(affordable["pv_capacity_kw"], 20.0)
        self.assertEqual(affordable["battery_energy_kwh"], 10.0)
        self.assertTrue(affordable["available"])

    def test_affordable_backup_design_can_fall_back_to_zero_capacity(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "pv_capacity_kw": 0.0,
                    "battery_energy_kwh": 0.0,
                    "mean_outage_load_served_ratio": 0.0,
                    "system_total_cost_sgd_per_year": 100.0,
                    "is_affordable": True,
                },
                {
                    "pv_capacity_kw": 10.0,
                    "battery_energy_kwh": 5.0,
                    "mean_outage_load_served_ratio": 0.80,
                    "system_total_cost_sgd_per_year": 120.0,
                    "is_affordable": False,
                },
            ]
        )
        affordable = select_affordable_backup_design(candidates)
        self.assertEqual(affordable["pv_capacity_kw"], 0.0)
        self.assertEqual(affordable["battery_energy_kwh"], 0.0)
        self.assertTrue(affordable["available"])

    def test_unavailable_design_and_available_reference_checks(self) -> None:
        candidates = pd.DataFrame(
            [{"pv_capacity_kw": 10.0, "battery_energy_kwh": 5.0, "mean_scenario_LOLP": 0.20, "system_total_cost_sgd_per_year": 80.0}]
        )
        unavailable = select_reliability_constrained_design(candidates, lolp_threshold=0.10)
        self.assertFalse(unavailable["available"])
        self.assertEqual(unavailable["design_type"], "reliability_constrained_design")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            available_design_row(
                pd.DataFrame([unavailable]),
                "reliability_constrained_design",
                missing_message="missing",
                unavailable_message="requested design is unavailable.",
            )


if __name__ == "__main__":
    unittest.main()
