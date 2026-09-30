from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.brisbane_c025_15y_optimization import _changed_config_paths
from src.figures.brisbane_c025_optimized_outage_balance import _event_scores, _plotted_design


class BrisbaneC025OptimizationTests(unittest.TestCase):
    def test_effective_config_diff_reports_only_nested_changes(self) -> None:
        before = {"analysis": {"years": 1}, "battery": {"max_c_rate": 0.5}, "outage": {"seed": 42}}
        after = {"analysis": {"years": 15}, "battery": {"max_c_rate": 0.25}, "outage": {"seed": 42}}
        self.assertEqual(
            _changed_config_paths(before, after),
            ["analysis.years", "battery.max_c_rate"],
        )

    def test_zero_battery_optimum_selects_cheapest_positive_candidate(self) -> None:
        optimal = pd.DataFrame([{"pv_capacity_kw": 2.0, "battery_energy_kwh": 0.0}])
        candidates = pd.DataFrame([
            {"pv_capacity_kw": 3.0, "battery_energy_kwh": 2.0,
             "pv_bess_system_annual_cost_sgd": 120.0, "search_stage": "a", "search_iteration": 1},
            {"pv_capacity_kw": 2.5, "battery_energy_kwh": 1.0,
             "pv_bess_system_annual_cost_sgd": 110.0, "search_stage": "a", "search_iteration": 2},
            {"pv_capacity_kw": 2.0, "battery_energy_kwh": 0.0,
             "pv_bess_system_annual_cost_sgd": 100.0, "search_stage": "a", "search_iteration": 3},
        ])
        chosen, is_optimum = _plotted_design(optimal, candidates)
        self.assertFalse(is_optimum)
        self.assertEqual(float(chosen["battery_energy_kwh"]), 1.0)

    def test_event_selection_uses_median_deficit_then_event_id(self) -> None:
        events = pd.DataFrame([
            {"event_id": 9, "start_hour": 100.0, "end_hour": 100.5},
            {"event_id": 2, "start_hour": 200.0, "end_hour": 200.5},
        ])
        load = np.zeros(480)
        load[100], load[200] = 2.0, 6.0
        selected, median, scored = _event_scores(events, load, np.zeros_like(load))
        self.assertEqual(float(median), 2.0)
        self.assertEqual(int(selected["event_id"]), 2)
        self.assertTrue(scored["eligible_72h"].all())

    def test_near_equal_median_distances_ignore_float_roundoff(self) -> None:
        events = pd.DataFrame([
            {"event_id": 12, "start_hour": 100.0, "end_hour": 100.5},
            {"event_id": 2, "start_hour": 200.0, "end_hour": 200.5},
        ])
        load = np.zeros(480)
        load[100] = 2 * 0.5252516063100837
        load[200] = 2 * 0.5219330662280652
        selected, _, _ = _event_scores(events, load, np.zeros_like(load))
        self.assertEqual(int(selected["event_id"]), 2)

    def test_event_with_another_outage_in_window_is_excluded(self) -> None:
        events = pd.DataFrame([
            {"event_id": 1, "start_hour": 100.0, "end_hour": 100.5},
            {"event_id": 2, "start_hour": 110.0, "end_hour": 110.5},
            {"event_id": 3, "start_hour": 200.0, "end_hour": 200.5},
        ])
        load = np.ones(480)
        selected, _, scored = _event_scores(events, load, np.zeros_like(load))
        self.assertEqual(int(selected["event_id"]), 3)
        self.assertEqual(int(scored["eligible_72h"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
