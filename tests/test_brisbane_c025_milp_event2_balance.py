from __future__ import annotations

import unittest

import pandas as pd

from src.dispatch import solve_dispatch_milp
from src.figures.brisbane_c025_milp_event2_balance import _dispatch_metrics


def _config() -> dict:
    return {
        "technical": {"cooling_cop": 5.0, "heating_cop": 3.5},
        "battery": {
            "roundtrip_efficiency": 0.9,
            "soc_min_fraction": 0.1,
            "soc_max_fraction": 0.9,
            "initial_soc_fraction": 0.5,
            "rated_cycle_life_to_eol": 9000.0,
        },
        "economics": {"battery_capex_sgd_per_kwh": 400.0,
                      "battery_power_capex_sgd_per_kw": 150.0},
        "dispatch": {"solver": "HiGHS", "include_degradation_penalty": True},
        "pv": {"export_enabled": False},
        "active_scenario": {"voll_sgd_per_kwh": 35.0},
    }


class BrisbaneMilpEvent2Tests(unittest.TestCase):
    def test_explicit_initial_soc_changes_window_boundary_without_changing_default(self) -> None:
        frame = pd.DataFrame({
            "hour_index": [0, 1], "grid_availability": [1.0, 1.0],
            "electric_load_kwh": [0.0, 0.0], "cooling_load_kwh_th": [0.0, 0.0],
            "heating_load_kwh_th": [0.0, 0.0], "pv_kwh": [0.0, 0.0],
            "tou_sgd_per_kwh": [0.2, 0.2], "pv_export_sgd_per_kwh": [0.0, 0.0],
        })
        sizing = {"battery_energy_kwh": 11.6, "battery_power_kw": 2.9}
        default = solve_dispatch_milp(frame, sizing, _config())
        matched = solve_dispatch_milp(frame, sizing, _config(), initial_soc_kwh=1.16)
        self.assertAlmostEqual(float(default["battery_soc_kwh"].iloc[-1]), 5.8)
        self.assertAlmostEqual(float(matched["battery_soc_kwh"].iloc[-1]), 1.16)
        self.assertTrue(matched["solver_status"].eq("Optimal").all())
        with self.assertRaisesRegex(ValueError, "initial SOC"):
            solve_dispatch_milp(frame, sizing, _config(), initial_soc_kwh=0.5)

    def test_switch_metrics_distinguish_bidirectional_hours_and_reversals(self) -> None:
        frame = pd.DataFrame({
            "hour_index": [0, 1, 2, 3],
            "battery_charge_kwh": [3.1, 0.0, 0.2, 0.0],
            "battery_discharge_kwh": [0.0, 0.3, 0.2, 0.4],
            "battery_soc_kwh": [2.0, 1.7, 1.7, 1.3],
            "tou_sgd_per_kwh": [0.1] * 4,
            "pv_export_sgd_per_kwh": [0.0] * 4,
            "grid_import_kwh": [0.0] * 4,
            "pv_export_kwh": [0.0] * 4,
            "unmet_load_kwh_equiv": [0.0] * 4,
        })
        metrics = _dispatch_metrics(frame, 1.16, _config(), 2.9, 0.0)
        self.assertEqual(metrics["charge_power_limit_exceeded_hour_indices"], [0])
        self.assertEqual(metrics["same_hour_charge_and_discharge_count"], 1)
        self.assertEqual(metrics["nonidle_direction_reversal_count"], 1)


if __name__ == "__main__":
    unittest.main()
