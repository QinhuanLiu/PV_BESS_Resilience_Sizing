from __future__ import annotations

import unittest

import pandas as pd

from src.dispatch import solve_dispatch_greedy


class ReserveDispatchCompatibilityTests(unittest.TestCase):
    def _dispatch_config(self, reserve_fraction: float) -> dict[str, object]:
        return {
            "technical": {"cooling_cop": 5.0, "heating_cop": 3.5},
            "economics": {
                "battery_capex_sgd_per_kwh": 100.0,
                "battery_power_capex_sgd_per_kw": 10.0,
            },
            "battery": {
                "roundtrip_efficiency": 1.0,
                "initial_soc_fraction": 0.5,
                "soc_min_fraction": 0.1,
                "soc_max_fraction": 0.95,
                "normal_operation_reserve_fraction": reserve_fraction,
                "rated_cycle_life_to_eol": 9000.0,
            },
            "dispatch": {
                "include_degradation_penalty": True,
            },
        }

    def test_grid_connected_discharge_respects_reserve_floor(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0, 1, 2],
                "grid_availability": [1.0, 1.0, 1.0],
                "tou_sgd_per_kwh": [2.0, 1.0, 0.0],
                "pv_kwh": [0.0, 0.0, 0.0],
                "electric_load_kwh": [30.0, 0.0, 0.0],
                "cooling_load_kwh_th": [0.0, 0.0, 0.0],
                "heating_load_kwh_th": [0.0, 0.0, 0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        dispatch = solve_dispatch_greedy(frame, sizing, self._dispatch_config(0.4))
        self.assertAlmostEqual(float(dispatch["battery_soc_kwh"].iloc[0]), 40.0)
        self.assertAlmostEqual(float(dispatch["battery_discharge_kwh"].iloc[0]), 10.0)

    def test_outage_discharge_can_go_below_reserve_to_technical_soc(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [0.0],
                "tou_sgd_per_kwh": [1.0],
                "pv_kwh": [0.0],
                "electric_load_kwh": [30.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        dispatch = solve_dispatch_greedy(frame, sizing, self._dispatch_config(0.4))
        self.assertAlmostEqual(float(dispatch["battery_soc_kwh"].iloc[0]), 20.0)
        self.assertAlmostEqual(float(dispatch["battery_discharge_kwh"].iloc[0]), 30.0)

    def test_reserve_above_soc_max_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [1.0],
                "tou_sgd_per_kwh": [1.0],
                "pv_kwh": [0.0],
                "electric_load_kwh": [1.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        with self.assertRaisesRegex(ValueError, "normal_operation_reserve_fraction"):
            solve_dispatch_greedy(frame, sizing, self._dispatch_config(0.99))


if __name__ == "__main__":
    unittest.main()
