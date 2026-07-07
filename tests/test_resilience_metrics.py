from __future__ import annotations

import unittest

import pandas as pd

from src.resilience_metrics import (
    achieved_lolp,
    chance_constraint_probability,
    electric_equivalent_building_load,
    eue,
    outage_load_served_ratio,
    outage_weighted_load_kwh,
    scenario_lolp,
)


class ResilienceMetricsTests(unittest.TestCase):
    def test_electric_equivalent_building_load_uses_full_building_load(self) -> None:
        frame = pd.DataFrame(
            {
                "electric_load_kwh": [10.0, 20.0],
                "cooling_load_kwh_th": [15.0, 30.0],
                "heating_load_kwh_th": [8.0, 12.0],
            }
        )
        load = electric_equivalent_building_load(frame, cooling_cop=5.0, heating_cop=4.0)
        self.assertEqual(load.tolist(), [15.0, 29.0])

    def test_electric_equivalent_building_load_rejects_invalid_cop(self) -> None:
        frame = pd.DataFrame(
            {
                "electric_load_kwh": [10.0],
                "cooling_load_kwh_th": [15.0],
                "heating_load_kwh_th": [8.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "cooling_cop must be positive"):
            electric_equivalent_building_load(frame, cooling_cop=0.0, heating_cop=4.0)
        with self.assertRaisesRegex(ValueError, "heating_cop must be positive"):
            electric_equivalent_building_load(frame, cooling_cop=5.0, heating_cop=-1.0)

    def test_outage_weighted_load_uses_partial_outage_fraction(self) -> None:
        load = pd.Series([15.0, 29.0, 10.0], name="total_load_e_equiv_kwh")
        outage = pd.Series([1.0, 0.5, 0.0], name="outage_fraction")
        weighted = outage_weighted_load_kwh(load, outage)
        self.assertEqual(weighted.tolist(), [15.0, 14.5, 0.0])
        self.assertEqual(weighted.name, "outage_weighted_load_kwh")

    def test_eue_sums_dispatch_unmet_load_above_tolerance(self) -> None:
        dispatch = pd.DataFrame({"unmet_load_kwh_equiv": [0.0, 2.0, 1e-8, 3.0]})
        self.assertAlmostEqual(eue(dispatch), 5.0)

    def test_outage_load_served_ratio_and_zero_load_default(self) -> None:
        dispatch = pd.DataFrame({"unmet_load_kwh_equiv": [0.0, 2.0, 3.0]})
        outage_load = pd.Series([10.0, 20.0, 20.0])
        self.assertAlmostEqual(outage_load_served_ratio(dispatch, outage_load), 0.9)
        self.assertAlmostEqual(outage_load_served_ratio(dispatch, pd.Series([0.0, 0.0])), 1.0)

    def test_scenario_lolp_uses_outage_weight_for_hours_with_unmet_load(self) -> None:
        dispatch = pd.DataFrame({"unmet_load_kwh_equiv": [0.0, 2.0, 1e-8, 3.0]})
        outage = pd.Series([1.0, 0.5, 0.25, 0.25])
        self.assertAlmostEqual(scenario_lolp(dispatch, outage), 0.75 / 2.0)
        self.assertAlmostEqual(scenario_lolp(dispatch, pd.Series([0.0, 0.0, 0.0, 0.0])), 0.0)

    def test_achieved_lolp_and_chance_constraint_probability(self) -> None:
        scenario_lolps = [0.0, 0.05, 0.2]
        self.assertAlmostEqual(achieved_lolp(scenario_lolps), 0.25 / 3.0)
        self.assertAlmostEqual(chance_constraint_probability(scenario_lolps), 2.0 / 3.0)
        self.assertAlmostEqual(chance_constraint_probability(scenario_lolps, lolp_threshold=0.0), 1.0 / 3.0)

    def test_cross_scenario_metrics_reject_empty_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "scenario_lolps must contain at least one scenario"):
            achieved_lolp([])
        with self.assertRaisesRegex(ValueError, "scenario_lolps must contain at least one scenario"):
            chance_constraint_probability([])


if __name__ == "__main__":
    unittest.main()
