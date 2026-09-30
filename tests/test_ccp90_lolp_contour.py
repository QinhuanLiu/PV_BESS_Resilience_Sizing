from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.figures.ccp90_lolp_contour import (
    CCP_TARGET, LEGACY_METRIC, LOLP_CONSTRAINT, METRIC, aggregate_q90_lolp,
    ccp_order_statistic_index,
)
from src.figures.capacity_sweep_contours import build_figure
from src.figures.core import matplotlib_context


def records() -> tuple[pd.DataFrame, dict[str, object]]:
    spec = {
        "case_key": "Brisbane_SingleFamily", "pv_capacities": [0.0, 1.0],
        "battery_capacities": [0.0, 2.5], "scenario_count": 100,
        "analysis_years": 15, "seed_start": 1,
    }
    rows = []
    for pv in spec["pv_capacities"]:
        for battery in spec["battery_capacities"]:
            values = np.full(100, 0.3)
            if (pv, battery) == (0.0, 0.0):
                values[:89], values[89] = 0.19, 0.20  # Strict CCP is only 89%.
            elif (pv, battery) == (0.0, 2.5):
                values[:90], values[90] = 0.19, 0.21  # Strict CCP reaches 90%.
            elif (pv, battery) == (1.0, 0.0):
                values[:] = np.linspace(0.0, 0.99, 100)
            else:
                values[:] = 0.0
            for scenario_id, lolp in enumerate(values, start=1):
                demand = 10.0
                rows.append({
                    "case_key": spec["case_key"], "pv_capacity_kw": pv,
                    "battery_energy_kwh": battery, "scenario_id": scenario_id,
                    "outage_seed": scenario_id, "outage_profile_sha256": f"profile-{scenario_id}",
                    "analysis_years": 15, "EUE_kwh_total": lolp * demand,
                    "EUE_kwh_per_year": lolp * demand / 15, "outage_load_kwh_total": demand,
                    "LOLP": lolp, "no_outage_load": False,
                })
    return pd.DataFrame(rows), spec


class Ccp90LlopContourTests(unittest.TestCase):
    def test_inverse_empirical_cdf_and_strict_constraint_are_distinct_at_equality(self):
        frame, spec = records()
        source = aggregate_q90_lolp(frame.sample(frac=1, random_state=3), spec)
        self.assertEqual(ccp_order_statistic_index(100), 89)
        tied = source[(source.pv_capacity_kw == 0) & (source.battery_energy_kwh == 0)].iloc[0]
        feasible = source[(source.pv_capacity_kw == 0) & (source.battery_energy_kwh == 2.5)].iloc[0]
        self.assertEqual(tied.Q90_LOLP, LOLP_CONSTRAINT)
        self.assertEqual(tied.CCP_LOLP_lt20, 0.89)
        self.assertFalse(tied.strict_CCP_LOLP_lt20_feasible)
        self.assertEqual(feasible.CCP_LOLP_lt20, CCP_TARGET)
        self.assertTrue(feasible.strict_CCP_LOLP_lt20_feasible)
        self.assertTrue(source.Q90_LOLP.between(0, 1).all())

    def test_figure_uses_existing_contour_style_and_ccp90_metric(self):
        frame, spec = records()
        source = aggregate_q90_lolp(frame, spec)
        _, plt = matplotlib_context()
        figure, metadata = build_figure(source, METRIC)
        try:
            self.assertEqual(metadata["colormap"], "YlGnBu")
            self.assertEqual(metadata["grid_shape"], [2, 2])
            self.assertEqual(metadata["fill_alpha"], 0.88)
            self.assertIn("Q90 LOLP", metadata["y_axis"])
            self.assertIsNone(metadata["design_marker"])
            self.assertEqual(METRIC["column"], "Q90_LOLP")
            self.assertEqual(METRIC["filename"], "Q90_LOLP_Contour")
            self.assertEqual(LEGACY_METRIC["filename"], "LOLP_at_90pct_CCP_Contour")
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
