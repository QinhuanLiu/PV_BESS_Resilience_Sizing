from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.figures.typical_outage_72h_energy_balance import _balance_columns, _window_bounds


class TypicalOutage72HourTests(unittest.TestCase):
    def test_selected_day_has_previous_and_recovery_day(self) -> None:
        self.assertEqual(_window_bounds(61141.56638545976, 61143.883545084005, 131400), (61104, 61176))
        with self.assertRaisesRegex(ValueError, "complete three-day window"):
            _window_bounds(47.5, 48.5, 8760)

    def test_ac_balance_closes_without_counting_unserved_energy_as_a_sink(self) -> None:
        frame = pd.DataFrame({
            "total_load_e_equiv_kwh": [5.0, 5.0],
            "unmet_load_kwh_equiv": [0.0, 1.0],
            "pv_available_kwh": [3.0, 1.0],
            "grid_import_kwh": [2.0, 0.0],
            "battery_discharge_kwh": [1.0, 3.0],
            "battery_charge_kwh": [1.0, 0.0],
            "pv_curtailed_kwh": [0.0, 0.0],
            "pv_export_kwh": [0.0, 0.0],
            "battery_soc_kwh": [5.0, 2.0],
        })
        config = {"battery": {"roundtrip_efficiency": 0.9, "soc_min_fraction": 0.1}}
        balanced = _balance_columns(frame, config)
        np.testing.assert_allclose(balanced["ac_balance_residual_kwh"], 0.0, atol=1e-12)
        self.assertEqual(balanced["served_load_kwh"].tolist(), [5.0, 4.0])
        self.assertTrue((balanced["battery_conversion_loss_kwh"] > 0).all())
        frame.loc[1, "grid_import_kwh"] = 0.2
        with self.assertRaisesRegex(ValueError, "AC energy balance"):
            _balance_columns(frame, config)


if __name__ == "__main__":
    unittest.main()
