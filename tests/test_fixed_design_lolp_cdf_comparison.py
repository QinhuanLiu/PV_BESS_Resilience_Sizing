from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures import fixed_design_lolp_cdf_comparison as comparison
from src.figures.core import FigureLayoutError, assert_publication_layout


def _records(pv: float, battery: float, offset: float) -> pd.DataFrame:
    seeds = np.arange(1, 1001)
    return pd.DataFrame({
        "outage_seed": seeds,
        "analysis_years": 15,
        "pv_capacity_kw": pv,
        "battery_energy_kwh": battery,
        "LOLP": np.clip((seeds % 101) / 200.0 + offset, 0.0, 0.95),
    })


class FixedDesignCdfComparisonTests(unittest.TestCase):
    def test_two_by_two_designs_use_the_requested_capacities_and_four_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, (pv, battery, _) in enumerate(comparison.TWO_BY_TWO_DESIGNS):
                path = root / f"two-by-two-{index}.csv"
                _records(pv, battery, index * 0.01).to_csv(path, index=False)
                paths.append(path)
            designs = comparison.load_designs(paths, comparison.TWO_BY_TWO_DESIGNS)
            figure, metadata = comparison.build_figure(
                designs, out=root, display_labels=comparison.TWO_BY_TWO_LEGEND_LABELS,
                legend_inside=True,
            )
            try:
                ax = figure.axes[0]
                self.assertEqual(len(ax.lines), 4)
                self.assertEqual(metadata["x_limits_percent"], [0.0, 100.0])
                self.assertEqual(metadata["y_limits_percent"], [0.0, 100.0])
                self.assertEqual(
                    [text.get_text() for text in ax.get_legend().get_texts()],
                    list(comparison.TWO_BY_TWO_LEGEND_LABELS),
                )
                self.assertEqual(
                    [(item["pv_capacity_kw"], item["battery_energy_kwh"])
                     for item in metadata["series"]],
                    [(pv, battery) for pv, battery, _ in comparison.TWO_BY_TWO_DESIGNS],
                )
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                legend_box = ax.get_legend().get_window_extent(renderer)
                axes_box = ax.get_window_extent(renderer)
                self.assertGreaterEqual(legend_box.x0, axes_box.x0)
                self.assertGreaterEqual(legend_box.y0, axes_box.y0)
                self.assertLessEqual(legend_box.x1, axes_box.x1)
                self.assertLessEqual(legend_box.y1, axes_box.y1)
                with self.assertRaisesRegex(FigureLayoutError, "legend overlaps the data area"):
                    assert_publication_layout(figure)
                assert_publication_layout(figure, allow_inside_legend=True)
                self.assertEqual(
                    [item["metadata"]["legend_label"] for item in designs],
                    ["PV 3.25 kW, BESS 11.5 kWh", "PV 3.25 kW, BESS 14.5 kWh",
                     "PV 6.5 kW, BESS 11.5 kWh", "PV 6.5 kW, BESS 14.5 kWh"],
                )
                for item in designs:
                    self.assertTrue(item["cdf"].CDF.is_monotonic_increasing)
                    self.assertEqual(float(item["cdf"].CDF.iloc[-1]), 1.0)
            finally:
                import matplotlib.pyplot as plt
                plt.close(figure)

    def test_four_design_cdfs_are_right_continuous_and_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, (pv, battery, _) in enumerate(comparison.DESIGNS):
                path = root / f"design-{index}.csv"
                _records(pv, battery, index * 0.01).to_csv(path, index=False)
                paths.append(path)
            designs = comparison.load_designs(paths)
            self.assertEqual(len(designs), 4)
            for item, (_, _, color) in zip(designs, comparison.DESIGNS, strict=True):
                cdf = item["cdf"]
                self.assertEqual(cdf["CDF"].iloc[-1], 1.0)
                self.assertTrue(cdf["CDF"].is_monotonic_increasing)
                self.assertEqual(item["color"], color)
            figure, metadata = comparison.build_figure(designs, out=root)
            try:
                self.assertEqual(len(metadata["series"]), 4)
                self.assertEqual(metadata["comparison"], "<=")
                self.assertEqual(metadata["x_limits_percent"], [0.0, 100.0])
                self.assertFalse(metadata["legend_inside_axes"])
                self.assertEqual(
                    [text.get_text() for text in figure.axes[0].get_legend().get_texts()],
                    [item["metadata"]["legend_label"] for item in designs],
                )
            finally:
                import matplotlib.pyplot as plt
                plt.close(figure)

    def test_rejects_a_design_in_the_wrong_input_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, (pv, battery, _) in enumerate(comparison.DESIGNS):
                path = root / f"design-{index}.csv"
                _records(pv, battery, index * 0.01).to_csv(path, index=False)
                paths.append(path)
            _records(6.0, 11.5, 0.01).to_csv(paths[-1], index=False)
            with self.assertRaisesRegex(ValueError, "Expected PV=5"):
                comparison.load_designs(paths)


if __name__ == "__main__":
    unittest.main()
