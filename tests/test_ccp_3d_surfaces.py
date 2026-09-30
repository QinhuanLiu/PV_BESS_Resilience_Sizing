from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures import _capacity_contour_style as style
from src.figures.ccp_3d_surfaces import (
    FIGURE_SIZE_INCHES,
    GRID_SPEC,
    SURFACE_STYLE,
    SURFACE_STYLES,
    VIEW_AZIMUTH,
    VIEW_ELEVATION,
    build_figure,
    dual_threshold_source,
    surface_grid,
)
from src.figures.core import matplotlib_context, save_publication_figure


def fixture_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for pv in (0.0, 10.0):
        for battery in (0.0, 20.0):
            for scenario_id, lolp in enumerate((0.05, 0.10, 0.15, 0.20, 0.25), start=1):
                rows.append(
                    {
                        "pv_capacity_kw": pv,
                        "battery_energy_kwh": battery,
                        "scenario_id": scenario_id,
                        "LOLP": lolp if (pv, battery) != (10.0, 20.0) else 0.05,
                    }
                )
    records = pd.DataFrame(rows)
    base = records.groupby(["pv_capacity_kw", "battery_energy_kwh"], as_index=False).agg(
        scenario_count=("scenario_id", "size")
    )
    base["LOLP_lt20_scenario_count"] = records.assign(_passes=records.LOLP.lt(0.20)).groupby(
        ["pv_capacity_kw", "battery_energy_kwh"], sort=True
    )["_passes"].sum().to_numpy(dtype=int)
    return records, base


class DualThresholdAggregationTests(unittest.TestCase):
    def test_strict_threshold_counts_and_nesting(self):
        records, base = fixture_records()
        source = dual_threshold_source(records, base)
        first = source.loc[(source.pv_capacity_kw == 0) & (source.battery_energy_kwh == 0)].iloc[0]
        self.assertEqual(first.LOLP_lt10_scenario_count, 1)
        self.assertEqual(first.LOLP_lt20_scenario_count, 3)
        self.assertAlmostEqual(first.CCP_LOLP_lt10, 0.2)
        self.assertAlmostEqual(first.CCP_LOLP_lt20, 0.6)
        top_right = source.loc[(source.pv_capacity_kw == 10) & (source.battery_energy_kwh == 20)].iloc[0]
        self.assertEqual(top_right.LOLP_lt10_scenario_count, 5)
        self.assertEqual(top_right.LOLP_lt20_scenario_count, 5)
        self.assertTrue((source.CCP_LOLP_lt10 <= source.CCP_LOLP_lt20).all())

    def test_lt20_reconciliation_and_invalid_nesting(self):
        records, base = fixture_records()
        base.loc[0, "LOLP_lt20_scenario_count"] = 2
        with self.assertRaisesRegex(ValueError, "established CCP"):
            dual_threshold_source(records, base)

    def test_strict_twenty_and_thirty_percent_counts_and_nesting(self):
        records, base = fixture_records()
        source = dual_threshold_source(records, base, surface_style=SURFACE_STYLES["20-30"])
        first = source.loc[(source.pv_capacity_kw == 0) & (source.battery_energy_kwh == 0)].iloc[0]
        self.assertEqual(first.LOLP_lt20_scenario_count, 3)
        self.assertEqual(first.LOLP_lt30_scenario_count, 5)
        self.assertAlmostEqual(first.CCP_LOLP_lt20, 0.6)
        self.assertAlmostEqual(first.CCP_LOLP_lt30, 1.0)
        self.assertTrue((source.CCP_LOLP_lt20 <= source.CCP_LOLP_lt30).all())


class ThreeDimensionalFigureTests(unittest.TestCase):
    def test_surface_grid_and_figure_style(self):
        records, base = fixture_records()
        source = dual_threshold_source(records, base)
        x, y, z = surface_grid(source, "CCP_LOLP_lt20")
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(z.shape, (2, 2))
        _, plt = matplotlib_context()
        figure, metadata = build_figure(source)
        try:
            axis = figure.axes[0]
            self.assertEqual(tuple(figure.get_size_inches()), FIGURE_SIZE_INCHES)
            self.assertEqual(metadata["grid_spec"], GRID_SPEC)
            self.assertEqual(axis.name, "3d")
            self.assertEqual(len(axis.collections), 2)
            self.assertEqual(metadata["view"], {"elevation": VIEW_ELEVATION, "azimuth": VIEW_AZIMUTH})
            self.assertEqual(metadata["z_tick_labels"], ["0%", "20%", "40%", "60%", "80%", "100%"])
            self.assertEqual(axis.title.get_fontsize(), style.SINGLE_OUTAGE_TITLE_FONT_SIZE)
            self.assertEqual(axis.xaxis.label.get_fontsize(), style.SINGLE_OUTAGE_BASE_FONT_SIZE)
            self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                             [SURFACE_STYLE["lt10"]["label"], SURFACE_STYLE["lt20"]["label"]])
            self.assertFalse(metadata["smoothing"])
            self.assertFalse(metadata["interpolation"])
        finally:
            plt.close(figure)

    def test_exported_three_dimensional_figure_has_all_formats(self):
        records, base = fixture_records()
        source = dual_threshold_source(records, base)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "surface"
            _, plt = matplotlib_context(output)
            figure, _ = build_figure(source, out=output)
            try:
                paths = save_publication_figure(figure, output)
            finally:
                plt.close(figure)
            self.assertEqual(set(paths), {"surface_png", "surface_svg", "surface_pdf"})
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
