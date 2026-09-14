from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.tail_resilience import (
    build_tail_resilience_plot_source,
    build_tail_resilience_source,
    example_tail_resilience_source,
    make_tail_resilience_publication_figure,
    validate_tail_resilience_source,
)


def _optimal_designs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "design_type": "minimum_total_cost_design",
                "available": True,
                "pv_capacity_kw": 1000.0,
                "battery_energy_kwh": 1000.0,
                "lolp_threshold": 0.10,
            },
            {
                "design_type": "reliability_constrained_design",
                "available": True,
                "pv_capacity_kw": 2000.0,
                "battery_energy_kwh": 3000.0,
                "lolp_threshold": 0.10,
            },
        ]
    )


def _scenario_results() -> pd.DataFrame:
    rows = []
    designs = [
        (1000.0, 1000.0, [7000.0, 2000.0, 0.0, 1000.0]),
        (2000.0, 3000.0, [2000.0, 500.0, 0.0, 0.0]),
        (3000.0, 5000.0, [0.0, 0.0, 0.0, 0.0]),
    ]
    loads = [20_000.0, 10_000.0, 5000.0, 5000.0]
    for pv_kw, bess_kwh, ues in designs:
        for scenario_id, (load, ue) in enumerate(zip(loads, ues, strict=True), start=1):
            lolp = ue / load if load > 0 else 0.0
            rows.append(
                {
                    "frontier_scenario_id": scenario_id,
                    "pv_capacity_kw": pv_kw,
                    "battery_energy_kwh": bess_kwh,
                    "outage_weighted_load_kwh": load,
                    "UE_kwh_per_year": ue,
                    "scenario_LOLP": lolp,
                    "outage_load_served_ratio": 1.0 - lolp,
                }
            )
    return pd.DataFrame(rows)


class PublicationTailResilienceTests(unittest.TestCase):
    def test_build_source_filters_selected_designs_and_adds_probabilities(self) -> None:
        source = build_tail_resilience_source(_scenario_results(), _optimal_designs())
        self.assertEqual(len(source), 8)
        self.assertEqual(source["design_case"].nunique(), 2)
        self.assertAlmostEqual(
            source[source["design_case"] == "minimum_total_cost_design"]["scenario_probability"].sum(),
            1.0,
        )
        self.assertIn("Economic optimum", source["design_case_label"].tolist())
        self.assertIn("Reliability-constrained", source["design_case_label"].tolist())

    def test_plot_source_adds_tail_statistics_and_failure_flags(self) -> None:
        source = build_tail_resilience_source(_scenario_results(), _optimal_designs())
        plot_source = build_tail_resilience_plot_source(source)
        economic = plot_source[plot_source["design_case"] == "minimum_total_cost_design"].iloc[0]
        reliability = plot_source[plot_source["design_case"] == "reliability_constrained_design"].iloc[0]
        self.assertAlmostEqual(economic["mean_UE_mwh_per_year"], 2.5)
        self.assertAlmostEqual(economic["p90_UE_mwh_per_year"], 7.0)
        self.assertAlmostEqual(economic["p95_UE_mwh_per_year"], 7.0)
        self.assertAlmostEqual(economic["worst_UE_mwh_per_year"], 7.0)
        self.assertGreater(economic["target_failure_probability"], reliability["target_failure_probability"])
        self.assertIn("normalized_UE_fraction", plot_source.columns)
        self.assertIn("load_cover_ratio", plot_source.columns)

    def test_validate_rejects_invalid_values(self) -> None:
        source = example_tail_resilience_source()
        bad = source.copy()
        bad.loc[0, "UE_kwh_per_year"] = -1.0
        with self.assertRaisesRegex(ValueError, "negative UE"):
            validate_tail_resilience_source(bad)

        bad = source.copy()
        bad.loc[0, "scenario_LOLP"] = 1.5
        with self.assertRaisesRegex(ValueError, "scenario_LOLP must be between 0 and 1"):
            validate_tail_resilience_source(bad)

        with self.assertRaisesRegex(ValueError, "missing required column"):
            validate_tail_resilience_source(source.drop(columns=["scenario_LOLP"]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_publication_figure_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_tail_resilience_publication_figure(
                Path(tmp),
                "SG_UTown",
                example_tail_resilience_source(),
            )
            expected = {
                "source_data",
                "SG_UTown_Figure_11_Mean_Tail_Resilience_Distribution_png",
                "SG_UTown_Figure_11_Mean_Tail_Resilience_Distribution_svg",
                "SG_UTown_Figure_11_Mean_Tail_Resilience_Distribution_pdf",
            }
            self.assertTrue(expected.issubset(paths.keys()))
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            svg_text = paths["SG_UTown_Figure_11_Mean_Tail_Resilience_Distribution_svg"].read_text()
            self.assertIn("Scenario-level UE distribution", svg_text)
            self.assertIn("Tail exceedance probability", svg_text)


if __name__ == "__main__":
    unittest.main()
