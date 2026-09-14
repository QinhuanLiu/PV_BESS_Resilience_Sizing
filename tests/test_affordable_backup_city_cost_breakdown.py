from __future__ import annotations

import unittest

import pandas as pd

from tools import affordable_backup_city_cost_breakdown as chart


class AffordableBackupCityCostBreakdownTests(unittest.TestCase):
    def test_night_window_chart_contract_covers_five_city_cases(self) -> None:
        self.assertEqual(
            chart.TARGET_CASE_KEYS,
            (
                "SG_SingleFamily",
                "Bangkok_SingleFamily",
                "Brisbane_SingleFamily",
                "Jakarta_SingleFamily",
                "Manila_SingleFamily",
            ),
        )
        self.assertEqual(chart.OUTPUT_DIR.name, "Affordable_Backup_City_Comparison_Night_Window")
        self.assertEqual(chart.NIGHT_WINDOW_START_HOUR, 19.0)
        self.assertEqual(chart.NIGHT_WINDOW_END_HOUR, 23.0)

    def test_validate_outputs_accepts_five_city_night_window_source_tables(self) -> None:
        cities = [case_key.split("_", 1)[0] for case_key in chart.TARGET_CASE_KEYS]
        labels = {"SG": "Singapore"}
        summary_rows = []
        breakdown_rows = []
        for city in cities:
            for design_order, (design_case, design_label) in enumerate(chart.DESIGN_ORDER, start=1):
                total = 1000.0 + 10.0 * design_order
                summary_rows.append(
                    {
                        "city_key": city,
                        "city_label": labels.get(city, city),
                        "design_case": design_case,
                        "design_case_label": design_label,
                        "design_case_order": design_order,
                        "pv_capacity_kw": float(design_order),
                        "battery_energy_kwh": float(design_order - 1),
                        "mean_outage_load_served_ratio": 1.0 if design_case == "affordable_backup_design" else 0.5,
                        "affordability_margin_sgd_per_year": 100.0,
                        "is_affordable": True,
                        "first_event_start_hour_of_day": 19.25,
                        "last_event_start_hour_of_day": 22.75,
                        "system_total_cost_sgd_per_year": total,
                    }
                )
                for component_order, (component_key, component_label, color) in enumerate(chart.COMPONENTS, start=1):
                    cost = total if component_order == 1 else 0.0
                    breakdown_rows.append(
                        {
                            "city_key": city,
                            "design_case": design_case,
                            "component_key": component_key,
                            "component_label": component_label,
                            "component_color": color,
                            "stack_order": component_order,
                            "cost_sgd": cost,
                            "cost_ksgd": cost / 1000.0,
                            "system_total_cost_sgd_per_year": total,
                        }
                    )

        summary = pd.DataFrame(summary_rows)
        breakdown = pd.DataFrame(breakdown_rows)
        chart._validate_outputs(summary, breakdown)
        svg = chart._build_svg(summary, breakdown)

        self.assertIn("Singapore", svg)
        self.assertIn("19:00-23:00", svg)
        self.assertEqual(len(summary["city_key"].drop_duplicates()), 5)
        self.assertTrue(
            summary.loc[summary["design_case"].eq("affordable_backup_design"), "is_affordable"].astype(bool).all()
        )


if __name__ == "__main__":
    unittest.main()
