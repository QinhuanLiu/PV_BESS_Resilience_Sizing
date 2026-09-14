from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.io.config import DEFAULT_CONFIG
from src.io.data_loader import InputCase
from src.optimizer import scenario_config
from src.outage import simulate_outages
from src.outage_evaluation import (
    OutageEvaluationContext,
    evaluate_outage_candidate,
    generate_outage_scenarios,
)


class OutageEvaluationTests(unittest.TestCase):
    def _case_and_config(self, root: Path) -> tuple[InputCase, dict[str, object]]:
        city_dir = root / "Input" / "SG"
        city_dir.mkdir(parents=True, exist_ok=True)
        load = pd.DataFrame(
            {
                "Cooling": [0.0] * 8760,
                "Heating": [0.0] * 8760,
                "Lighting": [6.0] * 8760,
                "Equipment": [4.0] * 8760,
            }
        )
        pv = pd.DataFrame({"PV_output_kWh_per_m2": [0.0] * 8760})
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.25] * 24})
        load_csv = city_dir / "SG_UTown_Load.csv"
        pv_csv = city_dir / "SG_UTown_PV.csv"
        tou_csv = city_dir / "SG_UTown_TOU.csv"
        parameter_csv = city_dir / "SG_UTown_Parameter.csv"
        load.to_csv(load_csv, index=False)
        pv.to_csv(pv_csv, index=False)
        tou.to_csv(tou_csv, index=False)
        pd.DataFrame().to_csv(parameter_csv, index=False)
        case = InputCase("SG", "UTown", "SG_UTown", city_dir, load_csv, pv_csv, tou_csv, parameter_csv)
        config = deepcopy(DEFAULT_CONFIG)
        config.setdefault("dispatch", {})["backend"] = "greedy"
        config["pv"]["capacity_kw_candidates"] = [0.0]
        config["pv"]["rooftop_area_m2"] = 1000.0
        config["pv"]["module_power_density_kw_per_m2"] = 0.2
        config["battery"]["max_c_rate"] = 0.0
        config["scenarios"] = {
            "SG_UTown": {
                "name": "Singapore UTown",
                "city_or_service_area": "Singapore",
                "saidi_min_per_year": 60.0,
                "saifi_per_year": 1.0,
                "voll_sgd_per_kwh": 50.0,
            }
        }
        return case, config

    def test_generate_outage_scenarios_is_reproducible_for_same_seed(self) -> None:
        with TemporaryDirectory() as tmp:
            case, config = self._case_and_config(Path(tmp))
            first = generate_outage_scenarios(case, config, 2, 123, "frontier_scenario_id")
            second = generate_outage_scenarios(case, config, 2, 123, "frontier_scenario_id")
            self.assertEqual(first["outage_seed"].tolist(), [123, 124])
            self.assertEqual(
                first[["frontier_scenario_id", "simulated_saidi_min_per_year", "simulated_saifi_per_year"]].values.tolist(),
                second[["frontier_scenario_id", "simulated_saidi_min_per_year", "simulated_saifi_per_year"]].values.tolist(),
            )

    def test_night_window_random_outages_start_in_requested_window(self) -> None:
        profile_a, events_a = simulate_outages(
            3.0,
            180.0,
            seed=123,
            forced_event_mode="night_window_random",
            night_window_start_hour=19.0,
            night_window_end_hour=23.0,
        )
        profile_b, events_b = simulate_outages(
            3.0,
            180.0,
            seed=123,
            forced_event_mode="night_window_random",
            night_window_start_hour=19.0,
            night_window_end_hour=23.0,
        )
        _, events_c = simulate_outages(
            3.0,
            180.0,
            seed=124,
            forced_event_mode="night_window_random",
            night_window_start_hour=19.0,
            night_window_end_hour=23.0,
        )

        start_hours = events_a["start_hour"].astype(float) % 24.0
        self.assertEqual(len(events_a), 3)
        self.assertTrue(((start_hours >= 19.0) & (start_hours < 23.0)).all())
        self.assertAlmostEqual(float(profile_a["outage_fraction"].sum() * 60.0), 180.0)
        self.assertEqual(events_a["start_hour"].tolist(), events_b["start_hour"].tolist())
        self.assertNotEqual(events_a["start_hour"].tolist(), events_c["start_hour"].tolist())

    def test_evaluate_outage_candidate_zero_outage_load_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            case, config = self._case_and_config(Path(tmp))
            cfg = scenario_config(config, "SG_UTown")
            context = OutageEvaluationContext(case, cfg, 0.0, retain_dispatch=True)
            profile = pd.DataFrame(
                {
                    "hour_index": range(8760),
                    "grid_availability": [1.0] * 8760,
                    "outage_fraction": [0.0] * 8760,
                }
            )
            evaluation = evaluate_outage_candidate(context, profile, 0.0, require_dispatch=False)
            self.assertAlmostEqual(evaluation.metrics["outage_weighted_load_kwh"], 0.0)
            self.assertAlmostEqual(evaluation.metrics["scenario_LOLP"], 0.0)
            self.assertAlmostEqual(evaluation.metrics["outage_load_served_ratio"], 1.0)

    def test_evaluate_outage_candidate_dispatch_and_no_dispatch_metrics_match(self) -> None:
        with TemporaryDirectory() as tmp:
            case, config = self._case_and_config(Path(tmp))
            cfg = scenario_config(config, "SG_UTown")
            context = OutageEvaluationContext(case, cfg, 0.0, retain_dispatch=True)
            profile = pd.DataFrame(
                {
                    "hour_index": range(8760),
                    "grid_availability": [0.0] + [1.0] * 8759,
                    "outage_fraction": [1.0] + [0.0] * 8759,
                }
            )
            no_dispatch = evaluate_outage_candidate(context, profile, 0.0, require_dispatch=False)
            with_dispatch = evaluate_outage_candidate(context, profile, 0.0, require_dispatch=True)
            self.assertIsNotNone(with_dispatch.dispatch)
            self.assertAlmostEqual(no_dispatch.metrics["UE_kwh_per_year"], with_dispatch.metrics["UE_kwh_per_year"])
            self.assertAlmostEqual(no_dispatch.metrics["scenario_LOLP"], with_dispatch.metrics["scenario_LOLP"])


if __name__ == "__main__":
    unittest.main()
