from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.fixed_design_paired_evaluation import (
    ALL_BATTERIES_KWH,
    ANALYSIS_YEARS,
    BASELINE_BATTERY_KWH,
    CASE_KEY,
    PAIRED_RUN_NAME,
    PV_CAPACITY_KW,
    SCENARIO_COUNT,
    TARGET_BATTERIES_KWH,
    load_paired_results,
    load_reference_records,
    run,
    strict_ccp_validation,
    target_records,
    validate_baseline_replay,
)


def reference_records() -> pd.DataFrame:
    seeds = np.arange(1, SCENARIO_COUNT + 1)
    return pd.DataFrame(
        {
            "outage_seed": seeds,
            "analysis_years": ANALYSIS_YEARS,
            "pv_capacity_kw": PV_CAPACITY_KW,
            "battery_energy_kwh": BASELINE_BATTERY_KWH,
            "LOLP": np.linspace(0.0, 0.6, SCENARIO_COUNT),
        }
    )


def paired_records(reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for battery in ALL_BATTERIES_KWH:
        rows.append(
            pd.DataFrame(
                {
                    "case_key": CASE_KEY,
                    "scenario_id": np.arange(1, SCENARIO_COUNT + 1),
                    "outage_seed": reference["outage_seed"].to_numpy(),
                    "outage_profile_sha256": [f"profile-{seed}" for seed in reference["outage_seed"]],
                    "analysis_years": ANALYSIS_YEARS,
                    "pv_capacity_kw": PV_CAPACITY_KW,
                    "battery_energy_kwh": battery,
                    "LOLP": (
                        reference["LOLP"].to_numpy()
                        if battery == BASELINE_BATTERY_KWH
                        else np.clip(reference["LOLP"].to_numpy() - (battery - BASELINE_BATTERY_KWH) * 0.03, 0, 1)
                    ),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class FixedDesignPairedEvaluationTests(unittest.TestCase):
    def write_csv(self, directory: Path, name: str, rows: pd.DataFrame) -> Path:
        path = directory / name
        rows.to_csv(path, index=False)
        return path

    def test_paired_records_validate_and_share_profile_identity(self):
        reference = reference_records()
        with tempfile.TemporaryDirectory() as temporary:
            source = self.write_csv(Path(temporary), "paired.csv", paired_records(reference))
            loaded = load_paired_results(source)
        self.assertEqual(len(loaded), 3000)
        self.assertEqual(set(loaded.battery_energy_kwh), set(ALL_BATTERIES_KWH))
        self.assertTrue(loaded.groupby("outage_seed").outage_profile_sha256.nunique().eq(1).all())
        report = validate_baseline_replay(reference, loaded)
        self.assertTrue(report["passed"])
        self.assertEqual(report["scenario_count"], 1000)

    def test_baseline_replay_rejects_a_single_seed_difference(self):
        reference = reference_records()
        records = paired_records(reference)
        records.loc[
            (records.battery_energy_kwh == BASELINE_BATTERY_KWH) & (records.outage_seed == 401), "LOLP"
        ] += 0.01
        with self.assertRaisesRegex(ValueError, "baseline replay differs"):
            validate_baseline_replay(reference, records)

    def test_invalid_paired_seed_or_profile_pairing_is_rejected(self):
        reference = reference_records()
        records = paired_records(reference)
        records.loc[(records.battery_energy_kwh == 8.5) & (records.outage_seed == 1000), "outage_seed"] = 999
        with tempfile.TemporaryDirectory() as temporary:
            source = self.write_csv(Path(temporary), "bad.csv", records)
            with self.assertRaisesRegex(ValueError, "outage seed"):
                load_paired_results(source)

        records = paired_records(reference)
        records.loc[(records.battery_energy_kwh == 14.5) & (records.outage_seed == 500), "outage_profile_sha256"] = "different"
        with tempfile.TemporaryDirectory() as temporary:
            source = self.write_csv(Path(temporary), "bad-profile.csv", records)
            with self.assertRaisesRegex(ValueError, "different outage profile hashes"):
                load_paired_results(source)

    def test_strict_ccp_excludes_lolp_equal_to_twenty_percent(self):
        records = reference_records()
        records.loc[:, "LOLP"] = 0.21
        records.loc[:900, "LOLP"] = 0.19
        records.loc[901, "LOLP"] = 0.20
        report = strict_ccp_validation(records)
        self.assertEqual(report["passing_scenarios"], 901)
        self.assertEqual(report["CCP"], 0.901)
        self.assertTrue(report["constraint_satisfied"])

    def test_run_publishes_both_target_designs_after_baseline_validation(self):
        reference = reference_records()
        paired = paired_records(reference)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_csv = self.write_csv(root, "reference.csv", reference)

            def fake_sweep(spec, output, *, workers, resume):
                self.assertEqual(spec.pv_capacities, (PV_CAPACITY_KW,))
                self.assertEqual(spec.battery_capacities, ALL_BATTERIES_KWH)
                self.assertEqual(spec.scenario_count, SCENARIO_COUNT)
                self.assertEqual(spec.analysis_years, ANALYSIS_YEARS)
                self.assertFalse(resume)
                output.mkdir(parents=True)
                paired.to_csv(output / "scenario_results.csv", index=False)
                (output / "manifest.json").write_text("{}", encoding="utf-8")
                return {"state": "complete"}

            def fake_figures(input_csv, output_dir):
                self.assertTrue(Path(input_csv).is_file())
                output_dir.mkdir(parents=True)
                artifact = output_dir / "mock-figure.txt"
                artifact.write_text("figure", encoding="utf-8")
                return {"validation": artifact}

            with patch("src.fixed_design_paired_evaluation.distribution.run", side_effect=fake_figures):
                outputs = run(
                    reference_csv=reference_csv,
                    output_root=root / "outputs",
                    workers=1,
                    sweep_runner=fake_sweep,
                )

            self.assertEqual(set(outputs), {"battery_8.5_kwh", "battery_14.5_kwh"})
            self.assertTrue((root / "outputs" / PAIRED_RUN_NAME / "baseline_replay_validation.json").is_file())
            for battery in TARGET_BATTERIES_KWH:
                path = outputs[f"battery_{battery:g}_kwh"]["scenario_results"]
                delivered = pd.read_csv(path)
                self.assertEqual(len(delivered), SCENARIO_COUNT)
                self.assertTrue(np.allclose(delivered.battery_energy_kwh, battery))
                report = json.loads(outputs[f"battery_{battery:g}_kwh"]["validation"].read_text())
                self.assertTrue(report["baseline_replay"]["passed"])
                self.assertEqual(report["strict_ccp_LOLP_lt_20"]["comparison"], "<")

    def test_reference_source_requires_the_expected_baseline_design(self):
        rows = reference_records()
        rows.loc[:, "battery_energy_kwh"] = 8.5
        with tempfile.TemporaryDirectory() as temporary:
            source = self.write_csv(Path(temporary), "wrong-baseline.csv", rows)
            with self.assertRaisesRegex(ValueError, "BESS=11.5"):
                load_reference_records(source)

    def test_target_extraction_contains_one_homogeneous_design(self):
        extracted = target_records(paired_records(reference_records()), 14.5)
        self.assertEqual(len(extracted), SCENARIO_COUNT)
        self.assertEqual(extracted.battery_energy_kwh.nunique(), 1)
        self.assertEqual(float(extracted.battery_energy_kwh.iloc[0]), 14.5)


if __name__ == "__main__":
    unittest.main()
