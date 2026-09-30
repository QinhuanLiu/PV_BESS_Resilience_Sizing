"""Independent final-deliverable checks; run after the sweep reports complete."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.capacity_sweep import _load_enrichment_source, KEYS
from src.sweep_economics import validate_economic_results


def audit(directory: Path) -> dict:
    loaded = _load_enrichment_source(directory)
    reference = json.loads((directory / "source_reference.json").read_text())
    original = _load_enrichment_source(Path(reference["directory"]))
    data, yearly, spec = loaded["results"], loaded["yearly"], loaded["spec"]
    status = json.loads((directory / "run_status.json").read_text())
    assert status["state"] == "complete" and status["tasks_failed_this_attempt"] == 0
    assert status["workers_requested"] == 8 and len(status["worker_pids"]) == 8
    assert len(data) == 9900 and len(yearly) == 148500
    assert set(data[KEYS].itertuples(index=False, name=None)) == set(itertools.product(
        range(11), [2.5 * i for i in range(9)], range(1, 101)))
    assert not data.duplicated(KEYS).any()
    assert data.groupby(KEYS[:2]).size().eq(100).all()
    assert data.groupby("scenario_id").outage_profile_sha256.nunique().eq(1).all()
    assert data.outage_seed.nunique() == 100 and set(data.outage_seed) == set(range(42, 142))
    assert data.analysis_years.eq(15).all() and data.dispatch_timesteps.eq(131400).all()
    pd.testing.assert_frame_equal(original["results"], data[original["results"].columns], check_exact=True)
    pd.testing.assert_frame_equal(original["yearly"], yearly, check_exact=True)
    assert loaded["payloads"] == original["payloads"]
    assert data.SCR.isna().sum() == 900
    assert data.loc[data.pv_capacity_kw.eq(0), "SCR"].isna().all()
    economic_report = validate_economic_results(data, loaded["config"])
    summary = pd.read_csv(directory / "capacity_summary.csv", float_precision="round_trip")
    assert len(summary) == 99
    summary = summary.set_index(KEYS[:2])
    for pair, group in data.groupby(KEYS[:2]):
        for metric in ("annual_cost_sgd_per_year", "EUE_kwh_total", "EUE_kwh_per_year", "LOLP", "SSR", "SCR"):
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values):
                expected = (np.mean(values), np.std(values, ddof=1), np.median(values), np.quantile(values, 0.95), np.min(values), np.max(values))
            else:
                expected = (np.nan,) * 6
            actual = summary.loc[pair, [metric + "_" + stat for stat in ("mean", "std", "median", "p95", "min", "max")]].to_numpy(dtype=float)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True)
    checkpoint_rows = []
    for path in sorted((directory / "checkpoints").glob("*.json")):
        checkpoint = json.loads(path.read_text())
        assert checkpoint["fingerprint"] == loaded["identity"]["source_fingerprint"]
        digest = hashlib.sha256(json.dumps(checkpoint["payload"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert checkpoint["payload_sha256"] == digest
        checkpoint_rows.extend(checkpoint["payload"]["scenario_rows"])
    recomputed = pd.DataFrame(checkpoint_rows).sort_values(KEYS).reset_index(drop=True)
    assert len(recomputed) == 9900
    max_replay_differences = {
        col: float(np.max(np.abs(recomputed[col].to_numpy() - original["results"][col].to_numpy())))
        for col in ("EUE_kwh_total", "EUE_kwh_per_year", "LOLP")
    }
    for difference in max_replay_differences.values():
        assert difference < 1e-9
    report = {
        "passed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_rows": len(data), "yearly_rows": len(yearly), "capacity_rows": len(summary),
        "unique_keys_complete": True, "shared_exact_outage_sequences": True,
        "original_reliability_columns_exact": True, "yearly_reliability_exact": True,
        "replayed_reliability_max_abs_differences": max_replay_differences,
        "summary_statistics_independently_verified": True,
        "checkpoint_rows_verified": len(recomputed),
        "elapsed_seconds": status["elapsed_seconds"], "worker_count": len(status["worker_pids"]),
        **economic_report,
    }
    (directory / "final_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, ensure_ascii=False))
