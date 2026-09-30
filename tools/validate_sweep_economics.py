"""Compare streaming metrics against the source run's frozen full-dispatch code.

Usage: python tools/validate_sweep_economics.py SOURCE_DIRECTORY REPORT_JSON
Only the small validation sample uses complete hourly dispatch tables.
"""
from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.capacity_sweep import _load_enrichment_source, _init_worker, _WORKER
from src.dispatch import solve_dispatch_greedy
from src.optimizer import battery_sizing
from src.sweep_economics import economic_metrics


def validate(source_directory: Path, report_path: Path) -> None:
    started = time.perf_counter()
    source = _load_enrichment_source(source_directory)
    config = source["config"]
    package = types.ModuleType("frozen_sweep_source")
    package.__path__ = [str(source_directory.resolve() / "source_snapshot/src")]
    sys.modules[package.__name__] = package
    old_dispatch = importlib.import_module(package.__name__ + ".dispatch").solve_dispatch_greedy
    old_cost = importlib.import_module(package.__name__ + ".costing").pv_bess_annual_cost
    _init_worker(source["case"], config, max(source["spec"].pv_capacities))
    checks = []
    # Extremes, a middle design, both generation/storage-only designs, two outages.
    combinations = [(0, 0), (0, 20), (5, 0), (5, 10), (10, 20)]
    for scenario in (source["payloads"][0], source["payloads"][-1]):
        frame = _WORKER["frame"].copy()
        available = np.ones(len(frame))
        available[scenario["hour_indices"]] = scenario["availability"]
        frame["grid_availability"] = available
        frame["outage_fraction"] = 1 - available
        for pv, battery in combinations:
            frame["pv_kwh"] = _WORKER["pv_per_kw"] * pv
            sizing = battery_sizing(battery, config)
            full = old_dispatch(frame, sizing, config)
            old = old_cost(frame, full, sizing, {"pv_capacity_kw": pv}, config)
            light = solve_dispatch_greedy(frame, sizing, config, metrics_only=True, collect_totals=True)
            pd.testing.assert_frame_equal(light, full[light.columns], check_exact=True)
            new = economic_metrics(light.attrs["energy_totals"], sizing, pv, config)
            for name, expected in old.items():
                np.testing.assert_allclose(new[name], expected, rtol=1e-10, atol=1e-8, err_msg=name)
            expected_ssr = full.pv_self_served_load_kwh.sum() / full.total_load_e_equiv_kwh.sum()
            np.testing.assert_allclose(new["SSR"], expected_ssr, rtol=1e-10, atol=1e-12)
            if pv:
                expected_scr = (full.pv_to_load_kwh.sum() + full.pv_to_battery_kwh.sum()) / full.pv_available_kwh.sum()
                np.testing.assert_allclose(new["SCR"], expected_scr, rtol=1e-10, atol=1e-12)
            else:
                assert new["SCR"] is None
            checks.append({
                "scenario_id": scenario["scenario_id"], "pv_capacity_kw": pv,
                "battery_energy_kwh": battery, "timesteps": len(frame),
                "full_hourly_reliability_exact": True, "all_original_cost_fields_agree": True,
                "annual_cost_absolute_difference_sgd": abs(new["annual_cost_sgd_per_year"] - old["pv_bess_system_annual_cost_sgd"]),
                "SSR_absolute_difference": abs(new["SSR"] - expected_ssr),
                "SCR_verified": True,
            })
            print(f"Verified frozen full dispatch: scenario {scenario['scenario_id']}, PV {pv}, battery {battery}", flush=True)
            del full, light
            gc.collect()
    report = {"passed": True, "source": str(source_directory.resolve()),
              "source_fingerprint": source["identity"]["source_fingerprint"],
              "checks": checks, "elapsed_seconds": time.perf_counter() - started}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
