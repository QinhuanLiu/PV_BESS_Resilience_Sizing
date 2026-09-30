"""Focused checks for the isolated sensitivity benchmark; no files are written."""

from copy import deepcopy
import csv
from pathlib import Path
import sys
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import pandas as pd  # noqa: E402

import benchmark_sensitivity_warm_start as benchmark  # noqa: E402
from src import optimizer  # noqa: E402
from src.design_evaluation import evaluate_design_case  # noqa: E402
from src.io.config import load_config  # noqa: E402
from src.io.data_loader import build_case_frame, load_input_case  # noqa: E402


def check_chains(case, parameters):
    runs = benchmark.ordered_runs(parameters)
    base = float(parameters.base[benchmark.PARAMETER])
    assert len(runs) == len(parameters.sensitivity[benchmark.PARAMETER])
    assert runs[0][0].value == base
    assert sum(run.value == base for run, _ in runs) == 1
    anchors = []

    def fake_evaluate(case, config, run, anchor):
        anchors.append(None if anchor is None else dict(anchor))
        return {"pv_capacity_kw": run.value / 100.0, "battery_energy_kwh": run.value / 50.0}

    with patch.object(benchmark, "evaluate_price", fake_evaluate):
        rows = list(benchmark.run_price_chain(case, {}, runs, "warm"))
    expected = {"lower": rows[0], "upper": rows[0]}
    assert anchors[0] is None
    for index, (_, side) in enumerate(runs[1:], start=1):
        assert set(anchors[index]) == set(benchmark.CAPACITY_KEYS)
        assert anchors[index] == {key: expected[side][key] for key in benchmark.CAPACITY_KEYS}
        expected[side] = rows[index]
    anchors.clear()
    with patch.object(benchmark, "evaluate_price", fake_evaluate):
        list(benchmark.run_price_chain(case, {}, runs, "full"))
    assert all(anchor is None for anchor in anchors)


def check_counting(case, config):
    cfg = optimizer.scenario_config(config, case.case_key)
    frame, meta = build_case_frame(case, cfg, 2.0)
    frame = frame.iloc[:24].copy()
    frame["grid_availability"] = 1.0
    frame.loc[6, "grid_availability"] = 0.5
    frame["outage_fraction"] = 1.0 - frame["grid_availability"]
    store = optimizer.CandidateStore(retain_dispatch=False)
    evaluator = optimizer.CandidateEvaluator(cfg, store)
    original = optimizer.solve_dispatch
    with benchmark.count_dispatches() as counts:
        evaluator.evaluate(frame, 2.0, meta, require_dispatch=False)
        assert counts.dispatch_calls == 2
        evaluator.evaluate(frame, 2.0, meta, require_dispatch=False)
        assert counts.dispatch_calls == 2
        _, dispatch = evaluator.evaluate(frame, 2.0, meta, require_dispatch=True)
    assert counts.summary() == {
        "dispatch_calls": 3, "reference_dispatch_calls": 1,
        "candidate_evaluations": 2, "candidate_cache_hits": 1, "unique_candidates": 1,
    }
    assert len(dispatch) == 24 and not store._dispatches
    assert optimizer.solve_dispatch is original


class BowlEvaluator:
    def __init__(self, target):
        self.target = target

    def evaluate(self, frame, battery_energy_kwh, pv_meta, require_dispatch=True):
        pv = float(pv_meta["pv_capacity_kw"])
        battery = float(battery_energy_kwh)
        result = {
            "pv_capacity_kw": pv, "battery_energy_kwh": battery,
            "pv_bess_system_annual_cost_sgd": (pv - self.target[0]) ** 2 + (battery - self.target[1]) ** 2,
        }
        return result, pd.DataFrame({"dummy": [0]}) if require_dispatch else None


def check_boundaries(config):
    cfg = deepcopy(config)
    cfg["pv"].update({"capacity_kw_candidates": [0, 4, 8, 10], "rooftop_capacity_limit_kw": 10,
                      "local_refine_window_kw": 1, "local_refine_step_kw": 0.5})
    cfg["battery"].update({"energy_kwh_candidates": [0, 4, 8, 10],
                           "local_refine_window_kwh": 1, "local_refine_step_kwh": 0.5})
    cfg["capacity_search"].update({"max_upper_expansions": 0, "warm_start_enabled": True})

    def builder(pv):
        return pd.DataFrame(), {"pv_capacity_kw": pv}

    summary, dispatches, best = optimizer.optimize_pv_battery_with_refinement_with_dispatches(
        builder, deepcopy(cfg), evaluator=BowlEvaluator((8, 8)), retain_candidate_dispatches=False,
        warm_start_best={"pv_capacity_kw": 0, "battery_energy_kwh": 0},
    )
    assert best["warm_start_fallback_triggered"] and best["warm_start_fallback_reason"] == "boundary"
    assert best["warm_start_candidate_count"] > 0 and "coarse" in set(summary["search_stage"])
    assert best["pv_capacity_kw"] == 8 and best["battery_energy_kwh"] == 8 and not dispatches
    for target in ((0, 0), (10, 0)):
        _, _, best = optimizer.optimize_pv_battery_with_refinement_with_dispatches(
            builder, deepcopy(cfg), evaluator=BowlEvaluator(target), retain_candidate_dispatches=False,
            warm_start_best=dict(zip(benchmark.CAPACITY_KEYS, target)),
        )
        assert not best["warm_start_fallback_triggered"]


def check_small_workflow(case, config):
    cfg = deepcopy(config)
    cfg["pv"].update({"capacity_kw_candidates": [0, 1, 2], "local_refine_window_kw": 0.5, "local_refine_step_kw": 0.5})
    cfg["battery"].update({"energy_kwh_candidates": [0, 1, 2], "local_refine_window_kwh": 0.5, "local_refine_step_kwh": 0.5})
    cfg["capacity_search"]["max_upper_expansions"] = 0
    anchors = None
    for mode in ("full", "warm"):
        with benchmark.count_dispatches() as counts:
            result = evaluate_design_case(
                case, cfg, case.case_key, frame_transform=lambda frame: frame.iloc[:24].copy(),
                warm_start_best=anchors,
            )
        row = result[0].iloc[0]
        assert bool(row["warm_start_used"]) == (mode == "warm")
        assert len(result[3]) == 24 and not result[5]
        assert row[list(benchmark.RESULT_METRICS)].notna().all()
        assert counts.dispatch_calls == counts.candidate_evaluations + counts.reference_dispatch_calls
        anchors = {key: float(row[key]) for key in benchmark.CAPACITY_KEYS}
        del result


def main():
    case = benchmark.input_case(REPO_ROOT / "Input")
    with case.parameter_csv.open(encoding="utf-8", newline="") as handle:
        assert all(len(row) == 8 for row in csv.reader(handle))
    config = load_config(REPO_ROOT / "config.yaml")
    config["dispatch"]["backend"] = "greedy"
    config["outputs"]["save_candidate_dispatch_workbooks"] = False
    frame, _, config, parameters = load_input_case(case, config)
    assert len(frame) == 8760
    check_chains(case, parameters)
    check_counting(case, config)
    check_boundaries(config)
    check_small_workflow(case, config)
    print("Sensitivity benchmark smoke passed: input, chain reset, dispatch counting, fallback, physical boundaries, and 24-hour workflow.")


if __name__ == "__main__":
    main()
