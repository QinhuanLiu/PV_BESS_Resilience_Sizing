"""Compare full and capacity-only warm-start searches in isolated processes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src import optimizer  # noqa: E402
from src.design_evaluation import evaluate_design_case  # noqa: E402
from src.io.config import load_config  # noqa: E402
from src.io.data_loader import discover_input_cases, load_case_parameters, load_input_case  # noqa: E402

CASE_KEY = "Brisbane_SingleFamily"
PARAMETER = "battery_capex_sgd_per_kwh"
CAPACITY_KEYS = ("pv_capacity_kw", "battery_energy_kwh")
RESULT_METRICS = (*CAPACITY_KEYS, "pv_bess_system_annual_cost_sgd", "unserved_energy_kwh_per_year", "energy_based_LOLP")
DEFAULT_OUTPUT = Path("/tmp/pv_bess_brisbane_sensitivity_warm_start")


@dataclass
class DispatchCounts:
    dispatch_calls: int = 0
    reference_dispatch_calls: int = 0
    candidate_evaluations: int = 0
    candidate_cache_hits: int = 0
    capacities: set[tuple[float, float]] = field(default_factory=set)

    def summary(self) -> dict[str, int]:
        return {
            "dispatch_calls": self.dispatch_calls,
            "reference_dispatch_calls": self.reference_dispatch_calls,
            "candidate_evaluations": self.candidate_evaluations,
            "candidate_cache_hits": self.candidate_cache_hits,
            "unique_candidates": len(self.capacities),
        }


@dataclass(frozen=True)
class BenchmarkRun:
    """A deliberately small price-chain point for the warm-start benchmark."""

    parameter: str
    value: float

    @property
    def override(self) -> dict[str, float]:
        return {self.parameter: self.value}


@contextmanager
def count_dispatches():
    """Instrument the optimizer's call sites without changing solver behavior."""
    counts = DispatchCounts()
    solve = optimizer.solve_dispatch
    evaluate = optimizer.evaluate_candidate
    reference = optimizer._zero_battery_reference
    cache_get = optimizer.CandidateStore.get

    def counted_solve(*args, **kwargs):
        counts.dispatch_calls += 1
        return solve(*args, **kwargs)

    def counted_evaluate(frame, battery_energy_kwh, pv_meta, *args, **kwargs):
        counts.candidate_evaluations += 1
        counts.capacities.add((float(pv_meta["pv_capacity_kw"]), float(battery_energy_kwh)))
        return evaluate(frame, battery_energy_kwh, pv_meta, *args, **kwargs)

    def counted_reference(*args, **kwargs):
        counts.reference_dispatch_calls += 1
        return reference(*args, **kwargs)

    def counted_get(store, *args, **kwargs):
        result = cache_get(store, *args, **kwargs)
        if result is not None:
            counts.candidate_cache_hits += 1
        return result

    with (
        patch.object(optimizer, "solve_dispatch", counted_solve),
        patch.object(optimizer, "evaluate_candidate", counted_evaluate),
        patch.object(optimizer, "_zero_battery_reference", counted_reference),
        patch.object(optimizer.CandidateStore, "get", counted_get),
    ):
        yield counts


def ordered_runs(parameters):
    runs = [BenchmarkRun(PARAMETER, float(value)) for value in parameters.sensitivity.get(PARAMETER, [])]
    base_value = float(parameters.base[PARAMETER])
    baseline = [run for run in runs if run.value == base_value]
    if len(baseline) != 1:
        raise ValueError("The sensitivity range must contain its base price exactly once.")
    lower = sorted((run for run in runs if run.value < base_value), key=lambda run: -run.value)
    upper = sorted((run for run in runs if run.value > base_value), key=lambda run: run.value)
    return [(baseline[0], "base"), *((run, "lower") for run in lower), *((run, "upper") for run in upper)]


def input_case(input_root: Path):
    return next(case for case in discover_input_cases(input_root) if case.case_key == CASE_KEY)


def manifest(case) -> dict[str, str]:
    inputs = [case.parameter_csv, case.load_csv, case.pv_csv, case.tou_csv]
    sources = [REPO_ROOT / "config.yaml", Path(__file__).resolve()]
    sources.extend(path for path in (REPO_ROOT / "src").rglob("*.py") if "figures" not in path.parts)
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(inputs + sources)}


def evaluate_price(case, config, run, anchor):
    started = perf_counter()
    frame, _, case_config, _ = load_input_case(case, config, parameter_overrides=run.override)
    if len(frame) != 8760 or case_config["analysis"]["years"] != 1:
        raise ValueError("This benchmark requires the configured one-year, 8760-hour case.")
    if case_config["outage"]["seed"] != 42:
        raise ValueError("This benchmark requires the configured outage seed 42.")
    if case_config["economics"]["battery_power_capex_sgd_per_kw"] != 150.0:
        raise ValueError("Keep the configured battery power cost at 150 SGD/kW.")
    del frame
    solve_started = perf_counter()
    with count_dispatches() as counts:
        result = evaluate_design_case(case, case_config, case.case_key, warm_start_best=anchor)
    solve_elapsed = perf_counter() - solve_started
    if result[5]:
        raise AssertionError("Candidate dispatches must not be retained.")
    if counts.dispatch_calls != counts.candidate_evaluations + counts.reference_dispatch_calls:
        raise AssertionError("Dispatch instrumentation missed a solver call.")
    row = dict(result[0].iloc[0])
    row.update({"sensitivity_parameter": run.parameter, "sensitivity_value": run.value})
    row.update(counts.summary())
    row.update({
        "solve_elapsed_s": solve_elapsed,
        "outage_profile_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(result[1], index=True).to_numpy().tobytes()
        ).hexdigest(),
        "pv_refine_step_kw": case_config["pv"]["local_refine_step_kw"],
        "battery_refine_step_kwh": case_config["battery"]["local_refine_step_kwh"],
    })
    if row["dispatch_backend"] != "greedy":
        raise AssertionError("The benchmark must use greedy dispatch throughout.")
    del result, case_config, counts
    gc.collect()
    row["price_elapsed_s"] = perf_counter() - started
    return row


def run_price_chain(case, config, runs, mode):
    previous = {"lower": None, "upper": None}
    for index, (run, direction) in enumerate(runs, start=1):
        anchor = previous.get(direction) if mode == "warm" else None
        row = evaluate_price(case, config, run, anchor)
        row.update({"mode": mode, "execution_index": index, "direction": direction})
        point = {key: float(row[key]) for key in CAPACITY_KEYS}
        if direction == "base":
            previous = {"lower": dict(point), "upper": dict(point)}
        else:
            previous[direction] = point
        yield row


def peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss) / (1024.0 ** 2 if sys.platform == "darwin" else 1024.0)


def worker(args):
    started = perf_counter()
    case = input_case(args.input_dir)
    source_manifest = manifest(case)
    parameters = load_case_parameters(case.parameter_csv)
    runs = ordered_runs(parameters)
    config = load_config(REPO_ROOT / "config.yaml")
    config["dispatch"]["backend"] = "greedy"
    config["outputs"]["save_candidate_dispatch_workbooks"] = False
    config["capacity_search"]["warm_start_enabled"] = args.worker_mode == "warm"
    rows = []
    for row in run_price_chain(case, config, runs, args.worker_mode):
        row["repeat"] = args.repeat_index
        rows.append(row)
        print(json.dumps({
            "event": "point", "repeat": args.repeat_index, "mode": args.worker_mode,
            "price": row["sensitivity_value"], "seconds": row["price_elapsed_s"],
            "dispatch_calls": row["dispatch_calls"],
            "fallback": bool(row["warm_start_fallback_triggered"]),
        }), flush=True)
    if source_manifest != manifest(case):
        raise RuntimeError("Inputs or source code changed during the benchmark batch.")
    frame = pd.DataFrame(rows).sort_values("sensitivity_value")
    if frame["outage_profile_sha256"].nunique() != 1:
        raise AssertionError("Sensitivity points did not use the same outage profile.")
    output_path = args.output_dir / f"repeat_{args.repeat_index}_{args.worker_mode}.csv"
    frame.to_csv(output_path, index=False)
    print(json.dumps({
        "event": "batch", "repeat": args.repeat_index, "mode": args.worker_mode,
        "point_count": len(rows), "result_file": output_path.name,
        "worker_elapsed_s": perf_counter() - started,
        "peak_rss_mib": peak_rss_mib(),
        "dispatch_calls": int(frame["dispatch_calls"].sum()),
        "reference_dispatch_calls": int(frame["reference_dispatch_calls"].sum()),
        "candidate_cache_hits": int(frame["candidate_cache_hits"].sum()),
        "fallback_count": int(frame["warm_start_fallback_triggered"].sum()),
        "source_manifest": source_manifest,
    }), flush=True)


def launch_batch(args, output_dir: Path, repeat: int, mode: str):
    command = [
        sys.executable, "-B", "-u", str(Path(__file__).resolve()),
        "--input-dir", str(args.input_dir), "--output-dir", str(output_dir),
        "--worker-mode", mode, "--repeat-index", str(repeat),
    ]
    started = perf_counter()
    receipt = None
    with subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True, encoding="utf-8") as process:
        try:
            for line in process.stdout:
                event = json.loads(line)
                if event["event"] == "point":
                    print(
                        f"Repeat {repeat} {mode}: price={event['price']:g}, "
                        f"{event['seconds']:.2f}s, dispatches={event['dispatch_calls']}, "
                        f"fallback={event['fallback']}", flush=True,
                    )
                elif event["event"] == "batch":
                    receipt = event
            returncode = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if returncode != 0 or receipt is None:
        raise RuntimeError(f"Benchmark batch failed: repeat={repeat}, mode={mode}, exit={returncode}")
    receipt["batch_elapsed_s"] = perf_counter() - started
    receipt.pop("event")
    return receipt


def compare_points(points: pd.DataFrame) -> pd.DataFrame:
    timing = ("price_elapsed_s", "solve_elapsed_s", "dispatch_calls", "unique_candidates", "candidate_cache_hits")
    rows = []
    for price, group in points.groupby("sensitivity_value", sort=True):
        full = group[group["mode"] == "full"]
        warm = group[group["mode"] == "warm"]
        row = {"battery_capex_sgd_per_kwh": float(price)}
        for mode, sample in (("full", full), ("warm", warm)):
            for metric in (*RESULT_METRICS, *timing):
                row[f"{mode}_{metric}"] = float(sample[metric].median())
        for metric in RESULT_METRICS:
            row[f"delta_{metric}"] = row[f"warm_{metric}"] - row[f"full_{metric}"]
        reference_cost = row["full_pv_bess_system_annual_cost_sgd"]
        row["cost_delta_pct"] = 100.0 * row["delta_pv_bess_system_annual_cost_sgd"] / max(abs(reference_cost), 1e-12)
        row["price_speedup"] = row["full_price_elapsed_s"] / row["warm_price_elapsed_s"]
        row["cost_screen_passed"] = abs(row["cost_delta_pct"]) <= 0.1
        row["capacity_review_required"] = (
            abs(row["delta_pv_capacity_kw"]) > float(group["pv_refine_step_kw"].iloc[0]) + 1e-6
            or abs(row["delta_battery_energy_kwh"]) > float(group["battery_refine_step_kwh"].iloc[0]) + 1e-6
        )
        row["reliability_changed"] = (
            abs(row["delta_unserved_energy_kwh_per_year"]) > 1e-9
            or abs(row["delta_energy_based_LOLP"]) > 1e-12
        )
        row["repeat_results_stable"] = all(
            sample[list(RESULT_METRICS)].nunique(dropna=False).le(1).all() for sample in (full, warm)
        )
        row["warm_fallback_repeats"] = int(warm["warm_start_fallback_triggered"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(output_dir, batches, comparison, source_manifest):
    full = batches[batches["mode"] == "full"]
    warm = batches[batches["mode"] == "warm"]
    full_seconds = float(full["batch_elapsed_s"].median())
    warm_seconds = float(warm["batch_elapsed_s"].median())
    savings = 100.0 * (1.0 - warm_seconds / full_seconds)
    summary = {
        "case": CASE_KEY, "parameter": PARAMETER, "repeats_per_mode": len(full),
        "point_count": len(comparison), "full_median_batch_seconds": full_seconds,
        "warm_median_batch_seconds": warm_seconds, "speedup": full_seconds / warm_seconds,
        "time_saved_pct": savings, "speed_target_passed": savings >= 20.0,
        "all_cost_screens_passed": bool(comparison["cost_screen_passed"].all()),
        "max_absolute_cost_delta_pct": float(comparison["cost_delta_pct"].abs().max()),
        "all_repeat_results_stable": bool(comparison["repeat_results_stable"].all()),
        "capacity_review_prices": comparison.loc[comparison["capacity_review_required"], PARAMETER].tolist(),
        "reliability_change_prices": comparison.loc[comparison["reliability_changed"], PARAMETER].tolist(),
        "full_median_dispatch_calls": float(full["dispatch_calls"].median()),
        "warm_median_dispatch_calls": float(warm["dispatch_calls"].median()),
        "full_median_peak_rss_mib": float(full["peak_rss_mib"].median()),
        "warm_median_peak_rss_mib": float(warm["peak_rss_mib"].median()),
        "warm_total_fallbacks": int(warm["fallback_count"].sum()),
        "source_manifest": source_manifest,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Brisbane Battery-Cost Warm-Start Benchmark", "",
        f"Full search: {full_seconds:.3f} s median; warm-start: {warm_seconds:.3f} s median.",
        f"Speedup: {summary['speedup']:.3f}x; time saved: {savings:.2f}%.", "",
        "## Method", "",
        "Prices come from the current Parameter.csv. Each batch evaluates the base price once,",
        "then lower prices descending and higher prices ascending. Each warm chain starts at its own base result.",
        "Only two capacity values pass between prices. Dispatch and candidate caches are not shared.",
        "Greedy dispatch; one year; seed 42; unchanged technical inputs and search resolution.",
        "Independent serial processes alternate full/warm and warm/full. Timing includes process startup,",
        "input loading, all nine prices, fallback attempts, and writing the batch result CSV.",
        "Cross-batch aggregation/report generation is outside timing. No candidate dispatch files are saved.",
        "Solver counts include no-battery reference dispatches and the final best-dispatch rerun, but not cache hits.",
        "Peak RSS is per isolated batch process, including interpreter/library memory.", "",
        "## Batch Results", "",
        "| Repeat | Mode | End-to-end s | Dispatch calls | Fallbacks | Peak MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in batches.itertuples(index=False):
        lines.append(f"| {row.repeat} | {row.mode} | {row.batch_elapsed_s:.3f} | {row.dispatch_calls} | {row.fallback_count} | {row.peak_rss_mib:.2f} |")
    lines += [
        "", "## Point Results", "",
        "Deltas are warm minus full. Capacities are kW/kWh; cost is SGD/year. Point times are medians.", "",
        "| Price | Full PV / BESS | Warm PV / BESS | Full / warm s | Cost delta % | UE delta kWh/year | LOLP delta | Warm fallbacks |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {getattr(row, PARAMETER):g} | {row.full_pv_capacity_kw:.6g} / {row.full_battery_energy_kwh:.6g} "
            f"| {row.warm_pv_capacity_kw:.6g} / {row.warm_battery_energy_kwh:.6g} "
            f"| {row.full_price_elapsed_s:.3f} / {row.warm_price_elapsed_s:.3f} "
            f"| {row.cost_delta_pct:.6f} | {row.delta_unserved_energy_kwh_per_year:.6g} "
            f"| {row.delta_energy_based_LOLP:.6g} | {row.warm_fallback_repeats} |"
        )
    lines += [
        "", "## Screening", "",
        f"- At least 20% time saved: {summary['speed_target_passed']}.",
        f"- All absolute cost differences within 0.1%: {summary['all_cost_screens_passed']}.",
        f"- Deterministic result metrics across repeats: {summary['all_repeat_results_stable']}.",
        f"- Capacity differences exceeding one configured refinement step: {summary['capacity_review_prices']}.",
        f"- Reliability differences: {summary['reliability_change_prices']}.",
        "- Full search is a reference heuristic, not a proof of global optimality.",
        "- Boundary fallback can miss better regions outside an interior local optimum.",
        "- Passing these screens is evidence for this case/backend/seed, not for other sensitivity analyses.",
        "- Detailed final rows, batch_performance.csv, point_comparison.csv, and source hashes are retained.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def benchmark(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run_", dir=args.output_dir))
    source_manifest = manifest(input_case(args.input_dir))
    receipts = []
    print(f"Results: {output_dir}", flush=True)
    for repeat in range(1, args.repeats + 1):
        modes = ("full", "warm") if repeat % 2 else ("warm", "full")
        for mode in modes:
            receipt = launch_batch(args, output_dir, repeat, mode)
            if receipt.pop("source_manifest") != source_manifest:
                raise RuntimeError("Cannot compare batches using different source/input versions.")
            receipts.append(receipt)
            pd.DataFrame(receipts).to_csv(output_dir / "batch_performance.csv", index=False)
            print(f"Completed repeat {repeat} {mode}: {receipt['batch_elapsed_s']:.2f}s", flush=True)
    batches = pd.DataFrame(receipts)
    points = pd.concat([pd.read_csv(output_dir / item["result_file"]) for item in receipts], ignore_index=True)
    if points["outage_profile_sha256"].nunique() != 1:
        raise AssertionError("Batches used different outage profiles.")
    comparison = compare_points(points)
    comparison.to_csv(output_dir / "point_comparison.csv", index=False)
    summary = write_report(output_dir, batches, comparison, source_manifest)
    print(f"Speedup: {summary['speedup']:.3f}x; time saved: {summary['time_saved_pct']:.2f}%", flush=True)
    print(f"Report: {output_dir / 'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "Input")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-mode", choices=("full", "warm"), help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    if args.worker_mode:
        worker(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
