"""Lightweight paired LOLP evaluation for the Brisbane 2-by-2 fixed designs.

The two PV=3.25 kW designs are reused. Only PV=6.5 kW with BESS=11.5 and
14.5 kWh is dispatched, using the frozen 1,000 outage profiles from the
completed paired replay. No hourly or yearly dispatch tables are retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from .capacity_sweep import (
    ROOT, _availability_hash, _atomic_csv, _atomic_json,
    _check_source_dependencies, _digest, _file_hash, _load_enrichment_source,
)
from .dispatch import solve_dispatch_greedy
from .figures import fixed_design_lolp_histogram_cdf as combined_figure
from .figures import fixed_design_lolp_distribution as distribution
from .fixed_design_paired_evaluation import (
    PAIRED_RUN_NAME, load_paired_results, load_reference_records,
    validate_baseline_replay,
)
from .io.data_loader import InputCase, build_case_frame
from .optimizer import battery_sizing


CASE_KEY = "Brisbane_SingleFamily"
ANALYSIS_YEARS = 15
SCENARIO_COUNT = 1000
LOW_PV_KW = 3.25
HIGH_PV_KW = 6.5
BATTERIES_KWH = (11.5, 14.5)
DESIGNS = tuple((pv, battery) for pv in (LOW_PV_KW, HIGH_PV_KW) for battery in BATTERIES_KWH)
CHUNK_SIZE = 10
DEFAULT_SOURCE = ROOT / "outputs/Brisbane" / PAIRED_RUN_NAME
DEFAULT_OUTPUT = (
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_2x2_"
    "PV3p25_PV6p5_Battery11p5_Battery14p5_15Y_1000Scenarios"
)
EXISTING_CSVS = {
    (3.25, 11.5): ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_"
    "PV3p25_Battery11p5_15Y_1000Scenarios/figures/Brisbane_FixedDesign_LOLP_Source_Data.csv",
    (3.25, 14.5): ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_"
    "PV3p25_Battery14p5_15Y_1000Scenarios/scenario_results.csv",
}
_WORKER: dict[str, Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label(pv: float, battery: float) -> str:
    return f"PV{str(pv).replace('.', 'p')}_Battery{str(battery).replace('.', 'p')}"


def _source_checks(source: dict[str, Any], source_dir: Path) -> tuple[pd.DataFrame, dict[tuple[float, float], Path]]:
    """Verify the retained designs against the canonical frozen paired run."""
    spec = source["spec"]
    if (spec.case_key != CASE_KEY or spec.analysis_years != ANALYSIS_YEARS
            or spec.scenario_count != SCENARIO_COUNT or spec.seed_start != 1):
        raise ValueError("Frozen source does not contain the expected Brisbane 1–1000, 15-year scenarios")
    paired = load_paired_results(source_dir / "scenario_results.csv")
    baseline = load_reference_records(EXISTING_CSVS[(LOW_PV_KW, 11.5)])
    validate_baseline_replay(baseline, paired)
    for battery in BATTERIES_KWH:
        path = EXISTING_CSVS[(LOW_PV_KW, battery)]
        saved = distribution.load_and_validate(path).sort_values("outage_seed").reset_index(drop=True)
        canonical = paired.loc[paired.battery_energy_kwh.eq(battery)].sort_values("outage_seed").reset_index(drop=True)
        if (len(saved) != SCENARIO_COUNT
                or not np.array_equal(saved.outage_seed.to_numpy(), canonical.outage_seed.to_numpy())
                or not np.allclose(saved.LOLP, canonical.LOLP, rtol=1e-10, atol=1e-9)):
            raise ValueError(f"Existing PV=3.25 kW, BESS={battery:g} kWh data differ from the paired source")
        if "outage_profile_sha256" in saved and not saved.outage_profile_sha256.equals(canonical.outage_profile_sha256):
            raise ValueError(f"Existing BESS={battery:g} kWh profile hashes differ from the paired source")
    return paired, EXISTING_CSVS


def _init_worker(case: InputCase, config: dict[str, Any]) -> None:
    frame, _ = build_case_frame(case, config, HIGH_PV_KW)
    keep = [
        "hour_index", "analysis_year", "electric_load_kwh", "cooling_load_kwh_th",
        "heating_load_kwh_th", "pv_kwh", "tou_sgd_per_kwh", "pv_export_sgd_per_kwh",
    ]
    _WORKER.clear()
    _WORKER.update(
        frame=frame[keep].copy(),
        pv_per_kw=frame["pv_kwh"].to_numpy(dtype=float) / HIGH_PV_KW,
        config=config,
    )


def _evaluate_profile(profile: dict[str, Any], pv_kw: float, batteries: tuple[float, ...]) -> list[dict[str, Any]]:
    frame = _WORKER["frame"].copy()
    availability = np.ones(len(frame), dtype=float)
    availability[np.asarray(profile["hour_indices"], dtype=int)] = np.asarray(profile["availability"], dtype=float)
    if _availability_hash(availability) != profile["outage_profile_sha256"]:
        raise ValueError(f"Frozen outage profile checksum mismatch for seed {profile['outage_seed']}")
    frame["grid_availability"] = availability
    frame["outage_fraction"] = 1.0 - availability
    frame["pv_kwh"] = _WORKER["pv_per_kw"] * pv_kw
    config = _WORKER["config"]
    rows: list[dict[str, Any]] = []
    for battery_kwh in batteries:
        sizing = battery_sizing(battery_kwh, config)
        dispatch = solve_dispatch_greedy(
            frame, sizing, config, metrics_only=True, streaming_only=True,
        )
        totals = dispatch.attrs["reliability_totals"]
        eue = float(totals["EUE_kwh_total"])
        demand = float(totals["outage_load_kwh_total"])
        if not (np.isfinite(eue) and np.isfinite(demand) and -1e-9 <= eue <= demand + 1e-6):
            raise ValueError(f"Invalid reliability totals for seed {profile['outage_seed']}")
        rows.append({
            "case_key": CASE_KEY,
            "scenario_id": int(profile["scenario_id"]),
            "outage_seed": int(profile["outage_seed"]),
            "outage_profile_sha256": str(profile["outage_profile_sha256"]),
            "analysis_years": ANALYSIS_YEARS,
            "pv_capacity_kw": pv_kw,
            "battery_energy_kwh": battery_kwh,
            "battery_power_kw": float(sizing["battery_power_kw"]),
            "EUE_kwh_total": eue,
            "EUE_kwh_per_year": eue / ANALYSIS_YEARS,
            "outage_load_kwh_total": demand,
            "LOLP": float(totals["LOLP"]),
            "no_outage_load": bool(totals["no_outage_load"]),
            "outage_hours_total": float(totals["outage_hours_total"]),
            "dispatch_backend": "greedy",
        })
    return rows


def _evaluate_chunk(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    return {
        "rows": [row for profile in profiles for row in _evaluate_profile(profile, HIGH_PV_KW, BATTERIES_KWH)],
        "worker_pid": os.getpid(),
        "compute_seconds": time.perf_counter() - started,
    }


def _spot_check_existing(source: dict[str, Any], paired: pd.DataFrame) -> dict[str, Any]:
    """Check that today's streaming dispatch reproduces saved baseline LOLP."""
    _init_worker(source["case"], source["config"])
    selected = {1, 500, 1000}
    rows = [
        row for profile in source["payloads"] if int(profile["outage_seed"]) in selected
        for row in _evaluate_profile(profile, LOW_PV_KW, BATTERIES_KWH)
    ]
    actual = pd.DataFrame(rows).sort_values(["battery_energy_kwh", "outage_seed"]).reset_index(drop=True)
    reference = paired.loc[
        paired.battery_energy_kwh.isin(BATTERIES_KWH) & paired.outage_seed.isin(selected)
    ].sort_values(["battery_energy_kwh", "outage_seed"]).reset_index(drop=True)
    if (len(actual) != len(selected) * len(BATTERIES_KWH)
            or not np.array_equal(actual.outage_seed, reference.outage_seed)
            or not np.array_equal(actual.battery_energy_kwh, reference.battery_energy_kwh)
            or not np.allclose(actual.LOLP, reference.LOLP, rtol=1e-10, atol=1e-9)):
        raise ValueError("Streaming dispatch did not reproduce saved 3.25 kW designs in the spot check")
    return {
        "seeds": sorted(selected), "designs": len(BATTERIES_KWH),
        "maximum_absolute_lolp_difference": float(np.max(np.abs(actual.LOLP - reference.LOLP))),
    }


def _chunk_path(output: Path, profiles: list[dict[str, Any]]) -> Path:
    return output / "checkpoints" / f"seeds_{profiles[0]['outage_seed']:04d}_{profiles[-1]['outage_seed']:04d}.json"


def _read_checkpoint(path: Path, fingerprint: str, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if envelope.get("fingerprint") != fingerprint or envelope.get("payload_sha256") != _digest(payload):
        raise ValueError(f"Checkpoint fingerprint or checksum mismatch: {path}")
    rows = pd.DataFrame(payload["rows"])
    expected_seeds = [int(profile["outage_seed"]) for profile in profiles]
    if (len(rows) != len(profiles) * len(BATTERIES_KWH)
            or not rows.groupby("outage_seed").size().reindex(expected_seeds, fill_value=0).eq(2).all()
            or set(rows.battery_energy_kwh) != set(BATTERIES_KWH)):
        raise ValueError(f"Checkpoint contains incomplete capacity or seed rows: {path}")
    return payload


def _validate_four_designs(paired: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    old = paired.loc[paired.battery_energy_kwh.isin(BATTERIES_KWH)].copy()
    columns = list(new_rows.columns)
    four = pd.concat([old[columns], new_rows], ignore_index=True)
    if len(four) != 4 * SCENARIO_COUNT or four.duplicated(["pv_capacity_kw", "battery_energy_kwh", "outage_seed"]).any():
        raise ValueError("Four-design result has missing or duplicate scenarios")
    for pv, battery in DESIGNS:
        group = four.loc[four.pv_capacity_kw.eq(pv) & four.battery_energy_kwh.eq(battery)]
        if (len(group) != SCENARIO_COUNT
                or not np.array_equal(np.sort(group.outage_seed.to_numpy(dtype=int)), np.arange(1, SCENARIO_COUNT + 1))
                or not group.analysis_years.eq(ANALYSIS_YEARS).all()
                or not np.allclose(group.LOLP, np.divide(
                    group.EUE_kwh_total, group.outage_load_kwh_total,
                    out=np.zeros(len(group)), where=group.outage_load_kwh_total.gt(0),
                ), rtol=1e-10, atol=1e-9)):
            raise ValueError(f"Invalid complete scenario set for PV={pv:g}, BESS={battery:g}")
    if not four.groupby("outage_seed").outage_profile_sha256.nunique().eq(1).all():
        raise ValueError("The four designs do not share identical frozen outage profiles")
    return four


def run(
    *, source_dir: Path | str = DEFAULT_SOURCE, output_dir: Path | str = DEFAULT_OUTPUT,
    workers: int = min(8, os.cpu_count() or 1), resume: bool = False,
) -> dict[str, Path]:
    if isinstance(workers, bool) or workers <= 0 or int(workers) != workers:
        raise ValueError("workers must be a positive integer")
    source_path = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if source_path == output:
        raise ValueError("Output cannot overwrite the frozen source")
    source = _load_enrichment_source(source_path)
    paired, existing_paths = _source_checks(source, source_path)
    spot_check = _spot_check_existing(source, paired)
    provenance = {
        "schema_version": 1,
        "source": source["identity"],
        "designs": DESIGNS,
        "new_pv_kw": HIGH_PV_KW,
        "new_batteries_kwh": BATTERIES_KWH,
        "chunk_size": CHUNK_SIZE,
        "existing_input_sha256": { _label(*key): _file_hash(path) for key, path in existing_paths.items() },
        "program_sha256": {
            name: _file_hash(ROOT / name) for name in (
                "src/fixed_design_four_way_evaluation.py", "src/dispatch.py", "src/optimizer.py",
                "src/io/data_loader.py", "src/figures/fixed_design_lolp_histogram_cdf.py",
            )
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    fingerprint = _digest(provenance)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"Output exists; use --resume: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resume refused because source, inputs, code or runtime changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Refusing to overwrite nonempty output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest_path, {"fingerprint": fingerprint, "created_at": _now(), **provenance})
    (output / "checkpoints").mkdir(exist_ok=True)
    profiles = sorted(source["payloads"], key=lambda item: item["outage_seed"])
    chunks = [profiles[start:start + CHUNK_SIZE] for start in range(0, len(profiles), CHUNK_SIZE)]
    completed: list[dict[str, Any]] = []
    pending: list[tuple[list[dict[str, Any]], Path]] = []
    worker_pids: set[int] = set()
    for chunk in chunks:
        path = _chunk_path(output, chunk)
        if path.exists():
            item = _read_checkpoint(path, fingerprint, chunk)
            completed.append(item)
            worker_pids.add(int(item["worker_pid"]))
        else:
            pending.append((chunk, path))
    started = time.perf_counter()
    resumed_chunks = len(completed)

    def status(state: str) -> None:
        _atomic_json(output / "run_status.json", {
            "state": state, "updated_at": _now(), "fingerprint": fingerprint,
            "workers_requested": int(workers), "worker_pids": sorted(worker_pids),
            "chunks_total": len(chunks), "chunks_completed": len(completed),
            "chunks_resumed": resumed_chunks, "new_scenario_rows_completed": len(completed) * CHUNK_SIZE * 2,
            "elapsed_seconds_this_attempt": time.perf_counter() - started,
        })

    def save(payload: dict[str, Any], path: Path) -> None:
        _atomic_json(path, {"fingerprint": fingerprint, "payload_sha256": _digest(payload), "payload": payload})
        completed.append(payload)
        worker_pids.add(int(payload["worker_pid"]))
        status("running")
        print(f"Completed {len(completed)}/{len(chunks)} chunks ({len(completed) * CHUNK_SIZE * 2}/2000 new design scenarios)", flush=True)

    status("running")
    if pending:
        if workers == 1:
            _init_worker(source["case"], source["config"])
            for chunk, path in pending:
                save(_evaluate_chunk(chunk), path)
        else:
            with ProcessPoolExecutor(
                max_workers=min(int(workers), len(pending)),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker, initargs=(source["case"], source["config"]),
            ) as executor:
                futures = {executor.submit(_evaluate_chunk, chunk): path for chunk, path in pending}
                for future in as_completed(futures):
                    save(future.result(), futures[future])
    _check_source_dependencies(source["identity"])
    if any(_file_hash(ROOT / name) != digest for name, digest in provenance["program_sha256"].items()):
        raise RuntimeError("Program changed during the run")
    new_rows = pd.DataFrame([row for item in completed for row in item["rows"]])
    new_rows = new_rows.sort_values(["pv_capacity_kw", "battery_energy_kwh", "outage_seed"]).reset_index(drop=True)
    four = _validate_four_designs(paired, new_rows)
    input_paths: dict[tuple[float, float], Path] = dict(existing_paths)
    for battery in BATTERIES_KWH:
        group = new_rows.loc[new_rows.battery_energy_kwh.eq(battery)].copy()
        path = output / f"{_label(HIGH_PV_KW, battery)}_scenario_results.csv"
        _atomic_csv(path, group)
        distribution.load_and_validate(path)
        input_paths[(HIGH_PV_KW, battery)] = path
    max_bin = max(
        float(combined_figure.fixed_histogram_bin_data(
            four.loc[four.pv_capacity_kw.eq(pv) & four.battery_energy_kwh.eq(battery), "LOLP"]
        ).frequency_percent.max())
        for pv, battery in DESIGNS
    )
    frequency_upper = min(100.0, max(40.0, math.ceil(max_bin / 10.0) * 10.0))
    summaries: list[dict[str, Any]] = []
    figure_paths: dict[str, Path] = {}
    for pv, battery in DESIGNS:
        design_rows = four.loc[four.pv_capacity_kw.eq(pv) & four.battery_energy_kwh.eq(battery)]
        label = _label(pv, battery)
        figures = combined_figure.run(
            input_paths[(pv, battery)], output / "figures" / label,
            frequency_upper_percent=frequency_upper,
        )
        figure_paths[label] = figures[f"{combined_figure.BASE_NAME}_png"]
        summaries.append({
            "design": label,
            "pv_capacity_kw": pv,
            "battery_energy_kwh": battery,
            "scenario_count": len(design_rows),
            "mean_LOLP": float(design_rows.LOLP.mean()),
            "median_LOLP": float(design_rows.LOLP.median()),
            "p95_LOLP": float(design_rows.LOLP.quantile(0.95)),
            "mean_EUE_kwh_per_year": float(design_rows.EUE_kwh_per_year.mean()),
            "LOLP_lt_20pct_passing_scenarios": int(design_rows.LOLP.lt(0.20).sum()),
            "CCP_LOLP_lt_20pct": float(design_rows.LOLP.lt(0.20).mean()),
            "reused_existing_simulation": pv == LOW_PV_KW,
            "source_csv": str(input_paths[(pv, battery)]),
            "combined_figure_png": str(figure_paths[label]),
        })
    summary_path = output / "four_design_summary.csv"
    _atomic_csv(summary_path, pd.DataFrame(summaries))
    readme_path = output / "README.md"
    readme_path.write_text(
        "# Brisbane 2×2 fixed-design LOLP comparison\n\n"
        "PV capacities are 3.25 and 6.5 kW; BESS capacities are 11.5 and 14.5 kWh. "
        "All four designs use the same frozen 1,000 outage seeds over 15 years. "
        "The PV=3.25 kW simulations are reused; only PV=6.5 kW is newly dispatched.\n\n"
        "New dispatch uses a streaming-only greedy path and saves no hourly or yearly tables. "
        "The figures are separate histogram-plus-CDF panels with a shared frequency axis. "
        "CDF uses P(LOLP ≤ x); the strict CCP column uses P(LOLP < 20%).\n\n"
        "PV=3.25 kW, BESS=14.5 kWh passes the strict 20% threshold in 889/1000 "
        "scenarios (88.9%), so it does not meet a 90% CCP target. The earlier near-equal "
        "annual-cost comparison uses 100 scenarios and is not recalculated here.\n",
        encoding="utf-8",
    )
    validation_path = output / "validation.json"
    report = {
        "passed": True, "created_at": _now(), "four_designs": [list(item) for item in DESIGNS],
        "scenario_count_per_design": SCENARIO_COUNT,
        "analysis_years": ANALYSIS_YEARS,
        "shared_outage_profile_hash_by_seed_verified": True,
        "existing_reliability_spot_check": spot_check,
        "new_dispatch": {"parallel_workers_requested": int(workers), "worker_pids": sorted(worker_pids),
                         "metrics_only": True, "streaming_only": True, "hourly_output": False,
                         "yearly_output": False, "new_scenario_rows": len(new_rows)},
        "figure_frequency_axis_upper_percent": frequency_upper,
        "maximum_histogram_bin_percent": max_bin,
        "summary_csv": str(summary_path),
        "figures_png": {key: str(path) for key, path in figure_paths.items()},
        "output_sha256": {path.name: _file_hash(path) for path in (
            summary_path, readme_path, *[input_paths[(HIGH_PV_KW, b)] for b in BATTERIES_KWH]
        )},
    }
    _atomic_json(validation_path, report)
    status("complete")
    return {"summary": summary_path, "validation": validation_path, **figure_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = run(source_dir=args.source_dir, output_dir=args.output_dir, workers=args.workers, resume=args.resume)
    print(f"Completed four-design comparison: {paths['validation']}")


if __name__ == "__main__":
    main()
