"""Paired fixed-design Brisbane outage evaluation and LOLP distribution figures.

The module replays the 1,000 seeds retained in the published 3.25 kW / 11.5
kWh design source.  It evaluates 8.5, 11.5, and 14.5 kWh BESS designs in one
capacity sweep so every design sees exactly the same fifteen-year outages.
The 11.5 kWh results must reproduce the retained source on a seed-by-seed
basis before distribution figures for the two new designs are published.

Run with::

    python -m src.fixed_design_paired_evaluation

Use ``--resume`` to continue or regenerate a prior paired sweep whose
provenance still matches its saved checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .capacity_sweep import ROOT, SweepSpec, run_sweep
from .figures import fixed_design_lolp_distribution as distribution


CASE_KEY = "Brisbane_SingleFamily"
ANALYSIS_YEARS = 15
SCENARIO_COUNT = 1000
SEED_START = 1
PV_CAPACITY_KW = 3.25
BASELINE_BATTERY_KWH = 11.5
TARGET_BATTERIES_KWH = (8.5, 14.5)
ALL_BATTERIES_KWH = (8.5, BASELINE_BATTERY_KWH, 14.5)
LOLP_THRESHOLD = 0.20
BASELINE_RTOL = 1e-10
BASELINE_ATOL = 1e-9

DEFAULT_REFERENCE_CSV = (
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_"
    "PV3p25_Battery11p5_15Y_1000Scenarios/figures/Brisbane_FixedDesign_LOLP_Source_Data.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/Brisbane"
PAIRED_RUN_NAME = (
    "Brisbane_SingleFamily_FixedDesign_PV3p25_"
    "Battery8p5_Battery11p5_Battery14p5_15Y_1000Scenarios_PairedReplay"
)
TARGET_DIRECTORY_NAMES = {
    8.5: "Brisbane_SingleFamily_FixedDesign_PV3p25_Battery8p5_15Y_1000Scenarios",
    14.5: "Brisbane_SingleFamily_FixedDesign_PV3p25_Battery14p5_15Y_1000Scenarios",
}
ADDITIONAL_DESIGN = (5.0, 11.5)
ADDITIONAL_REPLAY_NAME = (
    "Brisbane_SingleFamily_FixedDesign_PV5_Battery11p5_15Y_1000Scenarios_ProfileReplay"
)
ADDITIONAL_TARGET_NAME = "Brisbane_SingleFamily_FixedDesign_PV5_Battery11p5_15Y_1000Scenarios"
REQUIRED_PAIRED_COLUMNS = (
    "case_key",
    "scenario_id",
    "outage_seed",
    "outage_profile_sha256",
    "analysis_years",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "LOLP",
)


def file_sha256(path: Path) -> str:
    """Return the checksum of a delivered input or output file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _same_capacity(values: pd.Series, expected: float) -> bool:
    return bool(np.allclose(values.to_numpy(dtype=float), expected, rtol=0.0, atol=1e-12))


def load_reference_records(reference_csv: Path | str) -> pd.DataFrame:
    """Load the published 11.5 kWh source and enforce its fixed identity."""
    records = distribution.load_and_validate(reference_csv)
    if not _same_capacity(records["pv_capacity_kw"], PV_CAPACITY_KW):
        raise ValueError(f"Reference data must use PV={PV_CAPACITY_KW:g} kW")
    if not _same_capacity(records["battery_energy_kwh"], BASELINE_BATTERY_KWH):
        raise ValueError(f"Reference data must use BESS={BASELINE_BATTERY_KWH:g} kWh")
    return records.sort_values("outage_seed").reset_index(drop=True)


def load_paired_results(path: Path | str) -> pd.DataFrame:
    """Load and validate the 3,000 records emitted by the paired replay."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Paired scenario result CSV does not exist: {source}")
    results = pd.read_csv(source, float_precision="round_trip")
    missing = sorted(set(REQUIRED_PAIRED_COLUMNS) - set(results.columns))
    if missing:
        raise ValueError(f"Paired scenario results are missing required columns: {missing}")
    expected_rows = SCENARIO_COUNT * len(ALL_BATTERIES_KWH)
    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} paired scenario rows, found {len(results)}")
    required = results.loc[:, list(REQUIRED_PAIRED_COLUMNS)]
    if required.isna().any().any():
        raise ValueError("Paired scenario results contain missing required values")
    numeric_columns = [
        "scenario_id", "outage_seed", "analysis_years", "pv_capacity_kw",
        "battery_energy_kwh", "LOLP",
    ]
    if not np.isfinite(results[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Paired scenario results contain non-finite numeric values")
    if not results["case_key"].eq(CASE_KEY).all():
        raise ValueError("Paired scenario results have the wrong case key")
    if not _same_capacity(results["pv_capacity_kw"], PV_CAPACITY_KW):
        raise ValueError(f"Paired scenario results must use PV={PV_CAPACITY_KW:g} kW")
    if not results["analysis_years"].eq(ANALYSIS_YEARS).all():
        raise ValueError(f"Paired scenario results must cover {ANALYSIS_YEARS} years")
    if np.any((results["LOLP"] < 0.0) | (results["LOLP"] > 1.0)):
        raise ValueError("Paired scenario LOLP values must be fractions in [0, 1]")
    if not np.allclose(results["scenario_id"], np.rint(results["scenario_id"]), rtol=0.0, atol=1e-12):
        raise ValueError("scenario_id values must be integers")
    if not np.allclose(results["outage_seed"], np.rint(results["outage_seed"]), rtol=0.0, atol=1e-12):
        raise ValueError("outage_seed values must be integers")
    expected_batteries = np.asarray(ALL_BATTERIES_KWH, dtype=float)
    actual_batteries = np.sort(results["battery_energy_kwh"].unique().astype(float))
    if not np.array_equal(actual_batteries, expected_batteries):
        raise ValueError("Paired scenario results do not contain exactly 8.5, 11.5 and 14.5 kWh")

    expected_seeds = np.arange(SEED_START, SEED_START + SCENARIO_COUNT, dtype=int)
    expected_ids = np.arange(1, SCENARIO_COUNT + 1, dtype=int)
    for battery in ALL_BATTERIES_KWH:
        subset = results[np.isclose(results["battery_energy_kwh"], battery, rtol=0.0, atol=1e-12)]
        if len(subset) != SCENARIO_COUNT:
            raise ValueError(f"BESS={battery:g} kWh does not have {SCENARIO_COUNT} rows")
        if not np.array_equal(np.sort(subset["outage_seed"].astype(int).to_numpy()), expected_seeds):
            raise ValueError(f"BESS={battery:g} kWh does not contain each required outage seed exactly once")
        if not np.array_equal(np.sort(subset["scenario_id"].astype(int).to_numpy()), expected_ids):
            raise ValueError(f"BESS={battery:g} kWh does not contain each scenario id exactly once")
    profile_counts = results.groupby("outage_seed", sort=True)["outage_profile_sha256"].nunique()
    if not profile_counts.eq(1).all():
        raise ValueError("A paired outage seed maps to different outage profile hashes")
    row_counts = results.groupby("outage_seed", sort=True).size()
    if not row_counts.eq(len(ALL_BATTERIES_KWH)).all():
        raise ValueError("Each outage seed must be represented once for every BESS design")
    return results.sort_values(["battery_energy_kwh", "outage_seed"]).reset_index(drop=True)


def load_single_design_replay(
    path: Path | str, *, pv_capacity_kw: float, battery_kwh: float
) -> pd.DataFrame:
    """Load one replayed design and require its complete 1--1000 seed identity."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Replay scenario result CSV does not exist: {source}")
    records = pd.read_csv(source, float_precision="round_trip")
    missing = sorted(set(REQUIRED_PAIRED_COLUMNS) - set(records.columns))
    if missing:
        raise ValueError(f"Replay scenario results are missing required columns: {missing}")
    if len(records) != SCENARIO_COUNT:
        raise ValueError(f"Expected {SCENARIO_COUNT} replay scenario rows, found {len(records)}")
    required = records.loc[:, list(REQUIRED_PAIRED_COLUMNS)]
    if required.isna().any().any():
        raise ValueError("Replay scenario results contain missing required values")
    numeric = ["scenario_id", "outage_seed", "analysis_years", "pv_capacity_kw", "battery_energy_kwh", "LOLP"]
    if not np.isfinite(records[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Replay scenario results contain non-finite numeric values")
    if not records["case_key"].eq(CASE_KEY).all() or not records["analysis_years"].eq(ANALYSIS_YEARS).all():
        raise ValueError("Replay scenario results have an unexpected case or analysis horizon")
    if not _same_capacity(records["pv_capacity_kw"], pv_capacity_kw):
        raise ValueError(f"Replay scenario results must use PV={pv_capacity_kw:g} kW")
    if not _same_capacity(records["battery_energy_kwh"], battery_kwh):
        raise ValueError(f"Replay scenario results must use BESS={battery_kwh:g} kWh")
    if np.any((records["LOLP"] < 0.0) | (records["LOLP"] > 1.0)):
        raise ValueError("Replay scenario LOLP values must be fractions in [0, 1]")
    expected = np.arange(SEED_START, SEED_START + SCENARIO_COUNT, dtype=int)
    for column in ("scenario_id", "outage_seed"):
        values = records[column].to_numpy(dtype=float)
        if not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-12):
            raise ValueError(f"Replay {column} values must be integers")
        if not np.array_equal(np.sort(np.rint(values).astype(int)), expected):
            raise ValueError(f"Replay {column} must contain each integer from 1 through 1000 exactly once")
    if records["outage_profile_sha256"].duplicated().all():
        raise ValueError("Replay scenario results have invalid outage profile identities")
    return records.sort_values("outage_seed").reset_index(drop=True)


def validate_shared_profile_replay(paired_results: pd.DataFrame, replay_records: pd.DataFrame) -> dict[str, Any]:
    """Prove each new-design seed reuses precisely the paired run's outage profile."""
    source = paired_results.loc[:, ["outage_seed", "outage_profile_sha256"]].drop_duplicates()
    source = source.sort_values("outage_seed").reset_index(drop=True)
    target = replay_records.loc[:, ["outage_seed", "outage_profile_sha256"]].sort_values("outage_seed").reset_index(drop=True)
    if not source.equals(target):
        raise ValueError("Replay outage profile SHA-256 values do not match the frozen paired source")
    return {
        "verified": True,
        "comparison": "outage_profile_sha256 by outage_seed",
        "scenario_count": int(len(target)),
        "outage_seed_range": [int(target.outage_seed.min()), int(target.outage_seed.max())],
    }


def validate_baseline_replay(reference: pd.DataFrame, paired_results: pd.DataFrame) -> dict[str, Any]:
    """Compare the 11.5 kWh paired replay against the published source by seed."""
    baseline = paired_results[np.isclose(
        paired_results["battery_energy_kwh"], BASELINE_BATTERY_KWH, rtol=0.0, atol=1e-12
    )].loc[:, ["outage_seed", "LOLP"]].sort_values("outage_seed").reset_index(drop=True)
    expected = reference.loc[:, ["outage_seed", "LOLP"]].sort_values("outage_seed").reset_index(drop=True)
    if not np.array_equal(baseline["outage_seed"].to_numpy(dtype=int), expected["outage_seed"].to_numpy(dtype=int)):
        raise ValueError("Baseline replay does not cover exactly the reference outage seeds")
    differences = baseline["LOLP"].to_numpy(dtype=float) - expected["LOLP"].to_numpy(dtype=float)
    passed = bool(np.allclose(
        baseline["LOLP"].to_numpy(dtype=float), expected["LOLP"].to_numpy(dtype=float),
        rtol=BASELINE_RTOL, atol=BASELINE_ATOL,
    ))
    report = {
        "passed": passed,
        "comparison": "LOLP by outage_seed",
        "scenario_count": int(len(baseline)),
        "relative_tolerance": BASELINE_RTOL,
        "absolute_tolerance": BASELINE_ATOL,
        "maximum_absolute_difference": float(np.abs(differences).max()),
        "mean_absolute_difference": float(np.abs(differences).mean()),
    }
    if not passed:
        raise ValueError(
            "The 11.5 kWh baseline replay differs from the retained source; "
            f"maximum absolute LOLP difference is {report['maximum_absolute_difference']:.3g}"
        )
    return report


def target_records(paired_results: pd.DataFrame, battery_kwh: float) -> pd.DataFrame:
    """Return one complete fixed-design table suitable for the figure module."""
    records = paired_results[np.isclose(
        paired_results["battery_energy_kwh"], battery_kwh, rtol=0.0, atol=1e-12
    )].copy()
    if len(records) != SCENARIO_COUNT:
        raise ValueError(f"BESS={battery_kwh:g} kWh target extraction is incomplete")
    return records.sort_values("outage_seed").reset_index(drop=True)


def strict_ccp_validation(records: pd.DataFrame) -> dict[str, Any]:
    """Calculate the requested strict P(LOLP < 20%) requirement without plotting it."""
    passing = int(records["LOLP"].lt(LOLP_THRESHOLD).sum())
    count = int(len(records))
    ccp = passing / count
    return {
        "definition": "P(LOLP < 0.20); values exactly equal to 0.20 do not pass",
        "comparison": "<",
        "lolp_threshold": LOLP_THRESHOLD,
        "scenario_count": count,
        "passing_scenarios": passing,
        "CCP": ccp,
        "constraint": "CCP > 0.90",
        "constraint_satisfied": bool(ccp > 0.90),
    }


def _target_readme(
    *, battery_kwh: float, paired_run_dir: Path, reference_csv: Path, ccp: dict[str, Any]
) -> str:
    verdict = "满足" if ccp["constraint_satisfied"] else "不满足"
    return f"""# Brisbane fixed-design LOLP distribution

This directory contains the fixed PV=3.25 kW, BESS={battery_kwh:g} kWh result for
1,000 shared fifteen-year outage scenarios.  The paired simulation also evaluated
the retained BESS=11.5 kWh design and verified its scenario-by-scenario LOLP against
`{reference_csv.name}` before these figures were exported.

- The common paired replay is in `../{paired_run_dir.name}`.
- `scenario_results.csv` contains one unaggregated result for each outage seed.
- `figures/` contains a 5-percentage-point LOLP histogram and a right-continuous
  empirical CDF, both styled with the shared contour/core figure settings.
- The CDF uses `P(LOLP ≤ x)`.  The capacity reliability check is separate and uses
  the requested strict `P(LOLP < 20%)` definition.

For this design, {ccp['passing_scenarios']}/{ccp['scenario_count']} scenarios pass
the strict 20% threshold, giving CCP={ccp['CCP']:.1%}; it {verdict} CCP > 90%.
"""


def _target_directory(output_root: Path, battery_kwh: float) -> Path:
    try:
        return output_root / TARGET_DIRECTORY_NAMES[float(battery_kwh)]
    except KeyError as exc:
        raise ValueError(f"Unsupported target battery capacity: {battery_kwh}") from exc


def _assert_output_is_available(directory: Path, resume: bool) -> None:
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ValueError(f"Output directory already exists; use --resume: {directory}")


def _has_completed_paired_run(directory: Path) -> bool:
    """Return whether the saved shared replay is complete and independently validated."""
    status_path = directory / "run_status.json"
    validation_path = directory / "validation.json"
    results_path = directory / "scenario_results.csv"
    if not all(path.is_file() for path in (status_path, validation_path, results_path)):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Completed paired replay metadata is invalid: {directory}") from exc
    delivered_hashes = validation.get("file_sha256")
    if not isinstance(delivered_hashes, dict) or not delivered_hashes:
        return False
    files_match = all(
        (directory / name).is_file() and file_sha256(directory / name) == digest
        for name, digest in delivered_hashes.items()
    )
    return bool(status.get("state") == "complete" and validation.get("passed") is True and files_match)


def _publish_target(
    *,
    battery_kwh: float,
    records: pd.DataFrame,
    target_dir: Path,
    paired_run_dir: Path,
    reference_csv: Path,
    reference_sha256: str,
    paired_results_sha256: str,
    paired_manifest_sha256: str,
    baseline_validation: dict[str, Any],
) -> dict[str, Path]:
    """Write one target's raw data, validation record and publication figures."""
    target_dir.mkdir(parents=True, exist_ok=True)
    source_path = target_dir / "scenario_results.csv"
    _atomic_csv(source_path, records)
    distribution.load_and_validate(source_path)
    ccp = strict_ccp_validation(records)
    _target_readme_path = target_dir / "README.md"
    _target_readme_path.write_text(
        _target_readme(
            battery_kwh=battery_kwh, paired_run_dir=paired_run_dir,
            reference_csv=reference_csv, ccp=ccp,
        ),
        encoding="utf-8",
    )
    figure_paths = distribution.run(source_path, target_dir / "figures")
    report_path = target_dir / "fixed_design_paired_evaluation_validation.json"
    report = {
        "passed": True,
        "created_at": _utc_now(),
        "target_design": {"pv_capacity_kw": PV_CAPACITY_KW, "battery_energy_kwh": battery_kwh},
        "analysis_years": ANALYSIS_YEARS,
        "scenario_count": SCENARIO_COUNT,
        "shared_scenarios": {
            "verified": True,
            "outage_seed_range": [SEED_START, SEED_START + SCENARIO_COUNT - 1],
            "outage_profile_identity": "SHA-256 verified across BESS=8.5, 11.5 and 14.5 kWh",
            "paired_replay_directory": str(paired_run_dir),
            "paired_scenario_results_sha256": paired_results_sha256,
            "paired_manifest_sha256": paired_manifest_sha256,
        },
        "baseline_replay": {
            "reference_csv": str(reference_csv),
            "reference_csv_sha256": reference_sha256,
            **baseline_validation,
        },
        "strict_ccp_LOLP_lt_20": ccp,
        "figure_module": "src.figures.fixed_design_lolp_distribution",
        "figures": {key: str(path) for key, path in figure_paths.items()},
        "output_sha256": {
            "scenario_results.csv": file_sha256(source_path),
            "README.md": file_sha256(_target_readme_path),
            **{path.name: file_sha256(path) for path in figure_paths.values()},
        },
    }
    _atomic_json(report_path, report)
    return {
        "scenario_results": source_path,
        "readme": _target_readme_path,
        "validation": report_path,
        **{f"figure_{key}": path for key, path in figure_paths.items()},
    }


def run(
    *,
    reference_csv: Path | str = DEFAULT_REFERENCE_CSV,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    workers: int | None = None,
    resume: bool = False,
    sweep_runner: Callable[..., dict[str, Any]] = run_sweep,
) -> dict[str, dict[str, Path]]:
    """Replay the paired run, certify the baseline, and publish the two targets."""
    reference_path = Path(reference_csv).resolve()
    root = Path(output_root).resolve()
    reference = load_reference_records(reference_path)
    reference_sha256 = file_sha256(reference_path)
    paired_run_dir = root / PAIRED_RUN_NAME
    target_dirs = {battery: _target_directory(root, battery) for battery in TARGET_BATTERIES_KWH}
    _assert_output_is_available(paired_run_dir, resume)
    for directory in target_dirs.values():
        _assert_output_is_available(directory, resume)

    requested_workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    if requested_workers <= 0 or int(requested_workers) != requested_workers:
        raise ValueError("workers must be a positive integer")
    spec = SweepSpec(
        case_key=CASE_KEY,
        analysis_years=ANALYSIS_YEARS,
        scenario_count=SCENARIO_COUNT,
        seed_start=SEED_START,
        pv_capacities=(PV_CAPACITY_KW,),
        battery_capacities=ALL_BATTERIES_KWH,
    )
    if not (resume and _has_completed_paired_run(paired_run_dir)):
        sweep_runner(spec, paired_run_dir, workers=int(requested_workers), resume=resume)
    paired_results_path = paired_run_dir / "scenario_results.csv"
    paired_results = load_paired_results(paired_results_path)
    try:
        baseline_validation = validate_baseline_replay(reference, paired_results)
    except ValueError as exc:
        diagnostic = {
            "passed": False,
            "created_at": _utc_now(),
            "reference_csv": str(reference_path),
            "reference_csv_sha256": reference_sha256,
            "reason": str(exc),
        }
        _atomic_json(paired_run_dir / "baseline_replay_validation.json", diagnostic)
        raise
    _atomic_json(paired_run_dir / "baseline_replay_validation.json", {
        "passed": True,
        "created_at": _utc_now(),
        "reference_csv": str(reference_path),
        "reference_csv_sha256": reference_sha256,
        **baseline_validation,
    })

    paired_manifest = paired_run_dir / "manifest.json"
    if not paired_manifest.is_file():
        raise ValueError("Paired replay did not produce its provenance manifest")
    outputs: dict[str, dict[str, Path]] = {}
    for battery in TARGET_BATTERIES_KWH:
        outputs[f"battery_{battery:g}_kwh"] = _publish_target(
            battery_kwh=battery,
            records=target_records(paired_results, battery),
            target_dir=target_dirs[battery],
            paired_run_dir=paired_run_dir,
            reference_csv=reference_path,
            reference_sha256=reference_sha256,
            paired_results_sha256=file_sha256(paired_results_path),
            paired_manifest_sha256=file_sha256(paired_manifest),
            baseline_validation=baseline_validation,
        )
    return outputs


def run_additional_design(
    *,
    additional_design: tuple[float, float] = ADDITIONAL_DESIGN,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    workers: int | None = None,
    resume: bool = False,
    sweep_runner: Callable[..., dict[str, Any]] = run_sweep,
) -> dict[str, Path]:
    """Replay one extra PV/BESS design from the completed frozen paired profiles."""
    pv_capacity_kw, battery_kwh = (float(additional_design[0]), float(additional_design[1]))
    if (pv_capacity_kw, battery_kwh) != ADDITIONAL_DESIGN:
        raise ValueError("The approved additional design is PV=5 kW, BESS=11.5 kWh")
    root = Path(output_root).resolve()
    paired_run_dir = root / PAIRED_RUN_NAME
    if not _has_completed_paired_run(paired_run_dir):
        raise ValueError("A completed, validated three-BESS paired replay is required before the additional design")
    paired_results_path = paired_run_dir / "scenario_results.csv"
    paired_results = load_paired_results(paired_results_path)
    replay_dir = root / ADDITIONAL_REPLAY_NAME
    target_dir = root / ADDITIONAL_TARGET_NAME
    _assert_output_is_available(replay_dir, resume)
    _assert_output_is_available(target_dir, resume)
    requested_workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    if requested_workers <= 0 or int(requested_workers) != requested_workers:
        raise ValueError("workers must be a positive integer")
    spec = SweepSpec(
        case_key=CASE_KEY, analysis_years=ANALYSIS_YEARS, scenario_count=SCENARIO_COUNT,
        seed_start=SEED_START, pv_capacities=(pv_capacity_kw,), battery_capacities=(battery_kwh,),
    )
    if not (resume and _has_completed_paired_run(replay_dir)):
        sweep_runner(spec, replay_dir, workers=int(requested_workers), resume=resume, replay_from=paired_run_dir)
    replay_path = replay_dir / "scenario_results.csv"
    records = load_single_design_replay(replay_path, pv_capacity_kw=pv_capacity_kw, battery_kwh=battery_kwh)
    profile_validation = validate_shared_profile_replay(paired_results, records)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_path = target_dir / "scenario_results.csv"
    _atomic_csv(source_path, records)
    distribution.load_and_validate(source_path)
    ccp = strict_ccp_validation(records)
    readme_path = target_dir / "README.md"
    verdict = "满足" if ccp["constraint_satisfied"] else "不满足"
    readme_path.write_text(
        f"""# Brisbane fixed-design LOLP distribution

This directory contains PV={pv_capacity_kw:g} kW, BESS={battery_kwh:g} kWh for
1,000 shared fifteen-year outage scenarios. The exact outage profiles, seeds and
SHA-256 identities were replayed from `../{paired_run_dir.name}`.

- `scenario_results.csv` is the unaggregated result table.
- `figures/` contains the shared-style 5-percentage-point LOLP histogram and
  right-continuous empirical CDF `P(LOLP ≤ x)`.
- The separate capacity check uses strict `P(LOLP < 20%)`: {ccp['passing_scenarios']}/
  {ccp['scenario_count']} = {ccp['CCP']:.1%}; it {verdict} CCP > 90%.
""", encoding="utf-8"
    )
    figure_paths = distribution.run(source_path, target_dir / "figures")
    paired_manifest = paired_run_dir / "manifest.json"
    report_path = target_dir / "fixed_design_additional_evaluation_validation.json"
    report = {
        "passed": True,
        "created_at": _utc_now(),
        "target_design": {"pv_capacity_kw": pv_capacity_kw, "battery_energy_kwh": battery_kwh},
        "analysis_years": ANALYSIS_YEARS,
        "scenario_count": SCENARIO_COUNT,
        "shared_scenarios": {
            **profile_validation,
            "frozen_source_directory": str(paired_run_dir),
            "frozen_source_scenario_results_sha256": file_sha256(paired_results_path),
            "frozen_source_manifest_sha256": file_sha256(paired_manifest),
            "replay_directory": str(replay_dir),
            "replay_scenario_results_sha256": file_sha256(replay_path),
        },
        "strict_ccp_LOLP_lt_20": ccp,
        "figure_module": "src.figures.fixed_design_lolp_distribution",
        "figures": {key: str(path) for key, path in figure_paths.items()},
        "output_sha256": {
            "scenario_results.csv": file_sha256(source_path),
            "README.md": file_sha256(readme_path),
            **{path.name: file_sha256(path) for path in figure_paths.values()},
        },
    }
    _atomic_json(report_path, report)
    return {
        "scenario_results": source_path, "readme": readme_path, "validation": report_path,
        **{f"figure_{key}": path for key, path in figure_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV,
                        help="Existing PV=3.25 kW, BESS=11.5 kWh 1,000-scenario source CSV")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Directory that will hold the paired run and two fixed-design output folders")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true",
                        help="Resume a matching paired replay and regenerate its delivered artifacts")
    parser.add_argument("--additional-design", nargs=2, type=float, metavar=("PV_KW", "BESS_KWH"),
                        help="Replay the approved extra fixed design from the frozen paired outage profiles")
    parser.add_argument("--comparison-cdf", action="store_true",
                        help="After the additional design, export the four-design combined empirical CDF")
    args = parser.parse_args()
    if args.comparison_cdf and args.additional_design is None:
        parser.error("--comparison-cdf requires --additional-design 5 11.5")
    if args.additional_design is None:
        outputs = run(
            reference_csv=args.reference_csv,
            output_root=args.output_root,
            workers=args.workers,
            resume=args.resume,
        )
        for label, paths in outputs.items():
            print(f"Published {label}: {paths['figure_validation'].parent}")
    else:
        paths = run_additional_design(
            additional_design=(args.additional_design[0], args.additional_design[1]),
            output_root=args.output_root, workers=args.workers, resume=args.resume,
        )
        print(f"Published PV=5 kW, BESS=11.5 kWh: {paths['figure_validation'].parent}")
        if args.comparison_cdf:
            from .figures.fixed_design_lolp_cdf_comparison import run as run_comparison
            comparison = run_comparison(output_dir=Path(args.output_root) / (
                "Brisbane_SingleFamily_FixedDesign_LOLP_CDF_Comparison_1000Scenarios/figures"
            ))
            print(f"Published four-design CDF comparison: {comparison['validation'].parent}")


if __name__ == "__main__":
    main()
