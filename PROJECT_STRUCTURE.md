# Project Structure

This project has been reduced to a core PV-BESS resilience sizing engine. The previous
uncertainty, frontier, legacy fixed-grid sensitivity, city-comparison, publication-figure, manuscript, and old test
pipelines were removed from this branch. The full pre-simplification version is preserved by the
Git tag `snapshot/pre-simplification-2026-09-14`.

## Active Source Tree

```text
src/
  app/
    cli.py        # core command-line interface and sensitivity routing
    workflow.py   # baseline and TOU case orchestration
    outputs.py    # baseline CSV/XLSX output writing
    paths.py      # output path helpers
  tou_analysis.py # mean-preserving TOU spread analysis helpers
  sensitivity/
    catalog.py    # eligible numeric system sensitivity parameters
    planning.py   # OAT/factorial experiment planning
    runner.py     # preflight, execution, checkpoints, and CSV outputs
  design_evaluation.py # shared one-case optimization/evaluation service
  io/
    config.py     # defaults and YAML overrides
    data_loader.py# input discovery, validation, case frame construction
    reporting.py  # small CSV/report helpers
  figures/
    core.py       # retained shared plotting/style utilities
  costing.py
  degradation.py
  design_selection.py
  dispatch.py
  optimizer.py
  outage.py
  outage_evaluation.py
  resilience_metrics.py
```

## Runtime Flow

```text
pv_bess_sizing.py
  -> src.app.cli.main()
      -> load config
      -> discover/filter input cases
      -> baseline/TOU workflow OR unified sensitivity runner
          -> build hourly frame
          -> simulate outage profile
          -> optimize PV/BESS design
          -> write baseline, TOU, or sensitivity outputs
```

## Kept Modules

| Area | Files | Role |
| --- | --- | --- |
| Application | `src/app/*` | CLI, baseline workflow, output paths, result writing |
| Sensitivity | `src/sensitivity/*` | Numeric parameter catalogue, OAT/factorial plans, preflight, checkpointed execution |
| Input/config | `src/io/*` | Discover cases, validate CSV inputs, load parameters, merge config |
| Simulation | `dispatch.py`, `outage.py`, `outage_evaluation.py` | Dispatch and outage generation/evaluation |
| Optimization | `optimizer.py`, `design_selection.py` | PV/BESS candidate search and reusable design selectors |
| Economics | `costing.py`, `degradation.py` | Annualized cost and battery lifetime economics |
| Metrics | `resilience_metrics.py` | Reliability and outage-load metrics |
| TOU analysis | `tou_analysis.py` | Mean-preserving TOU spread runs and summary rows |
| Figure foundation | `src/figures/core.py` | Shared style utilities retained for future plotting work |

## Removed From This Branch

- `tests/`
- `outputs/`
- analysis modules for uncertainty, variable uncertainty, LOLP=0, frontier, and representative sensitivity
- publication/result figure modules under `src/figures/`, except `core.py`
- manuscript/deck/helper scripts that were tied to the old paper-output workflow

## Current Public Interface

```bash
python3 pv_bess_sizing.py --input-dir Input
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown
python3 pv_bess_sizing.py --input-dir Input --city SG
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --dispatch-backend greedy
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --tou-analysis
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --tou-spread-k 0 --tou-spread-k 1.25
```

The core workflow keeps figure-selection and old paper-pipeline flags removed. Current opt-in
analysis flags are unified parameter sensitivity and TOU spread analysis. Sensitivity supports
repeated `--sensitivity-parameter` options (or `all`), OAT by default, and explicit factorial
interaction plans.
