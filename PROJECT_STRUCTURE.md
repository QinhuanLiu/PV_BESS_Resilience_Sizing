# Project Structure And Module Map

This document summarizes the current PV-BESS resilience sizing project after recent additions and
module moves. It describes what each active module does and where the remaining structural clutter
comes from.

## Big Picture

The current codebase is an input-driven PV + BESS sizing workflow:

1. CSV case folders under `Input/` are discovered automatically.
2. Per-case parameters are merged into the default config from `src/io/config.py`.
3. An 8760-hour load/PV/TOU frame is built.
4. Outages are simulated from SAIDI/SAIFI assumptions.
5. PV and BESS candidates are evaluated by dispatch simulation and annualized cost calculation.
6. Baseline, sensitivity, uncertainty, resilience-frontier, and publication-figure outputs are
   written under `outputs/`.

The top-level script `pv_bess_sizing.py` is intentionally thin. The real program starts in
`src/app/cli.py`, then delegates to `src/app/workflow.py`.

## Current Directory Roles

```text
.
├── pv_bess_sizing.py                 # CLI entrypoint wrapper
├── config.yaml                       # Small user override config
├── README.md                         # User-facing run and input-format guide
├── requirements.txt                  # Python dependencies
├── manuscript_core_body_no_intro.md  # Manuscript text artifact
├── Input/                            # Source CSV case data
├── outputs/                          # Generated results and figures; ignored by git
├── src/                              # Application and analysis code
├── tests/                            # Pytest test suite
└── tools/                            # One-off utilities for documents, decks, and support outputs
```

There are also four currently empty directories:

```text
src/analysis/
src/economics/
src/optimization/
src/simulation/
```

These look like a planned layered architecture, but the active code has not been moved into them.
For now, they are structural noise unless they are kept as placeholders for a future refactor.

## Entry And Orchestration

| Module | Role |
| --- | --- |
| `pv_bess_sizing.py` | Imports and exposes the CLI, selected workflow functions, and ordinary sensitivity parameter list. Running this file calls `src.app.cli.main()`. |
| `src/app/cli.py` | Defines command-line arguments, maps deprecated flags to the current `--analysis` model, loads config, resolves input/output directories, and calls `run_selected()`. |
| `src/app/workflow.py` | Main orchestration layer. Discovers/filters cases, runs baseline optimization, ordinary sensitivity, uncertainty, LOLP=0 modes, resilience frontier, representative sensitivity, result figures, city-comparison figures, and figure archiving. |
| `src/app/paths.py` | Centralizes output folder naming for each case and analysis mode through `CaseOutputPaths`. |
| `src/app/outputs.py` | Writes CSV/XLSX outputs and calls uncertainty/LOLP figure writers. It is the output persistence layer for workflow results. |

Typical call flow:

```text
pv_bess_sizing.py
  -> src.app.cli.main()
      -> src.io.config.load_config()
      -> src.app.workflow.run_selected()
          -> src.io.data_loader.load_input_case()
          -> src.outage.simulate_outages()
          -> src.optimizer.resilience_design()
          -> src.app.outputs.write_*()
          -> src.figures.* figure builders/writers
```

## Input And Reporting Modules

| Module | Role |
| --- | --- |
| `src/io/config.py` | Holds `DEFAULT_CONFIG` and merges optional YAML overrides. This is where global defaults for inputs, outputs, dispatch, capacity search, technical assumptions, PV, BESS, economics, outage, uncertainty, and legacy built-in scenarios live. |
| `src/io/data_loader.py` | Discovers complete input cases, reads Load/PV/TOU/Parameter CSV files, validates formats, parses parameter/sensitivity ranges, builds per-case runtime config, generates adaptive PV/BESS search grids, and builds the 8760-hour case frame. |
| `src/io/reporting.py` | Builds parameter-source tables and compact case design summaries, and writes CSV files with parent-directory creation. |

Input case convention:

```text
Input/<City_or_Case_Folder>/
  <Case>_Load.csv
  <Case>_PV.csv
  <Case>_TOU.csv
  <Case>_Parameter.csv
```

Current input case families:

```text
Bangkok_SingleFamily
Brisbane_SingleFamily
Jakarta_SingleFamily
Manila_SingleFamily
SG_Apartment_Apartment
SG_SingleFamily
SG_UTown
```

`Jakarta_hourly_loads_kWh.csv` is handled as a special supported load-file name.

## Core Calculation Modules

| Module | Role |
| --- | --- |
| `src/dispatch.py` | Hourly dispatch engine. Provides greedy dispatch and MILP dispatch through PuLP/HiGHS, including PV-to-load, PV-to-battery, grid-to-load, grid-to-battery, battery SOC, unmet load, and outage-aware availability. |
| `src/optimizer.py` | PV/BESS candidate evaluation and search. Adds candidate caching, evaluates dispatch plus cost for each PV/BESS pair, expands coarse grids when the optimum hits the upper edge, refines around the best region, and returns optimal design rows plus candidate summaries. |
| `src/costing.py` | Annualizes PV cost, calls battery degradation/costing, computes grid bill, unserved-energy cost using VoLL, private annual cost, and total annual system cost. |
| `src/degradation.py` | Estimates annual battery state-of-health loss from throughput and calendar aging, estimates replacement interval, and annualizes BESS cost. |
| `src/outage.py` | Generates outage events and hourly grid availability profiles from SAIDI/SAIFI, including deterministic and night-window modes. |
| `src/outage_evaluation.py` | Reusable outage-scenario evaluation layer. Generates scenario tables, rehydrates saved outage seeds, computes outage diagnostics, and evaluates fixed PV/BESS candidates against a scenario. |
| `src/resilience_metrics.py` | Small metric library for electric-equivalent load, outage-weighted load, unserved energy, served ratio, scenario LOLP, mean LOLP, and chance-constraint probability. |
| `src/design_selection.py` | Selects named designs from candidate tables: minimum total cost, affordable backup, and reliability-constrained designs. Also provides helpers for availability flags. |

## Analysis Modules

| Module | Role |
| --- | --- |
| `src/uncertainty.py` | Runs stochastic outage uncertainty analysis. For each sampled outage scenario, re-optimizes PV/BESS, records scenario diagnostics, and summarizes distributions for cost, capacity, and unserved energy. |
| `src/lolp_zero_uncertainty.py` | Manila single-year LOLP=0 analysis. For each outage scenario, selects the minimum-cost design whose unserved energy is effectively zero, then reports coefficient-of-variation summaries. |
| `src/lolp_zero_three_year_uncertainty.py` | Manila three-year LOLP=0 analysis. Repeats the typical year into continuous multi-year dispatch paths, optimizes for zero unserved energy across the whole path, and annualizes multi-year operating terms. |
| `src/resilience_frontier.py` | Builds a resilience frontier by evaluating candidate PV/BESS designs across many outage scenarios, aggregating reliability/cost metrics, and selecting minimum-cost, affordable-backup, and reliability-constrained designs. |
| `src/representative_sensitivity.py` | Evaluates a selected frontier reference design under representative CAPEX and SAIDI/VoLL multiplier sweeps. |

## Figure Package

`src/figures/` is the largest and messiest area. It mixes reusable plotting infrastructure,
source-data builders, publication figure modules, and special-purpose city-comparison figure
pipelines.

### Figure Infrastructure

| Module | Role |
| --- | --- |
| `src/figures/core.py` | Shared Matplotlib style, figure saving, source-data writing, layout validation, tick helpers, and plotting helpers. |
| `src/figures/display_units.py` | Converts raw kW/kWh/SGD/kWh values into publication-friendly display units and labels. |
| `src/figures/names.py` | Central constants for result figure numbers, sensitivity parameter labels, and legacy design-name normalization. |
| `src/figures/registry.py` | Dispatches requested result figure numbers to the correct figure module. |
| `src/figures/sources.py` | Builds or reads source-data tables required by result figures. |
| `src/figures/versioning.py` | Archives generated figure files into per-case version folders and classifies figure-like files. |
| `src/figures/baseline.py` | Compatibility re-export for baseline design-space and LOLP plotting modules. |

### Baseline Figures

| Module | Role |
| --- | --- |
| `src/figures/baseline_design_space.py` | Builds and plots baseline PV/BESS design-space surfaces, contours, and PV-capacity slices. |
| `src/figures/baseline_cost_comparison.py` | Compares no-PV/no-BESS, PV-only/reference, and selected PV-BESS annualized cost components. |
| `src/figures/baseline_outage_balance.py` | Builds outage-event balance comparisons for selected baseline designs and dispatches fixed designs around a representative outage window. |
| `src/figures/baseline_lolp.py` | Builds LOLP/BESS slice and LOLP constraint scan figures from baseline candidate summaries. |

### Result / Publication Figures

| Module | Role |
| --- | --- |
| `src/figures/reliability_trajectory.py` | Publication figure for cost-reliability trajectory across designs and design-space context. |
| `src/figures/pareto.py` | Pareto frontier source transformation and plotting for cost vs. unserved energy / resilience. |
| `src/figures/sensitivity.py` | Ordinary one-factor sensitivity comparison figures. |
| `src/figures/saifi_saidi.py` | SAIFI-SAIDI response surfaces and contour maps. |
| `src/figures/uncertainty.py` | Uncertainty baseline violin distributions and capacity-cost bubble figures. |
| `src/figures/robust_uncertainty.py` | Robust uncertainty figure set: baseline distributions, robust sensitivity bands, mechanism plots, and tail robustness. |
| `src/figures/tail_resilience.py` | Tail-risk resilience distribution/exceedance/lollipop style publication figures. |
| `src/figures/robustness_matrix.py` | Robustness matrix figures showing capacities and reliability feasibility across probability/threshold settings. |
| `src/figures/ccp_bess_response.py` | Chance-constraint probability / BESS capacity response figure. |
| `src/figures/bess_interpretation.py` | Figure explaining BESS capacity interpretation using event, capacity-slice, and summary sources. |

### Outage Operation Figures

| Module | Role |
| --- | --- |
| `src/figures/outage_operation.py` | Builds/plots operational dispatch behavior for selected outage windows. |
| `src/figures/outage_balance.py` | Builds/plots outage energy-balance context figures and shared helpers reused by other figure modules. |

### Special Figure Pipelines

| Module | Role |
| --- | --- |
| `src/figures/city_comparison.py` | Very large city-comparison pipeline. It creates synthetic apartment-city sources or real single-family city-comparison sources, then plots multiple city comparison figures. This is currently the biggest module and mixes data construction, analysis, caching, and plotting. |
| `src/figures/single_outage_city_baseline.py` | Very large four-city single-outage baseline pipeline. It builds many related source datasets and figures for affordability, LOLP=0, day/night outages, TOU profiles, cost distributions, energy balance, scenario distributions, CVaR, and Brisbane sensitivity. This is the largest module. |
| `src/figures/singlefamily_preliminary_result.py` | Preliminary single-family figure pipeline. Reuses many helpers from `single_outage_city_baseline.py` for night-affordable LOLP contours, energy balance, TOU profiles, and LOLP=0 sensitivity. |
| `src/figures/lolp_zero_uncertainty.py` | Figure writers for single-year LOLP=0 uncertainty distributions. |
| `src/figures/lolp_zero_three_year_uncertainty.py` | Figure writer wrapper for three-year LOLP=0 uncertainty distributions. |
| `src/figures/lolp_zero_multiyear_synthetic.py` | Synthetic multi-year LOLP=0 distribution and CV-trend figure utilities. It appears to be a standalone/supplemental figure workflow rather than part of the main CLI path. |

## Tools

| File | Role |
| --- | --- |
| `tools/create_sg_singlefamily_staging.py` | Utility for preparing/staging SG single-family input data. |
| `tools/affordable_backup_city_cost_breakdown.py` | Utility for affordable-backup city cost-breakdown outputs. |
| `tools/build_sci_manuscript_docx.py` | Builds a scientific manuscript Word document artifact. |
| `tools/build_group_meeting_deck.mjs` | Builds a group-meeting slide deck using Node tooling. |

`tools/__pycache__/` is generated cache output and should remain ignored/cleaned locally.

## Tests

The test suite is broad and maps closely to the current modules:

| Test area | Main coverage |
| --- | --- |
| `tests/test_core.py` | Config, data loading, dispatch, costing, degradation, optimizer, outputs, uncertainty basics. |
| `tests/test_resilience_metrics.py` | Core resilience metric formulas. |
| `tests/test_outage_evaluation.py` | Outage scenario generation/evaluation behavior. |
| `tests/test_resilience_frontier.py` | Frontier generation and design selection. |
| `tests/test_representative_sensitivity.py` | Representative sensitivity workflows. |
| `tests/test_design_selection.py` | Named design selectors and availability handling. |
| `tests/test_reserve_sensitivity.py` | Dispatch reserve behavior. |
| `tests/test_lolp_zero_uncertainty.py` | LOLP=0 single-year and three-year analysis paths plus related workflow calls. |
| `tests/test_*figures.py` and `tests/test_publication_*.py` | Figure source-data validation, plotting behavior, and publication layout checks. |

## Current Structural Issues

1. `src/figures/` has become a mixed-purpose package. Some files are clean plotting modules, but
   `city_comparison.py` and `single_outage_city_baseline.py` are large end-to-end pipelines.
2. Analysis concepts are split between root-level modules (`src/uncertainty.py`,
   `src/resilience_frontier.py`, etc.) and figure modules that also compute source data.
3. Empty directories (`src/analysis`, `src/economics`, `src/optimization`, `src/simulation`) suggest
   a future architecture but are not currently used.
4. Git status shows old modules were moved or removed (`src/config.py`, `src/data_loader.py`,
   `src/reporting.py`, old visualization files), while the active replacements live under
   `src/io/` and `src/figures/`. Until committed, this looks noisy.
5. `outputs/` is large but generated and ignored. It should not be treated as source structure.

## Suggested Future Organization

If you want to clean the codebase further, the lowest-risk target structure would be:

```text
src/
  app/              # CLI, workflow orchestration, paths, output writing
  io/               # Config, input validation/loading, reporting helpers
  simulation/       # Dispatch and outage generation/evaluation
  optimization/     # Candidate search, design selection, resilience frontier
  economics/        # Costing and degradation
  analysis/         # Uncertainty, LOLP=0, representative sensitivity
  figures/          # Plot-only modules plus shared figure style/source IO
```

The safest cleanup order:

1. Remove or populate the empty placeholder directories.
2. Move `dispatch.py`, `outage.py`, and `outage_evaluation.py` into `src/simulation/`.
3. Move `optimizer.py`, `design_selection.py`, and `resilience_frontier.py` into `src/optimization/`.
4. Move `costing.py` and `degradation.py` into `src/economics/`.
5. Move `uncertainty.py`, `lolp_zero_*.py`, and `representative_sensitivity.py` into `src/analysis/`.
6. Split the two largest figure pipelines into `source_builders`, `plotters`, and `workflow`
   submodules before making deeper behavior changes.

That refactor should be done in small commits with import-compatibility shims or coordinated test
updates, because the current tests import many modules directly.
