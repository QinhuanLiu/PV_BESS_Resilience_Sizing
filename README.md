# PV-BESS Resilience Sizing Core

This repository is now a compact PV + BESS resilience sizing engine. It keeps the
input-driven baseline workflow and removes the previous paper-figure, uncertainty,
frontier, and reporting pipelines.

The archived full version is available at:

```bash
git checkout snapshot/pre-simplification-2026-09-14
```

## What The Core Engine Does

For each complete case under `Input/`, the workflow:

1. Reads load, PV, TOU, and parameter CSV files.
2. Builds an hourly annual case frame.
3. Simulates the configured outage profile from SAIDI/SAIFI assumptions.
4. Searches PV capacity and BESS energy candidates.
5. Runs outage-aware dispatch for each candidate.
6. Selects the least-cost resilience sizing result.
7. Writes baseline CSV outputs under `outputs/`.

## Project Layout

```text
.
├── pv_bess_sizing.py       # CLI entrypoint
├── config.yaml             # Small user override config
├── requirements.txt        # Runtime dependencies
├── Input/                  # Source case CSV files
├── src/
│   ├── app/                # CLI, workflow, paths, output writing
│   ├── io/                 # Config, input loading, reporting helpers
│   ├── sensitivity/        # Parameter catalogue, experiment planner, and runner
│   ├── design_evaluation.py# Shared one-case optimization/evaluation service
│   ├── figures/core.py     # Shared plotting style utilities for future use
│   ├── costing.py          # Annualized PV-BESS and outage cost calculations
│   ├── degradation.py      # Battery degradation and replacement economics
│   ├── design_selection.py # Reusable design-selection helpers
│   ├── dispatch.py         # Greedy and MILP dispatch engines
│   ├── optimizer.py        # PV/BESS candidate search and refinement
│   ├── outage.py           # Outage event/profile generation
│   ├── outage_evaluation.py# Fixed-candidate outage evaluation helpers
│   └── resilience_metrics.py
└── tools/
    └── smoke_core.py       # Lightweight core workflow check
```

## Input Structure

Each case folder under `Input/` must provide four CSV files with the same case prefix:

```text
Input/
  SG/
    SG_UTown_Load.csv
    SG_UTown_PV.csv
    SG_UTown_TOU.csv
    SG_UTown_Parameter.csv
```

Required file roles:

- `*_Load.csv`: 8760 hourly rows with `Cooling`, `Heating`, `Lighting`, and `Equipment`.
- `*_PV.csv`: 8760 hourly rows with `PV_output_kWh_per_m2`.
- `*_TOU.csv`: 24 hourly rows with `hour`, `tou_sgd_per_kwh`, and optional `pv_export_sgd_per_kwh`.
- `*_Parameter.csv`: case-specific technical, economic, outage, and sizing parameters.

Parameter files keep the standard 8-column CSV schema and group rows in this order: case identity
(`city_name`, `building_name`), PV parameters, battery parameters, and other system parameters.
A numeric system parameter can participate in sensitivity analysis when its row contains all three
of `sensitivity_start`, `sensitivity_end`, and `sensitivity_step`. Empty range cells mean that the
parameter stays fixed. The unified catalogue currently accepts model-used numeric PV, battery,
economic, reliability, and technical parameters; it excludes identity/text/boolean fields,
`outage_seed`, `analysis_years`, and local-search tuning controls.

The previous uncertainty/parallel parameters are no longer part of the core input table. The core
engine keeps internal defaults for future extensions, but baseline sizing does not read
`uncertainty_baseline_scenario_count`, `uncertainty_sensitivity_scenario_count`,
`uncertainty_seed_start`, or `parallel_workers` from `Parameter.csv`.

Capacity search is split into adaptive coarse search and optional local refinement:

- Coarse PV/BESS capacity candidates are always generated from the 8760 load/PV inputs and outage assumptions.
- When `pv_rooftop_area_limit_enabled=true`, the PV search upper bound is capped by `rooftop_area_m2 * pv_rooftop_area_utilization_fraction * pv_module_power_density_kw_per_m2`.
- The current input cases set `pv_rooftop_area_limit_enabled=true` and `pv_rooftop_area_utilization_fraction=0.8`.
- Manual coarse candidate lists such as `pv_capacity_kw_candidates` and `battery_energy_kwh_candidates` are not accepted in `Parameter.csv`.
- `local_refine_window_kw`, `local_refine_step_kw`, `local_refine_window_kwh`, and `local_refine_step_kwh` can override the local refinement window and step.
- If any local refinement value is absent, the engine derives it from the adaptive coarse step and rounds fallback steps to nice values.
- By default, local refinement uses coordinate local search rather than evaluating the full PV x BESS refinement grid.
- After coordinate search, a small validation grid around the final point is evaluated to catch nearby diagonal improvements.
- Set `capacity_search.local_search_method: grid` in config overrides to restore full local-grid refinement for comparison.

PV rooftop sizing parameters are controlled per case:

- `rooftop_area_m2` is the rooftop area available for PV installation.
- `pv_rooftop_area_utilization_fraction` is the usable fraction of that rooftop area.
- `pv_module_power_density_kw_per_m2` converts usable rooftop area into the PV capacity limit.
- If `pv_rooftop_area_limit_enabled` is absent in older inputs, the engine defaults to `true`; if `pv_rooftop_area_utilization_fraction` is absent, it defaults to `0.8`.

Capital cost annualization is controlled per case:

- `discount_rate` in `*_Parameter.csv` defines the real annual discount rate used to annualize PV and BESS capital costs.
- `pv_project_years` defines the PV capital recovery period.
- The BESS capital recovery period is derived from battery degradation and capped by `battery_max_replacement_years`.
- If `discount_rate` is absent, the engine defaults to `0`, which preserves the previous straight-line annualization.

PV export policy is controlled per case:

- `pv_export_enabled` in `*_Parameter.csv` controls whether surplus PV can be exported to the grid.
- `pv_export_sgd_per_kwh` in `*_TOU.csv` defines the hourly export price.
- If `pv_export_enabled` is absent, the engine defaults to `false`.
- If `pv_export_sgd_per_kwh` is absent, the engine defaults to `0`.

The current input cases explicitly set `pv_export_enabled=false` and `pv_export_sgd_per_kwh=0`,
so the default policy remains no PV export.

Multi-year baseline analysis is controlled per case:

- `analysis_years` in `*_Parameter.csv` defines how many years to simulate.
- If `analysis_years` is absent or blank, the engine defaults to `1`.
- Values must be positive integers.
- For `analysis_years>1`, the 8760-hour load, PV, and TOU inputs are repeated for each analysis year.
- Outage events are generated across the whole analysis horizon, so individual years can have different
  interruption counts and durations while the full-horizon random process is parameterized by the input
  SAIDI and SAIFI values.
- Optimization uses annual-average operating and outage costs, while PV and BESS capital costs remain
  annualized.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all discovered cases:

```bash
python3 pv_bess_sizing.py --input-dir Input
```

Run one case:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown
```

Run one city:

```bash
python3 pv_bess_sizing.py --input-dir Input --city SG
```

Force the greedy dispatch backend:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --dispatch-backend greedy
```

Write optional candidate dispatch workbooks:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --save-candidate-dispatch-workbooks
```

Run one parameter sensitivity (the default OAT design; every non-varied parameter remains at its base value):

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --sensitivity-parameter pv_capex_sgd_per_kw
```

Run several independent OAT sensitivities:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --sensitivity-parameter pv_capex_sgd_per_kw --sensitivity-parameter voll_sgd_per_kwh
```

Run every eligible parameter that has a complete range in the selected case's `Parameter.csv`:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --sensitivity-parameter all
```

Run a factorial interaction study (the final parameter is the fast axis in each warm-start chain):

```bash
python3 pv_bess_sizing.py --input-dir Input --case Brisbane_SingleFamily \
  --sensitivity-parameter pv_capex_sgd_per_kw \
  --sensitivity-parameter battery_capex_sgd_per_kwh \
  --sensitivity-mode factorial --sensitivity-workers 4
```

Sensitivity analysis is opt-in and does not write a normal `Baseline` folder. Each selected case
writes one sensitivity run directory containing a manifest, frozen input copies, checkpoints,
effective configuration, validation report, parameter catalogue, and `sensitivity_results.csv`.
Every run has exactly one baseline/reference row. OAT points always run complete searches. In
factorial mode, the first point in each chain runs a complete search and later points warm-start
from the prior point; use `--disable-sensitivity-warm-start` only with
`--sensitivity-mode factorial` to turn this off. The analysis horizon defaults to the case's
`analysis_years`; use `--sensitivity-analysis-years N` to override it. Factorial plans over 500
unique points require `--allow-large-sensitivity`.

The legacy fixed-grid figure modules can still read their historical grid results, but they do not
read the new generic sensitivity table yet.

Run TOU spread analysis:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --tou-analysis
```

TOU spread analysis is opt-in. It keeps the 24-hour average TOU price unchanged and changes the
price spread with `k`: `k=0` is a flat tariff, `k=1` is the input TOU profile, and `k>1` expands
the peak-valley spread. By default it runs `k = 0, 0.25, 0.5, 0.75, 1, 1.25`.

Specify custom TOU spread values:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --tou-spread-k 0 --tou-spread-k 1 --tou-spread-k 1.25
```

## Outputs

### Fixed-capacity, multi-scenario evaluation

Run the Bangkok SingleFamily 11 x 9 fixed-capacity grid with 100 shared 15-year outage
scenarios (seeds 42–141), using eight processes. All five requested metrics are calculated in
the first and only dispatch pass:

```bash
.venv/bin/python -m src.capacity_sweep --case Bangkok_SingleFamily --workers 8 \
  --include-economic-metrics
```

This separate entrypoint evaluates every requested pair using the existing greedy dispatch;
it does not optimize capacities. It overrides the analysis horizon in memory and preserves
the input files. Defaults are PV 0–10 kW in 1 kW steps and battery 0–20 kWh in 2.5 kWh steps.
Other grids, horizons, cases and seeds can be set with the options shown by `--help`.

The run produces 9,900 individual scenario results, 148,500 yearly results and 99 capacity
summaries under `outputs/Bangkok/Bangkok_SingleFamily_Capacity_Sweep_15Y_100Scenarios/`.
EUE is saved both as horizon-total kWh and annual-average kWh/year. LOLP retains the project's
energy-based definition: total unmet energy divided by total outage demand over the same
horizon. Each capacity uses the exact same 100 outage profiles. The output README documents
units, scenario sampling, zero-demand treatment and the fixed-capacity battery assumption.
`scenario_results.csv` also retains annual cost, SSR and SCR for each of the 9,900 records; no
full hourly dispatch tables are written.

Plot the completed sweep's mean annual EUE, mean energy-based LOLP and CCP without rerunning dispatch:

```bash
.venv/bin/python -m src.figures.capacity_sweep_contours
```

This recomputes the equal-weight means from the 9,900 raw scenario records and checks them
against the saved capacity summary. It retains the previous multi-city contour functions,
with battery energy on the horizontal axis and PV capacity on the vertical axis. The two
independent mean figures and a purple CCP figure are saved under the run's `figures/` directory as 600 dpi PNG, SVG and PDF,
alongside the 99-row source table and validation report. Use `--results-dir` to select a
completed sweep or `--output-dir` to choose a different figure destination.
CCP is the fraction of all scenarios at each capacity with LOLP strictly below 20%;
scenarios exactly at 20% do not pass. The source table includes the threshold, passing
scenario count and CCP as a 0–1 fraction; the plot uses a 0–100% colorbar.

Plot the empirical Q90(LOLP) value from the same 9,900 raw records:

```bash
.venv/bin/python -m src.figures.ccp90_lolp_contour
```

This adds a YlGnBu contour whose value at each capacity point is the 90th ascending
LOLP observation among its 100 paired scenarios. Its source table separately records
the strict chance constraint `P(LOLP < 20%) >= 90%`, including its pass/fail flag, so
an exact 20% value is never treated as a strict pass.

For the completed Brisbane fixed design (PV 3.25 kW, BESS 11.5 kWh), create a
LOLP frequency histogram and a standard empirical LOLP CDF directly
from its 1,000 raw fifteen-year scenario records:

```bash
.venv/bin/python -m src.figures.fixed_design_lolp_distribution \
  --input-csv /path/to/brisbane_fixed_3p25pv_11p5bess_1000seed_15yr_eue_lolp.csv
```

The module validates all 1,000 seeds and exports two standalone 600 dpi PNG, SVG and
PDF figures, ranked source data, histogram bin counts, CDF source data and a validation
report. The curve uses `F(x)=P(LOLP≤x)` as a right-continuous empirical CDF; LOLP remains
a 0–1 fraction in source data and is shown as a percentage. The fixed design is shown in
each figure's series legend rather than the title, supporting later comparisons.

To reproduce the retained 3.25 kW PV / 11.5 kWh BESS source and generate paired
1,000-scenario distribution figures for BESS=8.5 and 14.5 kWh, run:

```bash
.venv/bin/python -m src.fixed_design_paired_evaluation --workers 8
```

The command replays seeds 1–1000 over 15 years for BESS=8.5, 11.5 and 14.5 kWh using
the same outage profiles. It compares the recomputed 11.5 kWh LOLP against the retained
source seed by seed before publishing either new design. Each new design has its own output
directory containing 1,000 raw records, strict `P(LOLP < 20%)` CCP validation, a histogram,
and a standard empirical CDF. Use `--resume` after an interrupted matching paired run.

To reproduce the typical-outage energy-balance figure from that completed paired run:

```bash
.venv/bin/python -m src.figures.typical_outage_energy_balance
```

The figure uses seed 999, event 6, whose pre-storage net deficit is the median of the
run's 9,805 outage events. It replays the exact saved 15-year profile for 3.25 kW PV and
11.5 kWh BESS, then exports PNG, SVG, PDF, hourly source data and an event summary in
the paired run's `figures/` directory. The main panel accounts only for interrupted
load within the three affected model hours; SOC is shown at model-hour ends.

For the same median-deficit event, generate a Figure 5-style energy-balance view
covering the complete day before the outage, the outage day, and the recovery day:

```bash
.venv/bin/python -m src.figures.typical_outage_72h_energy_balance
```

This single-design 72-hour figure uses the frozen 3.25 kW PV / 11.5 kWh BESS
replay. Hourly positive bars show available PV, grid import and BESS discharge;
negative bars show served load, BESS charge and PV curtailment. The two sides
close at the AC bus. Battery conversion loss affects SOC and is retained in the
source data rather than stacked as a second AC load. The run exports PNG, SVG,
PDF, 72 hourly source rows and an event summary into the paired run's `figures/`
directory without replacing the existing short-outage figure.

For a separate 15-year Brisbane capacity optimization with the original seed 42,
temporarily set only the analysis horizon to 15 years and the battery charge/discharge
limit to 0.25C in memory, then draw its median-deficit outage:

```bash
.venv/bin/python -m src.brisbane_c025_15y_optimization
.venv/bin/python -m src.figures.brisbane_c025_optimized_outage_balance
```

The independent results directory is
`outputs/Brisbane/Brisbane_SingleFamily_15Y_C0p25_Optimization/`. It holds the
effective configuration, original input checksums, optimal design, all candidates,
15-year outage events/profile and hourly dispatch. Its `figures/` subdirectory
contains the selected event's Figure 5-style PNG, SVG, PDF, 72 hourly source rows
and selection summary. The selection minimizes distance to the median
pre-storage net deficit across this run's events; distances within 1e-9 kWh
tie and use the smaller event ID.
If the global optimum has no battery, the figure instead uses the cheapest
positive-battery candidate and labels that choice explicitly.

To compare the saved Greedy dispatch with the current MILP backend for that same
event 2 and fixed 3.5 kW PV / 11.6 kWh BESS design, run:

```bash
.venv/bin/python -m src.figures.brisbane_c025_milp_event2_balance
```

This solves only the selected 72 hours, starting at the saved Greedy SOC of
1.16 kWh and requiring the final SOC to be at least that value. The backend
named MILP currently contains no integer variables, so it is a continuous
linear optimization with full 72-hour foresight. Separate PNG, SVG, PDF, hourly
source data, event summary and Greedy comparison files are written to `figures/`;
the original 15-year optimization and Greedy figure are not changed. The
comparison records power-limit violations, same-hour bidirectional operation,
and direction reversals at a 0.01 kWh activity threshold. A boundary hour can
contain grid-available and outage minutes, so both directions in one model hour
do not necessarily mean simultaneous instantaneous operation.

To replay the same frozen 1,000 outage profiles for PV=5 kW, BESS=11.5 kWh and
export its distribution figures plus a four-design empirical CDF comparison, run:

```bash
.venv/bin/python -m src.fixed_design_paired_evaluation \
  --additional-design 5 11.5 --comparison-cdf --workers 8
```

The comparison overlays PV=3.25 kW/BESS=8.5, 11.5 and 14.5 kWh with PV=5 kW/
BESS=11.5 kWh. All four lines are unsmoothed right-continuous `F(x)=P(LOLP≤x)`
curves on shared 0–100% axes.

To generate one combined 5%-bin LOLP frequency histogram and empirical CDF for
each of the same four fixed designs, run:

```bash
.venv/bin/python -m src.figures.fixed_design_lolp_histogram_cdf
```

Each new dual-Y-axis figure uses a common 0–100% LOLP axis, a teal 0–40%
frequency axis and a blue 0–100% right-continuous CDF axis. The individual
histogram and CDF figures remain available unchanged.

To overlay the strict `LOLP < 20%` and `LOLP < 10%` chance-constraint probabilities
as two three-dimensional capacity surfaces, run:

```bash
.venv/bin/python -m src.figures.ccp_3d_surfaces
```

The output uses the completed Brisbane capacity sweep directly: BESS energy is the x axis,
PV capacity is the y axis, and CCP is the 0–100% z axis. It preserves the original 11×9
capacity grid, uses no smoothing or interpolation, and writes PNG, SVG, PDF, source data
and a separate validation report without replacing the two-dimensional contour figures.
Use `--threshold-pair 20-30` to export the analogous strict `LOLP < 20%` and
`LOLP < 30%` surface pair with independent filenames.

To show how strict `LOLP < 20%` CCP responds to BESS energy from 0 to 20 kWh for
PV capacities 2, 5 and 8 kW, run:

```bash
.venv/bin/python -m src.figures.ccp_battery_response
```

The resulting three marker-free curves use 200-sample PCHIP interpolation between the
nine original battery capacities, retain the 27 source points in a CSV, and do not
extrapolate outside the plotted 0–20 kWh range.

To generate the analogous PV=3, 6 and 9 kW figure with high-contrast blue, red and
purple curves, plus a star at PV=3 kW and BESS=15 kWh, run:

```bash
.venv/bin/python -m src.figures.ccp_battery_response --profile pv3-pv6-pv9
```

This profile uses the same strict `LOLP < 20%` definition, nine battery capacities,
and PCHIP procedure, while writing separate figure, source-data and validation files.

To show how strict `LOLP < 20%` CCP responds to PV capacity from 0 to 8 kW for
BESS capacities 10, 12.5 and 15 kWh, run:

```bash
.venv/bin/python -m src.figures.ccp_pv_response
```

The resulting three marker-free curves use 200-sample PCHIP interpolation between
the nine original PV capacities. They retain all 27 source points in a CSV and do
not extrapolate outside the 0--8 kW PV domain.

To compare the cost-minimising PV and BESS designs at CCP targets under fixed strict
LOLP constraints of 10%, 20% and 30%, using the completed 0.25 kW × 0.25 kWh adaptive
Brisbane sizing result, run:

```bash
.venv/bin/python -m src.figures.ccp_lolp_target_line_plots
```

The output is three independent English 16:9 figures with a shared 50%--95% CCP-target axis:
optimal PV capacity, optimal BESS energy, and mean annual system cost. Each figure shows one
marker-free PCHIP-smoothed curve for each LOLP constraint. The input is the completed adaptive
curve, whose final capacity resolution is 0.25 kW PV and 0.25 kWh BESS; it is not recomputed
from the 1 kW × 2.5 kWh coarse sweep. CCP uses the strict definition `P(LOLP < threshold)`, so
equality does not pass. The y axes use the data range plus 5% padding to make the capacity and
cost changes legible; smoothing connects the selected points without changing their values or
extrapolating beyond the requested CCP targets. The figure bundle contains 600 dpi PNG, SVG,
PDF, shared source data, README and validation report. Use `--sizing-results-dir` for another
completed fine sizing result, or `--use-coarse-sweep` only for exploratory coarse-grid plotting.

To show the denser 2%-CCP sizing result as discrete scatter points with descriptive
quadratic trends for BESS and annual cost, run:

```bash
.venv/bin/python -m src.figures.ccp_lolp_target_scatter_plots
```

To plot the evaluated incremental annual-cost--reliability frontier under the strict
`LOLP < 20%` condition, relative to the 3.25 kW PV and 11.5 kWh BESS global
minimum-cost reference design, run:

```bash
.venv/bin/python -m src.figures.incremental_cost_reliability_frontier
```

This export includes the original non-dominated-design scatter frontier and a companion
scatter-plus-power-law figure constrained through the origin. Both use only capacity
designs already evaluated by the adaptive 0.25 kW × 0.25 kWh sizing result. The fitted
curve is descriptive within the observed incremental-cost range; it neither interpolates
between capacity designs nor claims a continuous capacity optimum.

This produces three separate figures with 16:9 plotting areas for PV, BESS and mean
annual system cost. Each point is an optimal design from the completed 0.25 kW ×
0.25 kWh sizing result; the three strict LOLP constraints use the same blue, purple
and gold colours as the 5%-CCP figures. All 2%-CCP points are shown, while axis labels
are simplified. PV remains marker-only; BESS and annual cost add same-colour quadratic
least-squares trends limited to the observed CCP range. The fit coefficients and R² are
written to a separate source-data summary, not printed inside the figures.

Resume after interruption with the same options plus `--resume`. Completed task checkpoints
are verified by checksum, and input/configuration/program/runtime fingerprints must match.
The worker count may change between attempts. Failures are recorded without silently falling
back to serial execution; final outputs are marked complete only after validation passes.

For compatibility, annual system cost, SSR and SCR can also be added to an older completed
reliability-only run using its frozen inputs and exact outage sequences (without resampling):

```bash
.venv/bin/python -m src.capacity_sweep --workers 8 \
  --enrich-from outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios
```

The new sibling directory ends in `_Cost_SSR_SCR`; the source run is preserved. Add
`--resume` to reuse verified checkpoints. Scenario/grid/input overrides are rejected in
this mode. The lightweight dispatch accumulates cost and energy statistics without saving
complete hourly tables. `annual_cost_sgd_per_year` includes annualized investment, O&M,
net grid purchases and outage losses. SSR excludes grid-origin and initial battery energy;
SCR counts PV directly serving demand or charging the battery. Zero-PV SCR is blank and
explicitly labelled not applicable. The original EUE/LOLP columns are checked and retained.
Each output contains a Chinese metric guide and machine-readable validation report.

To plot the enriched records, reuse the contour entrypoint:

```bash
.venv/bin/python -m src.figures.capacity_sweep_contours \
  --results-dir outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios_Cost_SSR_SCR
```

When all enriched fields are present, this adds blue mean annual cost (kSGD/year),
orange mean SCR, and green mean SSR figures to the existing three resilience figures.
SCR and SSR use 0–100% colorbars. Zero-PV SCR remains undefined in the raw and source
mean columns, with zero defined scenarios. For display only, the SCR contour copies the
matching PV=1 kW value to each PV=0 kW row and records it in `SCR_display_mean`. All other
capacity means use the full 100 scenarios. Outputs are saved inside the enriched run's `figures/`
directory; the original run remains unchanged.

The default output root is `outputs/`. Each case writes a baseline folder:

```text
outputs/<City>/<Case>_Baseline/
```

Main files:

- `<Case>_Baseline_Timeseries_Max_PV_Capacity.csv`
- `<Case>_Baseline_Parameter_Table.csv`
- `<Case>_Baseline_Optimal_Design.csv`
- `<Case>_Baseline_Candidate_Summary.csv`
- `<Case>_Baseline_Dispatch.csv`
- `<Case>_Baseline_Outage_Profile.csv`
- `<Case>_Baseline_Outage_Events.csv`
- `<Case>_Baseline_Outage_Summary.csv`
- `<Case>_Baseline_Yearly_Summary.csv`

When PV export is enabled, dispatch outputs include `pv_export_kwh`; design summaries include
`pv_export_revenue_sgd_per_year` and `pv_bess_net_grid_cost_sgd`.

Cost outputs include `discount_rate`, `pv_capital_recovery_factor`, and
`battery_capital_recovery_factor` so annualized investment assumptions are visible in each result.

When sensitivity analysis is enabled, each case also writes:

```text
outputs/<City>/<Case>_Sensitivity_<mode>_<parameter-spec>_<years>Y/
```

The directory includes `sensitivity_results.csv`, `sensitivity_parameter_catalog.csv`,
`manifest.json`, `effective_config.json`, `run_status.json`, `validation.json`, frozen `inputs/`,
and verified `checkpoints/`. The result table has one row per experiment point, including the
baseline flag, actual values of every selected parameter, optimal PV/BESS sizing, annual cost, EUE,
LOLP, SSR, SCR, and warm-start diagnostics. A sensitivity-only command does not create the normal
baseline folder.

When TOU spread analysis is enabled, each case also writes:

```text
outputs/<City>/<Case>_TOU_Analysis/<Case>_TOU_Analysis_Summary.csv
```

The TOU analysis summary records `tou_spread_k`, the base and adjusted mean/min/max/spread prices,
optimal PV/BESS sizing, annual system cost, unserved energy, outage cost, LOLP, outage served ratio,
reliability assumptions, and search-stage diagnostics for each TOU spread run.

## Sensitivity Warm-Start Benchmark

An isolated experiment compares full search with capacity-only warm-start for the
`Brisbane_SingleFamily` battery energy CAPEX sensitivity range in its parameter CSV:

```bash
.venv/bin/python tools/benchmark_sensitivity_warm_start.py
```

The experiment uses greedy dispatch, the configured one-year horizon and seed 42,
and the existing search resolution. It evaluates the base price once, then follows
separate decreasing-price and increasing-price chains. Only PV/BESS capacities
pass between prices; dispatches and cost caches are not shared across prices.
Boundary fallback remains enabled and its full cost is included in measurements. This benchmark is
separate from the unified sensitivity runner, whose factorial chains use the same lower-level
capacity-only warm-start facility.

By default, three paired repetitions run in isolated serial processes in A/B,
B/A, A/B order. `--input-dir`, `--output-dir`, and `--repeats` can be supplied.
Results go to a new run directory under
`/tmp/pv_bess_brisbane_sensitivity_warm_start/`, containing final per-price rows,
`batch_performance.csv`, `point_comparison.csv`, `benchmark_summary.json`, and
`REPORT.md`. No candidate dispatch files are written. Timings include startup,
input reading, all prices, and batch CSV writing; report aggregation is excluded.
The report compares costs, capacities, reliability, actual dispatch calls, and
peak process memory. A 20% time reduction and 0.1% absolute relative cost
difference are preliminary screens, not guarantees of global optimality.

Validate the experiment's chain routing, real dispatch counters, and boundary
fallback using a small scenario without writing result files:

```bash
.venv/bin/python tools/smoke_sensitivity_warm_start.py
```

Validate the unified factorial runner with a tiny two-year 2 x 2 case:

```bash
.venv/bin/python tools/smoke_sensitivity_grid.py
```

## Smoke Check

Run a lightweight validation without writing into the repository:

```bash
python3 tools/smoke_core.py
```

The smoke check discovers input cases, validates the SG_UTown case can be loaded, and runs the
core workflow into `/tmp/pv_bess_core_smoke`.
