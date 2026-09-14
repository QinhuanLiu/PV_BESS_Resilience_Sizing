# PV-BESS Resilience Sizing

Standalone PV + BESS resilience-informed sizing workflow. This repository is an independent
input-driven project and does not modify any legacy reference project.

This version is input-first: all city/building cases are read from `Input/`, and synthetic city
load/PV generation is no longer used by the main workflow.

## Input Structure

Each city has one folder under `Input/`. Each city/building case must provide four CSV files with
the same `<City>_<Building>` prefix.

```text
Input/
  SG/
    SG_UTown_Load.csv
    SG_UTown_PV.csv
    SG_UTown_TOU.csv
    SG_UTown_Parameter.csv
```

The program scans all folders under `Input/` automatically. Missing `Load`, `PV`, `TOU`, or
`Parameter` files cause a validation error.

## File Formats

`<City>_<Building>_Load.csv`

- 8760 rows.
- Required columns: `Cooling`, `Heating`, `Lighting`, `Equipment`.
- `Cooling` and `Heating` use `kWh_th`; `Lighting` and `Equipment` use `kWh_e`.
- Optional `timestamp`; if omitted, the program generates hourly timestamps.

`<City>_<Building>_PV.csv`

- 8760 rows.
- Required column: `PV_output_kWh_per_m2`.
- The PV output is the hourly generation from 1 m2 of PV module area, in `kWh/m2/hour`.

`<City>_<Building>_TOU.csv`

- 24 rows.
- Required columns: `hour`, `tou_sgd_per_kwh`.
- `hour` must cover `0..23`; the 24-hour curve is repeated every day.

`<City>_<Building>_Parameter.csv`

Required columns:

```text
parameter,base_value,unit,sensitivity_start,sensitivity_end,sensitivity_step,description,中文说明
```

The first six rows must be the sensitivity parameters, in this exact order:

1. `pv_capex_sgd_per_kw`
2. `battery_capex_sgd_per_kwh`
3. `battery_power_capex_sgd_per_kw`
4. `saidi_min_per_year`
5. `saifi_per_year`
6. `voll_sgd_per_kwh`

Only these six parameters may define `sensitivity_start`, `sensitivity_end`, and
`sensitivity_step`. Sensitivity runs are single-factor: one parameter changes while all other
parameters remain at baseline. Ranges are inclusive. If a step is zero/negative, or end is below
start, validation fails.

Uncertainty analysis also requires the non-sensitivity parameters
`uncertainty_baseline_scenario_count`, `uncertainty_sensitivity_scenario_count`, and
`uncertainty_seed_start`. It can optionally use `parallel_workers` to run outage scenarios across
multiple processes. These are read from `Parameter.csv`, not from CLI flags.

PV rooftop coverage is not used as a sizing variable. PV is optimized directly by installed capacity
(`kW`). The coarse PV and BESS capacity ranges are generated automatically from the 8760 load and
1 m2 PV input, then refined near the lowest-cost region. The parameter file must include:

- `rooftop_area_m2`: effective rooftop area available for PV installation.
- `pv_module_power_density_kw_per_m2`: PV module rated power density, e.g. `0.20 kW/m2`.

The old parameters `base_pv_coverage_fraction`, `pv_capacity_kw_at_base_coverage`, and
`pv_coverage_candidates` are no longer accepted. The old manual candidate-list parameters
`pv_capacity_kw_candidates` and `battery_energy_kwh_candidates` are also no longer accepted in
`Parameter.csv`. For each candidate PV capacity `C_pv`, the model calculates PV as:

```text
PV_output_kWh_per_kw = PV_output_kWh_per_m2 / pv_module_power_density_kw_per_m2
pv_kwh = PV_output_kWh_per_kw * C_pv
pv_installed_area_m2 = C_pv / pv_module_power_density_kw_per_m2
pv_rooftop_coverage_fraction = pv_installed_area_m2 / rooftop_area_m2
```

`pv_rooftop_coverage_fraction` is retained only as a derived diagnostic field for rooftop-area
analysis; it does not define the PV sizing search grid.

The optimizer searches an automatically generated coarse `PV capacity x BESS energy capacity`
matrix, expands the upper bound if the best point is on the upper edge, then builds a regular
optimal-region grid around the final best point for refinement and plotting.

## SG Example

The repository includes:

```text
Input/SG/SG_UTown_Load.csv
Input/SG/SG_UTown_PV.csv
Input/SG/SG_UTown_TOU.csv
Input/SG/SG_UTown_Parameter.csv
```

For Singapore, `Heating = 0`. TOU uses the Keppel weekday-only curve for all days:

- 00:00-08:00: `0.239 SGD/kWh`
- 09:00-20:00: `0.329 SGD/kWh`
- 21:00-23:00: `0.239 SGD/kWh`

The PV file should represent 1 m2 of module generation. The PV and BESS search ranges are generated
from the SG load and PV files at runtime.

## Run

```bash
python3 pv_bess_sizing.py --input-dir Input
```

Default output directory is `outputs/`. Each case is written under its city folder:

- `outputs/<City>/<City>_<Building>_Baseline/`
- `outputs/<City>/<City>_<Building>_Sensitive_Analysis/`
- `outputs/<City>/<City>_<Building>_Uncertainty_Analysis/` when uncertainty modes are used.

Baseline runs also write `Baseline_Result_Figures/`, including single-outage Figure 3 design-space
plots at fixed PV capacities of 8, 20, and 32 MW, plus a Figure 4 annual-cost comparison between
the no-PV/no-BESS reference and the least-cost PV-BESS design. These figures export PNG/SVG/PDF
files with matching `_Source_Data.csv` tables.

The default annualized-cost metric is `pv_bess_system_annual_cost_sgd`, which equals
`pv_bess_private_annual_cost_sgd + unserved_energy_cost_sgd_per_year`. Baseline sizing reports a
single outage-resilience optimal design under the active outage profile and VoLL assumptions.
Per-candidate dispatch workbooks are disabled by default because writing one XLSX per capacity
candidate is I/O intensive; enable them only when needed with
`--save-candidate-dispatch-workbooks` or `outputs.save_candidate_dispatch_workbooks: true`.

To validate baseline cases only:

```bash
python3 pv_bess_sizing.py --input-dir Input --analysis baseline
```

To run only one case or one ordinary sensitivity parameter:

```bash
python3 pv_bess_sizing.py --input-dir Input --case LA_UTown --analysis baseline
python3 pv_bess_sizing.py --input-dir Input --case LA_UTown --analysis sensitivity --sensitivity-parameter voll_sgd_per_kwh
python3 pv_bess_sizing.py --input-dir Input --city LA --analysis sensitivity --sensitivity-parameter pv_capex_sgd_per_kw --sensitivity-parameter battery_capex_sgd_per_kwh
```

`--city`, `--case`, and sensitivity parameter options can be repeated. If both `--city` and
`--case` are provided, the program runs their intersection. `--skip-sensitivity` remains a
compatibility alias for `--analysis baseline`, and deprecated `--run-mode` commands are mapped to
the new `--analysis` / `--uncertainty-analysis` hierarchy.

## Uncertainty Analysis

Uncertainty analysis evaluates how random outage timing changes the optimal PV capacity, BESS
capacity, annualized system cost, and unserved-energy risk under the same SAIDI/SAIFI/VoLL inputs.
Full `--analysis all` includes uncertainty baseline and uncertainty sensitivity, so it can be much
slower than baseline and ordinary sensitivity runs. The scenario counts and seed start are read
from each case's `Parameter.csv`:

- `uncertainty_baseline_scenario_count`
- `uncertainty_sensitivity_scenario_count`
- `uncertainty_seed_start`
- `parallel_workers`, default `1`; set to `4`, `6`, or `8` to enable multiprocessing for outage
  scenarios.

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --analysis uncertainty --uncertainty-analysis baseline
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --analysis uncertainty --uncertainty-analysis sensitivity --uncertainty-sensitivity-parameter battery_capex_sgd_per_kwh
```

Uncertainty analysis uses common outage generation settings and re-optimizes the resilience sizing
design separately for each sampled outage scenario.
When `parallel_workers > 1`, outage scenarios are evaluated in separate worker processes and then
sorted back into deterministic scenario order before CSV output.

Main uncertainty outputs:

- `<Case>_Uncertainty_Analysis/<Case>_Uncertainty_Baseline/`
- `<Case>_Uncertainty_Analysis/<Case>_Uncertainty_Sensitive_Analysis/`
- Baseline CSVs for all results, outage scenarios, and distribution summary.
- Baseline PNG/SVG/PDF figures with matching `_Source_Data.csv` files:
  - `<Case>_Uncertainty_Baseline_Distribution_Violins` for total cost, PV capacity, BESS capacity,
    UE, and scenario LOLP across the resilience-informed outage scenarios.
  - `<Case>_Uncertainty_Baseline_Capacity_Cost_Bubble` for the PV-BESS capacity scatter, with point
    size scaled by annualized total system cost.
- Uncertainty sensitivity PNG figures with median lines and P10-P90 bands for annualized cost,
  BESS capacity, and PV capacity, separated into one figure per design case.

The uncertainty module intentionally does not output explanatory scatter plots, regret fields,
regret summary tables, or regret figures.

### Variable uncertainty analysis

Use `variable-uncertainty` when you want one configurable interface for random-outage uncertainty
analysis. It can run any discovered input case, override the random outage count, and optionally
apply an energy-based LOLP threshold:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --analysis variable-uncertainty --uncertainty-scenario-count 100
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --analysis variable-uncertainty --uncertainty-scenario-count 100 --lolp-threshold 0.10
```

When `--lolp-threshold` is omitted, each random outage scenario selects the least-cost PV/BESS
design. When it is provided, each scenario selects the least-cost design satisfying
`scenario_LOLP <= threshold`; infeasible scenarios are retained with `available = false`.
The same run also evaluates all configured candidate PV/BESS pairs across the sampled outages and
writes a cross-scenario candidate summary plus selected designs.

This mode writes `outputs/<City>/<Case>_Uncertainty_Analysis/<Case>_Variable_Uncertainty/`,
including scenario optima, outage scenarios, distribution summary, candidate-scenario results,
candidate summary, and selected-design CSV files.
It also exports five histogram + CDF distribution figures for total cost, optimal PV capacity,
optimal battery capacity, unserved energy, and LOLP, plus one combined five-panel figure. To redraw
those figures from existing CSV results without rerunning the uncertainty simulation:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --analysis variable-uncertainty-figures
```

### Manila LOLP=0 uncertainty analysis

For the Manila single-family typical year, run the 30-scenario zero-unserved-energy uncertainty
module with:

```bash
.venv/bin/python pv_bess_sizing.py --input-dir Input --analysis lolp-zero-uncertainty
```

This mode writes `outputs/Manila/Manila_SingleFamily_Uncertainty_Analysis/Manila_SingleFamily_LOLP_Zero_Uncertainty/`.
For each random outage scenario, it selects the minimum-cost PV/BESS design satisfying
`unserved_energy_kwh_per_year <= 1e-6`, then reports annualized cost, PV capacity, BESS energy,
and their coefficient of variation. It also exports separate histogram-density distribution
figures for annualized cost, PV capacity, and BESS energy.

### Figure 2 Manila three-year LOLP=0 uncertainty analysis

For Figure 2, run 30 continuous three-year Manila paths. Each path repeats the Manila
`Manila_SingleFamily` 8760-hour typical year three times, uses a different outage realization for
each simulated year, forces greedy dispatch, and selects the minimum mean annual cost design
satisfying zero unserved energy across the full three-year path:

```bash
.venv/bin/python pv_bess_sizing.py --input-dir Input --analysis lolp-zero-three-year-uncertainty --dispatch-backend greedy
```

This mode writes
`outputs/Manila/Manila_SingleFamily_Uncertainty_Analysis/Manila_SingleFamily_Figure2_LOLP_Zero_Three_Year_Uncertainty/`.
PV and BESS fixed costs remain annualized; three-year grid energy cost, unserved-energy cost, and
cycle aging are averaged per year before computing the reported annualized cost. It exports 30
optimal designs, 90 year-level outage scenario rows, a CV summary, and three histogram-density
distribution figures for annualized cost, PV capacity, and BESS energy.

## Resilience Frontier Figures

`--analysis frontier` writes publication-ready PNG figures and matching source-data CSVs for
the full-building-load resilience frontier. Optional SAIDI-VoLL heatmap figures are left as a
future extension because they require an additional two-dimensional computational sweep.

## Full-Load Resilience Framework

The frontier-based framework uses the full electric-equivalent building load as the outage service
target. It does not split load into critical and non-critical portions, and it does not use load
priority classes.

Reliability metrics use an energy-based LOLP definition. For one outage scenario,
`scenario_LOLP` is `UE_kWh / total outage-weighted electric-equivalent building load`; therefore
`outage_load_served_ratio` is retained as an auxiliary audit field and equals `1 - scenario_LOLP`
for that scenario. Across stochastic outage scenarios, `mean_scenario_LOLP` is the mean scenario LOLP,
and `chance_constraint_probability` is the share of scenarios with `scenario_LOLP <= lolp_threshold`.

Build the PV/BESS cost-reliability frontier with:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --analysis frontier
```

Run the lightweight representative single-scenario sensitivity figures with:

```bash
python3 pv_bess_sizing.py --input-dir Input --case SG_UTown --analysis representative-sensitivity
```

Frontier outputs are written to `outputs/<City>/<Case>_Resilience_Frontier/` and include candidate
metrics, scenario-level frontier results, selected optimal designs, outage scenarios, and the
frontier figures with matching `_Source_Data.csv` files.

Representative sensitivity outputs are written to
`outputs/<City>/<Case>_Representative_Sensitivity/` and include:

- `<Case>_Representative_Sensitivity_All_Results.csv`
- `<Case>_Representative_Sensitivity_Cost_Reliability.csv`
- Figure outputs with matching `_Source_Data.csv` files when those figures are generated.

Figure 1 summarizes the research framework. Figure 11 evaluates fixed-design PV and battery CAPEX
sensitivity. Figure 12 evaluates a deterministic SAIDI-VoLL sensitivity grid. These representative
sensitivity figures use one typical outage scenario and fixed PV-BESS capacity; they should not be
interpreted as replacements for the 100-scenario frontier results.
