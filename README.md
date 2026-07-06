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

The first five rows must be the sensitivity parameters, in this exact order:

1. `pv_capex_sgd_per_kwp`
2. `battery_capex_sgd_per_kwh`
3. `battery_power_capex_sgd_per_kw`
4. `saidi_min_per_year`
5. `voll_sgd_per_kwh`

Only these five parameters may define `sensitivity_start`, `sensitivity_end`, and
`sensitivity_step`. Sensitivity runs are single-factor: one parameter changes while all other
parameters remain at baseline. Ranges are inclusive. If a step is zero/negative, or end is below
start, validation fails.

Uncertainty analysis also requires the non-sensitivity parameters
`uncertainty_baseline_scenario_count`, `uncertainty_sensitivity_scenario_count`, and
`uncertainty_seed_start`. It can optionally use `parallel_workers` to run outage scenarios across
multiple processes. These are read from `Parameter.csv`, not from CLI flags.

PV coverage is no longer used as a sizing variable. PV is optimized directly by installed capacity
(`kWp`). The coarse PV and BESS capacity ranges are generated automatically from the 8760 load and
1 m2 PV input, then refined near the lowest-cost region. The parameter file must include:

- `rooftop_area_m2`: effective rooftop area available for PV installation.
- `pv_module_power_density_kwp_per_m2`: PV module rated power density, e.g. `0.20 kWp/m2`.

The old parameters `base_pv_coverage_fraction`, `pv_capacity_kwp_at_base_coverage`, and
`pv_coverage_candidates` are no longer accepted. The old manual candidate-list parameters
`pv_capacity_kwp_candidates` and `battery_energy_kwh_candidates` are also no longer accepted in
`Parameter.csv`. For each candidate PV capacity `C_pv`, the model calculates PV as:

```text
PV_output_kWh_per_kwp = PV_output_kWh_per_m2 / pv_module_power_density_kwp_per_m2
pv_kwh = PV_output_kWh_per_kwp * C_pv
pv_installed_area_m2 = C_pv / pv_module_power_density_kwp_per_m2
pv_rooftop_utilization_fraction = pv_installed_area_m2 / rooftop_area_m2
```

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

The default annualized-cost metric is `pv_bess_system_annual_cost_sgd`, which equals
`pv_bess_private_annual_cost_sgd + unserved_energy_cost_sgd_per_year`. Flexibility-only sizing
is still optimized without outages, but it is evaluated under the actual outage profile and VoLL.

To validate baseline cases only:

```bash
python3 pv_bess_sizing.py --input-dir Input --analysis baseline
```

To run only one case or one ordinary sensitivity parameter:

```bash
python3 pv_bess_sizing.py --input-dir Input --case LA_UTown --analysis baseline
python3 pv_bess_sizing.py --input-dir Input --case LA_UTown --analysis sensitivity --sensitivity-parameter voll_sgd_per_kwh
python3 pv_bess_sizing.py --input-dir Input --city LA --analysis sensitivity --sensitivity-parameter pv_capex_sgd_per_kwp --sensitivity-parameter battery_capex_sgd_per_kwh
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

For common random numbers, the same outage scenario is used to evaluate both design cases.
`Flexibility-only` is optimized once under no-outage conditions and then re-evaluated across all
outage scenarios. `Resilience-informed` is re-optimized separately for each outage scenario.
When `parallel_workers > 1`, outage scenarios are evaluated in separate worker processes and then
sorted back into deterministic scenario order before CSV output.

Main uncertainty outputs:

- `<Case>_Uncertainty_Analysis/<Case>_Uncertainty_Baseline/`
- `<Case>_Uncertainty_Analysis/<Case>_Uncertainty_Sensitive_Analysis/`
- Baseline CSVs for all results, outage scenarios, and distribution summary.
- Baseline PNG figures with matching `_Source_Data.csv` files for distributions and ECDF risk curves.
- Uncertainty sensitivity PNG figures with median lines and P10-P90 bands for annualized cost,
  BESS capacity, and PV capacity, separated into one figure per design case.

The uncertainty module intentionally does not output explanatory scatter plots, regret fields,
regret summary tables, or regret figures.
