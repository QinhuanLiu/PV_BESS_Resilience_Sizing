from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any

from ..io.data_loader import CaseParameters
from .catalog import is_numeric_system_parameter, numeric_system_parameter_names, parameter_definition


ALL_SENSITIVITY_PARAMETERS = "all"
SENSITIVITY_MODES = ("oat", "factorial")
DEFAULT_MAX_RUNS = 500


@dataclass(frozen=True)
class SensitivitySpec:
    parameters: tuple[str, ...]
    mode: str = "oat"
    analysis_years: int | None = None
    workers: int = 1
    allow_large: bool = False
    warm_start_enabled: bool = True

    def validate(self) -> None:
        if not self.parameters:
            raise ValueError("Sensitivity analysis requires at least one --sensitivity-parameter.")
        if self.mode not in SENSITIVITY_MODES:
            raise ValueError(f"Unknown sensitivity mode {self.mode!r}. Available modes: {list(SENSITIVITY_MODES)}")
        if self.analysis_years is not None and self.analysis_years <= 0:
            raise ValueError("Sensitivity analysis years must be a positive integer.")
        if self.workers <= 0:
            raise ValueError("Sensitivity workers must be a positive integer.")


@dataclass(frozen=True)
class SensitivityPoint:
    run_id: str
    execution_index: int
    overrides: dict[str, float]
    is_baseline: bool
    chain_id: str
    chain_position: int


def normalize_requested_parameters(
    requested: list[str] | tuple[str, ...] | None,
    parameters: CaseParameters,
) -> tuple[str, ...]:
    values = [str(value).strip() for value in (requested or []) if str(value).strip()]
    if not values:
        return ()
    if any(value.lower() == ALL_SENSITIVITY_PARAMETERS for value in values):
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with named sensitivity parameters.")
        values = [name for name in parameters.sensitivity if is_numeric_system_parameter(name)]
    selected: list[str] = []
    unknown: list[str] = []
    unavailable: list[str] = []
    for name in values:
        if not is_numeric_system_parameter(name):
            unknown.append(name)
        elif not parameters.sensitivity.get(name):
            unavailable.append(name)
        elif name not in selected:
            selected.append(name)
    if unknown:
        available = ", ".join(numeric_system_parameter_names())
        raise ValueError(f"Unknown or non-system sensitivity parameter(s): {', '.join(sorted(set(unknown)))}. Available: {available}")
    if unavailable:
        raise ValueError(
            "Sensitivity parameter(s) need complete sensitivity_start/end/step values in Parameter.csv: "
            + ", ".join(sorted(set(unavailable)))
        )
    if not selected:
        raise ValueError("No selected sensitivity parameter has a configured numeric range.")
    return tuple(selected)


def build_sensitivity_plan(parameters: CaseParameters, spec: SensitivitySpec) -> list[SensitivityPoint]:
    spec.validate()
    selected = tuple(spec.parameters)
    for name in selected:
        if not is_numeric_system_parameter(name):
            raise ValueError(f"{name} is not a numeric system sensitivity parameter.")
        if not parameters.sensitivity.get(name):
            raise ValueError(f"Sensitivity parameter {name} has no configured range.")
        _base_value(parameters, name)

    base = {name: _base_value(parameters, name) for name in selected}
    combinations: list[dict[str, float]]
    if spec.mode == "oat":
        combinations = []
        for name in selected:
            for value in parameters.sensitivity[name]:
                overrides = dict(base)
                overrides[name] = float(value)
                combinations.append(overrides)
    else:
        combinations = [
            dict(zip(selected, values))
            for values in product(*(tuple(float(value) for value in parameters.sensitivity[name]) for name in selected))
        ]

    deduplicated: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    base_key = _value_key(base, selected)
    for values in combinations:
        key = _value_key(values, selected)
        if key not in seen and key != base_key:
            seen.add(key)
            deduplicated.append(values)
    point_count = 1 + len(deduplicated)
    if spec.mode == "factorial" and point_count > DEFAULT_MAX_RUNS and not spec.allow_large:
        raise ValueError(
            f"Sensitivity plan contains {point_count} unique runs, exceeding the {DEFAULT_MAX_RUNS}-run limit. "
            "Pass --allow-large-sensitivity to confirm."
        )

    points = [
        SensitivityPoint(
            run_id="baseline",
            execution_index=0,
            overrides=base,
            is_baseline=True,
            chain_id="baseline",
            chain_position=0,
        )
    ]
    if spec.mode == "oat":
        for index, values in enumerate(deduplicated, start=1):
            changed = next(
                name for name in selected if round(float(values[name]), 12) != round(float(base[name]), 12)
            )
            points.append(
                SensitivityPoint(
                    run_id=f"oat_{changed}_{index:04d}",
                    execution_index=index,
                    overrides=values,
                    is_baseline=False,
                    chain_id=f"oat_{index:04d}",
                    chain_position=0,
                )
            )
        return points

    fast_axis = selected[-1]
    prefix = selected[:-1]
    ordered = sorted(deduplicated, key=lambda values: tuple(values[name] for name in (*prefix, fast_axis)))
    chain_positions: dict[tuple[float, ...], int] = {}
    for index, values in enumerate(ordered, start=1):
        prefix_values = tuple(values[name] for name in prefix)
        chain_position = chain_positions.get(prefix_values, 0)
        chain_positions[prefix_values] = chain_position + 1
        chain_label = "factorial" if not prefix else "factorial_" + "_".join(f"{value:g}" for value in prefix_values)
        points.append(
            SensitivityPoint(
                run_id=f"factorial_{index:04d}",
                execution_index=index,
                overrides=values,
                is_baseline=False,
                chain_id=chain_label,
                chain_position=chain_position,
            )
        )
    return points


def catalog_rows(parameters: CaseParameters, selected: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = parameters.raw.set_index("parameter")
    for name in selected:
        definition = parameter_definition(name)
        source = raw.loc[name]
        rows.append(
            {
                "parameter": name,
                "category": definition.category if definition else "",
                "base_value": _base_value(parameters, name),
                "unit": definition.unit if definition else ("-" if str(source["unit"]) == "nan" else str(source["unit"])),
                "sensitivity_start": float(source["sensitivity_start"]),
                "sensitivity_end": float(source["sensitivity_end"]),
                "sensitivity_step": float(source["sensitivity_step"]),
                "value_count": len(parameters.sensitivity[name]),
            }
        )
    return rows


def _base_value(parameters: CaseParameters, name: str) -> float:
    value = parameters.base.get(name)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Sensitivity parameter {name} must have a numeric base_value.")
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sensitivity parameter {name} must have a numeric base_value.") from exc
    if not math.isfinite(value_f):
        raise ValueError(f"Sensitivity parameter {name} must have a finite base_value.")
    return value_f


def _value_key(values: dict[str, float], selected: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(round(float(values[name]), 12) for name in selected)
