from __future__ import annotations

import pandas as pd


RESULT_FIGURE_NUMBERS = (4, 5, 6, 7, 8, 9, 10, 11, 12)
RESULT_FIGURE_STATUS = "under reconstruction"
_LEGACY_RESILIENCE_KEY = "resilience_" + "aware"
LEGACY_DESIGN_CASES = {
    _LEGACY_RESILIENCE_KEY: "resilience_informed",
    "Resilience-" + "aware": "Resilience-informed",
    "Resilience_" + "Aware": "Resilience_Informed",
}
RESULT_SENSITIVITY_PARAMETERS = [
    "battery_capex_sgd_per_kwh",
    "pv_capex_sgd_per_kw",
    "saidi_min_per_year",
    "saifi_per_year",
    "voll_sgd_per_kwh",
]
PARAMETER_LABELS = {
    "battery_capex_sgd_per_kwh": "Battery CAPEX",
    "pv_capex_sgd_per_kw": "PV CAPEX",
    "saidi_min_per_year": "SAIDI",
    "saifi_per_year": "SAIFI",
    "voll_sgd_per_kwh": "VoLL",
}


def normalize_design_case_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized = normalized.rename(columns=lambda column: str(column).replace(_LEGACY_RESILIENCE_KEY, "resilience_informed"))
    if "design_case" in normalized.columns:
        normalized["design_case"] = normalized["design_case"].replace(LEGACY_DESIGN_CASES)
    if "design_case_label" in normalized.columns:
        normalized["design_case_label"] = normalized["design_case_label"].replace(LEGACY_DESIGN_CASES)
    return normalized
