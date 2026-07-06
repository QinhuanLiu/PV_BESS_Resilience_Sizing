from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutageEvent:
    event_id: int
    start_minute: float
    end_minute: float

    @property
    def duration_minutes(self) -> float:
        return max(0.0, self.end_minute - self.start_minute)


def simulate_outages(
    saifi_per_year: float,
    saidi_min_per_year: float,
    hours: int = 8760,
    seed: int = 42,
    forced_event_mode: str = "match_saidi",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective_saifi = max(0.0, float(saifi_per_year))
    effective_saidi = max(0.0, float(saidi_min_per_year))
    if effective_saifi <= 0 or effective_saidi <= 0:
        return _profile_from_events([], hours), pd.DataFrame(columns=["event_id", "start_hour", "end_hour", "duration_minutes"])

    if forced_event_mode == "match_saidi":
        event_count = max(1, int(round(effective_saifi)))
    else:
        rng = np.random.default_rng(seed)
        event_count = int(rng.poisson(effective_saifi))
        if event_count == 0:
            event_count = 1

    rng = np.random.default_rng(seed)
    mean_duration = effective_saidi / max(event_count, 1)
    raw = rng.gamma(shape=1.6, scale=mean_duration / 1.6, size=event_count)
    raw = np.maximum(raw, 1e-6)
    durations = raw * effective_saidi / raw.sum()

    total_minutes = hours * 60.0
    slot = total_minutes / event_count
    events: list[OutageEvent] = []
    for idx, duration in enumerate(durations, start=1):
        start_min = (idx - 1) * slot + float(rng.uniform(0.05 * slot, 0.45 * slot))
        start_min = min(start_min, total_minutes - float(duration))
        events.append(OutageEvent(idx, max(0.0, start_min), min(total_minutes, start_min + float(duration))))

    profile = _profile_from_events(events, hours)
    event_rows = [
        {
            "event_id": event.event_id,
            "start_hour": event.start_minute / 60.0,
            "end_hour": event.end_minute / 60.0,
            "duration_minutes": event.duration_minutes,
        }
        for event in events
    ]
    return profile, pd.DataFrame(event_rows)


def _profile_from_events(events: Iterable[OutageEvent], hours: int) -> pd.DataFrame:
    availability = events_to_availability(events, hours)
    return pd.DataFrame(
        {
            "hour_index": np.arange(hours, dtype=int),
            "grid_availability": availability,
            "outage_fraction": 1.0 - availability,
        }
    )


def events_to_availability(events: Iterable[OutageEvent], hours: int = 8760) -> np.ndarray:
    outage_minutes = np.zeros(hours, dtype=float)
    for event in events:
        if event.end_minute <= event.start_minute:
            continue
        first_hour = max(0, int(np.floor(event.start_minute / 60.0)))
        last_hour = min(hours - 1, int(np.floor((event.end_minute - 1e-9) / 60.0)))
        for hour in range(first_hour, last_hour + 1):
            hour_start = hour * 60.0
            hour_end = hour_start + 60.0
            overlap = max(0.0, min(event.end_minute, hour_end) - max(event.start_minute, hour_start))
            outage_minutes[hour] = min(60.0, outage_minutes[hour] + overlap)
    return np.clip(1.0 - outage_minutes / 60.0, 0.0, 1.0)


def summarize_outages(profile: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    return {
        "simulated_events": float(len(events)),
        "simulated_saifi_per_year": float(len(events)),
        "simulated_saidi_min_per_year": float(profile["outage_fraction"].sum() * 60.0),
        "min_grid_availability": float(profile["grid_availability"].min()),
    }
