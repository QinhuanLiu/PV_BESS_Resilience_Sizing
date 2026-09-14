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
    night_window_start_hour: float = 19.0,
    night_window_end_hour: float = 23.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective_saifi = max(0.0, float(saifi_per_year))
    effective_saidi = max(0.0, float(saidi_min_per_year))
    if effective_saifi <= 0 or effective_saidi <= 0:
        return _profile_from_events([], hours), pd.DataFrame(columns=["event_id", "start_hour", "end_hour", "duration_minutes"])

    if forced_event_mode in {"match_saidi", "night_window_random"}:
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
        if forced_event_mode == "night_window_random":
            start_min = _night_window_start_minute(
                idx,
                event_count,
                float(duration),
                hours,
                rng,
                night_window_start_hour,
                night_window_end_hour,
            )
        else:
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


def _night_window_start_minute(
    event_index: int,
    event_count: int,
    duration_minutes: float,
    hours: int,
    rng: np.random.Generator,
    night_window_start_hour: float,
    night_window_end_hour: float,
) -> float:
    start_hour = float(night_window_start_hour)
    end_hour = float(night_window_end_hour)
    if not 0.0 <= start_hour < end_hour <= 24.0:
        raise ValueError("night_window_start_hour and night_window_end_hour must satisfy 0 <= start < end <= 24")

    total_minutes = float(hours) * 60.0
    latest_start = total_minutes - float(duration_minutes)
    if latest_start < start_hour * 60.0:
        raise ValueError("Outage duration is too long to start inside the requested night window.")

    total_days = max(1, int(np.ceil(float(hours) / 24.0)))
    slot_days = total_days / max(int(event_count), 1)
    slot_start_day = int(np.floor((int(event_index) - 1) * slot_days))
    slot_end_day = int(np.ceil(int(event_index) * slot_days)) - 1
    slot_start_day = max(0, min(total_days - 1, slot_start_day))
    slot_end_day = max(slot_start_day, min(total_days - 1, slot_end_day))

    valid_days = [
        day
        for day in range(slot_start_day, slot_end_day + 1)
        if day * 24.0 * 60.0 + start_hour * 60.0 <= latest_start
    ]
    if not valid_days:
        latest_valid_day = int(np.floor((latest_start - start_hour * 60.0) / (24.0 * 60.0)))
        valid_days = [max(0, min(total_days - 1, latest_valid_day))]

    day = int(valid_days[int(rng.integers(0, len(valid_days)))])
    hour_of_day = float(rng.uniform(start_hour, end_hour))
    start_minute = day * 24.0 * 60.0 + hour_of_day * 60.0
    if start_minute > latest_start:
        start_minute = day * 24.0 * 60.0 + start_hour * 60.0
    return float(start_minute)


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
