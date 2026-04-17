from __future__ import annotations

import re

import pandas as pd


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    match = re.fullmatch(r"(\d+)([mhdw])", str(timeframe).strip().lower())
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    amount = int(match.group(1))
    unit = match.group(2)
    unit_map = {
        "m": "min",
        "h": "h",
        "d": "d",
        "w": "W",
    }
    return pd.to_timedelta(amount, unit=unit_map[unit])


def normalize_bars_for_timeframe(
    bars: int | float,
    target_timeframe: str,
    reference_timeframe: str,
    minimum: int = 1,
) -> int:
    raw_bars = float(bars)
    if raw_bars <= 0:
        return max(0, minimum)

    reference_duration = raw_bars * timeframe_to_timedelta(reference_timeframe)
    target_duration = timeframe_to_timedelta(target_timeframe)
    normalized = int(round(reference_duration / target_duration))
    return max(minimum, normalized)
