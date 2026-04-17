from __future__ import annotations

import pandas as pd

import config as cfg

CONTROL_KEYS = {"enabled", "side", "required_columns"}


def _normalize_filter_config(raw_config: dict | None = None) -> dict:
    config = {
        "enabled": False,
        "side": "both",
        "required_columns": [],
    }
    if raw_config:
        config.update(raw_config)

    config["enabled"] = bool(config.get("enabled", False))
    config["side"] = str(config.get("side", "both")).strip().lower()

    required_columns = list(config.get("required_columns", []))
    for key, value in config.items():
        if value is None or key in CONTROL_KEYS:
            continue
        if key.startswith("min_") or key.startswith("max_"):
            required_columns.append(key[4:])
    config["required_columns"] = list(dict.fromkeys(required_columns))
    return config


def _iter_threshold_rules(config: dict):
    for key, value in config.items():
        if value is None or key in CONTROL_KEYS:
            continue
        if key.startswith("min_"):
            yield "min", key[4:], float(value)
        elif key.startswith("max_"):
            yield "max", key[4:], float(value)


def resolve_event_filter_config(override: dict | None = None) -> dict:
    base_config = getattr(cfg, "EVENT_FILTER_CONFIG", None)
    resolved = _normalize_filter_config(base_config)
    if override:
        merged = resolved.copy()
        merged.update(override)
        return _normalize_filter_config(merged)
    return resolved


def validate_event_gate_columns(frame: pd.DataFrame, event_filter_config: dict | None = None) -> None:
    config = resolve_event_filter_config(event_filter_config)
    if not config.get("enabled", False):
        return

    missing = [column for column in config["required_columns"] if column not in frame.columns]
    if missing:
        raise ValueError("Event filter requires missing columns: " + ", ".join(missing))


def build_event_gate_mask(frame: pd.DataFrame, event_filter_config: dict | None = None) -> pd.Series:
    config = resolve_event_filter_config(event_filter_config)
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)
    if not config.get("enabled", False):
        return pd.Series(True, index=frame.index)

    validate_event_gate_columns(frame, config)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for rule_type, column, threshold in _iter_threshold_rules(config):
        series = pd.to_numeric(frame[column], errors="coerce")
        if rule_type == "min":
            mask &= series >= threshold
        else:
            mask &= series <= threshold

    return mask.fillna(False)
