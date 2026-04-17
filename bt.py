import argparse
from collections import Counter
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from config import *
from execution_engine import (
    EngineLogConfig,
    EntrySignal,
    ExecutionConfig,
    compute_net_pnl_pct as engine_compute_net_pnl_pct,
    compute_portfolio_equity as engine_compute_portfolio_equity,
    compute_trade_outcome as engine_compute_trade_outcome,
    simulate_portfolio,
    update_drawdown_stats as engine_update_drawdown_stats,
)
from signal_filter import build_event_gate_mask, resolve_event_filter_config
from src.features.indicators import normalize_bars_for_timeframe
from src.persistence.feature_store_factory import create_feature_store
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

# Futures settings come from config.py
TAKER_COM = globals().get("TAKER_COM", 0.0004)
MAKER_COM = globals().get("MAKER_COM", 0.0002)
SLIPPAGE = globals().get("SLIPPAGE", 0.0003)
LEVERAGE = globals().get("LEVERAGE", 1)
RISK_PER_TRADE = globals().get("RISK_PER_TRADE", 0.01)
MIN_STOP_PCT = float(globals().get("BARRIER_MIN_PCT", 0.005))
BACKTEST_SL_COOLDOWN_BARS = normalize_bars_for_timeframe(
    int(globals().get("BACKTEST_SL_COOLDOWN_BARS", 0)),
    str(globals().get("TIMEFRAME", "1h")),
    str(globals().get("WINDOW_REFERENCE_TIMEFRAME", "5m")),
    minimum=0,
)
BACKTEST_MAX_SL_PER_DAY = int(globals().get("BACKTEST_MAX_SL_PER_DAY", 0))
BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = int(
    globals().get("BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES", 0)
)
BACKTEST_REDUCED_RISK_PER_TRADE = float(globals().get("BACKTEST_REDUCED_RISK_PER_TRADE", RISK_PER_TRADE))
BACKTEST_MAX_HOLDING_BARS = normalize_bars_for_timeframe(
    int(globals().get("BACKTEST_MAX_HOLDING_BARS", globals().get("HORIZON", 0))),
    str(globals().get("TIMEFRAME", "1h")),
    str(globals().get("WINDOW_REFERENCE_TIMEFRAME", "5m")),
    minimum=0,
)
BACKTEST_INITIAL_BALANCE = float(globals().get("BACKTEST_INITIAL_BALANCE", 100.0))
BARRIER_MODE = str(globals().get("BARRIER_MODE", "")).strip().lower()
MODEL_NAME = globals().get("MODEL_NAME", "lightgbm_long_only")
BACKTEST_CHARTS_DIR = Path(globals().get("BACKTEST_CHARTS_DIR", "backtest_charts"))
BACKTEST_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
EQUITY_CURVE_PATH = BACKTEST_CHARTS_DIR / "equity_curve.png"

ANSI_RESET = "\033[0m"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest the saved LightGBM long-only model.")
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=None,
        help="Override the entry threshold saved in the model metadata.",
    )
    return parser.parse_args()


def get_end_date_cutoff():
    if not globals().get("END_DATE"):
        return None
    return pd.to_datetime(globals().get("END_DATE"), errors="coerce")


def resolve_barrier_mode() -> str:
    return BARRIER_MODE


def apply_end_date_cutoff(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty or timestamp_column not in df.columns:
        return df

    end_cutoff = get_end_date_cutoff()
    if end_cutoff is None or pd.isna(end_cutoff):
        return df

    return df.loc[df[timestamp_column] <= end_cutoff].copy()


def parse_period_payload(period_payload: dict | None, period_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not period_payload:
        raise RuntimeError(
            f"Model metadata does not include {period_name}. Re-run train.py with holdout-aware artifacts first."
        )

    start = pd.to_datetime(period_payload.get("start"), errors="coerce")
    end = pd.to_datetime(period_payload.get("end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise RuntimeError(f"Model metadata has an invalid {period_name}: {period_payload}")
    if start > end:
        raise RuntimeError(f"Model metadata has {period_name} start after end: {period_payload}")
    return start, end

def timeframe_to_ms(timeframe: str) -> int:
    if timeframe not in TF_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TF_MS[timeframe]


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def format_pnl_pct(pnl_pct: float) -> str:
    color = ANSI_GREEN if pnl_pct >= 0 else ANSI_RED
    return colorize(f"{pnl_pct:+.2f}%", color)


def format_reason(reason: str) -> str:
    if reason == "TP":
        return colorize("TP", ANSI_GREEN)
    if reason == "SL":
        return colorize("SL", ANSI_RED)
    return reason


def should_open_long(p_long: float, entry_threshold: float) -> bool:
    return p_long >= entry_threshold


def get_barrier_pcts(feature_row: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if feature_row is None or feature_row.empty:
        return None, None

    if "barrier_stop_pct" not in feature_row.columns or "barrier_take_pct" not in feature_row.columns:
        return None, None

    stop_pct = float(feature_row["barrier_stop_pct"].iloc[0])
    take_pct = float(feature_row["barrier_take_pct"].iloc[0])
    if not np.isfinite(stop_pct) or not np.isfinite(take_pct):
        return None, None
    return stop_pct, take_pct


def compact_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def format_signed_dollars(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def format_percent_value(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def compute_net_pnl_pct(entry_price: float, exit_price: float) -> float:
    return engine_compute_net_pnl_pct(entry_price, exit_price, float(TAKER_COM))


def compute_trade_outcome(position: dict, exit_price: float) -> tuple[float, float, float]:
    return engine_compute_trade_outcome(position, exit_price, build_execution_config(BACKTEST_INITIAL_BALANCE, 0.0))


def compute_portfolio_equity(balance: float, positions: dict, mark_prices: dict[str, float]) -> float:
    return engine_compute_portfolio_equity(
        balance,
        positions,
        mark_prices,
        build_execution_config(BACKTEST_INITIAL_BALANCE, 0.0),
    )


def update_drawdown_stats(equity: float, peak_equity: float, max_drawdown: float) -> tuple[float, float]:
    return engine_update_drawdown_stats(equity, peak_equity, max_drawdown)


def print_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> None:
    right_align = right_align or set()
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def format_row(row_values):
        formatted = []
        for idx, cell in enumerate(row_values):
            text = str(cell)
            if idx in right_align:
                formatted.append(text.rjust(widths[idx]))
            else:
                formatted.append(text.ljust(widths[idx]))
        return "| " + " | ".join(formatted) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print(separator)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)


def load_raw_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    repository = HistoricalKlineRepository()
    df = repository.load_candles(symbol, timeframe)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df = apply_end_date_cutoff(df)
    return df


def load_all_raw_data(symbols):
    all_data = {}

    print(f"Loading raw data for {len(symbols)} symbols...")
    for sym in symbols:
        try:
            df_main = load_raw_candles(sym, TIMEFRAME)

            if df_main.empty:
                print(f"Warning: {sym} has no main TF data ({TIMEFRAME})")
                continue

            all_data[sym] = {"main": df_main}
            print(f"{sym}: main={len(df_main)} candles")
        except Exception as e:
            print(f"Warning: failed loading {sym}: {e}")

    return all_data


def load_precomputed_features(symbol: str, symbol_categories=None, required_columns: list | None = None) -> pd.DataFrame:
    feature_store = create_feature_store()
    read_columns = None
    if required_columns:
        read_columns = list(dict.fromkeys(["timestamp"] + [column for column in required_columns if column != "symbol"]))
    try:
        df = feature_store.load_features(symbol, columns=read_columns)
    except Exception as exc:
        print(f"Warning: failed loading prepared dataset for {symbol}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    if required_columns:
        required_order = list(dict.fromkeys(["timestamp"] + required_columns + ["symbol"]))
        missing_columns = [column for column in required_order if column not in df.columns]
        if missing_columns:
            preview = ", ".join(missing_columns[:10])
            suffix = "..." if len(missing_columns) > 10 else ""
            print(
                f"Warning: {symbol} prepared dataset is missing required columns: "
                f"{preview}{suffix}"
            )
            return pd.DataFrame()
        df = df[required_order].copy()

    df = apply_end_date_cutoff(df)
    if symbol_categories is None:
        df["symbol"] = df["symbol"].astype("category")
    else:
        df["symbol"] = pd.Categorical(df["symbol"], categories=symbol_categories)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_timestamp_index(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    return df.set_index("timestamp", drop=False).to_dict("index")


def get_common_main_timestamps(all_data: dict) -> list:
    if not all_data:
        return []

    ts_sets = [set(payload["main"]["timestamp"]) for payload in all_data.values()]
    return sorted(list(set.intersection(*ts_sets))) if ts_sets else []


def get_market_timestamps(all_data: dict, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[pd.Timestamp]:
    if not all_data:
        return []

    timestamps = set()
    for payload in all_data.values():
        main_frame = payload.get("main")
        if main_frame is None or main_frame.empty:
            continue
        period_rows = main_frame.loc[
            (main_frame["timestamp"] >= start_ts) & (main_frame["timestamp"] <= end_ts),
            "timestamp",
        ]
        timestamps.update(period_rows.tolist())
    return sorted(timestamps)


def filter_symbols_with_period_overlap(all_data: dict, start_ts: pd.Timestamp, end_ts: pd.Timestamp, min_candles: int = 2):
    if not all_data:
        return {}, []

    filtered = {}
    dropped = []
    for symbol, payload in all_data.items():
        period_rows = payload["main"].loc[
            (payload["main"]["timestamp"] >= start_ts) & (payload["main"]["timestamp"] <= end_ts)
        ]
        if len(period_rows) < min_candles:
            dropped.append(symbol)
            continue
        filtered[symbol] = payload

    return filtered, dropped


def get_feature_row_precomputed(df: pd.DataFrame, ts: pd.Timestamp, feature_names: list):
    row = df[df["timestamp"] == ts]
    if row.empty:
        return None

    latest_row = row.iloc[[-1]].copy()
    missing = [f for f in feature_names if f not in latest_row.columns]
    if missing:
        return None

    if latest_row[feature_names].isna().any(axis=None):
        return None

    return latest_row


def get_last_exec_row_on_or_before(df: pd.DataFrame, ts: pd.Timestamp):
    if df is None or df.empty or "timestamp" not in df.columns:
        return None

    eligible_rows = df.loc[df["timestamp"] <= ts]
    if eligible_rows.empty:
        return None
    return eligible_rows.iloc[-1].to_dict()


def get_exec_row_by_ts_index(indexed_rows: dict, ts: pd.Timestamp):
    row = indexed_rows.get(ts)
    if row is None:
        return None
    return row


def get_feature_row_precomputed_index(indexed_rows: dict, ts: pd.Timestamp, feature_names: list):
    row = indexed_rows.get(ts)
    if row is None:
        return None

    latest_row = pd.DataFrame([row])
    missing = [f for f in feature_names if f not in latest_row.columns]
    if missing:
        return None

    if latest_row[feature_names].isna().any(axis=None):
        return None

    return latest_row


def passes_event_gate(feature_row: pd.DataFrame, event_filter_config: dict) -> bool:
    if feature_row is None or feature_row.empty:
        return False
    mask = build_event_gate_mask(feature_row, event_filter_config)
    return bool(mask.iloc[0]) if not mask.empty else False


def format_event_filter_summary(event_filter_config: dict) -> str:
    if not event_filter_config or not event_filter_config.get("enabled", False):
        return "Event filter: disabled"

    side = str(event_filter_config.get("side", "both")).upper()
    rule_parts = []
    for key in sorted(event_filter_config.keys()):
        if key in {"enabled", "side", "required_columns"}:
            continue
        value = event_filter_config.get(key)
        if value is None:
            continue
        if key.startswith("min_"):
            rule_parts.append(f"{key[4:]}>={float(value):.6f}")
        elif key.startswith("max_"):
            rule_parts.append(f"{key[4:]}<={float(value):.6f}")

    rules = ", ".join(rule_parts) if rule_parts else "no threshold rules"
    return f"Event filter: side={side}, {rules}"


def normalize_features_for_model(latest_row: pd.DataFrame, feature_names: list, symbol_categories=None):
    features = latest_row[feature_names].copy()
    if "symbol" in features.columns:
        if symbol_categories is None:
            features["symbol"] = features["symbol"].astype("category")
        else:
            features["symbol"] = pd.Categorical(features["symbol"], categories=symbol_categories)
    return features


def apply_feature_clip_bounds(frame: pd.DataFrame, clip_bounds: dict):
    if not clip_bounds:
        return frame

    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def prepare_precomputed_feature_store(
    feat_df: pd.DataFrame,
    feature_names: list,
    symbol_categories=None,
    clip_bounds: dict | None = None,
) -> pd.DataFrame:
    if feat_df is None or feat_df.empty:
        return pd.DataFrame(columns=feature_names)

    missing = [feature for feature in feature_names if feature not in feat_df.columns]
    if missing:
        return pd.DataFrame(columns=feature_names)

    timestamps = feat_df["timestamp"].copy()
    prepared = normalize_features_for_model(
        feat_df,
        feature_names,
        symbol_categories=symbol_categories,
    )
    prepared = apply_feature_clip_bounds(prepared, clip_bounds or {})
    prepared.insert(0, "timestamp", timestamps.values)
    prepared = prepared.dropna(subset=feature_names)
    prepared = prepared.drop_duplicates(subset=["timestamp"], keep="last")
    return prepared.set_index("timestamp", drop=True).sort_index()


def get_feature_batch_precomputed(
    feature_store: dict,
    symbols: list,
    ts: pd.Timestamp,
):
    batch_frames = []
    batch_symbols = []

    for symbol in symbols:
        prepared = feature_store.get(symbol)
        if prepared is None or prepared.empty or ts not in prepared.index:
            continue

        batch_frames.append(prepared.loc[[ts]])
        batch_symbols.append(symbol)

    if not batch_frames:
        return [], pd.DataFrame()

    return batch_symbols, pd.concat(batch_frames, axis=0)


def build_execution_config(initial_balance: float, entry_threshold: float) -> ExecutionConfig:
    return ExecutionConfig(
        initial_balance=float(initial_balance),
        taker_com=float(TAKER_COM),
        slippage=float(SLIPPAGE),
        leverage=float(LEVERAGE),
        risk_per_trade=float(RISK_PER_TRADE),
        min_stop_pct=float(MIN_STOP_PCT),
        entry_threshold=float(entry_threshold),
        max_new_positions_per_bar=int(BACKTEST_MAX_NEW_POSITIONS_PER_BAR),
        max_open_positions=int(BACKTEST_MAX_OPEN_POSITIONS),
        sl_cooldown_bars=int(BACKTEST_SL_COOLDOWN_BARS),
        max_sl_per_day=int(BACKTEST_MAX_SL_PER_DAY),
        reduce_risk_after_consecutive_losses=int(BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES),
        reduced_risk_per_trade=float(BACKTEST_REDUCED_RISK_PER_TRADE),
        max_holding_bars=int(BACKTEST_MAX_HOLDING_BARS),
    )


def backtest(entry_threshold_override: float | None = None):
    print("Loading model and prepared inputs...")

    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    features_meta_path = MODELS_DIR / f"{MODEL_NAME}_features.json"

    if not model_path.exists():
        print(f"Error: model not found at {model_path}. Run train.py first.")
        return

    if not features_meta_path.exists():
        print(f"Error: model input metadata not found at {features_meta_path}. Run train.py first.")
        return

    model = joblib.load(model_path)

    with open(features_meta_path, "r", encoding="utf-8") as f:
        features_meta = json.load(f)

    feature_names = features_meta["feature_columns"]
    if features_meta.get("task_type") != "binary_long_only":
        print(
            "Error: loaded model metadata is not long-only (`task_type != binary_long_only`). "
            "Re-run train.py and rebuild the long-only artifact first."
        )
        return
    if bool(features_meta.get("prod_train")):
        print(
            "Error: this model artifact was retrained on the full dataset (`prod_train=true`), "
            "so the holdout window is no longer out-of-sample. Re-run train.py without --prod-train."
        )
        return
    if "entry_threshold" not in features_meta:
        print("Error: model metadata does not include entry_threshold. Re-run train.py first.")
        return
    saved_entry_threshold = float(features_meta["entry_threshold"])
    entry_threshold = saved_entry_threshold
    if entry_threshold_override is not None:
        entry_threshold = float(entry_threshold_override)

    try:
        test_start_ts, test_end_ts = parse_period_payload(features_meta.get("test_period"), "test_period")
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return
    event_filter_config = resolve_event_filter_config()
    runtime_required_columns = list(dict.fromkeys(feature_names + ["barrier_stop_pct", "barrier_take_pct"]))
    if event_filter_config.get("enabled", False):
        runtime_required_columns.extend(
            column
            for column in event_filter_config.get("required_columns", [])
            if column not in runtime_required_columns
        )

    trained_symbols = list(features_meta.get("symbols", SYMBOLS))
    use_symbol_feature = "symbol" in feature_names
    unseen_symbols = [symbol for symbol in SYMBOLS if symbol not in trained_symbols]
    if use_symbol_feature:
        symbol_categories = list(dict.fromkeys(trained_symbols + list(SYMBOLS)))
        if unseen_symbols:
            print(
                "Warning: backtest includes symbols absent from training metadata: "
                + ", ".join(unseen_symbols)
            )
    else:
        symbol_categories = None
    feature_clip_meta = features_meta.get("feature_clip", {})
    clip_bounds = feature_clip_meta.get("bounds", {})
    print(f"Loaded LightGBM model with {len(feature_names)} input columns")
    if clip_bounds:
        print(
            "Feature clipping: "
            f"{len(clip_bounds)} columns "
            f"[{feature_clip_meta.get('lower_q', 0.01) * 100:.2f}%, "
            f"{feature_clip_meta.get('upper_q', 0.99) * 100:.2f}%]"
        )
    print(f"Holdout test window: {test_start_ts.isoformat()} to {test_end_ts.isoformat()}")
    if entry_threshold_override is not None:
        print(f"Entry threshold override: {entry_threshold:.4f} (metadata={saved_entry_threshold:.4f})")
    print(format_event_filter_summary(event_filter_config))
    all_raw = load_all_raw_data(SYMBOLS)
    if not all_raw:
        print("Error: no raw data for backtest.")
        return

    print("Input mode: prepared DB dataset")

    all_features = {}
    all_main_index = {}
    for sym in list(all_raw.keys()):
        feat_df = load_precomputed_features(
            sym,
            symbol_categories=symbol_categories,
            required_columns=runtime_required_columns,
        )
        if feat_df.empty:
            print(f"Warning: {sym} has no prepared dataset rows.")
            continue
        all_features[sym] = feat_df

    all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features}
    if not all_raw:
        print("Error: no symbols with prepared dataset rows available in DB. Run etl.py first.")
        return

    for sym, payload in all_raw.items():
        all_main_index[sym] = build_timestamp_index(payload["main"])

    all_features_prepared = {}
    for sym, feat_df in all_features.items():
        prepared = prepare_precomputed_feature_store(
            feat_df,
            feature_names,
            symbol_categories=symbol_categories,
            clip_bounds=clip_bounds,
        )
        if prepared.empty:
            print(f"Warning: {sym} has no usable prepared rows after input preparation.")
            continue
        all_features_prepared[sym] = prepared

    all_raw = {sym: payload for sym, payload in all_raw.items() if sym in all_features_prepared}
    if not all_raw:
        print("Error: no symbols with usable prepared rows remain after input preparation.")
        return

    test_timestamps = get_market_timestamps(all_raw, test_start_ts, test_end_ts)

    if len(test_timestamps) < 2:
        print("Error: too little market data inside the saved holdout test period.")
        return

    print("\nBacktest Configuration:")
    print(f"   Period: {test_timestamps[0].isoformat()} to {test_timestamps[-1].isoformat()}")
    print(f"   Symbols: {', '.join(compact_symbol(sym) for sym in all_raw.keys())}")
    print(f"   Initial Balance: ${BACKTEST_INITIAL_BALANCE:.2f}")
    print(f"   Risk per Trade: {RISK_PER_TRADE * 100:.0f}%")
    print(f"   Leverage: {LEVERAGE:.0f}x")
    print(f"   Main TF: {TIMEFRAME}")
    print("   Direction mode: LONG ONLY")
    print(f"   Entry threshold: {entry_threshold:.2f}")
    print(f"   Batch entries per bar: {BACKTEST_MAX_NEW_POSITIONS_PER_BAR}")
    print(f"   Max open positions: {BACKTEST_MAX_OPEN_POSITIONS}")
    print(f"   Max holding bars: {BACKTEST_MAX_HOLDING_BARS}")
    print(f"   SL cooldown bars: {BACKTEST_SL_COOLDOWN_BARS}")
    print(f"   Max SL per day: {BACKTEST_MAX_SL_PER_DAY}")
    print(
        "   Reduced risk after consecutive losses: "
        f"{BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES} -> {BACKTEST_REDUCED_RISK_PER_TRADE * 100:.2f}%"
    )
    barrier_mode = resolve_barrier_mode()
    if barrier_mode == "dynamic":
        print(
            "   Dynamic barriers: "
            f"ATRx{globals().get('BARRIER_ATR_MULTIPLIER', 1.25):.2f}, "
            f"TP/SL={globals().get('BARRIER_TP_TO_SL_RATIO', 2.0):.2f}"
        )
    else:
        print(f"   TP: {TP_PCT:.4f} | SL: {SL_PCT:.4f}")
    print(f"\nBacktest on {len(test_timestamps)} candles")
    print("-" * 80)

    signal_diagnostics = Counter()
    p_long_values = []

    def saved_model_signal_provider(current_ts, candidate_symbols, market_batch):
        requested_symbols = [sym for sym in candidate_symbols if sym in all_features_prepared]
        batch_symbols, batch_features = get_feature_batch_precomputed(
            all_features_prepared,
            requested_symbols,
            current_ts,
        )
        if batch_features.empty:
            signal_diagnostics["missing_feature_batch"] += len(requested_symbols)
            return []
        signal_diagnostics["missing_feature_timestamp"] += max(0, len(requested_symbols) - len(batch_symbols))

        signals = []
        batch_proba = model.predict_proba(batch_features[feature_names])
        for sym, proba in zip(batch_symbols, batch_proba):
            p_long = float(proba[1])
            p_long_values.append(p_long)
            signal_diagnostics["prediction_seen"] += 1
            if p_long >= entry_threshold:
                signal_diagnostics["above_entry_threshold"] += 1
            else:
                signal_diagnostics["below_entry_threshold_provider"] += 1

            feature_row = get_feature_row_precomputed(
                all_features[sym],
                current_ts,
                runtime_required_columns,
            )
            stop_pct, take_pct = get_barrier_pcts(feature_row)
            if stop_pct is None or take_pct is None:
                signal_diagnostics["missing_barrier"] += 1
                continue
            if not passes_event_gate(feature_row, event_filter_config):
                signal_diagnostics["event_gate"] += 1
                continue
            signals.append(
                EntrySignal(
                    symbol=sym,
                    p_long=p_long,
                    stop_pct=float(stop_pct),
                    take_pct=float(take_pct),
                )
            )
        return signals

    simulation = simulate_portfolio(
        all_raw=all_raw,
        all_main_index=all_main_index,
        start_ts=test_start_ts,
        end_ts=test_end_ts,
        config=build_execution_config(BACKTEST_INITIAL_BALANCE, entry_threshold),
        signal_provider=saved_model_signal_provider,
        log_config=EngineLogConfig(
            verbose_trades=True,
            close_prefix="[{ts}] No. {trade_number}",
            open_prefix="[{ts}] No. {trade_number}",
            final_prefix="[{ts}] No. {trade_number}",
        ),
        include_month_start_balance=True,
    )

    balance = simulation["ending_balance"]
    initial_balance = BACKTEST_INITIAL_BALANCE
    trades = simulation["trades"]
    equity_curve = simulation["equity_curve"]
    equity_timestamps = simulation["equity_timestamps"]
    monthly_stats = simulation["monthly_stats"]
    max_drawdown = simulation["summary"]["max_drawdown_pct"]

    summary = simulation["summary"]
    engine_diagnostics = Counter(simulation.get("candidate_diagnostics", {}))
    combined_diagnostics = Counter(signal_diagnostics)
    combined_diagnostics.update(engine_diagnostics)
    if p_long_values:
        p_long_series = pd.Series(p_long_values)
        print("\nSignal probability diagnostics:")
        print(
            "   p_long: "
            f"min={p_long_series.min():.4f} | "
            f"median={p_long_series.median():.4f} | "
            f"p95={p_long_series.quantile(0.95):.4f} | "
            f"max={p_long_series.max():.4f}"
        )
    if combined_diagnostics:
        print("Candidate diagnostics:")
        for key, value in sorted(combined_diagnostics.items()):
            print(f"   {key}: {int(value)}")

    total_trades = summary["total_trades"]
    total_wins = summary["total_wins"]
    total_losses = summary["total_losses"]
    win_rate_pct = summary["win_rate_pct"]
    total_pnl_abs = summary["total_pnl_abs"]
    total_fees = summary["total_fees"]
    total_return_pct = summary["total_return_pct"]
    expectancy = summary["expectancy_abs"]
    pf = summary["profit_factor"]
    sharpe = summary["sharpe"]

    print("\nSimulation finished.\n")
    print("=" * 64)
    print("PORTFOLIO BACKTEST RESULTS")
    print("=" * 64)
    print(f"\nTrades: {total_trades} (W: {total_wins} / L: {total_losses})")
    print(f"Winrate: {win_rate_pct:.2f}%")
    print("Equity:")
    print(f"   Start: ${initial_balance:.2f}")
    print(f"   End:   ${balance:.2f}")
    print(f"   PnL:   {format_signed_dollars(total_pnl_abs)} ({format_percent_value(total_return_pct)})")
    print(f"   Fees:  ${total_fees:.2f}")
    print("Risk:")
    print(f"   Max DD: {max_drawdown:.2f}%")
    print(f"   Profit Factor: {pf:.2f}")
    print(f"   Expectancy: {format_signed_dollars(expectancy)}")
    print(f"   Sharpe: {sharpe:.2f}")

    monthly_rows = []
    for month_key in sorted(monthly_stats.keys()):
        stats = monthly_stats[month_key]
        monthly_rows.append(
            [
                month_key,
                format_signed_dollars(stats["pnl_abs"]),
                str(stats["trades"]),
                str(stats["wins"]),
                str(stats["losses"]),
            ]
        )

    if monthly_rows:
        print("\nMonthly Performance Extended:")
        print_table(
            ["Month", "PnL", "Total", "Wins", "Losses"],
            monthly_rows,
            right_align={1, 2, 3, 4},
        )

    symbol_stats = {}
    for trade in trades:
        stats = symbol_stats.setdefault(
            trade["sym"],
            {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        )
        stats["trades"] += 1
        stats["pnl_abs"] += trade["pnl_abs"]
        if trade["reason"] == "TP":
            stats["tp"] += 1
        if trade["reason"] == "SL":
            stats["sl"] += 1
        if trade["pnl_abs"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    if symbol_stats:
        coin_rows = []
        for symbol in sorted(symbol_stats.keys()):
            stats = symbol_stats[symbol]
            winrate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0.0
            coin_rows.append(
                [
                    compact_symbol(symbol),
                    str(stats["trades"]),
                    str(stats["tp"]),
                    str(stats["sl"]),
                    f"{winrate:.1f}%",
                    format_signed_dollars(stats["pnl_abs"]),
                ]
            )

        print("\nSummary by Coin:")
        print_table(
            ["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"],
            coin_rows,
            right_align={1, 2, 3, 4, 5},
        )

    direction_stats = {"LONG": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0}}
    for trade in trades:
        stats = direction_stats[trade["direction"]]
        stats["total"] += 1
        stats["pnl_abs"] += trade["pnl_abs"]
        if trade["pnl_abs"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    direction_rows = []
    for direction in ("LONG",):
        stats = direction_stats[direction]
        direction_rows.append(
            [
                direction,
                str(stats["total"]),
                str(stats["wins"]),
                str(stats["losses"]),
                format_signed_dollars(stats["pnl_abs"]),
            ]
        )

    print("\nLong Summary:")
    print_table(
        ["Direction", "Total", "Wins", "Losses", "PnL"],
        direction_rows,
        right_align={1, 2, 3, 4},
    )

    if len(equity_curve) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_timestamps, equity_curve, label="Portfolio Equity")
        plt.axhline(y=initial_balance, linestyle="--")
        plt.title(f"Multi-Symbol Equity Curve | {total_trades} trades | DD: {max_drawdown:.1f}%")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(EQUITY_CURVE_PATH, dpi=150)
        plt.show()
        print(f"\nSaved chart: {EQUITY_CURVE_PATH}")


def backtest_saved_model():
    args = parse_args()
    return backtest(entry_threshold_override=args.entry_threshold)


if __name__ == "__main__":
    backtest_saved_model()
