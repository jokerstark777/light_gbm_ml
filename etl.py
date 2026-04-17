import logging
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.binance.binance_service import BinanceService
from src.exchanges.bybit.bybit_service import BybitService
from src.features import MasterFeatureBuilder
from src.features.indicators import normalize_bars_for_timeframe
from src.persistence.feature_store_factory import create_feature_store
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]
LONG_TARGET_OUTPUT_COLUMNS = [
    "target_long_label",
    "target_long_outcome",
    "target_long_pnl",
    "target_long_exit_reason",
    "target_long_mfe",
    "target_long_mae",
    "target_long_bars_to_tp",
    "target_long_bars_to_sl",
    "target_long_holding_bars",
]
SHORT_TARGET_OUTPUT_COLUMNS = [
    "target_short_label",
    "target_short_outcome",
    "target_short_pnl",
    "target_short_exit_reason",
    "target_short_mfe",
    "target_short_mae",
    "target_short_bars_to_tp",
    "target_short_bars_to_sl",
    "target_short_holding_bars",
]

OUTCOME_TP_FIRST = "TP_FIRST"
OUTCOME_SL_FIRST = "SL_FIRST"
OUTCOME_NEITHER = "NEITHER"


def normalize_main_timeframe_bars(raw_bars: int, minimum: int = 1) -> int:
    return normalize_bars_for_timeframe(
        raw_bars,
        str(getattr(cfg, "TIMEFRAME", "1h")),
        str(getattr(cfg, "WINDOW_REFERENCE_TIMEFRAME", "5m")),
        minimum=minimum,
    )


def resolve_horizon_bars() -> int:
    return normalize_main_timeframe_bars(int(getattr(cfg, "HORIZON", 16)))


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    return numerator / safe_denominator


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length, min_periods=length).mean()


def compute_dynamic_barrier_stop_pct(
    close: pd.Series,
    atr_14: pd.Series,
) -> pd.Series:
    atr_pct = safe_ratio(atr_14, close).abs()
    stop_pct = atr_pct * float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 1.25))
    min_pct = float(getattr(cfg, "BARRIER_MIN_PCT", getattr(cfg, "SL_PCT", 0.015)))
    max_pct = float(getattr(cfg, "BARRIER_MAX_PCT", getattr(cfg, "TP_PCT", 0.03)))
    return stop_pct.clip(lower=min_pct, upper=max_pct)


def compute_dynamic_barrier_take_pct(stop_pct: pd.Series) -> pd.Series:
    return stop_pct * float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 2.0))


def resolve_barrier_mode() -> str:
    return str(getattr(cfg, "BARRIER_MODE", "")).strip().lower()


def attach_barrier_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    close = output["close"]
    atr_14 = compute_atr(output["high"], output["low"], close, length=14)
    barrier_mode = resolve_barrier_mode()

    if barrier_mode == "dynamic":
        output["barrier_stop_pct"] = compute_dynamic_barrier_stop_pct(close, atr_14)
        output["barrier_take_pct"] = compute_dynamic_barrier_take_pct(output["barrier_stop_pct"])
    else:
        output["barrier_stop_pct"] = float(getattr(cfg, "SL_PCT", 0.015))
        output["barrier_take_pct"] = float(getattr(cfg, "TP_PCT", 0.03))
    return output


def compute_clean_pnl(entry_price: float, exit_price: float, side: str = "long") -> float:
    if side == "short":
        raw_pnl = (entry_price - exit_price) / entry_price
    else:
        raw_pnl = (exit_price - entry_price) / entry_price
    taker_com = float(getattr(cfg, "TAKER_COM", 0.0004))
    return raw_pnl - (taker_com + taker_com)


def resolve_vertical_barrier_exit(final_close: float, side: str = "long") -> float:
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
    if side == "short":
        return final_close * (1 + slippage)
    return final_close * (1 - slippage)


def simulate_trade_event(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    stop_pcts: np.ndarray,
    take_pcts: np.ndarray,
    start_idx: int,
    horizon: int,
    side: str = "long",
) -> dict[str, object]:
    slippage = float(getattr(cfg, "SLIPPAGE", 0.0003))
    if start_idx + 1 >= len(opens):
        return {
            "outcome": None,
            "label": np.nan,
            "pnl": np.nan,
            "exit_reason": None,
            "mfe": np.nan,
            "mae": np.nan,
            "bars_to_tp": np.nan,
            "bars_to_sl": np.nan,
            "holding_bars": np.nan,
        }

    stop_pct = stop_pcts[start_idx]
    take_pct = take_pcts[start_idx]

    if pd.isna(stop_pct) or pd.isna(take_pct):
        return {
            "outcome": None,
            "label": np.nan,
            "pnl": np.nan,
            "exit_reason": None,
            "mfe": np.nan,
            "mae": np.nan,
            "bars_to_tp": np.nan,
            "bars_to_sl": np.nan,
            "holding_bars": np.nan,
        }

    base_open = opens[start_idx + 1]
    entry_price = base_open * (1 - slippage) if side == "short" else base_open * (1 + slippage)
    mfe = 0.0
    mae = 0.0
    bars_to_tp = np.nan
    bars_to_sl = np.nan

    for j in range(1, horizon + 1):
        candle_idx = start_idx + j
        if candle_idx >= len(opens):
            break

        candle_open = opens[candle_idx]
        candle_high = highs[candle_idx]
        candle_low = lows[candle_idx]

        if side == "short":
            favorable = max(0.0, (entry_price - candle_low) / entry_price)
            adverse = max(0.0, (candle_high - entry_price) / entry_price)
            stop_price = entry_price * (1 + stop_pct)
            take_price = entry_price * (1 - take_pct)

            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            if np.isnan(bars_to_sl) and candle_high >= stop_price:
                bars_to_sl = float(j)
            if np.isnan(bars_to_tp) and candle_low <= take_price:
                bars_to_tp = float(j)

            if candle_high >= stop_price:
                exit_price = (candle_open if candle_open > stop_price else stop_price) * (1 + slippage)
                return {
                    "outcome": OUTCOME_SL_FIRST,
                    "label": 0.0,
                    "pnl": compute_clean_pnl(entry_price, exit_price, side=side),
                    "exit_reason": "SL",
                    "mfe": mfe,
                    "mae": mae,
                    "bars_to_tp": bars_to_tp,
                    "bars_to_sl": float(j),
                    "holding_bars": float(j),
                }

            if candle_low <= take_price:
                exit_price = take_price * (1 + slippage)
                return {
                    "outcome": OUTCOME_TP_FIRST,
                    "label": 1.0,
                    "pnl": compute_clean_pnl(entry_price, exit_price, side=side),
                    "exit_reason": "TP",
                    "mfe": mfe,
                    "mae": mae,
                    "bars_to_tp": float(j),
                    "bars_to_sl": bars_to_sl,
                    "holding_bars": float(j),
                }

            continue

        favorable = max(0.0, (candle_high - entry_price) / entry_price)
        adverse = max(0.0, (entry_price - candle_low) / entry_price)
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        if np.isnan(bars_to_sl) and candle_low <= stop_price:
            bars_to_sl = float(j)
        if np.isnan(bars_to_tp) and candle_high >= take_price:
            bars_to_tp = float(j)

        if candle_low <= stop_price:
            exit_price = (candle_open if candle_open < stop_price else stop_price) * (1 - slippage)
            return {
                "outcome": OUTCOME_SL_FIRST,
                "label": 0.0,
                "pnl": compute_clean_pnl(entry_price, exit_price, side=side),
                "exit_reason": "SL",
                "mfe": mfe,
                "mae": mae,
                "bars_to_tp": bars_to_tp,
                "bars_to_sl": float(j),
                "holding_bars": float(j),
            }

        if candle_high >= take_price:
            exit_price = take_price * (1 - slippage)
            return {
                "outcome": OUTCOME_TP_FIRST,
                "label": 1.0,
                "pnl": compute_clean_pnl(entry_price, exit_price, side=side),
                "exit_reason": "TP",
                "mfe": mfe,
                "mae": mae,
                "bars_to_tp": float(j),
                "bars_to_sl": bars_to_sl,
                "holding_bars": float(j),
            }

    final_candle_idx = min(start_idx + horizon, len(closes) - 1)
    final_exit_price = resolve_vertical_barrier_exit(closes[final_candle_idx], side=side)
    return {
        "outcome": OUTCOME_NEITHER,
        "label": 0.0,
        "pnl": compute_clean_pnl(entry_price, final_exit_price, side=side),
        "exit_reason": "TIME",
        "mfe": mfe,
        "mae": mae,
        "bars_to_tp": bars_to_tp,
        "bars_to_sl": bars_to_sl,
        "holding_bars": float(min(horizon, len(opens) - 1 - start_idx)),
    }


def triple_barrier_labeling(df: pd.DataFrame) -> pd.DataFrame:
    long_labels = []
    long_outcomes = []
    long_pnls = []
    long_exit_reasons = []
    long_mfes = []
    long_maes = []
    long_bars_to_tps = []
    long_bars_to_sls = []
    long_holding_bars = []
    short_labels = []
    short_outcomes = []
    short_pnls = []
    short_exit_reasons = []
    short_mfes = []
    short_maes = []
    short_bars_to_tps = []
    short_bars_to_sls = []
    short_holding_bars = []
    horizon = resolve_horizon_bars()

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    stop_pcts = df["barrier_stop_pct"].values
    take_pcts = df["barrier_take_pct"].values

    for i in range(len(df) - horizon):
        long_event = simulate_trade_event(
            opens,
            highs,
            lows,
            closes,
            stop_pcts,
            take_pcts,
            i,
            horizon,
            side="long",
        )
        short_event = simulate_trade_event(
            opens,
            highs,
            lows,
            closes,
            stop_pcts,
            take_pcts,
            i,
            horizon,
            side="short",
        )
        long_labels.append(long_event["label"])
        long_outcomes.append(long_event["outcome"])
        long_pnls.append(long_event["pnl"])
        long_exit_reasons.append(long_event["exit_reason"])
        long_mfes.append(long_event["mfe"])
        long_maes.append(long_event["mae"])
        long_bars_to_tps.append(long_event["bars_to_tp"])
        long_bars_to_sls.append(long_event["bars_to_sl"])
        long_holding_bars.append(long_event["holding_bars"])
        short_labels.append(short_event["label"])
        short_outcomes.append(short_event["outcome"])
        short_pnls.append(short_event["pnl"])
        short_exit_reasons.append(short_event["exit_reason"])
        short_mfes.append(short_event["mfe"])
        short_maes.append(short_event["mae"])
        short_bars_to_tps.append(short_event["bars_to_tp"])
        short_bars_to_sls.append(short_event["bars_to_sl"])
        short_holding_bars.append(short_event["holding_bars"])

    unavailable_count = len(df) - len(long_labels)
    long_labels.extend([np.nan] * unavailable_count)
    long_outcomes.extend([None] * unavailable_count)
    long_pnls.extend([np.nan] * unavailable_count)
    long_exit_reasons.extend([None] * unavailable_count)
    long_mfes.extend([np.nan] * unavailable_count)
    long_maes.extend([np.nan] * unavailable_count)
    long_bars_to_tps.extend([np.nan] * unavailable_count)
    long_bars_to_sls.extend([np.nan] * unavailable_count)
    long_holding_bars.extend([np.nan] * unavailable_count)
    short_labels.extend([np.nan] * unavailable_count)
    short_outcomes.extend([None] * unavailable_count)
    short_pnls.extend([np.nan] * unavailable_count)
    short_exit_reasons.extend([None] * unavailable_count)
    short_mfes.extend([np.nan] * unavailable_count)
    short_maes.extend([np.nan] * unavailable_count)
    short_bars_to_tps.extend([np.nan] * unavailable_count)
    short_bars_to_sls.extend([np.nan] * unavailable_count)
    short_holding_bars.extend([np.nan] * unavailable_count)
    output = df.copy()
    output["target_long_label"] = long_labels
    output["target_long_outcome"] = long_outcomes
    output["target_long_pnl"] = long_pnls
    output["target_long_exit_reason"] = long_exit_reasons
    output["target_long_mfe"] = long_mfes
    output["target_long_mae"] = long_maes
    output["target_long_bars_to_tp"] = long_bars_to_tps
    output["target_long_bars_to_sl"] = long_bars_to_sls
    output["target_long_holding_bars"] = long_holding_bars
    output["target_short_label"] = short_labels
    output["target_short_outcome"] = short_outcomes
    output["target_short_pnl"] = short_pnls
    output["target_short_exit_reason"] = short_exit_reasons
    output["target_short_mfe"] = short_mfes
    output["target_short_mae"] = short_maes
    output["target_short_bars_to_tp"] = short_bars_to_tps
    output["target_short_bars_to_sl"] = short_bars_to_sls
    output["target_short_holding_bars"] = short_holding_bars
    return output


def finalize_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    output_columns = (
        BASE_OUTPUT_COLUMNS
        + feature_columns
        + BARRIER_OUTPUT_COLUMNS
        + LONG_TARGET_OUTPUT_COLUMNS
        + SHORT_TARGET_OUTPUT_COLUMNS
    )

    for column in output_columns:
        if column not in output.columns:
            output[column] = np.nan

    output = output[output_columns].copy()
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    required_columns = feature_columns + BARRIER_OUTPUT_COLUMNS + ["target_long_label"]
    output.dropna(subset=required_columns, inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output


def create_exchange_service() -> ExchangeContract:
    exchange_name = str(getattr(cfg, "ACTIVE_EXCHANGE", "bybit")).strip().lower()
    if exchange_name == "bybit":
        return BybitService()
    if exchange_name == "binance":
        return BinanceService()
    raise ValueError(f"Unsupported ACTIVE_EXCHANGE: {exchange_name}")


def format_yellow_warning(message: str) -> str:
    return f"{ANSI_YELLOW}{message}{ANSI_RESET}"


def parse_iso_datetime_to_utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp() * 1000)


def timeframe_to_ms(timeframe: str) -> int:
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe.strip().lower())
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    amount = int(match.group(1))
    unit = match.group(2)
    unit_to_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    return amount * unit_to_ms[unit]


def align_to_next_candle_open(timestamp_ms: int, timeframe: str) -> int:
    timeframe_ms = timeframe_to_ms(timeframe)
    remainder = timestamp_ms % timeframe_ms
    if remainder == 0:
        return timestamp_ms
    return timestamp_ms + (timeframe_ms - remainder)


def warn_if_history_starts_late(
    repository: HistoricalKlineRepository,
    symbol: str,
    timeframe: str,
    requested_start_date: str,
) -> None:
    requested_start_ts = parse_iso_datetime_to_utc_ms(requested_start_date)
    expected_first_open_ts = align_to_next_candle_open(requested_start_ts, timeframe)
    first_open_time = repository.get_first_open_time(symbol, timeframe)
    if first_open_time is None or first_open_time <= expected_first_open_ts:
        return

    logger.warning(
        format_yellow_warning(
            f"[{symbol}-{timeframe}] incomplete history: requested from "
            f"{datetime.fromtimestamp(requested_start_ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}, "
            f"expected first candle at "
            f"{datetime.fromtimestamp(expected_first_open_ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}, "
            f"but first available candle starts at "
            f"{datetime.fromtimestamp(first_open_time / 1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S UTC}. "
            "The asset was likely listed after the requested start date."
        )
    )


def load_symbol_frame(
    repository: HistoricalKlineRepository,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    frame = repository.load_candles(symbol, timeframe)
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in BASE_OUTPUT_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().sort_values("timestamp").reset_index(drop=True)


def build_candle_maps(
    repository: HistoricalKlineRepository,
    symbols_to_load: list,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_candle_map: dict[str, pd.DataFrame] = {}
    htf_candle_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        df = load_symbol_frame(repository, symbol_name, str(getattr(cfg, "TIMEFRAME", "1h")))
        htf_df = load_symbol_frame(repository, symbol_name, str(getattr(cfg, "HTF_TIMEFRAME", "4h")))
        if df.empty:
            logger.warning("%s: no data in DB (main=%s)", symbol_name, len(df))
            continue

        if htf_df.empty:
            logger.warning("%s: no HTF data in DB (htf=%s)", symbol_name, len(htf_df))
        else:
            htf_candle_map[symbol_name] = htf_df

        logger.info("%s: main=%s, htf=%s rows", symbol_name, len(df), len(htf_df))
        base_candle_map[symbol_name] = df

    return base_candle_map, htf_candle_map


def main() -> None:
    for warning in cfg.validate_config(context="etl"):
        logger.warning("Config: %s", warning)

    exchange_service = create_exchange_service()
    repository = HistoricalKlineRepository(exchange_code=exchange_service.get_exchange_code())
    feature_store = create_feature_store(exchange_code=exchange_service.get_exchange_code())
    repository.init_schema()

    symbols_to_load = [exchange_service.normalize_symbol(symbol) for symbol in dict.fromkeys(getattr(cfg, "SYMBOLS", []))]
    timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
    htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
    start_date = str(getattr(cfg, "START_DATE", "2023-01-01"))
    end_date = getattr(cfg, "END_DATE", None)

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        logger.info("Loading %s %s from %s...", symbol_name, timeframe, start_date)
        loaded = repository.sync_candles(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s %s: %s new candles", symbol_name, timeframe, loaded)
        warn_if_history_starts_late(repository, symbol_name, timeframe, start_date)

        logger.info("Loading %s %s from %s...", symbol_name, htf_timeframe, start_date)
        htf_loaded = repository.sync_candles(exchange_service, symbol, htf_timeframe, start_date, end_date)
        logger.info("%s %s: %s new candles", symbol_name, htf_timeframe, htf_loaded)
        warn_if_history_starts_late(repository, symbol_name, htf_timeframe, start_date)

    base_candle_map, htf_candle_map = build_candle_maps(repository, symbols_to_load)
    pipeline_result = MasterFeatureBuilder().build(base_candle_map, htf_candle_map)
    logger.info(
        "Feature build request resolved: profile=%s | blocks=%s | features=%s",
        pipeline_result.profile_name,
        ", ".join(pipeline_result.active_blocks) or "none",
        len(pipeline_result.feature_columns),
    )

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        feature_df = pipeline_result.feature_map.get(symbol_name)
        if feature_df is None or feature_df.empty:
            logger.warning("%s: skipped, missing prepared feature inputs", symbol_name)
            continue

        feature_df = attach_barrier_columns(feature_df)
        feature_df = triple_barrier_labeling(feature_df)
        feature_df = finalize_feature_frame(feature_df, list(pipeline_result.feature_columns))
        feature_store.save_features(symbol, feature_df)
        logger.info("%s: saved %s rows with %s requested features", symbol_name, len(feature_df), len(pipeline_result.feature_columns))


if __name__ == "__main__":
    main()
