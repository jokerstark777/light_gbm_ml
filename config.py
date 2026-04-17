import os
import runpy
from copy import deepcopy
from pathlib import Path


# ============================================================
# 0. RUN IDENTITY
# ============================================================
# CONFIG_STAGE documents intent:
# - scaffold: feature/building blocks may be present but intentionally empty.
# - research: active experiment settings, not production-safe by default.
# - production: saved artifacts/backtests should be treated carefully.
EXPERIMENT_NAME = "baseline_ohlcv_htf_long_only_1h"
CONFIG_STAGE = "research"  # "scaffold" | "research" | "production"
CONFIG_PRESET_ENV = "LIGHT_LWTI_PRESET"
CONFIG_PRESET = os.environ.get(CONFIG_PRESET_ENV, "")
CONFIG_PRESETS_DIR = Path("configs") / "presets"
CONFIG_PRESET_SOURCE = None
CONFIG_PRESET_KEYS: set[str] = set()
CONFIG_OVERRIDE_ENV = "LIGHT_LWTI_CONFIG"
CONFIG_OVERRIDE_PATH = os.environ.get(CONFIG_OVERRIDE_ENV, "config.local.py")
CONFIG_OVERRIDE_SOURCE = None
CONFIG_OVERRIDDEN_KEYS: set[str] = set()


# ============================================================
# 1. MARKET UNIVERSE AND DATE RANGE
# ============================================================
ACTIVE_EXCHANGE = "bybit"  # "bybit" | "binance"

SYMBOLS = [

    "BTC/USDT",
    # "ETH/USDT",
    # "SOL/USDT",
    # "XRP/USDT",
    # "ADA/USDT",
    # "TRX/USDT",
    # "BNB/USDT",
    # "XLM/USDT",

    "1000PEPE/USDT",
    "LTC/USDT",
    "1000FLOKI/USDT",
    "SHIB1000/USDT",


    "WIF/USDT",
    "1000000MOG/USDT",
    "1000BONK/USDT",
    "DOGE/USDT",

    # "BTC/USDT",
    # "BNB/USDT",
    # "ETH/USDT",
    # "SOL/USDT",
    # "XRP/USDT",
    # "XLM/USDT",
    # "ADA/USDT",
    # "TRX/USDT",
    # "XMR/USDT",

    # "ATOM/USDT",
    # "DOGE/USDT",
    # "LINK/USDT",
    # "NEAR/USDT",
    # "RENDER/USDT",
    # "ARB/USDT",
    # "HBAR/USDT",
    # "MATIC/USDT",
    # "OP/USDT",
    # "TIA/USDT",
    # "FET/USDT",
    # "SEI/USDT",
    # "WLD/USDT",
    # "INJ/USDT",
    # "AVAX/USDT",
    # "SUI/USDT",
    # "STX/USDT",
    # "TON/USDT",
    # "APT/USDT",
    # "TAO/USDT",
]

TIMEFRAME = "1h"
HTF_TIMEFRAME = "1d"
WINDOW_REFERENCE_TIMEFRAME = TIMEFRAME
HTF_WINDOW_REFERENCE_TIMEFRAME = HTF_TIMEFRAME

START_DATE = "2025-01-01"
END_DATE = "2026-03-27 21:00:00"


# ============================================================
# 2. PATHS AND STORAGE
# ============================================================
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
BACKTEST_CHARTS_DIR = Path("backtest_charts")

DB_PATH = str(DATA_DIR / "market_data.db")

# Raw candles and sync state remain in SQLite. Prepared ML feature frames can
# be stored in Parquet for faster train/backtest IO.
FEATURE_STORAGE = "parquet"  # "sqlite" | "parquet"
PARQUET_FEATURES_DIR = DATA_DIR / "ml_features"
PARQUET_DATASET_VERSION = "default"


# ============================================================
# 3. FEATURE BUILD
# ============================================================
# Feature builders are intentionally scaffolded first. Empty profiles are valid
# while the ML feature layer is being wired, but they are not trainable presets.
FEATURE_PIPELINE_STAGE = CONFIG_STAGE
ALLOW_EMPTY_FEATURE_PROFILE = CONFIG_STAGE == "scaffold"

FEATURE_PROFILES = {
    "empty": [],
    "scaffold_empty": [],
    "baseline_ohlcv": [
        "atr_pct_14",
        "range_expansion_20",
        "trend_efficiency_16",
        "close_pos_range_32",
        "pullback_ema_20_atr",
        "dist_to_prev_day_high_atr",
        "dist_to_prev_day_low_atr",
        "body_to_range",
        "lower_wick_to_range",
        "upper_wick_to_range",
        "volume_rel_median_20",
        "signed_volume_pressure_8",
    ],
    "baseline_ohlcv_htf": [
        "atr_pct_14",
        "volatility_regime_change_1h",
        "range_expansion_20",
        "return_4h_atr_norm",
        "zscore_vs_vwap_4h",
        "bb_width_pct_rank_120",
        "range_compression_ratio_20_100",
        "trend_efficiency_16",
        "close_pos_range_32",
        "pullback_ema_20_atr",
        "dist_to_prev_day_high_atr",
        "dist_to_prev_day_low_atr",
        "body_to_range",
        "lower_wick_to_range",
        "upper_wick_to_range",
        "volume_rel_median_20",
        "signed_volume_pressure_8",
        "htf_trend_direction",
        "htf_close_pos_range_60",
        "htf_atr_pct_rank_90",
        "market_dispersion_return_4h",
    ],
    "baseline_ohlcv_htf_market": [
        "atr_pct_14",
        "volatility_regime_change_1h",
        "range_expansion_20",
        "return_4h_atr_norm",
        "zscore_vs_vwap_4h",
        "bb_width_pct_rank_120",
        "range_compression_ratio_20_100",
        "trend_efficiency_16",
        "close_pos_range_32",
        "pullback_ema_20_atr",
        "dist_to_prev_day_high_atr",
        "dist_to_prev_day_low_atr",
        "body_to_range",
        "lower_wick_to_range",
        "upper_wick_to_range",
        "volume_rel_median_20",
        "signed_volume_pressure_8",
        "htf_trend_direction",
        "htf_close_pos_range_60",
        "htf_atr_pct_rank_90",
        "rel_day_return_vs_btc",
        "rel_day_return_vs_market",
        "market_dispersion_return_4h",
        "market_breadth_pos_intraday",
        "market_breadth_avg_sign_intraday",
        "btc_daily_trend_direction",
        "btc_intraday_return",
    ],
}

FEATURE_BUILD_REQUEST = {
    "profile": "baseline_ohlcv_htf",
    "include_features": [],
    "exclude_features": [],
    "exclude_blocks": [],
}

EVENT_FILTER_CONFIG = {
    "enabled": False,
    "side": "long",
    "required_columns": [],
}

USE_SYMBOL_FEATURE = False
MANUAL_DISABLED_FEATURE_COLUMNS = [
    
    "price_acceleration_8",
    "consecutive_direction_count",
    "body_to_range",
    "lower_wick_to_range",
    "upper_wick_to_range",
    "bb_width_pct_rank_120",
    "range_compression_ratio_20_100",
    "return_4h_atr_norm",
    "zscore_vs_vwap_4h",
    "volatility_regime_change_1h",
    "market_dispersion_return_4h",
    "dist_to_prev_day_high_atr",
    "dist_to_prev_day_low_atr",
]


# ============================================================
# 4. LABELING AND BARRIERS
# ============================================================
SIDE_MODE = "long_only"
HORIZON = 16

# Legacy fixed barriers are kept for compatibility when BARRIER_MODE = "fixed".
TP_PCT = 0.03
SL_PCT = 0.015

# First-hit outcome labeling:
# target_*_label = 1 only when TP is reached before SL inside HORIZON.
# If neither barrier is reached before the vertical horizon, outcome is NEITHER
# and the binary label is 0.
BARRIER_MODE = "dynamic"  # "dynamic" | "fixed"
BARRIER_ATR_MULTIPLIER = 1.25
BARRIER_TP_TO_SL_RATIO = 2.0
BARRIER_MIN_PCT = 0.0075
BARRIER_MAX_PCT = 0.06


# ============================================================
# 5. TRAINING
# ============================================================
MODEL_NAME = "lightgbm_long_only"
ENABLE_PROD_TRAINING = False

# Baseline keeps NEITHER as part of the negative class. Enable this for strict
# first-hit training, for example allowed_outcomes=["TP_FIRST", "SL_FIRST"].
TRAIN_EVENT_FILTER_CONFIG = {
    "enabled": False,
    "side": "long",
    "allowed_outcomes": ["TP_FIRST", "SL_FIRST"],
}

ENABLE_FEATURE_CLIP = True
FEATURE_CLIP_LOWER_Q = 0.01
FEATURE_CLIP_UPPER_Q = 0.99

LGBM_N_ESTIMATORS = 4000
LGBM_LEARNING_RATE = 0.01
LGBM_NUM_LEAVES = 31
LGBM_MIN_CHILD_SAMPLES = 40
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8
LGBM_REG_ALPHA = 0.1
LGBM_REG_LAMBDA = 0.5
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_LOG_EVAL_PERIOD = 50

TRAIN_SAMPLE_WEIGHT_CONFIG = {
    "enabled": False,
    "outcome_weights": {
        "TP_FIRST": 1.0,
        "SL_FIRST": 1.0,
        "NEITHER": 0.35,
    },
}

TRAIN_VALIDATION_MODE = "walk_forward"  # "single" | "walk_forward"
WF_TRAIN_MONTHS = 6
WF_VAL_MONTHS = 1
WF_TEST_MONTHS = 1
WF_STEP_MONTHS = 1
WF_EMBARGO_BARS = HORIZON


# ============================================================
# 6. BACKTEST AND EXECUTION
# ============================================================
BACKTEST_INITIAL_BALANCE = 100
TAKER_COM = 0.0004
MAKER_COM = 0.0002
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01
ENTRY_THRESHOLD = 0.58

BACKTEST_MAX_NEW_POSITIONS_PER_BAR = 10
BACKTEST_MAX_OPEN_POSITIONS = 100
BACKTEST_MAX_HOLDING_BARS = HORIZON
BACKTEST_SL_COOLDOWN_BARS = 12
BACKTEST_MAX_SL_PER_DAY = 0
BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES = 0
BACKTEST_REDUCED_RISK_PER_TRADE = 0.005


# ============================================================
# 7. SAFETY SWITCHES
# ============================================================
ALLOW_REBUILD_RAW_FROM_FEATURE_ONLY = False


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_override_file(path: str | Path) -> dict:
    if not str(path).strip():
        return {}

    override_path = Path(path)
    if not override_path.exists():
        return {}

    loaded = runpy.run_path(str(override_path))
    if "CONFIG_OVERRIDES" in loaded:
        overrides = loaded["CONFIG_OVERRIDES"]
        if not isinstance(overrides, dict):
            raise ValueError("CONFIG_OVERRIDES must be a dict")
        return dict(overrides)

    return {
        key: value
        for key, value in loaded.items()
        if key.isupper() and not key.startswith("_") and key not in {"CONFIG_OVERRIDES"}
    }


def _preset_path(preset_name: str) -> Path:
    preset = str(preset_name).strip()
    if not preset:
        return Path()
    candidate = Path(preset)
    if candidate.suffix == ".py" or candidate.parent != Path("."):
        return candidate
    return CONFIG_PRESETS_DIR / f"{preset}.py"


def _apply_config_layer(path: str | Path, source_name: str) -> tuple[str | None, set[str]]:
    overrides = _read_override_file(path)
    if not overrides:
        return None, set()

    unknown_keys = sorted(key for key in overrides if key not in globals())
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys in {source_name}: "
            + ", ".join(unknown_keys)
            + ". Add the setting to config.py first to make typos loud."
        )

    for key, value in overrides.items():
        current_value = globals()[key]
        if isinstance(current_value, dict) and isinstance(value, dict):
            globals()[key] = _deep_merge_dict(current_value, value)
        else:
            globals()[key] = value

    return str(Path(path)), set(overrides)


def _apply_config_overrides() -> None:
    global CONFIG_PRESET_SOURCE, CONFIG_PRESET_KEYS
    global CONFIG_OVERRIDE_SOURCE, CONFIG_OVERRIDDEN_KEYS

    if str(CONFIG_PRESET).strip():
        preset_path = _preset_path(CONFIG_PRESET)
        CONFIG_PRESET_SOURCE, CONFIG_PRESET_KEYS = _apply_config_layer(
            preset_path,
            f"{CONFIG_PRESET_ENV} preset",
        )
        if CONFIG_PRESET_SOURCE is None:
            raise ValueError(f"Config preset not found: {CONFIG_PRESET} ({preset_path})")

    CONFIG_OVERRIDE_SOURCE, CONFIG_OVERRIDDEN_KEYS = _apply_config_layer(
        CONFIG_OVERRIDE_PATH,
        CONFIG_OVERRIDE_ENV,
    )


def _sync_derived_defaults() -> None:
    global DATA_DIR, MODELS_DIR, BACKTEST_CHARTS_DIR, DB_PATH, PARQUET_FEATURES_DIR
    global WINDOW_REFERENCE_TIMEFRAME, HTF_WINDOW_REFERENCE_TIMEFRAME
    global WF_EMBARGO_BARS, BACKTEST_MAX_HOLDING_BARS
    global FEATURE_PIPELINE_STAGE, ALLOW_EMPTY_FEATURE_PROFILE

    DATA_DIR = Path(DATA_DIR)
    MODELS_DIR = Path(MODELS_DIR)
    BACKTEST_CHARTS_DIR = Path(BACKTEST_CHARTS_DIR)
    changed_keys = CONFIG_PRESET_KEYS | CONFIG_OVERRIDDEN_KEYS

    if "DB_PATH" not in changed_keys:
        DB_PATH = str(DATA_DIR / "market_data.db")
    else:
        DB_PATH = str(DB_PATH)
    if "PARQUET_FEATURES_DIR" not in changed_keys:
        PARQUET_FEATURES_DIR = DATA_DIR / "ml_features"
    else:
        PARQUET_FEATURES_DIR = Path(PARQUET_FEATURES_DIR)

    if "WINDOW_REFERENCE_TIMEFRAME" not in changed_keys:
        WINDOW_REFERENCE_TIMEFRAME = TIMEFRAME
    if "HTF_WINDOW_REFERENCE_TIMEFRAME" not in changed_keys:
        HTF_WINDOW_REFERENCE_TIMEFRAME = HTF_TIMEFRAME
    if "WF_EMBARGO_BARS" not in changed_keys:
        WF_EMBARGO_BARS = HORIZON
    if "BACKTEST_MAX_HOLDING_BARS" not in changed_keys:
        BACKTEST_MAX_HOLDING_BARS = HORIZON
    if "FEATURE_PIPELINE_STAGE" not in changed_keys:
        FEATURE_PIPELINE_STAGE = CONFIG_STAGE
    if "ALLOW_EMPTY_FEATURE_PROFILE" not in changed_keys:
        ALLOW_EMPTY_FEATURE_PROFILE = CONFIG_STAGE == "scaffold"


def validate_config(context: str = "runtime", strict: bool = False) -> list[str]:
    """Return human-readable config warnings; raise on hard errors."""
    warnings: list[str] = []

    allowed_stages = {"scaffold", "research", "production"}
    if CONFIG_STAGE not in allowed_stages:
        raise ValueError(f"Unsupported CONFIG_STAGE: {CONFIG_STAGE}")

    if ACTIVE_EXCHANGE not in {"bybit", "binance"}:
        raise ValueError(f"Unsupported ACTIVE_EXCHANGE: {ACTIVE_EXCHANGE}")

    if FEATURE_STORAGE not in {"sqlite", "parquet"}:
        raise ValueError(f"Unsupported FEATURE_STORAGE: {FEATURE_STORAGE}")

    profile = str(FEATURE_BUILD_REQUEST.get("profile", "empty") or "empty")
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown FEATURE_BUILD_REQUEST profile: {profile}")

    if not FEATURE_PROFILES.get(profile) and not FEATURE_BUILD_REQUEST.get("include_features"):
        if ALLOW_EMPTY_FEATURE_PROFILE:
            message = (
                f"Feature profile '{profile}' is empty. This is expected in scaffold mode, "
                "but train.py cannot fit a model until real features are enabled."
            )
        else:
            message = f"Feature profile '{profile}' is empty; train.py cannot fit a model until real features are enabled."
        if strict and not ALLOW_EMPTY_FEATURE_PROFILE:
            raise ValueError(message)
        warnings.append(message)

    if TRAIN_VALIDATION_MODE not in {"single", "walk_forward"}:
        raise ValueError(f"Unsupported TRAIN_VALIDATION_MODE: {TRAIN_VALIDATION_MODE}")

    if SIDE_MODE != "long_only":
        raise ValueError(f"Unsupported SIDE_MODE: {SIDE_MODE}")

    train_event_filter_config = TRAIN_EVENT_FILTER_CONFIG or {}
    if not isinstance(train_event_filter_config, dict):
        raise ValueError("TRAIN_EVENT_FILTER_CONFIG must be a dict")
    train_event_filter_side = str(train_event_filter_config.get("side", "long")).strip().lower()
    if train_event_filter_side != "long":
        raise ValueError(f"Unsupported TRAIN_EVENT_FILTER_CONFIG side: {train_event_filter_side}")
    allowed_train_outcomes = train_event_filter_config.get("allowed_outcomes", [])
    if allowed_train_outcomes is None:
        allowed_train_outcomes = []
    if not isinstance(allowed_train_outcomes, (list, tuple, set)):
        raise ValueError("TRAIN_EVENT_FILTER_CONFIG allowed_outcomes must be a list, tuple, or set")
    unknown_train_outcomes = sorted(
        {str(outcome).strip().upper() for outcome in allowed_train_outcomes}
        - {"TP_FIRST", "SL_FIRST", "NEITHER"}
    )
    if unknown_train_outcomes:
        raise ValueError("Unsupported TRAIN_EVENT_FILTER_CONFIG allowed_outcomes: " + ", ".join(unknown_train_outcomes))

    if BARRIER_MODE not in {"dynamic", "fixed"}:
        raise ValueError(f"Unsupported BARRIER_MODE: {BARRIER_MODE}")

    if HORIZON <= 0:
        raise ValueError("HORIZON must be > 0")

    if not 0 <= FEATURE_CLIP_LOWER_Q < FEATURE_CLIP_UPPER_Q <= 1:
        raise ValueError("FEATURE_CLIP_LOWER_Q and FEATURE_CLIP_UPPER_Q must satisfy 0 <= lower < upper <= 1")

    if not 0 < ENTRY_THRESHOLD < 1:
        raise ValueError("ENTRY_THRESHOLD must be between 0 and 1")

    if CONFIG_STAGE == "production" and ENABLE_PROD_TRAINING:
        warnings.append("ENABLE_PROD_TRAINING is enabled; saved artifacts will include the full dataset.")

    if CONFIG_PRESET_SOURCE:
        warnings.append(
            "Loaded config preset from "
            f"{CONFIG_PRESET_SOURCE}: {', '.join(sorted(CONFIG_PRESET_KEYS)) or 'no keys'}"
        )

    if CONFIG_OVERRIDE_SOURCE:
        warnings.append(
            "Loaded config override from "
            f"{CONFIG_OVERRIDE_SOURCE}: {', '.join(sorted(CONFIG_OVERRIDDEN_KEYS)) or 'no keys'}"
        )

    if context == "train" and ALLOW_EMPTY_FEATURE_PROFILE:
        warnings.append("CONFIG_STAGE=scaffold: ETL can prepare labels, but training still requires feature columns.")

    return warnings


_apply_config_overrides()
_sync_derived_defaults()

for _directory in (DATA_DIR, PARQUET_FEATURES_DIR, MODELS_DIR, BACKTEST_CHARTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
