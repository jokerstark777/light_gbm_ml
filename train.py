import argparse
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

import config as cfg
from src.features.builders.compression_expansion_feature_builder import CompressionExpansionFeatureBuilder
from src.features.builders.entry_location_feature_builder import EntryLocationFeatureBuilder
from src.features.builders.htf_context_feature_builder import HtfContextFeatureBuilder
from src.features.builders.interaction_feature_builder import InteractionFeatureBuilder
from src.features.builders.market_context_feature_builder import MarketContextFeatureBuilder
from src.features.builders.ohlcv_baseline_feature_builder import OhlcvBaselineFeatureBuilder
from src.features.builders.regime_feature_builder import RegimeFeatureBuilder
from src.features.builders.session_context_feature_builder import SessionContextFeatureBuilder
from src.features.builders.trend_feature_builder import TrendFeatureBuilder
from src.features.builders.volume_flow_feature_builder import VolumeFlowFeatureBuilder
from src.features.indicators import normalize_bars_for_timeframe
from src.features.models.feature_request import resolve_feature_request
from src.persistence.feature_store_factory import create_feature_store
from src.utils.validation import generate_walk_forward_splits, validate_walk_forward_args, resolve_timestamp_index_on_or_after

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "target_long_label"
TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
OUTCOME_TP_FIRST = "TP_FIRST"
OUTCOME_SL_FIRST = "SL_FIRST"
OUTCOME_NEITHER = "NEITHER"
RESERVED_COLUMNS = {
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    "barrier_stop_pct",
    "barrier_take_pct",
    "target_long_outcome",
    "target_long_pnl",
    "target_long_exit_reason",
    "target_long_mfe",
    "target_long_mae",
    "target_long_bars_to_tp",
    "target_long_bars_to_sl",
    "target_long_holding_bars",
    "target_short_label",
    "target_short_outcome",
    "target_short_pnl",
    "target_short_exit_reason",
    "target_short_mfe",
    "target_short_mae",
    "target_short_bars_to_tp",
    "target_short_bars_to_sl",
    "target_short_holding_bars",
}
EXCLUDED_RAW_FEATURE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}
VALID_TRAIN_OUTCOMES = {OUTCOME_TP_FIRST, OUTCOME_SL_FIRST, OUTCOME_NEITHER}


def resolve_requested_feature_columns():
    block_features: dict[str, set[str]] = {}
    main_timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
    htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
    main_reference_timeframe = str(getattr(cfg, "WINDOW_REFERENCE_TIMEFRAME", "5m"))
    htf_reference_timeframe = str(getattr(cfg, "HTF_WINDOW_REFERENCE_TIMEFRAME", htf_timeframe))
    builders = [
        OhlcvBaselineFeatureBuilder(),
        TrendFeatureBuilder(timeframe_label=main_timeframe, reference_timeframe=main_reference_timeframe),
        RegimeFeatureBuilder(timeframe_label=main_timeframe, reference_timeframe=main_reference_timeframe),
        CompressionExpansionFeatureBuilder(
            timeframe_label=main_timeframe,
            reference_timeframe=main_reference_timeframe,
        ),
        EntryLocationFeatureBuilder(
            timeframe_label=main_timeframe,
            reference_timeframe=main_reference_timeframe,
        ),
        InteractionFeatureBuilder(),
        MarketContextFeatureBuilder(),
        SessionContextFeatureBuilder(),
        VolumeFlowFeatureBuilder(
            timeframe_label=main_timeframe,
            reference_timeframe=main_reference_timeframe,
        ),
        HtfContextFeatureBuilder(),
        TrendFeatureBuilder(
            timeframe_label=htf_timeframe,
            use_htf=True,
            reference_timeframe=htf_reference_timeframe,
        ),
        RegimeFeatureBuilder(
            timeframe_label=htf_timeframe,
            use_htf=True,
            reference_timeframe=htf_reference_timeframe,
        ),
    ]
    for builder in builders:
        block_features.setdefault(builder.block_name, set()).update(builder.provides())

    resolved = resolve_feature_request(
        getattr(cfg, "FEATURE_BUILD_REQUEST", {}),
        getattr(cfg, "FEATURE_PROFILES", {}),
        block_features,
    )
    return set(resolved.active_features), resolved.profile


def get_end_date_cutoff():
    end_date = getattr(cfg, "END_DATE", None)
    if not end_date:
        return None
    return pd.to_datetime(end_date, errors="coerce")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM classifier on prepared ETL datasets.")
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="Symbols to load, for example ETH/USDT SOL/USDT.",
    )
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation share for chronological split.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Holdout test share for chronological split.")
    parser.add_argument(
        "--validation-mode",
        choices=["single", "walk_forward"],
        default=str(getattr(cfg, "TRAIN_VALIDATION_MODE", "single")),
        help="Validation mode: single chronological split or walk-forward validation.",
    )
    parser.add_argument(
        "--wf-train-months",
        type=int,
        default=int(getattr(cfg, "WF_TRAIN_MONTHS", 6)),
        help="Walk-forward train window size in months.",
    )
    parser.add_argument(
        "--wf-val-months",
        type=int,
        default=int(getattr(cfg, "WF_VAL_MONTHS", 1)),
        help="Walk-forward validation window size in months.",
    )
    parser.add_argument(
        "--wf-test-months",
        type=int,
        default=int(getattr(cfg, "WF_TEST_MONTHS", 1)),
        help="Walk-forward test window size in months.",
    )
    parser.add_argument(
        "--wf-step-months",
        type=int,
        default=int(getattr(cfg, "WF_STEP_MONTHS", 1)),
        help="Walk-forward step size in months.",
    )
    parser.add_argument(
        "--wf-embargo-bars",
        type=int,
        default=normalize_bars_for_timeframe(
            int(getattr(cfg, "WF_EMBARGO_BARS", getattr(cfg, "HORIZON", 0))),
            str(getattr(cfg, "TIMEFRAME", "1h")),
            str(getattr(cfg, "WINDOW_REFERENCE_TIMEFRAME", "5m")),
            minimum=0,
        ),
        help="Number of unique timestamps to skip between train/validation and validation/test windows.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=getattr(cfg, "TIMEFRAME", "1h"),
        help="Chart timeframe string (e.g., '1h', '1d') for time-based embargo calculation.",
    )
    parser.add_argument(
        "--model-name",
        default=getattr(cfg, "MODEL_NAME", "lightgbm_long_only"),
        help="Base filename for saved artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--prod-train",
        action="store_true",
        default=bool(getattr(cfg, "ENABLE_PROD_TRAINING", False)),
        help="After validation, retrain the final model on the full dataset.",
    )
    parser.add_argument(
        "--wf-start-fold",
        type=int,
        default=1,
        help="1-based walk-forward fold number to start from. Default keeps the full run from fold 1.",
    )
    parser.add_argument(
        "--wf-max-folds",
        type=int,
        default=0,
        help="Maximum number of walk-forward folds to run. 0 means all available folds.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast research mode: run only the most recent folds with a smaller LightGBM budget.",
    )
    parser.add_argument(
        "--quick-folds",
        type=int,
        default=5,
        help="When --quick is enabled, run only the last N walk-forward folds.",
    )
    parser.add_argument(
        "--quick-n-estimators",
        type=int,
        default=800,
        help="When --quick is enabled, use this LightGBM tree budget.",
    )
    parser.add_argument(
        "--quick-log-eval-period",
        type=int,
        default=100,
        help="When --quick is enabled, log LightGBM evaluation every N rounds.",
    )

    args = parser.parse_args()
    if args.wf_start_fold <= 0:
        raise ValueError("--wf-start-fold must be >= 1.")
    if args.wf_max_folds < 0:
        raise ValueError("--wf-max-folds must be >= 0.")
    if args.quick_folds <= 0:
        raise ValueError("--quick-folds must be > 0.")
    if args.quick_n_estimators <= 0:
        raise ValueError("--quick-n-estimators must be > 0.")
    if args.quick_log_eval_period <= 0:
        raise ValueError("--quick-log-eval-period must be > 0.")

    if args.quick and args.validation_mode != "walk_forward":
        raise ValueError("--quick is only supported with --validation-mode walk_forward.")

    default_model_name = getattr(cfg, "MODEL_NAME", "lightgbm_long_only")
    model_name_was_default = args.model_name == default_model_name
    if args.quick and model_name_was_default:
        args.model_name = f"{args.model_name}_quick"
    return args


def resolve_side_mode() -> str:
    side_mode = str(getattr(cfg, "SIDE_MODE", "long_only")).strip().lower()
    if side_mode != "long_only":
        raise ValueError(f"Unsupported SIDE_MODE: {side_mode}")
    return side_mode


def _clean_json_key(value) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _clean_optional_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def value_counts_payload(series: pd.Series) -> dict[str, int]:
    return {
        _clean_json_key(key): int(count)
        for key, count in series.value_counts(dropna=False).items()
    }


def resolve_train_event_filter_config() -> dict:
    raw_config = getattr(cfg, "TRAIN_EVENT_FILTER_CONFIG", None) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("TRAIN_EVENT_FILTER_CONFIG must be a dict")

    enabled = bool(raw_config.get("enabled", False))
    side = str(raw_config.get("side", "long")).strip().lower()
    if side != "long":
        raise ValueError(f"Unsupported TRAIN_EVENT_FILTER_CONFIG side: {side}")

    allowed_outcomes = raw_config.get("allowed_outcomes", [OUTCOME_TP_FIRST, OUTCOME_SL_FIRST])
    if allowed_outcomes is None:
        allowed_outcomes = []
    if isinstance(allowed_outcomes, str):
        allowed_outcomes = [allowed_outcomes]
    allowed_outcomes = [str(outcome).strip().upper() for outcome in allowed_outcomes if str(outcome).strip()]
    if enabled and not allowed_outcomes:
        allowed_outcomes = [OUTCOME_TP_FIRST, OUTCOME_SL_FIRST]

    unknown_outcomes = sorted(set(allowed_outcomes) - VALID_TRAIN_OUTCOMES)
    if unknown_outcomes:
        raise ValueError("Unsupported TRAIN_EVENT_FILTER_CONFIG allowed_outcomes: " + ", ".join(unknown_outcomes))

    outcome_column = f"target_{side}_outcome"
    target_column = f"target_{side}_label"
    if target_column != TARGET_COLUMN:
        raise ValueError(f"Unsupported train target column from TRAIN_EVENT_FILTER_CONFIG: {target_column}")

    return {
        "enabled": enabled,
        "side": side,
        "allowed_outcomes": allowed_outcomes,
        "outcome_column": outcome_column,
        "target_column": target_column,
    }


def build_target_event_summary(dataset: pd.DataFrame, side: str = "long") -> dict:
    outcome_column = f"target_{side}_outcome"
    label_column = f"target_{side}_label"
    if outcome_column not in dataset.columns:
        return {"available": False, "reason": f"missing {outcome_column}"}

    numeric_columns = [
        f"target_{side}_pnl",
        f"target_{side}_mfe",
        f"target_{side}_mae",
        f"target_{side}_bars_to_tp",
        f"target_{side}_bars_to_sl",
        f"target_{side}_holding_bars",
    ]
    available_numeric_columns = [column for column in numeric_columns if column in dataset.columns]
    by_outcome = {}

    for outcome, group in dataset.groupby(outcome_column, dropna=False):
        outcome_key = _clean_json_key(outcome)
        record = {"rows": int(len(group))}
        if label_column in group.columns:
            record["positive_rate"] = _clean_optional_float(pd.to_numeric(group[label_column], errors="coerce").mean())
        for column in available_numeric_columns:
            series = pd.to_numeric(group[column], errors="coerce")
            record[f"{column}_mean"] = _clean_optional_float(series.mean())
            record[f"{column}_median"] = _clean_optional_float(series.median())
        by_outcome[outcome_key] = record

    return {
        "available": True,
        "side": side,
        "rows": int(len(dataset)),
        "outcome_counts": value_counts_payload(dataset[outcome_column]),
        "by_outcome": by_outcome,
    }


def apply_train_event_filter(dataset: pd.DataFrame) -> pd.DataFrame:
    filter_config = resolve_train_event_filter_config()
    outcome_column = filter_config["outcome_column"]
    before_rows = len(dataset)

    if outcome_column in dataset.columns:
        before_outcome_counts = value_counts_payload(dataset[outcome_column])
    else:
        before_outcome_counts = {}

    if filter_config["enabled"]:
        if outcome_column not in dataset.columns:
            raise ValueError(f"Train event filter requires {outcome_column}. Re-run etl.py to rebuild event targets.")
        mask = dataset[outcome_column].isin(filter_config["allowed_outcomes"])
        dataset = dataset.loc[mask].copy()
        if dataset.empty:
            raise ValueError(
                "Train event filter removed all rows. Check TRAIN_EVENT_FILTER_CONFIG allowed_outcomes and ETL targets."
            )

    excluded_rows = before_rows - len(dataset)
    dataset.attrs["training_universe_rows"] = int(before_rows)
    dataset.attrs["excluded_by_train_event_filter_rows"] = int(excluded_rows)
    dataset.attrs["train_event_filter_config"] = filter_config
    dataset.attrs["target_long_outcome_counts_before_filter"] = before_outcome_counts
    dataset.attrs["target_long_outcome_counts"] = (
        value_counts_payload(dataset["target_long_outcome"]) if "target_long_outcome" in dataset.columns else {}
    )
    dataset.attrs["target_event_summary"] = build_target_event_summary(dataset, side="long")
    if filter_config["enabled"] and set(filter_config["allowed_outcomes"]) == {OUTCOME_TP_FIRST, OUTCOME_SL_FIRST}:
        label_mode = "strict_long_tp_first_vs_sl_first"
    elif filter_config["enabled"]:
        label_mode = "filtered_long_tp_first_vs_allowed_non_tp"
    else:
        label_mode = "baseline_long_tp_first_vs_all_non_tp"
    dataset.attrs["train_label_mode"] = label_mode
    return dataset


def load_training_frame(db_path, symbols):
    feature_store = create_feature_store(db_path=db_path)
    dataset = feature_store.load_feature_dataset(symbols)
    dataset = dataset.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    end_cutoff = get_end_date_cutoff()
    if end_cutoff is not None and not pd.isna(end_cutoff):
        before_rows = len(dataset)
        dataset = dataset.loc[dataset[TIMESTAMP_COLUMN] <= end_cutoff].copy()
        logger.info(
            "Applied END_DATE cutoff at %s: kept %s/%s rows",
            end_cutoff,
            len(dataset),
            before_rows,
        )

    raw_labels = dataset[TARGET_COLUMN].astype(int)
    legacy_directional_labels = sorted(set(raw_labels.unique()).intersection({-1}))
    if legacy_directional_labels:
        raise ValueError(
            f"Legacy directional labels detected in {TARGET_COLUMN} (-1/0/1). Re-run etl.py to rebuild binary long-only labels (0/1)."
        )
    unknown_labels = sorted(set(raw_labels.unique()) - {0, 1})
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {TARGET_COLUMN}: {unknown_labels}")

    dataset = apply_train_event_filter(dataset)
    dataset[SYMBOL_COLUMN] = dataset[SYMBOL_COLUMN].astype("category")
    dataset.attrs["target_long_positive_rows"] = int(dataset[TARGET_COLUMN].sum())
    dataset.attrs["target_long_negative_rows"] = int((dataset[TARGET_COLUMN] == 0).sum())
    return dataset


def select_feature_columns(dataset):
    feature_columns = []
    use_symbol_feature = bool(getattr(cfg, "USE_SYMBOL_FEATURE", False))
    disabled_feature_columns = set(getattr(cfg, "MANUAL_DISABLED_FEATURE_COLUMNS", []))
    requested_feature_columns, profile_name = resolve_requested_feature_columns()

    for column in dataset.columns:
        if column in RESERVED_COLUMNS:
            continue
        if column in EXCLUDED_RAW_FEATURE_COLUMNS:
            continue
        if column in disabled_feature_columns:
            continue
        if column != SYMBOL_COLUMN and column not in requested_feature_columns:
            continue
        if column == SYMBOL_COLUMN:
            if use_symbol_feature:
                feature_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise RuntimeError("No usable feature columns found in the dataset.")
    if disabled_feature_columns:
        disabled_present = sorted(disabled_feature_columns.intersection(dataset.columns))
        if disabled_present:
            logger.info(
                "Config disabled %s feature columns, excluding them from training: %s",
                len(disabled_present),
                ", ".join(disabled_present),
            )
    feature_columns, dropped_constant_columns = drop_constant_feature_columns(dataset, feature_columns)
    if dropped_constant_columns:
        logger.info(
            "Detected %s globally constant feature columns in the training universe, excluding them from training: %s",
            len(dropped_constant_columns),
            ", ".join(sorted(dropped_constant_columns)),
        )
    logger.info(
        "Training feature profile resolved: %s | selected %s configured feature columns",
        profile_name,
        len([column for column in feature_columns if column != SYMBOL_COLUMN]),
    )
    return feature_columns


def get_clippable_feature_columns(dataset, feature_columns):
    clippable_columns = []
    for column in feature_columns:
        if column == SYMBOL_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            clippable_columns.append(column)
    return clippable_columns


def _finite_feature_series(frame, column):
    return frame[column].replace([np.inf, -np.inf], np.nan).dropna()


def build_feature_clip_bounds(train_df, feature_columns):
    if not bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)):
        return {}, []

    lower_q = float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01))
    upper_q = float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99))
    min_unique_values = int(getattr(cfg, "FEATURE_CLIP_MIN_UNIQUE_VALUES", 5))
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("FEATURE_CLIP_LOWER_Q and FEATURE_CLIP_UPPER_Q must satisfy 0 <= lower < upper <= 1.")

    clip_bounds = {}
    skipped_low_cardinality = []
    for column in get_clippable_feature_columns(train_df, feature_columns):
        series = _finite_feature_series(train_df, column)
        if series.empty:
            continue
        if int(series.nunique(dropna=True)) < min_unique_values:
            skipped_low_cardinality.append(column)
            continue
        lower = series.quantile(lower_q)
        upper = series.quantile(upper_q)
        if pd.isna(lower) or pd.isna(upper):
            continue
        if float(lower) >= float(upper):
            skipped_low_cardinality.append(column)
            continue
        clip_bounds[column] = {"lower": float(lower), "upper": float(upper)}
    return clip_bounds, skipped_low_cardinality


def apply_feature_clip_bounds(frame, clip_bounds):
    if not clip_bounds:
        return frame

    clipped = frame.copy()
    for column, bounds in clip_bounds.items():
        if column not in clipped.columns:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["lower"], upper=bounds["upper"])
    return clipped


def drop_constant_feature_columns(frame, feature_columns):
    active_feature_columns = []
    dropped_feature_columns = []
    for column in feature_columns:
        if column == SYMBOL_COLUMN:
            active_feature_columns.append(column)
            continue
        series = _finite_feature_series(frame, column)
        if series.empty or int(series.nunique(dropna=True)) <= 1:
            dropped_feature_columns.append(column)
            continue
        active_feature_columns.append(column)

    numeric_feature_count = len([column for column in active_feature_columns if column != SYMBOL_COLUMN])
    if numeric_feature_count == 0:
        if SYMBOL_COLUMN in active_feature_columns:
            logger.info("No non-constant numeric features remain; continuing with symbol-only feature set.")
        else:
            raise RuntimeError("No non-constant numeric feature columns remain after train-window preparation.")
    return active_feature_columns, dropped_feature_columns


def prepare_modeling_frames(train_df, valid_df, test_df, feature_columns, split_label):
    clip_bounds, skipped_clip_columns = build_feature_clip_bounds(train_df, feature_columns)
    train_df = apply_feature_clip_bounds(train_df, clip_bounds)
    valid_df = apply_feature_clip_bounds(valid_df, clip_bounds)
    test_df = apply_feature_clip_bounds(test_df, clip_bounds)
    active_feature_columns, dropped_constant_columns = drop_constant_feature_columns(train_df, feature_columns)

    if clip_bounds:
        logger.info(
            "%s | feature clipping enabled: %s numeric columns clipped to [%.2f%%, %.2f%%] train percentiles",
            split_label,
            len(clip_bounds),
            float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)) * 100,
            float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)) * 100,
        )
    if skipped_clip_columns:
        logger.info(
            "%s | skipped clipping for %s low-cardinality features: %s",
            split_label,
            len(skipped_clip_columns),
            ", ".join(sorted(skipped_clip_columns)),
        )
    if dropped_constant_columns:
        logger.info(
            "%s | dropped %s constant features after train-window preparation: %s",
            split_label,
            len(dropped_constant_columns),
            ", ".join(sorted(dropped_constant_columns)),
        )

    return train_df, valid_df, test_df, active_feature_columns, clip_bounds


def build_period_payload(frame):
    if frame.empty:
        return None
    return {
        "start": str(frame[TIMESTAMP_COLUMN].iloc[0]),
        "end": str(frame[TIMESTAMP_COLUMN].iloc[-1]),
    }


def time_split(dataset, val_size, test_size):
    if not 0 < val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if (val_size + test_size) >= 1:
        raise ValueError("--val-size + --test-size must be less than 1.")

    unique_timestamps = dataset[TIMESTAMP_COLUMN].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_timestamps) < 3:
        raise RuntimeError("Need at least 3 unique timestamps for a chronological train/validation/test split.")

    train_end_idx = int(len(unique_timestamps) * (1 - val_size - test_size))
    valid_end_idx = int(len(unique_timestamps) * (1 - test_size))

    train_end_idx = max(1, train_end_idx)
    valid_end_idx = max(train_end_idx + 1, valid_end_idx)
    valid_end_idx = min(valid_end_idx, len(unique_timestamps) - 1)
    if train_end_idx >= valid_end_idx:
        raise RuntimeError("Chronological split is too small for separate validation and test windows.")

    valid_start_ts = unique_timestamps.iloc[train_end_idx]
    test_start_ts = unique_timestamps.iloc[valid_end_idx]

    train_df = dataset.loc[dataset[TIMESTAMP_COLUMN] < valid_start_ts].copy()
    valid_df = dataset.loc[
        (dataset[TIMESTAMP_COLUMN] >= valid_start_ts) & (dataset[TIMESTAMP_COLUMN] < test_start_ts)
    ].copy()
    test_df = dataset.loc[dataset[TIMESTAMP_COLUMN] >= test_start_ts].copy()
    return train_df, valid_df, test_df


def validate_split(train_df, valid_df, test_df):
    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(
            "Train/validation/test split produced an empty part. Adjust split sizes or prepare more data."
        )

    train_last_ts = train_df[TIMESTAMP_COLUMN].max()
    valid_first_ts = valid_df[TIMESTAMP_COLUMN].min()
    valid_last_ts = valid_df[TIMESTAMP_COLUMN].max()
    test_first_ts = test_df[TIMESTAMP_COLUMN].min()
    if train_last_ts >= valid_first_ts:
        raise RuntimeError("Train/validation split has overlapping timestamps, which would leak validation context.")
    if valid_last_ts >= test_first_ts:
        raise RuntimeError("Validation/test split has overlapping timestamps, which would leak holdout context.")

    train_classes = sorted(train_df[TARGET_COLUMN].unique().tolist())
    if len(train_classes) < 2:
        raise RuntimeError(f"Training split has too few classes for LightGBM: {train_classes}")


def select_walk_forward_folds(folds, args):
    if not folds:
        return folds

    total_folds = len(folds)
    if args.quick:
        selected = folds[-int(args.quick_folds):]
        logger.info(
            "Quick mode enabled: using last %s/%s walk-forward folds with n_estimators=%s and log_eval_period=%s",
            len(selected),
            total_folds,
            int(args.quick_n_estimators),
            int(args.quick_log_eval_period),
        )
        return selected

    start_index = int(args.wf_start_fold) - 1
    if start_index >= total_folds:
        raise RuntimeError(
            f"--wf-start-fold={args.wf_start_fold} is out of range. Available folds: 1..{total_folds}."
        )

    selected = folds[start_index:]
    if int(args.wf_max_folds) > 0:
        selected = selected[: int(args.wf_max_folds)]

    logger.info(
        "Walk-forward fold selection: using folds %s..%s of %s total",
        int(selected[0]["fold_id"]),
        int(selected[-1]["fold_id"]),
        total_folds,
    )
    return selected


def build_model(seed, n_estimators=None):
    if n_estimators is None:
        n_estimators = int(getattr(cfg, "LGBM_N_ESTIMATORS", 2000))
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=float(getattr(cfg, "LGBM_LEARNING_RATE", 0.01)),
        num_leaves=int(getattr(cfg, "LGBM_NUM_LEAVES", 31)),
        min_child_samples=int(getattr(cfg, "LGBM_MIN_CHILD_SAMPLES", 40)),
        subsample=float(getattr(cfg, "LGBM_SUBSAMPLE", 0.8)),
        colsample_bytree=float(getattr(cfg, "LGBM_COLSAMPLE_BYTREE", 0.8)),
        reg_alpha=float(getattr(cfg, "LGBM_REG_ALPHA", 0.1)),
        reg_lambda=float(getattr(cfg, "LGBM_REG_LAMBDA", 0.5)),
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def build_sample_weight(frame: pd.DataFrame) -> np.ndarray | None:
    weight_config = getattr(cfg, "TRAIN_SAMPLE_WEIGHT_CONFIG", {}) or {}
    if not bool(weight_config.get("enabled", False)):
        return None
    if "target_long_outcome" not in frame.columns:
        return None

    outcome_weights = {
        str(key).strip().upper(): float(value)
        for key, value in (weight_config.get("outcome_weights", {}) or {}).items()
    }
    default_weight = float(weight_config.get("default_weight", 1.0))
    weights = (
        frame["target_long_outcome"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(outcome_weights)
        .fillna(default_weight)
        .astype(float)
    )
    return weights.values


def train_validation_model(train_df, valid_df, feature_columns, seed, args=None):
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN]
    train_weight = build_sample_weight(train_df)
    valid_weight = build_sample_weight(valid_df)

    n_estimators = int(args.quick_n_estimators) if args is not None and bool(getattr(args, "quick", False)) else None
    log_eval_period = (
        int(args.quick_log_eval_period)
        if args is not None and bool(getattr(args, "quick", False))
        else int(getattr(cfg, "LGBM_LOG_EVAL_PERIOD", 50))
    )

    model = build_model(seed=seed, n_estimators=n_estimators)
    model.fit(
        x_train,
        y_train,
        sample_weight=train_weight,
        eval_set=[(x_valid, y_valid)],
        eval_sample_weight=[valid_weight] if valid_weight is not None else None,
        eval_metric="binary_logloss",
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(getattr(cfg, "LGBM_EARLY_STOPPING_ROUNDS", 50)),
                verbose=False,
            ),
            lgb.log_evaluation(period=log_eval_period),
        ],
    )
    return model


def build_prediction_frame(model, eval_df, feature_columns):
    x_eval = eval_df[feature_columns]
    y_true = eval_df[TARGET_COLUMN].astype(int)
    y_pred = model.predict(x_eval)
    y_proba = model.predict_proba(x_eval)
    p_signal = y_proba[:, 1]

    predictions = eval_df[[TIMESTAMP_COLUMN]].copy()
    if SYMBOL_COLUMN in eval_df.columns:
        predictions[SYMBOL_COLUMN] = eval_df[SYMBOL_COLUMN].astype(str)
    predictions["y_true"] = y_true.values
    predictions["y_pred"] = np.asarray(y_pred).astype(int)
    predictions["p_signal"] = np.asarray(p_signal, dtype=float)
    return predictions.reset_index(drop=True)


def compute_metrics_from_prediction_frame(predictions, split_name):
    if predictions.empty:
        raise RuntimeError(f"{split_name} evaluation frame is empty.")

    y_true_values = predictions["y_true"].astype(int).values
    y_pred_values = predictions["y_pred"].astype(int).values
    p_signal_values = predictions["p_signal"].astype(float).values
    signal_label = "long_tp_first"
    no_signal_label = "not_long_tp_first"

    report = classification_report(
        y_true_values,
        y_pred_values,
        labels=[0, 1],
        target_names=[no_signal_label, signal_label],
        output_dict=True,
        zero_division=0,
    )

    confidence_thresholds = sorted(
        {
            round(
                max(
                    0.5,
                    float(cfg.ENTRY_THRESHOLD),
                ),
                2,
            ),
            0.55,
            0.60,
            0.65,
            0.70,
        }
    )
    probability_threshold_metrics = {}
    for threshold in confidence_thresholds:
        threshold = float(threshold)
        signal = (p_signal_values >= threshold).astype(int)
        signal_count = int(signal.sum())
        no_trade = int(len(predictions) - signal_count)
        coverage = float(signal_count / len(predictions)) if len(predictions) else 0.0
        threshold_report = classification_report(
            y_true_values,
            signal,
            labels=[0, 1],
            target_names=[no_signal_label, signal_label],
            output_dict=True,
            zero_division=0,
        )
        signal_tp = int(((signal == 1) & (y_true_values == 1)).sum())
        total_true_signal = int((y_true_values == 1).sum())

        probability_threshold_metrics[f"{threshold:.2f}"] = {
            "rows": int(len(predictions)),
            "coverage": coverage,
            "long_signals": signal_count,
            "no_trade": no_trade,
            "signal_accuracy": float(accuracy_score(y_true_values, signal)),
            "signal_balanced_accuracy": float(balanced_accuracy_score(y_true_values, signal)),
            "signal_f1_macro": float(f1_score(y_true_values, signal, average="macro")),
            "signal_confusion_matrix": confusion_matrix(y_true_values, signal, labels=[0, 1]).tolist(),
            "signal_classification_report": threshold_report,
            "long_precision": float(signal_tp / signal_count) if signal_count > 0 else None,
            "long_recall_all": float(signal_tp / total_true_signal) if total_true_signal > 0 else 0.0,
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true_values, y_pred_values)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_values, y_pred_values)),
        "f1_macro": float(f1_score(y_true_values, y_pred_values, average="macro")),
        "roc_auc": float(roc_auc_score(y_true_values, p_signal_values)),
        "pr_auc": float(average_precision_score(y_true_values, p_signal_values)),
        "mcc": float(matthews_corrcoef(y_true_values, y_pred_values)),
        "confusion_matrix": confusion_matrix(y_true_values, y_pred_values, labels=[0, 1]).tolist(),
        "classification_report": report,
        f"{split_name}_rows": int(len(predictions)),
        "probability_threshold_metrics": probability_threshold_metrics,
    }
    return metrics


def evaluate_model(model, eval_df, feature_columns, split_name):
    predictions = build_prediction_frame(model, eval_df, feature_columns)
    return compute_metrics_from_prediction_frame(predictions, split_name=split_name)


def retrain_full_model(dataset, feature_columns, seed, best_iteration):
    n_estimators = int(best_iteration) if best_iteration and best_iteration > 0 else 200
    final_model = build_model(seed=seed, n_estimators=n_estimators)
    final_model.fit(
        dataset[feature_columns],
        dataset[TARGET_COLUMN],
        sample_weight=build_sample_weight(dataset),
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
    )
    return final_model


def top_feature_importance(model, feature_columns, limit=25):
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    return importance.head(limit)


def build_model_feature_importance(model, feature_columns, fold_id: int | None = None) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    )
    importance["gain_rank"] = importance["importance_gain"].rank(method="min", ascending=False).astype(int)
    importance["split_rank"] = importance["importance_split"].rank(method="min", ascending=False).astype(int)
    if fold_id is not None:
        importance.insert(0, "fold_id", int(fold_id))
    return importance.sort_values(["importance_gain", "importance_split"], ascending=False).reset_index(drop=True)


def summarize_feature_stability(fold_importance: pd.DataFrame) -> pd.DataFrame:
    if fold_importance.empty:
        return pd.DataFrame()

    total_folds = int(fold_importance["fold_id"].nunique())
    prepared = fold_importance.copy()
    prepared["used"] = prepared["importance_gain"] > 0
    prepared["top5"] = prepared["gain_rank"] <= 5
    prepared["top10"] = prepared["gain_rank"] <= 10

    summary = (
        prepared.groupby("feature", as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            mean_gain=("importance_gain", "mean"),
            median_gain=("importance_gain", "median"),
            std_gain=("importance_gain", "std"),
            total_gain=("importance_gain", "sum"),
            mean_split=("importance_split", "mean"),
            median_split=("importance_split", "median"),
            nonzero_fold_ratio=("used", "mean"),
            top5_fold_ratio=("top5", "mean"),
            top10_fold_ratio=("top10", "mean"),
            median_rank=("gain_rank", "median"),
            mean_rank=("gain_rank", "mean"),
            std_rank=("gain_rank", "std"),
        )
    )
    summary["fold_coverage"] = summary["folds"] / max(total_folds, 1)
    summary["std_gain"] = summary["std_gain"].fillna(0.0)
    summary["std_rank"] = summary["std_rank"].fillna(0.0)
    summary["stability_score"] = (
        summary["median_gain"]
        * summary["nonzero_fold_ratio"]
        * summary["top10_fold_ratio"].clip(lower=0.05)
        / (1.0 + summary["std_rank"])
    )
    return summary.sort_values(
        ["stability_score", "nonzero_fold_ratio", "median_gain"],
        ascending=False,
    ).reset_index(drop=True)


def log_feature_stability_ranking(feature_stability: pd.DataFrame, limit: int = 20) -> None:
    if feature_stability.empty:
        return
    logger.info("Walk-forward feature stability ranking:")
    for rank, row in enumerate(feature_stability.head(limit).itertuples(index=False), start=1):
        logger.info(
            "%s. %s | score=%.6f | nonzero=%.2f | top5=%.2f | top10=%.2f | median_gain=%.6f | median_rank=%.1f",
            rank,
            row.feature,
            float(row.stability_score),
            float(row.nonzero_fold_ratio),
            float(row.top5_fold_ratio),
            float(row.top10_fold_ratio),
            float(row.median_gain),
            float(row.median_rank),
        )


def log_feature_importance_ranking(model, feature_columns):
    importance = build_model_feature_importance(model, feature_columns)

    logger.info("Feature importance ranking:")
    for rank, row in enumerate(importance.itertuples(index=False), start=1):
        logger.info(
            "%s. %s | gain=%.6f | split=%s",
            rank,
            row.feature,
            float(row.importance_gain),
            int(row.importance_split),
        )


def summarize_single_split(dataset, feature_columns, args):
    train_df, valid_df, test_df = time_split(dataset, args.val_size, args.test_size)
    validate_split(train_df, valid_df, test_df)
    train_df, valid_df, test_df, active_feature_columns, clip_bounds = prepare_modeling_frames(
        train_df,
        valid_df,
        test_df,
        feature_columns,
        split_label="Single split",
    )
    logger.info(
        "Chronological split: train=%s rows, valid=%s rows, test=%s rows | valid starts at %s | test starts at %s",
        len(train_df),
        len(valid_df),
        len(test_df),
        valid_df[TIMESTAMP_COLUMN].iloc[0],
        test_df[TIMESTAMP_COLUMN].iloc[0],
    )

    model = train_validation_model(train_df, valid_df, active_feature_columns, args.seed, args=args)
    validation_metrics = evaluate_model(model, valid_df, active_feature_columns, split_name="validation")
    test_metrics = evaluate_model(model, test_df, active_feature_columns, split_name="test")
    metrics = {
        "validation_mode": "single",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(model.best_iteration_ or model.n_estimators_),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(active_feature_columns)),
        "side_mode": resolve_side_mode(),
        "target_column": TARGET_COLUMN,
        "train_label_mode": dataset.attrs.get("train_label_mode"),
        "target_long_positive_rows": int(dataset.attrs.get("target_long_positive_rows", 0)),
        "target_long_negative_rows": int(dataset.attrs.get("target_long_negative_rows", 0)),
        "target_long_outcome_counts": dataset.attrs.get("target_long_outcome_counts", {}),
        "target_long_outcome_counts_before_filter": dataset.attrs.get("target_long_outcome_counts_before_filter", {}),
        "target_event_summary": dataset.attrs.get("target_event_summary"),
        "training_universe_rows": int(dataset.attrs.get("training_universe_rows", len(dataset))),
        "excluded_by_train_event_filter_rows": int(dataset.attrs.get("excluded_by_train_event_filter_rows", 0)),
        "training_event_filter": dataset.attrs.get("train_event_filter_config"),
        "event_filter_role": "execution_gate",
        "train_period": build_period_payload(train_df),
        "validation_period": build_period_payload(valid_df),
        "test_period": build_period_payload(test_df),
        "split_sizes": {
            "validation": float(args.val_size),
            "test": float(args.test_size),
        },
    }

    logger.info(
        "Validation metrics | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
        validation_metrics["accuracy"],
        validation_metrics["balanced_accuracy"],
        validation_metrics["f1_macro"],
        validation_metrics["roc_auc"],
        validation_metrics["pr_auc"],
        validation_metrics["mcc"],
    )
    logger.info(
        "Holdout test metrics | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
        test_metrics["accuracy"],
        test_metrics["balanced_accuracy"],
        test_metrics["f1_macro"],
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["mcc"],
    )

    return {
        "model": model,
        "metrics": metrics,
        "clip_bounds": clip_bounds,
        "dataset_to_save": dataset,
        "feature_columns": active_feature_columns,
    }


def summarize_walk_forward(dataset, feature_columns, args):
    all_folds = generate_walk_forward_splits(dataset, args)
    folds = select_walk_forward_folds(all_folds, args)
    logger.info(
        "Walk-forward validation: %s folds | train=%sM | valid=%sM | test=%sM | step=%sM | embargo=%s bars",
        len(folds),
        args.wf_train_months,
        args.wf_val_months,
        args.wf_test_months,
        args.wf_step_months,
        args.wf_embargo_bars,
    )

    fold_summaries = []
    test_prediction_frames = []
    validation_prediction_frames = []
    fold_importance_frames = []
    last_fold_model = None
    last_fold_clip_bounds = {}
    last_fold_feature_columns = feature_columns
    last_fold_train_df = None
    last_fold_valid_df = None
    last_fold_test_df = None

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_df = fold["train_df"].copy()
        valid_df = fold["valid_df"].copy()
        test_df = fold["test_df"].copy()
        train_df, valid_df, test_df, active_feature_columns, clip_bounds = prepare_modeling_frames(
            train_df,
            valid_df,
            test_df,
            feature_columns,
            split_label=f"WF fold {fold_id}",
        )

        logger.info(
            "WF fold %s/%s | train=%s rows %s -> %s | valid=%s rows %s -> %s | test=%s rows %s -> %s",
            fold_id,
            len(folds),
            len(train_df),
            fold["train_period"]["start"],
            fold["train_period"]["end"],
            len(valid_df),
            fold["validation_period"]["start"],
            fold["validation_period"]["end"],
            len(test_df),
            fold["test_period"]["start"],
            fold["test_period"]["end"],
        )

        model = train_validation_model(train_df, valid_df, active_feature_columns, args.seed + fold_id - 1, args=args)
        fold_importance_frames.append(build_model_feature_importance(model, active_feature_columns, fold_id=fold_id))
        validation_predictions = build_prediction_frame(model, valid_df, active_feature_columns)
        test_predictions = build_prediction_frame(model, test_df, active_feature_columns)
        validation_metrics = compute_metrics_from_prediction_frame(validation_predictions, split_name="validation")
        test_metrics = compute_metrics_from_prediction_frame(test_predictions, split_name="test")
        best_iteration = int(model.best_iteration_ or model.n_estimators_)

        validation_prediction_frames.append(validation_predictions.assign(fold_id=fold_id))
        test_prediction_frames.append(test_predictions.assign(fold_id=fold_id))
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(valid_df)),
                "test_rows": int(len(test_df)),
                "best_iteration": best_iteration,
                "train_period": fold["train_period"],
                "validation_period": fold["validation_period"],
                "test_period": fold["test_period"],
                "embargo_bars": int(fold["embargo_bars"]),
                "feature_count": int(len(active_feature_columns)),
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            }
        )

        logger.info(
            "WF fold %s metrics | validation roc_auc=%.4f pr_auc=%.4f mcc=%.4f | test roc_auc=%.4f pr_auc=%.4f mcc=%.4f",
            fold_id,
            validation_metrics["roc_auc"],
            validation_metrics["pr_auc"],
            validation_metrics["mcc"],
            test_metrics["roc_auc"],
            test_metrics["pr_auc"],
            test_metrics["mcc"],
        )

        last_fold_model = model
        last_fold_clip_bounds = clip_bounds
        last_fold_feature_columns = active_feature_columns
        last_fold_train_df = train_df
        last_fold_valid_df = valid_df
        last_fold_test_df = test_df

    if not test_prediction_frames or last_fold_model is None:
        raise RuntimeError("Walk-forward validation failed to produce test predictions.")

    all_validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True).sort_values(
        [TIMESTAMP_COLUMN, SYMBOL_COLUMN] if SYMBOL_COLUMN in validation_prediction_frames[0].columns else [TIMESTAMP_COLUMN]
    ).reset_index(drop=True)
    all_test_predictions = pd.concat(test_prediction_frames, ignore_index=True).sort_values(
        [TIMESTAMP_COLUMN, SYMBOL_COLUMN] if SYMBOL_COLUMN in test_prediction_frames[0].columns else [TIMESTAMP_COLUMN]
    ).reset_index(drop=True)

    validation_metrics = compute_metrics_from_prediction_frame(all_validation_predictions, split_name="validation")
    test_metrics = compute_metrics_from_prediction_frame(all_test_predictions, split_name="test")
    best_iterations = [int(summary["best_iteration"]) for summary in fold_summaries]
    fold_feature_importance = (
        pd.concat(fold_importance_frames, ignore_index=True)
        if fold_importance_frames
        else pd.DataFrame()
    )
    feature_stability = summarize_feature_stability(fold_feature_importance)
    aggregate_best_iteration = (
        int(round(float(np.median(best_iterations))))
        if best_iterations
        else int(getattr(cfg, "LGBM_N_ESTIMATORS", 2000))
    )

    metrics = {
        "validation_mode": "walk_forward",
        "walk_forward": {
            "fold_count": int(len(fold_summaries)),
            "selected_fold_ids": [int(summary["fold_id"]) for summary in fold_summaries],
            "train_months": int(args.wf_train_months),
            "validation_months": int(args.wf_val_months),
            "test_months": int(args.wf_test_months),
            "step_months": int(args.wf_step_months),
            "embargo_bars": int(args.wf_embargo_bars),
            "folds": fold_summaries,
            "oos_validation_period": {
                "start": str(all_validation_predictions[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(all_validation_predictions[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "oos_test_period": {
                "start": str(all_test_predictions[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(all_test_predictions[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "aggregate_best_iteration": aggregate_best_iteration,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "best_iteration": aggregate_best_iteration,
        "train_rows": int(sum(summary["train_rows"] for summary in fold_summaries)),
        "validation_rows": int(sum(summary["validation_rows"] for summary in fold_summaries)),
        "test_rows": int(sum(summary["test_rows"] for summary in fold_summaries)),
        "feature_count": int(len(last_fold_feature_columns)),
        "side_mode": resolve_side_mode(),
        "target_column": TARGET_COLUMN,
        "train_label_mode": dataset.attrs.get("train_label_mode"),
        "target_long_positive_rows": int(dataset.attrs.get("target_long_positive_rows", 0)),
        "target_long_negative_rows": int(dataset.attrs.get("target_long_negative_rows", 0)),
        "target_long_outcome_counts": dataset.attrs.get("target_long_outcome_counts", {}),
        "target_long_outcome_counts_before_filter": dataset.attrs.get("target_long_outcome_counts_before_filter", {}),
        "target_event_summary": dataset.attrs.get("target_event_summary"),
        "training_universe_rows": int(dataset.attrs.get("training_universe_rows", len(dataset))),
        "excluded_by_train_event_filter_rows": int(dataset.attrs.get("excluded_by_train_event_filter_rows", 0)),
        "training_event_filter": dataset.attrs.get("train_event_filter_config"),
        "event_filter_role": "execution_gate",
        "train_period": build_period_payload(last_fold_train_df),
        "validation_period": build_period_payload(last_fold_valid_df),
        "test_period": build_period_payload(last_fold_test_df),
        "split_sizes": {
            "walk_forward": {
                "train_months": int(args.wf_train_months),
                "validation_months": int(args.wf_val_months),
                "test_months": int(args.wf_test_months),
                "step_months": int(args.wf_step_months),
                "embargo_bars": int(args.wf_embargo_bars),
            }
        },
    }

    logger.info(
        "Walk-forward aggregate validation | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
        validation_metrics["accuracy"],
        validation_metrics["balanced_accuracy"],
        validation_metrics["f1_macro"],
        validation_metrics["roc_auc"],
        validation_metrics["pr_auc"],
        validation_metrics["mcc"],
    )
    logger.info(
        "Walk-forward stitched OOS test | accuracy=%.4f | balanced_accuracy=%.4f | f1_macro=%.4f | roc_auc=%.4f | pr_auc=%.4f | mcc=%.4f",
        test_metrics["accuracy"],
        test_metrics["balanced_accuracy"],
        test_metrics["f1_macro"],
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["mcc"],
    )
    logger.warning(
        "Walk-forward mode saves the latest fold model artifact for runtime compatibility; stitched OOS metrics are stored in *_metrics.json."
    )
    log_feature_stability_ranking(feature_stability)

    return {
        "model": last_fold_model,
        "metrics": metrics,
        "clip_bounds": last_fold_clip_bounds,
        "dataset_to_save": dataset,
        "feature_columns": last_fold_feature_columns,
        "fold_feature_importance": fold_feature_importance,
        "feature_stability": feature_stability,
    }


def save_binary_long_only_artifacts(
    model,
    metrics,
    dataset,
    feature_columns,
    clip_bounds,
    args,
    fold_feature_importance: pd.DataFrame | None = None,
    feature_stability: pd.DataFrame | None = None,
):
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    side_mode = resolve_side_mode()
    no_signal_label = "not_long_tp_first"
    signal_label = "long_tp_first"

    model_path = cfg.MODELS_DIR / f"{args.model_name}.joblib"
    metrics_path = cfg.MODELS_DIR / f"{args.model_name}_metrics.json"
    features_path = cfg.MODELS_DIR / f"{args.model_name}_features.json"
    importance_path = cfg.MODELS_DIR / f"{args.model_name}_feature_importance.csv"
    fold_importance_path = cfg.MODELS_DIR / f"{args.model_name}_fold_feature_importance.csv"
    stability_path = cfg.MODELS_DIR / f"{args.model_name}_feature_stability.csv"

    payload = {
        "feature_columns": feature_columns,
        "label_mapping": {no_signal_label: 0, signal_label: 1},
        "inverse_label_mapping": {"0": no_signal_label, "1": signal_label},
        "symbols": list(args.symbols),
        "rows": int(len(dataset)),
        "prod_train": bool(args.prod_train),
        "task_type": "binary_long_only",
        "side_mode": side_mode,
        "target_column": TARGET_COLUMN,
        "train_label_mode": metrics.get("train_label_mode"),
        "entry_threshold": float(cfg.ENTRY_THRESHOLD),
        "validation_mode": metrics.get("validation_mode", "single"),
        "quick_mode": bool(getattr(args, "quick", False)),
        "train_period": metrics.get("train_period"),
        "validation_period": metrics.get("validation_period"),
        "test_period": metrics.get("test_period"),
        "split_sizes": metrics.get("split_sizes"),
        "walk_forward": metrics.get("walk_forward"),
        "event_filter_role": metrics.get("event_filter_role", "execution_gate"),
        "training_event_filter": metrics.get("training_event_filter"),
        "sample_weight": getattr(cfg, "TRAIN_SAMPLE_WEIGHT_CONFIG", {}),
        "target_long_outcome_counts": metrics.get("target_long_outcome_counts"),
        "target_event_summary": metrics.get("target_event_summary"),
        "feature_storage": {
            "type": str(getattr(cfg, "FEATURE_STORAGE", "sqlite")),
            "parquet_features_dir": str(getattr(cfg, "PARQUET_FEATURES_DIR", "")),
            "parquet_dataset_version": str(getattr(cfg, "PARQUET_DATASET_VERSION", "")),
        },
        "feature_clip": {
            "enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.01)),
            "upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 0.99)),
            "bounds": clip_bounds,
        },
    }

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    top_feature_importance(model, feature_columns).to_csv(importance_path, index=False)
    if fold_feature_importance is not None and not fold_feature_importance.empty:
        fold_feature_importance.to_csv(fold_importance_path, index=False)
    if feature_stability is not None and not feature_stability.empty:
        feature_stability.to_csv(stability_path, index=False)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved feature metadata to %s", features_path)
    logger.info("Saved feature importance to %s", importance_path)
    if fold_feature_importance is not None and not fold_feature_importance.empty:
        logger.info("Saved fold feature importance to %s", fold_importance_path)
    if feature_stability is not None and not feature_stability.empty:
        logger.info("Saved feature stability to %s", stability_path)


def main():
    try:
        args = parse_args()
        for warning in cfg.validate_config(context="train"):
            logger.warning("Config: %s", warning)
        if args.quick:
            logger.info(
                "Quick research mode is active. Artifacts will be saved under model name: %s",
                args.model_name,
            )
        dataset = load_training_frame(args.db_path, args.symbols)
        feature_columns = select_feature_columns(dataset)

        logger.info("Loaded %s rows with %s model input columns", len(dataset), len(feature_columns))
        logger.info("Using symbols: %s", ", ".join(args.symbols))
        logger.info(
            "Training universe: kept %s/%s rows; train-side event filter %s",
            len(dataset),
            int(dataset.attrs.get("training_universe_rows", len(dataset))),
            "enabled" if dataset.attrs.get("train_event_filter_config", {}).get("enabled", False) else "disabled",
        )
        if dataset.attrs.get("train_event_filter_config", {}).get("enabled", False):
            logger.info(
                "Train-side event filter: allowed_outcomes=%s | excluded=%s rows",
                ", ".join(dataset.attrs.get("train_event_filter_config", {}).get("allowed_outcomes", [])),
                int(dataset.attrs.get("excluded_by_train_event_filter_rows", 0)),
            )
        else:
            logger.info(
                "Baseline label mode is active: NEITHER remains in the negative class unless TRAIN_EVENT_FILTER_CONFIG is enabled."
            )
        logger.info(
            "Target outcome counts after train filter: %s",
            dataset.attrs.get("target_long_outcome_counts", {}),
        )
        logger.info(
            "Long-only target distribution inside training universe: positives=%s | negatives=%s",
            int(dataset.attrs.get("target_long_positive_rows", 0)),
            int(dataset.attrs.get("target_long_negative_rows", 0)),
        )
        sample_weight_config = getattr(cfg, "TRAIN_SAMPLE_WEIGHT_CONFIG", {}) or {}
        if bool(sample_weight_config.get("enabled", False)):
            logger.info(
                "Sample weighting enabled: %s",
                sample_weight_config.get("outcome_weights", {}),
            )

        if args.validation_mode == "walk_forward":
            training_result = summarize_walk_forward(dataset, feature_columns, args)
        else:
            training_result = summarize_single_split(dataset, feature_columns, args)

        model_to_save = training_result["model"]
        metrics = training_result["metrics"]
        clip_bounds_to_save = training_result["clip_bounds"]
        dataset_to_save = training_result["dataset_to_save"]
        feature_columns_to_save = training_result["feature_columns"]
        fold_feature_importance = training_result.get("fold_feature_importance")
        feature_stability = training_result.get("feature_stability")

        if args.prod_train:
            logger.warning(
                "ENABLE_PROD_TRAINING is enabled: the saved model will be retrained on the full dataset, "
                "including the holdout test window. Use the saved test metrics for evaluation, but do not treat "
                "subsequent backtests with this retrained artifact as out-of-sample."
            )
            clip_bounds_to_save, _ = build_feature_clip_bounds(dataset_to_save, feature_columns)
            dataset_to_save = apply_feature_clip_bounds(dataset_to_save, clip_bounds_to_save)
            feature_columns_to_save, dropped_constant_columns = drop_constant_feature_columns(
                dataset_to_save, feature_columns
            )
            if dropped_constant_columns:
                logger.info(
                    "Full-dataset retrain | dropped %s constant features after clipping: %s",
                    len(dropped_constant_columns),
                    ", ".join(sorted(dropped_constant_columns)),
                )
            model_to_save = retrain_full_model(
                dataset_to_save,
                feature_columns_to_save,
                args.seed,
                metrics["best_iteration"],
            )

        log_feature_importance_ranking(model_to_save, feature_columns_to_save)
        save_binary_long_only_artifacts(
            model_to_save,
            metrics,
            dataset_to_save,
            feature_columns_to_save,
            clip_bounds_to_save,
            args,
            fold_feature_importance=fold_feature_importance,
            feature_stability=feature_stability,
        )
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
