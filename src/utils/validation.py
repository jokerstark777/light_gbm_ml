"""Shared validation utilities for walk-forward splits and chronological data splitting."""

import pandas as pd
from typing import Any


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target_long_label"


def compute_embargo_timedelta(horizon_bars: int, timeframe: str) -> pd.Timedelta:
    """Compute a time-based embargo duration from horizon bars and timeframe.
    
    Args:
        horizon_bars: Number of bars in the forecast horizon (e.g., HORIZON config).
        timeframe: The chart timeframe string (e.g., "1h", "1d", "15m").
    
    Returns:
        pd.Timedelta representing the embargo duration.
    """
    if "d" in timeframe:
        return pd.Timedelta(days=horizon_bars)
    elif "h" in timeframe:
        return pd.Timedelta(hours=horizon_bars)
    elif "m" in timeframe:
        return pd.Timedelta(minutes=horizon_bars)
    elif "s" in timeframe:
        return pd.Timedelta(seconds=horizon_bars)
    else:
        # Default to hours if timeframe is unrecognized
        return pd.Timedelta(hours=horizon_bars)


def validate_walk_forward_args(args: Any) -> None:
    """Validate walk-forward configuration arguments.
    
    Args:
        args: Object containing wf_train_months, wf_val_months, wf_test_months, 
              wf_step_months, and wf_embargo_bars attributes.
    
    Raises:
        ValueError: If any argument has an invalid value.
    """
    if args.wf_train_months <= 0:
        raise ValueError("--wf-train-months must be > 0.")
    if args.wf_val_months <= 0:
        raise ValueError("--wf-val-months must be > 0.")
    if args.wf_test_months <= 0:
        raise ValueError("--wf-test-months must be > 0.")
    if args.wf_step_months <= 0:
        raise ValueError("--wf-step-months must be > 0.")
    if args.wf_embargo_bars < 0:
        raise ValueError("--wf-embargo-bars must be >= 0.")


def resolve_timestamp_index_on_or_after(unique_timestamps: pd.Series, boundary_ts: pd.Timestamp) -> int | None:
    """Find the index of the first timestamp on or after a boundary.
    
    Args:
        unique_timestamps: Sorted series of unique timestamps.
        boundary_ts: The boundary timestamp to search for.
    
    Returns:
        Index of the first timestamp >= boundary_ts, or None if not found.
    """
    idx = int(unique_timestamps.searchsorted(boundary_ts, side="left"))
    if idx >= len(unique_timestamps):
        return None
    return idx


def generate_walk_forward_splits(dataset: pd.DataFrame, args: Any) -> list[dict]:
    """Generate walk-forward validation splits based on time periods.
    
    Implements a rolling window approach based on actual time (e.g., Train 6M, Val 1M, Test 1M, Step 1M).
    This ensures consistency between training and backtesting evaluations.
    
    Args:
        dataset: DataFrame with TIMESTAMP_COLUMN and TARGET_COLUMN.
        args: Configuration object with walk-forward parameters:
            - wf_train_months: Training window size in months
            - wf_val_months: Validation window size in months
            - wf_test_months: Test window size in months
            - wf_step_months: Step size between folds in months
            - wf_embargo_bars: Number of bars to embargo between splits
            - timeframe: Optional chart timeframe string (e.g., "1h", "1d") for time-based embargo.
                         If provided, embargo is computed as timedelta instead of index offset.
    
    Returns:
        List of fold dictionaries containing train_df, valid_df, test_df, and metadata.
    
    Raises:
        RuntimeError: If no valid folds can be generated.
    """
    from train import validate_split, build_period_payload
    
    validate_walk_forward_args(args)
    unique_timestamps = dataset[TIMESTAMP_COLUMN].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_timestamps) < 10:
        raise RuntimeError("Need more unique timestamps for walk-forward validation.")

    first_ts = pd.Timestamp(unique_timestamps.iloc[0])
    last_ts = pd.Timestamp(unique_timestamps.iloc[-1])
    cursor_ts = first_ts
    folds = []
    fold_id = 1
    
    # Determine if we should use time-based embargo (preferred) or index-based embargo
    timeframe = getattr(args, "timeframe", None)
    use_time_based_embargo = timeframe is not None
    
    if use_time_based_embargo:
        embargo_timedelta = compute_embargo_timedelta(int(args.wf_embargo_bars), timeframe)

    while True:
        train_start_boundary = cursor_ts
        train_end_boundary = train_start_boundary + pd.DateOffset(months=int(args.wf_train_months))
        train_end_idx_exclusive = resolve_timestamp_index_on_or_after(unique_timestamps, train_end_boundary)
        if train_end_idx_exclusive is None or train_end_idx_exclusive <= 0:
            break

        # Compute validation start using either time-based or index-based embargo
        if use_time_based_embargo:
            val_start_boundary = unique_timestamps.iloc[train_end_idx_exclusive] + embargo_timedelta
            val_start_idx = resolve_timestamp_index_on_or_after(unique_timestamps, pd.Timestamp(val_start_boundary))
            if val_start_idx is None:
                break
        else:
            val_start_idx = train_end_idx_exclusive + int(args.wf_embargo_bars)
        
        if val_start_idx >= len(unique_timestamps):
            break
        val_start_ts = pd.Timestamp(unique_timestamps.iloc[val_start_idx])
        val_end_boundary = val_start_ts + pd.DateOffset(months=int(args.wf_val_months))
        val_end_idx_exclusive = resolve_timestamp_index_on_or_after(unique_timestamps, val_end_boundary)
        if val_end_idx_exclusive is None or val_end_idx_exclusive <= val_start_idx:
            break

        # Compute test start using either time-based or index-based embargo
        if use_time_based_embargo:
            test_start_boundary = unique_timestamps.iloc[val_end_idx_exclusive] + embargo_timedelta
            test_start_idx = resolve_timestamp_index_on_or_after(unique_timestamps, pd.Timestamp(test_start_boundary))
            if test_start_idx is None:
                break
        else:
            test_start_idx = val_end_idx_exclusive + int(args.wf_embargo_bars)
        
        if test_start_idx >= len(unique_timestamps):
            break
        test_start_ts = pd.Timestamp(unique_timestamps.iloc[test_start_idx])
        test_end_boundary = test_start_ts + pd.DateOffset(months=int(args.wf_test_months))
        test_end_idx_exclusive = resolve_timestamp_index_on_or_after(unique_timestamps, test_end_boundary)
        if test_end_idx_exclusive is None or test_end_idx_exclusive <= test_start_idx:
            break

        next_cursor_ts = train_start_boundary + pd.DateOffset(months=int(args.wf_step_months))
        if next_cursor_ts <= cursor_ts:
            raise RuntimeError("Walk-forward step did not advance the cursor.")

        train_df = dataset.loc[
            (dataset[TIMESTAMP_COLUMN] >= train_start_boundary) & (dataset[TIMESTAMP_COLUMN] < val_start_ts)
        ].copy()
        valid_df = dataset.loc[
            (dataset[TIMESTAMP_COLUMN] >= val_start_ts) & (dataset[TIMESTAMP_COLUMN] < test_start_ts)
        ].copy()

        test_end_ts_exclusive = (
            pd.Timestamp(unique_timestamps.iloc[test_end_idx_exclusive])
            if test_end_idx_exclusive < len(unique_timestamps)
            else (last_ts + pd.Timedelta(microseconds=1))
        )
        test_df = dataset.loc[
            (dataset[TIMESTAMP_COLUMN] >= test_start_ts) & (dataset[TIMESTAMP_COLUMN] < test_end_ts_exclusive)
        ].copy()

        if train_df.empty or valid_df.empty or test_df.empty:
            cursor_ts = next_cursor_ts
            continue

        validate_split(train_df, valid_df, test_df)
        folds.append(
            {
                "fold_id": fold_id,
                "train_df": train_df,
                "valid_df": valid_df,
                "test_df": test_df,
                "train_period": build_period_payload(train_df),
                "validation_period": build_period_payload(valid_df),
                "test_period": build_period_payload(test_df),
                "embargo_bars": int(args.wf_embargo_bars),
            }
        )
        fold_id += 1
        cursor_ts = next_cursor_ts
        if cursor_ts >= last_ts:
            break

    if not folds:
        raise RuntimeError(
            "Walk-forward split produced no usable folds. Adjust WF_* window sizes, embargo, or prepare more data."
        )
    return folds
