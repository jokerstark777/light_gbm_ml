import argparse
from collections import Counter
import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import bt as runtime_bt
import config as cfg
import train
from bt_walk_forward import load_runtime_inputs, plot_equity_curve
from execution_engine import EngineLogConfig, EntrySignal, simulate_portfolio
from signal_filter import build_event_gate_mask, resolve_event_filter_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run portfolio backtest using only walk-forward out-of-sample predictions."
    )
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument("--symbols", nargs="+", default=cfg.SYMBOLS, help="Symbols to load.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=int(getattr(cfg, "WF_EMBARGO_BARS", getattr(cfg, "HORIZON", 0))),
        help="Purge gap in unique timestamps between train and test folds.",
    )
    parser.add_argument(
        "--predictions-name",
        default="walk_forward_oos_predictions",
        help="Base filename for saved OOS predictions and backtest artifacts.",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=float(getattr(cfg, "ENTRY_THRESHOLD", 0.55)),
        help="Minimum p_long required to open a long position.",
    )
    parser.add_argument("--verbose-trades", action="store_true", help="Print every open/close event.")
    parser.add_argument("--quiet-trades", action="store_true", help="Do not print per-trade logs.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print backtest progress every N candles.")
    return parser.parse_args()


def load_candidate_and_training_frames(db_path: str, symbols: list[str]):
    frame = train.load_training_frame(db_path, symbols)
    event_filter_config = resolve_event_filter_config()
    candidate_mask = build_event_gate_mask(frame, event_filter_config)
    candidate_frame = frame.loc[candidate_mask].copy()

    symbol_categories = list(dict.fromkeys(symbols))
    frame[train.SYMBOL_COLUMN] = pd.Categorical(frame[train.SYMBOL_COLUMN], categories=symbol_categories)
    candidate_frame[train.SYMBOL_COLUMN] = pd.Categorical(
        candidate_frame[train.SYMBOL_COLUMN],
        categories=symbol_categories,
    )

    feature_columns = train.select_feature_columns(frame)
    return frame, candidate_frame, feature_columns, event_filter_config


def fit_fold_model(train_df: pd.DataFrame, feature_columns: list[str], seed: int):
    clip_bounds, _ = train.build_feature_clip_bounds(train_df, feature_columns)
    clipped_train = train.apply_feature_clip_bounds(train_df, clip_bounds)
    active_feature_columns, dropped_columns = train.drop_constant_feature_columns(clipped_train, feature_columns)
    if dropped_columns:
        print(f"Fold seed {seed}: dropped constant features={len(dropped_columns)}")

    clipped_train = clipped_train.dropna(subset=active_feature_columns + [train.TARGET_COLUMN])
    if clipped_train.empty:
        raise RuntimeError("Fold training frame is empty after feature NaN cleanup.")

    x_train = clipped_train[active_feature_columns]
    y_train = clipped_train[train.TARGET_COLUMN].astype(int)
    model = train.build_model(seed=seed)
    categorical_feature = [train.SYMBOL_COLUMN] if train.SYMBOL_COLUMN in active_feature_columns else "auto"

    if len(x_train) >= 20:
        internal_eval_size = max(1, int(len(x_train) * 0.15))
        train_part_y = y_train.iloc[:-internal_eval_size]
        if internal_eval_size < len(x_train) and train_part_y.nunique(dropna=True) >= 2:
            model.fit(
                x_train.iloc[:-internal_eval_size],
                train_part_y,
                eval_set=[(x_train.iloc[-internal_eval_size:], y_train.iloc[-internal_eval_size:])],
                eval_metric="binary_logloss",
                categorical_feature=categorical_feature,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=int(getattr(cfg, "LGBM_EARLY_STOPPING_ROUNDS", 50)),
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )
            return model, clip_bounds, active_feature_columns

    model.fit(x_train, y_train, categorical_feature=categorical_feature)
    return model, clip_bounds, active_feature_columns


def build_walk_forward_predictions(
    full_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    purge_gap: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if n_splits <= 1:
        raise ValueError("--n-splits must be > 1.")
    if purge_gap < 0:
        raise ValueError("--purge-gap must be >= 0.")

    unique_ts = np.sort(full_frame[train.TIMESTAMP_COLUMN].dropna().unique())
    if len(unique_ts) < n_splits + 1:
        raise RuntimeError(
            f"Only {len(unique_ts)} unique timestamps, need at least {n_splits + 1} for {n_splits} folds."
        )

    predictions = []
    fold_details = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_idx, (train_ts_idx, test_ts_idx) in enumerate(splitter.split(unique_ts), start=1):
        train_timestamps = unique_ts[train_ts_idx]
        test_timestamps = unique_ts[test_ts_idx]

        purged_count = 0
        if purge_gap > 0:
            purged_count = min(int(purge_gap), len(train_timestamps))
            train_timestamps = train_timestamps[:-purged_count] if purged_count else train_timestamps

        train_df = candidate_frame.loc[candidate_frame[train.TIMESTAMP_COLUMN].isin(set(train_timestamps))].copy()
        test_df = candidate_frame.loc[candidate_frame[train.TIMESTAMP_COLUMN].isin(set(test_timestamps))].copy()
        train_classes = sorted(train_df[train.TARGET_COLUMN].dropna().astype(int).unique().tolist())
        if train_df.empty or test_df.empty or len(train_classes) < 2:
            print(f"Fold {fold_idx}: skipped (train={len(train_df)}, test={len(test_df)}, classes={train_classes})")
            continue

        model, clip_bounds, active_feature_columns = fit_fold_model(train_df, feature_columns, seed + fold_idx)
        clipped_test = train.apply_feature_clip_bounds(test_df, clip_bounds)
        clipped_test = clipped_test.dropna(subset=active_feature_columns + [train.TARGET_COLUMN])
        if clipped_test.empty:
            print(f"Fold {fold_idx}: skipped after feature NaN cleanup")
            continue

        proba = model.predict_proba(clipped_test[active_feature_columns])
        fold_predictions = pd.DataFrame(
            {
                "timestamp": clipped_test[train.TIMESTAMP_COLUMN].values,
                "symbol": clipped_test[train.SYMBOL_COLUMN].astype(str).values,
                "y_true": clipped_test[train.TARGET_COLUMN].astype(int).values,
                "p_not_long": proba[:, 0],
                "p_long": proba[:, 1],
                "fold": fold_idx,
            }
        )
        fold_predictions["y_pred"] = (fold_predictions["p_long"] >= 0.5).astype(int)
        predictions.append(fold_predictions)

        fold_info = {
            "fold": int(fold_idx),
            "train_rows": int(len(train_df)),
            "prediction_rows": int(len(fold_predictions)),
            "purged_timestamps": int(purged_count),
            "train_start": str(train_df[train.TIMESTAMP_COLUMN].min()),
            "train_end": str(train_df[train.TIMESTAMP_COLUMN].max()),
            "test_start": str(clipped_test[train.TIMESTAMP_COLUMN].min()),
            "test_end": str(clipped_test[train.TIMESTAMP_COLUMN].max()),
            "feature_count": int(len(active_feature_columns)),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or getattr(model, "n_estimators_", 0)),
        }
        fold_details.append(fold_info)
        print(
            f"Fold {fold_idx}/{n_splits}: train={fold_info['train_rows']} "
            f"pred={fold_info['prediction_rows']} "
            f"[{fold_info['test_start']} -> {fold_info['test_end']}]"
        )

    if not predictions:
        raise RuntimeError("All walk-forward folds were skipped.")
    return pd.concat(predictions, ignore_index=True), fold_details


def save_walk_forward_payload(predictions: pd.DataFrame, fold_details: list[dict], args) -> dict:
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    predictions_path = cfg.MODELS_DIR / f"{args.predictions_name}.csv"
    summary_path = cfg.MODELS_DIR / f"{args.predictions_name}_summary.json"
    predictions.to_csv(predictions_path, index=False)

    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "symbols": list(args.symbols),
        "n_splits": int(args.n_splits),
        "purge_gap": int(args.purge_gap),
        "entry_threshold": float(args.entry_threshold),
        "prediction_rows": int(len(predictions)),
        "prediction_period": {
            "start": str(pd.to_datetime(predictions["timestamp"]).min()),
            "end": str(pd.to_datetime(predictions["timestamp"]).max()),
        },
        "fold_details": fold_details,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved OOS predictions: {predictions_path}")
    print(f"Saved WFV summary: {summary_path}")
    return summary


def build_prediction_lookup(predictions: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], dict]:
    prepared = predictions.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared = prepared.dropna(subset=["timestamp", "symbol", "p_long"])
    prepared = prepared.sort_values(["timestamp", "symbol", "fold"]).drop_duplicates(
        subset=["timestamp", "symbol"],
        keep="last",
    )
    return {
        (pd.Timestamp(row.timestamp), str(row.symbol)): {"p_long": float(row.p_long), "fold": int(row.fold)}
        for row in prepared.itertuples(index=False)
    }


def get_feature_row_from_index(indexed_rows: dict, ts: pd.Timestamp, required_columns: list[str]) -> dict | None:
    row = indexed_rows.get(pd.Timestamp(ts))
    if row is None:
        return None

    for column in required_columns:
        if column not in row or pd.isna(row[column]):
            return None
    return row


def get_barrier_pcts_from_row(feature_row: dict | None) -> tuple[float | None, float | None]:
    if not feature_row:
        return None, None

    stop_pct = feature_row.get("barrier_stop_pct")
    take_pct = feature_row.get("barrier_take_pct")
    if stop_pct is None or take_pct is None:
        return None, None

    stop_pct = float(stop_pct)
    take_pct = float(take_pct)
    if not np.isfinite(stop_pct) or not np.isfinite(take_pct):
        return None, None
    return stop_pct, take_pct


def passes_event_gate_row(feature_row: dict | None, event_filter_config: dict) -> bool:
    if not feature_row:
        return False
    if not event_filter_config.get("enabled", False):
        return True

    for key, value in event_filter_config.items():
        if value is None or key in {"enabled", "side", "required_columns"}:
            continue
        if not (key.startswith("min_") or key.startswith("max_")):
            continue

        column = key[4:]
        row_value = feature_row.get(column)
        if row_value is None or pd.isna(row_value):
            return False
        row_value = float(row_value)
        threshold = float(value)
        if key.startswith("min_") and row_value < threshold:
            return False
        if key.startswith("max_") and row_value > threshold:
            return False

    return True


def run_predictions_backtest(
    predictions: pd.DataFrame,
    symbols: list[str],
    event_filter_config: dict,
    entry_threshold: float,
    verbose_trades: bool,
    progress_every: int,
) -> dict:
    event_filter_config = resolve_event_filter_config(event_filter_config)
    runtime_required_columns = list(dict.fromkeys(["barrier_stop_pct", "barrier_take_pct"]))
    if event_filter_config.get("enabled", False):
        runtime_required_columns.extend(
            column
            for column in event_filter_config.get("required_columns", [])
            if column not in runtime_required_columns
        )

    all_raw, all_features, all_main_index = load_runtime_inputs(
        symbols,
        runtime_required_columns,
        symbol_categories=None,
    )
    all_feature_index = {
        symbol: runtime_bt.build_timestamp_index(feature_frame)
        for symbol, feature_frame in all_features.items()
    }
    prediction_lookup = build_prediction_lookup(predictions)
    prediction_timestamps = pd.to_datetime(predictions["timestamp"], errors="coerce").dropna()
    prediction_timestamp_set = set(pd.Timestamp(timestamp) for timestamp in prediction_timestamps.unique())
    threshold_prediction_rows = int((pd.to_numeric(predictions["p_long"], errors="coerce") >= entry_threshold).sum())
    start_ts = pd.Timestamp(prediction_timestamps.min())
    end_ts = pd.Timestamp(prediction_timestamps.max())
    test_timestamps = runtime_bt.get_market_timestamps(all_raw, start_ts, end_ts)
    if len(test_timestamps) < 2:
        raise RuntimeError("Too little market data inside the OOS prediction period.")

    print(
        f"OOS prediction backtest: {test_timestamps[0]} -> {test_timestamps[-1]} | "
        f"symbols={', '.join(runtime_bt.compact_symbol(symbol) for symbol in all_raw)} | "
        f"entry_threshold={entry_threshold:.2f}"
    )
    print(
        f"OOS prediction rows: {len(predictions)} | "
        f"timestamps={len(prediction_timestamp_set)} | "
        f"p_long >= threshold: {threshold_prediction_rows}"
    )

    rejection_counts = Counter()

    def prediction_signal_provider(current_ts, candidate_symbols, market_batch):
        signals = []
        for symbol in candidate_symbols:
            prediction = prediction_lookup.get((pd.Timestamp(current_ts), symbol))
            if prediction is None:
                rejection_counts["no_symbol_prediction"] += 1
                continue
            rejection_counts["prediction_seen"] += 1

            feature_row = get_feature_row_from_index(
                all_feature_index.get(symbol, {}),
                current_ts,
                runtime_required_columns,
            )
            stop_pct, take_pct = get_barrier_pcts_from_row(feature_row)
            if stop_pct is None or take_pct is None:
                rejection_counts["missing_barrier"] += 1
                continue
            if not passes_event_gate_row(feature_row, event_filter_config):
                rejection_counts["event_gate"] += 1
                continue

            fold = int(prediction["fold"])
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    p_long=float(prediction["p_long"]),
                    stop_pct=float(stop_pct),
                    take_pct=float(take_pct),
                    extra_position_fields={"fold": fold},
                    extra_trade_fields={"fold": fold},
                )
            )
        return signals

    result = simulate_portfolio(
        all_raw=all_raw,
        all_main_index=all_main_index,
        start_ts=start_ts,
        end_ts=end_ts,
        config=runtime_bt.build_execution_config(runtime_bt.BACKTEST_INITIAL_BALANCE, entry_threshold),
        signal_provider=prediction_signal_provider,
        log_config=EngineLogConfig(
            verbose_trades=bool(verbose_trades),
            progress_every=int(progress_every),
            progress_label="Backtest progress",
            close_prefix="[{ts}] No. {trade_number}",
            open_prefix="[{ts}] No. {trade_number}",
            final_prefix="[{ts}] No. {trade_number}",
        ),
        skip_empty_signal_timestamps=prediction_timestamp_set,
    )
    engine_diagnostics = Counter(result.get("candidate_diagnostics", {}))
    engine_diagnostics.update(rejection_counts)
    result["summary"].update(
        {
            "period_start": str(test_timestamps[0]),
            "period_end": str(test_timestamps[-1]),
            "symbols": list(all_raw.keys()),
            "entry_threshold": float(entry_threshold),
            "prediction_rows": int(len(predictions)),
            "candidate_diagnostics": {key: int(value) for key, value in sorted(engine_diagnostics.items())},
        }
    )
    if engine_diagnostics:
        print("OOS candidate diagnostics:")
        for key, value in sorted(engine_diagnostics.items()):
            print(f"   {key}: {int(value)}")
    return result


def save_backtest_payload(backtest_result: dict, summary: dict, args) -> None:
    output_json = cfg.MODELS_DIR / f"{args.predictions_name}_backtest.json"
    output_trades = cfg.MODELS_DIR / f"{args.predictions_name}_trades.csv"
    output_chart = cfg.BACKTEST_CHARTS_DIR / f"{args.predictions_name}_equity_curve.png"

    trades = pd.DataFrame(backtest_result["trades"])
    if not trades.empty:
        trades.to_csv(output_trades, index=False)

    output_json.write_text(
        json.dumps(
            {
                "predictions": summary,
                "backtest": backtest_result["summary"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_equity_curve(backtest_result["equity_timestamps"], backtest_result["equity_curve"], output_chart)
    print(f"Saved OOS backtest summary: {output_json}")
    if not trades.empty:
        print(f"Saved OOS trades: {output_trades}")
    print(f"Saved OOS equity chart: {output_chart}")


def main():
    args = parse_args()
    full_frame, candidate_frame, feature_columns, event_filter_config = load_candidate_and_training_frames(
        args.db_path,
        args.symbols,
    )
    print(
        f"Loaded rows={len(full_frame)}, candidate rows={len(candidate_frame)}, "
        f"features={len(feature_columns)}"
    )

    predictions, fold_details = build_walk_forward_predictions(
        full_frame=full_frame,
        candidate_frame=candidate_frame,
        feature_columns=feature_columns,
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
        seed=args.seed,
    )
    summary = save_walk_forward_payload(predictions, fold_details, args)
    backtest_result = run_predictions_backtest(
        predictions=predictions,
        symbols=args.symbols,
        event_filter_config=event_filter_config,
        entry_threshold=float(args.entry_threshold),
        verbose_trades=bool(args.verbose_trades or not args.quiet_trades),
        progress_every=int(args.progress_every),
    )
    save_backtest_payload(backtest_result, summary, args)
    bt_summary = backtest_result["summary"]
    print(
        f"WALK-FORWARD OOS BACKTEST | trades={bt_summary['total_trades']} | "
        f"winrate={bt_summary['win_rate_pct']:.2f}% | "
        f"end_balance=${bt_summary['ending_balance']:.2f} | "
        f"return={bt_summary['total_return_pct']:.2f}% | "
        f"max_dd={bt_summary['max_drawdown_pct']:.2f}% | "
        f"pf={bt_summary['profit_factor']:.2f} | "
        f"sharpe={bt_summary['sharpe']:.2f}"
    )


def backtest_from_oos_predictions():
    return main()


if __name__ == "__main__":
    backtest_from_oos_predictions()
