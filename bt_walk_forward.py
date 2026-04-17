import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from execution_engine import EngineLogConfig, EntrySignal, simulate_portfolio, summarize_performance as engine_summarize_performance
import bt as runtime_bt
import config as cfg
import train
from src.features.indicators import normalize_bars_for_timeframe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a stitched walk-forward trading backtest by retraining the model on each fold."
    )
    parser.add_argument("--db-path", default=cfg.DB_PATH, help="Path to SQLite database.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="Symbols to include in the walk-forward backtest.",
    )
    parser.add_argument(
        "--model-name",
        default=getattr(cfg, "MODEL_NAME", "lightgbm_long_only"),
        help="Base name used for saved walk-forward backtest artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        help="Number of unique timestamps to skip between windows.",
    )
    parser.add_argument(
        "--wf-start-fold",
        type=int,
        default=1,
        help="1-based walk-forward fold number to start from.",
    )
    parser.add_argument(
        "--wf-max-folds",
        type=int,
        default=0,
        help="Maximum number of folds to run. 0 means all available folds.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the same reduced LightGBM budget as train.py --quick.",
    )
    parser.add_argument(
        "--quick-folds",
        type=int,
        default=5,
        help="When --quick is enabled, run only the most recent N folds.",
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
    parser.add_argument(
        "--verbose-trades",
        action="store_true",
        help="Print every open/close event during the stitched OOS simulation.",
    )
    return parser.parse_args()


def resolve_runtime_required_columns(feature_columns: list[str], event_filter_config: dict | None) -> list[str]:
    required_columns = list(dict.fromkeys(feature_columns + ["barrier_stop_pct", "barrier_take_pct"]))
    if event_filter_config and event_filter_config.get("enabled", False):
        required_columns.extend(
            column
            for column in event_filter_config.get("required_columns", [])
            if column not in required_columns
        )
    return required_columns


def load_runtime_inputs(symbols: list[str], required_columns: list[str], symbol_categories: list[str] | None):
    all_raw = runtime_bt.load_all_raw_data(symbols)
    if not all_raw:
        raise RuntimeError("No raw data for walk-forward backtest.")

    all_features = {}
    for symbol in list(all_raw.keys()):
        feat_df = runtime_bt.load_precomputed_features(
            symbol,
            symbol_categories=symbol_categories,
            required_columns=required_columns,
        )
        if feat_df.empty:
            print(f"Warning: {symbol} has no prepared dataset rows.")
            continue
        all_features[symbol] = feat_df

    all_raw = {symbol: payload for symbol, payload in all_raw.items() if symbol in all_features}
    if not all_raw:
        raise RuntimeError("No symbols with prepared dataset rows available in DB. Run etl.py first.")

    all_main_index = {
        symbol: runtime_bt.build_timestamp_index(payload["main"])
        for symbol, payload in all_raw.items()
    }
    return all_raw, all_features, all_main_index


def prepare_feature_store_for_fold(
    all_features: dict[str, pd.DataFrame],
    feature_columns: list[str],
    clip_bounds: dict,
    symbol_categories: list[str] | None,
) -> dict[str, pd.DataFrame]:
    feature_store = {}
    for symbol, feat_df in all_features.items():
        prepared = runtime_bt.prepare_precomputed_feature_store(
            feat_df,
            feature_columns,
            symbol_categories=symbol_categories,
            clip_bounds=clip_bounds,
        )
        if prepared.empty:
            continue
        feature_store[symbol] = prepared
    return feature_store


def summarize_performance(
    initial_balance: float,
    ending_balance: float,
    trades: list[dict],
    equity_curve: list[float],
    equity_timestamps: list[pd.Timestamp],
    max_drawdown: float | None,
) -> dict:
    return engine_summarize_performance(
        initial_balance=initial_balance,
        ending_balance=ending_balance,
        trades=trades,
        equity_curve=equity_curve,
        equity_timestamps=equity_timestamps,
        max_drawdown=max_drawdown,
    )


def run_fold_backtest(
    fold_id: int,
    model,
    feature_columns: list[str],
    runtime_required_columns: list[str],
    event_filter_config: dict,
    all_raw: dict[str, dict],
    all_features: dict[str, pd.DataFrame],
    all_main_index: dict[str, dict],
    prepared_feature_store: dict[str, pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    initial_balance: float,
    trade_number_start: int,
    entry_threshold: float,
    verbose_trades: bool = False,
):
    fold_raw = {
        symbol: payload
        for symbol, payload in all_raw.items()
        if symbol in prepared_feature_store and symbol in all_features
    }
    if not fold_raw:
        raise RuntimeError(f"WF fold {fold_id}: no symbols with prepared rows available for the test window.")

    test_timestamps = runtime_bt.get_market_timestamps(fold_raw, start_ts, end_ts)
    if len(test_timestamps) < 2:
        raise RuntimeError(f"WF fold {fold_id}: too little market data inside the test window.")

    print(
        f"WF fold {fold_id} backtest: {test_timestamps[0]} -> {test_timestamps[-1]} | "
        f"symbols={', '.join(runtime_bt.compact_symbol(symbol) for symbol in fold_raw)} | "
        f"starting_balance=${float(initial_balance):.2f}"
    )

    def fold_signal_provider(current_ts, candidate_symbols, market_batch):
        batch_symbols, batch_features = runtime_bt.get_feature_batch_precomputed(
            prepared_feature_store,
            [symbol for symbol in candidate_symbols if symbol in prepared_feature_store],
            current_ts,
        )
        if batch_features.empty:
            return []

        signals = []
        batch_proba = model.predict_proba(batch_features[feature_columns])
        for symbol, proba in zip(batch_symbols, batch_proba):
            feature_row = runtime_bt.get_feature_row_precomputed(
                all_features[symbol],
                current_ts,
                runtime_required_columns,
            )
            stop_pct, take_pct = runtime_bt.get_barrier_pcts(feature_row)
            if stop_pct is None or take_pct is None:
                continue
            if not runtime_bt.passes_event_gate(feature_row, event_filter_config):
                continue
            signals.append(
                EntrySignal(
                    symbol=symbol,
                    p_long=float(proba[1]),
                    stop_pct=float(stop_pct),
                    take_pct=float(take_pct),
                    extra_trade_fields={"fold_id": int(fold_id)},
                )
            )
        return signals

    result = simulate_portfolio(
        all_raw=fold_raw,
        all_main_index={symbol: all_main_index[symbol] for symbol in fold_raw},
        start_ts=start_ts,
        end_ts=end_ts,
        config=runtime_bt.build_execution_config(float(initial_balance), entry_threshold),
        signal_provider=fold_signal_provider,
        trade_number_start=int(trade_number_start),
        log_config=EngineLogConfig(
            verbose_trades=bool(verbose_trades),
            fold_id=int(fold_id),
            close_prefix=f"[fold {fold_id}][{{ts}}]",
            open_prefix=f"[fold {fold_id}][{{ts}}]",
            final_prefix=f"[fold {fold_id}][{{ts}}]",
        ),
    )
    result["summary"].update(
        {
            "fold_id": int(fold_id),
            "period_start": str(test_timestamps[0]),
            "period_end": str(test_timestamps[-1]),
            "symbols": list(fold_raw.keys()),
            "ending_trade_number": int(result["next_trade_number"] - 1),
        }
    )
    return result


def plot_equity_curve(timestamps: list[pd.Timestamp], equity_curve: list[float], output_path: Path):
    if len(equity_curve) <= 1:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(timestamps), equity_curve, label="WF OOS Equity")
    plt.axhline(y=equity_curve[0], linestyle="--")
    plt.title("Walk-Forward OOS Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
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

    dataset = train.load_training_frame(args.db_path, args.symbols)
    feature_columns = train.select_feature_columns(dataset)
    event_filter_config = dataset.attrs.get("event_filter_config", {})
    use_symbol_feature = "symbol" in feature_columns
    symbol_categories = list(dict.fromkeys(args.symbols)) if use_symbol_feature else None
    runtime_required_columns = resolve_runtime_required_columns(feature_columns, event_filter_config)

    all_raw, all_features, all_main_index = load_runtime_inputs(
        args.symbols,
        runtime_required_columns,
        symbol_categories,
    )

    all_folds = train.generate_walk_forward_splits(dataset, args)
    folds = train.select_walk_forward_folds(all_folds, args)
    if not folds:
        raise RuntimeError("No walk-forward folds selected.")

    print(
        f"Walk-forward backtest: {len(folds)} folds | "
        f"train={args.wf_train_months}M valid={args.wf_val_months}M test={args.wf_test_months}M "
        f"step={args.wf_step_months}M embargo={args.wf_embargo_bars}"
    )
    print(f"Feature profile: {getattr(cfg, 'FEATURE_BUILD_REQUEST', {}).get('profile', 'unknown')}")
    print(f"Using {len(feature_columns)} model inputs")
    entry_threshold = float(cfg.ENTRY_THRESHOLD)
    print(f"Entry threshold: {entry_threshold:.2f}")

    balance = float(runtime_bt.BACKTEST_INITIAL_BALANCE)
    next_trade_number = 1
    all_trades = []
    all_equity_curve = []
    all_equity_timestamps = []
    fold_summaries = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_df = fold["train_df"].copy()
        valid_df = fold["valid_df"].copy()
        test_df = fold["test_df"].copy()
        train_df, valid_df, test_df, active_feature_columns, clip_bounds = train.prepare_modeling_frames(
            train_df,
            valid_df,
            test_df,
            feature_columns,
            split_label=f"WF backtest fold {fold_id}",
        )
        model = train.train_validation_model(
            train_df,
            valid_df,
            active_feature_columns,
            args.seed + fold_id - 1,
            args=args,
        )
        fold_test_predictions = train.build_prediction_frame(model, test_df, active_feature_columns)
        fold_test_metrics = train.compute_metrics_from_prediction_frame(fold_test_predictions, split_name="test")
        prepared_feature_store = prepare_feature_store_for_fold(
            all_features,
            active_feature_columns,
            clip_bounds,
            symbol_categories,
        )
        if not prepared_feature_store:
            raise RuntimeError(f"WF fold {fold_id}: no usable prepared rows remain after input preparation.")

        fold_result = run_fold_backtest(
            fold_id=fold_id,
            model=model,
            feature_columns=active_feature_columns,
            runtime_required_columns=runtime_required_columns,
            event_filter_config=event_filter_config,
            all_raw=all_raw,
            all_features=all_features,
            all_main_index=all_main_index,
            prepared_feature_store=prepared_feature_store,
            start_ts=pd.to_datetime(fold["test_period"]["start"]),
            end_ts=pd.to_datetime(fold["test_period"]["end"]),
            initial_balance=balance,
            trade_number_start=next_trade_number,
            entry_threshold=entry_threshold,
            verbose_trades=bool(args.verbose_trades),
        )

        balance = float(fold_result["ending_balance"])
        next_trade_number = int(fold_result["next_trade_number"])
        all_trades.extend(fold_result["trades"])
        all_equity_curve.extend(fold_result["equity_curve"])
        all_equity_timestamps.extend(fold_result["equity_timestamps"])

        fold_summary = fold_result["summary"]
        fold_summary["model_test_metrics"] = {
            "roc_auc": fold_test_metrics["roc_auc"],
            "pr_auc": fold_test_metrics["pr_auc"],
            "mcc": fold_test_metrics["mcc"],
            "accuracy": fold_test_metrics["accuracy"],
        }
        fold_summary["feature_count"] = int(len(active_feature_columns))
        fold_summaries.append(fold_summary)

        print(
            f"WF fold {fold_id} summary | trades={fold_summary['total_trades']} | "
            f"winrate={fold_summary['win_rate_pct']:.2f}% | "
            f"return={fold_summary['total_return_pct']:.2f}% | "
            f"max_dd={fold_summary['max_drawdown_pct']:.2f}% | "
            f"test_roc_auc={fold_test_metrics['roc_auc']:.4f} | "
            f"test_pr_auc={fold_test_metrics['pr_auc']:.4f}"
        )

    aggregate_summary = summarize_performance(
        initial_balance=float(runtime_bt.BACKTEST_INITIAL_BALANCE),
        ending_balance=balance,
        trades=all_trades,
        equity_curve=all_equity_curve,
        equity_timestamps=all_equity_timestamps,
        max_drawdown=None,
    )
    aggregate_summary["fold_count"] = int(len(fold_summaries))
    aggregate_summary["symbols"] = list(args.symbols)

    print(
        f"WF stitched OOS backtest | trades={aggregate_summary['total_trades']} | "
        f"winrate={aggregate_summary['win_rate_pct']:.2f}% | "
        f"end_balance=${aggregate_summary['ending_balance']:.2f} | "
        f"return={aggregate_summary['total_return_pct']:.2f}% | "
        f"max_dd={aggregate_summary['max_drawdown_pct']:.2f}% | "
        f"pf={aggregate_summary['profit_factor']:.2f} | "
        f"sharpe={aggregate_summary['sharpe']:.2f}"
    )

    output_json = cfg.MODELS_DIR / f"{args.model_name}_walk_forward_backtest.json"
    output_chart = cfg.BACKTEST_CHARTS_DIR / f"{args.model_name}_walk_forward_equity_curve.png"
    output_json.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "feature_profile": getattr(cfg, "FEATURE_BUILD_REQUEST", {}).get("profile", "unknown"),
                "walk_forward": {
                    "train_months": int(args.wf_train_months),
                    "validation_months": int(args.wf_val_months),
                    "test_months": int(args.wf_test_months),
                    "step_months": int(args.wf_step_months),
                    "embargo_bars": int(args.wf_embargo_bars),
                    "folds": fold_summaries,
                },
                "aggregate": aggregate_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_equity_curve(all_equity_timestamps, all_equity_curve, output_chart)
    print(f"Saved walk-forward backtest summary to {output_json}")
    print(f"Saved walk-forward equity chart to {output_chart}")


def backtest_walk_forward():
    return main()


if __name__ == "__main__":
    backtest_walk_forward()
