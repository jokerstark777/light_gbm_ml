from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ANSI_RESET = "\033[0m"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"


@dataclass(frozen=True)
class ExecutionConfig:
    initial_balance: float
    taker_com: float
    slippage: float
    leverage: float
    risk_per_trade: float
    min_stop_pct: float
    entry_threshold: float
    max_new_positions_per_bar: int
    max_open_positions: int
    sl_cooldown_bars: int = 0
    max_sl_per_day: int = 0
    reduce_risk_after_consecutive_losses: int = 0
    reduced_risk_per_trade: float = 0.0
    min_position_notional: float = 10.0
    max_holding_bars: int = 0


@dataclass(frozen=True)
class EngineLogConfig:
    verbose_trades: bool = True
    fold_id: int | None = None
    progress_every: int = 0
    progress_label: str = "Backtest progress"
    close_prefix: str = ""
    open_prefix: str = ""
    final_prefix: str = ""


@dataclass
class EntrySignal:
    symbol: str
    p_long: float
    stop_pct: float
    take_pct: float
    extra_position_fields: dict = field(default_factory=dict)
    extra_trade_fields: dict = field(default_factory=dict)


SignalProvider = Callable[[pd.Timestamp, Iterable[str], dict[str, dict]], list[EntrySignal]]


def compute_net_pnl_pct(entry_price: float, exit_price: float, taker_com: float) -> float:
    raw_pnl = (exit_price - entry_price) / entry_price
    return raw_pnl - (taker_com + taker_com)


def estimate_long_stop_loss_pct(stop_pct: float, config: ExecutionConfig) -> float:
    """Return planned net loss fraction at SL, including exit slippage and both taker fees."""
    stop_exit_multiplier = (1 - float(stop_pct)) * (1 - config.slippage)
    net_pnl_pct = stop_exit_multiplier - 1 - (config.taker_com + config.taker_com)
    return abs(float(net_pnl_pct))


def compute_trade_outcome(position: dict, exit_price: float, config: ExecutionConfig) -> tuple[float, float, float]:
    """Compute trade PnL including slippage (already in exit_price) and commissions.
    
    Args:
        position: Dict with 'entry' price and 'size' (position notional)
        exit_price: Exit price (should already include slippage adjustment)
        config: Execution configuration
        
    Returns:
        Tuple of (pnl_pct after fees, absolute PnL, commission amount)
    """
    # Raw price-based PnL percentage (exit_price should already include slippage)
    raw_pnl_pct = (exit_price - position["entry"]) / position["entry"]
    
    # Commission is based on position notional (both entry and exit)
    position_notional = float(position["size"])
    commission = position_notional * (config.taker_com + config.taker_com)
    
    # Absolute PnL from price movement minus commission
    gross_pnl = position_notional * raw_pnl_pct
    trade_profit = gross_pnl - commission
    
    # Net PnL percentage relative to position notional
    pnl_clean = trade_profit / position_notional if position_notional > 0 else 0.0
    
    return pnl_clean, trade_profit, commission


def compute_portfolio_equity(
    balance: float,
    positions: dict[str, dict | None],
    mark_prices: dict[str, float],
    config: ExecutionConfig,
) -> float:
    """Compute portfolio equity including unrealized PnL and accounting for commissions.
    
    Args:
        balance: Current cash balance
        positions: Dict of open positions
        mark_prices: Current mark prices for each symbol
        config: Execution configuration
        
    Returns:
        Total equity (balance + unrealized PnL)
    """
    equity = float(balance)
    for symbol, position in positions.items():
        if position is None:
            continue
        mark_price = mark_prices.get(symbol)
        if mark_price is None or not np.isfinite(mark_price):
            continue
        
        # Raw price-based PnL percentage
        raw_pnl_pct = (float(mark_price) - position["entry"]) / position["entry"]
        
        # Commission already paid at entry (and would be paid at exit)
        position_notional = float(position["size"])
        commission = position_notional * (config.taker_com + config.taker_com)
        
        # Unrealized PnL from price movement minus commission (already paid)
        gross_unrealized = position_notional * raw_pnl_pct
        net_unrealized = gross_unrealized - commission
        
        equity += net_unrealized
    return equity


def update_drawdown_stats(equity: float, peak_equity: float, max_drawdown: float) -> tuple[float, float]:
    if equity > peak_equity:
        peak_equity = equity

    if peak_equity > 0:
        current_dd = (peak_equity - equity) / peak_equity * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd

    return peak_equity, max_drawdown


def summarize_performance(
    initial_balance: float,
    ending_balance: float,
    trades: list[dict],
    equity_curve: list[float],
    equity_timestamps: list[pd.Timestamp],
    max_drawdown: float | None,
) -> dict:
    total_trades = len(trades)
    total_wins = sum(1 for trade in trades if trade["pnl_abs"] > 0)
    total_losses = total_trades - total_wins
    total_pnl_abs = float(sum(trade["pnl_abs"] for trade in trades))
    total_fees = float(sum(trade.get("commission", 0.0) for trade in trades))
    total_return_pct = float(((ending_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0)
    expectancy = float((total_pnl_abs / total_trades) if total_trades > 0 else 0.0)
    win_rate = float((total_wins / total_trades) * 100) if total_trades > 0 else 0.0

    peak_equity = float(initial_balance)
    computed_max_drawdown = 0.0
    sharpe = 0.0
    sortino = 0.0
    calmar = 0.0
    cagr = 0.0
    profit_factor = 0.0

    if equity_curve:
        equity_series = pd.Series(equity_curve, index=pd.to_datetime(equity_timestamps)).sort_index()
        for equity_value in equity_series.values.astype(float):
            peak_equity, computed_max_drawdown = update_drawdown_stats(
                equity_value,
                peak_equity,
                computed_max_drawdown,
            )
        daily_equity = equity_series.resample("D").last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 1 and float(daily_returns.std()) > 0:
            total_days = int((daily_equity.index[-1] - daily_equity.index[0]).days)
            if total_days > 0 and float(daily_equity.iloc[0]) > 0:
                cagr = float((daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1)
            mean_daily_return = float(daily_returns.mean())
            std_daily_return = float(daily_returns.std())
            sharpe = float((mean_daily_return / std_daily_return) * np.sqrt(365))

            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1 and float(downside_returns.std()) > 0:
                sortino = float((mean_daily_return / float(downside_returns.std())) * np.sqrt(365))

            active_max_drawdown = float(max_drawdown) if max_drawdown is not None else float(computed_max_drawdown)
            if active_max_drawdown > 0:
                calmar = float(cagr / (active_max_drawdown / 100))

    if trades:
        returns = np.array([trade["pnl_abs"] for trade in trades], dtype=float)
        gross_profit = float(sum(value for value in returns if value > 0))
        gross_loss = float(abs(sum(value for value in returns if value < 0)))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "initial_balance": float(initial_balance),
        "ending_balance": float(ending_balance),
        "total_trades": int(total_trades),
        "total_wins": int(total_wins),
        "total_losses": int(total_losses),
        "win_rate_pct": win_rate,
        "total_pnl_abs": total_pnl_abs,
        "total_return_pct": total_return_pct,
        "total_fees": total_fees,
        "expectancy_abs": expectancy,
        "max_drawdown_pct": float(max_drawdown) if max_drawdown is not None else float(computed_max_drawdown),
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
    }


def build_market_timestamps(all_raw: dict[str, dict], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[pd.Timestamp]:
    timestamps = set()
    for payload in all_raw.values():
        main_frame = payload.get("main")
        if main_frame is None or main_frame.empty:
            continue
        period_rows = main_frame.loc[
            (main_frame["timestamp"] >= start_ts) & (main_frame["timestamp"] <= end_ts),
            "timestamp",
        ]
        timestamps.update(period_rows.tolist())
    return sorted(timestamps)


def get_last_exec_row_on_or_before(df: pd.DataFrame, ts: pd.Timestamp):
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    eligible_rows = df.loc[df["timestamp"] <= ts]
    if eligible_rows.empty:
        return None
    return eligible_rows.iloc[-1].to_dict()


def _build_market_batch(
    symbols: Iterable[str],
    all_main_index: dict[str, dict],
    current_ts: pd.Timestamp,
    next_ts: pd.Timestamp,
) -> dict[str, dict]:
    market_batch = {}
    for symbol in symbols:
        curr_exec = all_main_index.get(symbol, {}).get(current_ts)
        next_exec = all_main_index.get(symbol, {}).get(next_ts)
        if curr_exec is None or next_exec is None:
            continue
        market_batch[symbol] = {
            "current_close": float(curr_exec["close"]),
            "next_open": float(next_exec["open"]),
            "next_high": float(next_exec["high"]),
            "next_low": float(next_exec["low"]),
            "next_close": float(next_exec["close"]),
        }
    return market_batch


def _effective_risk_per_trade(consecutive_loss_count: int, config: ExecutionConfig) -> float:
    if (
        config.reduce_risk_after_consecutive_losses > 0
        and 0 < config.reduced_risk_per_trade < config.risk_per_trade
        and consecutive_loss_count >= config.reduce_risk_after_consecutive_losses
    ):
        return config.reduced_risk_per_trade
    return config.risk_per_trade


def _record_month(monthly_stats: dict, month_key: str, balance: float, include_start_balance: bool) -> None:
    if month_key in monthly_stats:
        return
    monthly_stats[month_key] = {
        "pnl_abs": 0.0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
    }
    if include_start_balance:
        monthly_stats[month_key]["start_balance"] = balance


def _add_trade_to_month(monthly_stats: dict, month_key: str, trade_profit: float, pnl_clean: float) -> None:
    monthly_stats[month_key]["pnl_abs"] += trade_profit
    monthly_stats[month_key]["trades"] += 1
    if pnl_clean > 0:
        monthly_stats[month_key]["wins"] += 1
    else:
        monthly_stats[month_key]["losses"] += 1


def _format_prefix(prefix: str, timestamp: pd.Timestamp, trade_number: int | None = None) -> str:
    if prefix:
        return prefix.format(ts=timestamp, trade_number=trade_number or "")
    return f"[{timestamp}]"


def _exit_emoji(reason: str) -> str:
    if reason == "TP":
        return "✅"
    if reason == "SL":
        return "❌"
    if reason == "TIME":
        return "⏱"
    if reason == "FINAL":
        return "🏁"
    return "•"


def _display_reason(reason: str) -> str:
    if reason == "SL":
        return "SL/LIQ"
    return reason


def _format_signed_dollars(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def _format_pnl_pct(value: float) -> str:
    color = ANSI_GREEN if value >= 0 else ANSI_RED
    return f"{color}{value * 100:+.2f}%{ANSI_RESET}"


def _format_balance(value: float) -> str:
    return f"{ANSI_YELLOW}{value:.2f}{ANSI_RESET}"


def simulate_portfolio(
    *,
    all_raw: dict[str, dict],
    all_main_index: dict[str, dict],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    config: ExecutionConfig,
    signal_provider: SignalProvider,
    trade_number_start: int = 1,
    log_config: EngineLogConfig | None = None,
    skip_empty_signal_timestamps: set[pd.Timestamp] | None = None,
    include_month_start_balance: bool = False,
) -> dict:
    log_config = log_config or EngineLogConfig()
    symbols = list(all_raw.keys())
    test_timestamps = build_market_timestamps(all_raw, start_ts, end_ts)
    if len(test_timestamps) < 2:
        raise RuntimeError("Too little market data inside the simulation period.")

    balance = float(config.initial_balance)
    initial_balance = float(balance)
    positions = {symbol: None for symbol in symbols}
    trades = []
    equity_curve = []
    equity_timestamps = []
    monthly_stats = {}
    peak_equity = float(balance)
    max_drawdown = 0.0
    used_margin = 0.0
    next_trade_number = int(trade_number_start)
    stop_cooldown_until_index = {symbol: -1 for symbol in symbols}
    current_trade_day = None
    daily_sl_count = 0
    daily_stop_announced = False
    consecutive_loss_count = 0
    latest_mark_prices = {}
    rejection_counts = Counter()

    signal_timestamps = set(skip_empty_signal_timestamps or [])

    for i in range(len(test_timestamps) - 1):
        current_ts = pd.Timestamp(test_timestamps[i])
        next_ts = pd.Timestamp(test_timestamps[i + 1])
        has_open_positions = any(position is not None for position in positions.values())
        if signal_timestamps and not has_open_positions and current_ts not in signal_timestamps:
            continue

        if log_config.progress_every > 0 and i > 0 and i % log_config.progress_every == 0:
            print(
                f"{log_config.progress_label}: {i}/{len(test_timestamps) - 1} candles | "
                f"ts={current_ts} | trades={len(trades)} | balance={balance:.2f}",
                flush=True,
            )

        trade_day = next_ts.normalize()
        if current_trade_day is None or trade_day != current_trade_day:
            current_trade_day = trade_day
            daily_sl_count = 0
            daily_stop_announced = False

        month_key = next_ts.strftime("%Y-%m")
        _record_month(monthly_stats, month_key, balance, include_month_start_balance)

        market_batch = _build_market_batch(symbols, all_main_index, current_ts, next_ts)
        for symbol, ctx in market_batch.items():
            latest_mark_prices[symbol] = ctx["current_close"]

        current_equity = compute_portfolio_equity(balance, positions, latest_mark_prices, config)
        equity_curve.append(current_equity)
        equity_timestamps.append(current_ts)
        peak_equity, max_drawdown = update_drawdown_stats(current_equity, peak_equity, max_drawdown)

        for symbol, ctx in market_batch.items():
            if positions[symbol] is None:
                continue

            position = positions[symbol]
            entry_price = position["entry"]
            stop_price = entry_price * (1 - position["stop_pct"])
            take_price = entry_price * (1 + position["take_pct"])
            exit_price = 0.0
            reason = ""

            if ctx["next_low"] <= stop_price:
                exit_price = (ctx["next_open"] if ctx["next_open"] < stop_price else stop_price) * (1 - config.slippage)
                reason = "SL"
            elif ctx["next_high"] >= take_price:
                exit_price = take_price * (1 - config.slippage)
                reason = "TP"
            elif config.max_holding_bars > 0:
                signal_bar_index = int(position.get("signal_bar_index", i))
                bars_since_signal = (i + 1) - signal_bar_index
                if bars_since_signal >= config.max_holding_bars:
                    exit_price = ctx["next_close"] * (1 - config.slippage)
                    reason = "TIME"

            if not reason:
                continue

            pnl_clean, trade_profit, commission = compute_trade_outcome(position, exit_price, config)
            previous_loss_streak = consecutive_loss_count
            used_margin = max(0.0, used_margin - position["margin"])
            balance += trade_profit

            trade = {
                **position.get("extra_trade_fields", {}),
                "trade_number": int(position["trade_number"]),
                "sym": symbol,
                "direction": "LONG",
                "reason": reason,
                "pnl_pct": float(pnl_clean),
                "pnl_abs": float(trade_profit),
                "commission": float(commission),
                "risk_loss_pct": float(position.get("risk_loss_pct", 0.0)),
                "ts": str(next_ts),
            }
            trades.append(trade)
            _add_trade_to_month(monthly_stats, month_key, trade_profit, pnl_clean)

            if pnl_clean > 0:
                consecutive_loss_count = 0
            else:
                consecutive_loss_count += 1

            if reason == "SL":
                if config.sl_cooldown_bars > 0:
                    stop_cooldown_until_index[symbol] = i + config.sl_cooldown_bars
                daily_sl_count += 1

            positions[symbol] = None

            if log_config.verbose_trades:
                prefix = _format_prefix(log_config.close_prefix, next_ts, position["trade_number"])
                print(
                    f"{prefix} {_exit_emoji(reason)} CLOSE {symbol}: {_display_reason(reason)} | "
                    f"PnL={_format_pnl_pct(pnl_clean)} | Com={commission:.2f}$ | "
                    f"Bal={_format_balance(balance)}",
                    flush=True,
                )

            if (
                config.reduce_risk_after_consecutive_losses > 0
                and config.reduced_risk_per_trade < config.risk_per_trade
                and pnl_clean > 0
                and previous_loss_streak >= config.reduce_risk_after_consecutive_losses
                and log_config.verbose_trades
            ):
                prefix = _format_prefix(log_config.close_prefix, next_ts)
                print(
                    f"{prefix} Loss streak reset: "
                    f"risk per trade restored to {config.risk_per_trade * 100:.2f}%",
                    flush=True,
                )

            if (
                reason == "SL"
                and config.max_sl_per_day > 0
                and daily_sl_count >= config.max_sl_per_day
                and not daily_stop_announced
            ):
                daily_stop_announced = True
                if log_config.verbose_trades:
                    prefix = _format_prefix(log_config.close_prefix, next_ts)
                    print(
                        f"{prefix} Daily SL limit reached ({daily_sl_count}), "
                        "new entries are paused until next day",
                        flush=True,
                    )

        effective_risk_per_trade = _effective_risk_per_trade(consecutive_loss_count, config)
        if config.max_sl_per_day > 0 and daily_sl_count >= config.max_sl_per_day:
            continue

        candidate_symbols = [
            symbol
            for symbol in market_batch
            if positions[symbol] is None and i >= stop_cooldown_until_index.get(symbol, -1)
        ]
        signals = signal_provider(current_ts, candidate_symbols, market_batch)
        entry_candidates = []
        snapshot_balance = balance
        for signal in signals:
            if signal.symbol not in market_batch:
                rejection_counts["missing_market"] += 1
                continue
            if signal.symbol not in candidate_symbols:
                rejection_counts["blocked_symbol"] += 1
                continue
            if signal.stop_pct <= 0 or signal.stop_pct < config.min_stop_pct:
                rejection_counts["stop_too_small"] += 1
                continue
            if signal.take_pct <= signal.stop_pct:
                rejection_counts["take_not_above_stop"] += 1
                continue
            if signal.p_long < config.entry_threshold:
                rejection_counts["below_entry_threshold"] += 1
                continue

            ctx = market_batch[signal.symbol]
            entry_price = ctx["next_open"] * (1 + config.slippage)
            risk_capital = snapshot_balance * effective_risk_per_trade
            stop_loss_pct_with_costs = estimate_long_stop_loss_pct(signal.stop_pct, config)
            if stop_loss_pct_with_costs <= 0:
                rejection_counts["invalid_cost_aware_risk"] += 1
                continue

            position_notional = min(risk_capital / stop_loss_pct_with_costs, snapshot_balance * config.leverage)
            required_margin = position_notional / config.leverage
            if position_notional < config.min_position_notional:
                rejection_counts["position_too_small"] += 1
                continue

            entry_candidates.append(
                {
                    "symbol": signal.symbol,
                    "entry_price": entry_price,
                    "position_notional": position_notional,
                    "required_margin": required_margin,
                    "stop_pct": float(signal.stop_pct),
                    "take_pct": float(signal.take_pct),
                    "risk_loss_pct": float(stop_loss_pct_with_costs),
                    "p_long": float(signal.p_long),
                    "extra_position_fields": dict(signal.extra_position_fields),
                    "extra_trade_fields": dict(signal.extra_trade_fields),
                }
            )

        if not entry_candidates:
            continue

        entry_candidates.sort(key=lambda candidate: candidate["p_long"], reverse=True)
        opened_this_bar = 0
        open_positions_count = sum(position is not None for position in positions.values())
        for candidate in entry_candidates:
            if opened_this_bar >= config.max_new_positions_per_bar:
                break
            if open_positions_count >= config.max_open_positions:
                break

            available_balance = balance - used_margin
            if available_balance <= 0:
                break

            required_margin = min(candidate["required_margin"], available_balance)
            position_notional = min(candidate["position_notional"], required_margin * config.leverage)
            if position_notional < config.min_position_notional or required_margin <= 0:
                rejection_counts["insufficient_margin_or_size"] += 1
                continue

            trade_number = next_trade_number
            next_trade_number += 1
            used_margin += required_margin
            positions[candidate["symbol"]] = {
                "trade_number": int(trade_number),
                "entry": float(candidate["entry_price"]),
                "size": float(position_notional),
                "margin": float(required_margin),
                "stop_pct": float(candidate["stop_pct"]),
                "take_pct": float(candidate["take_pct"]),
                "risk_loss_pct": float(candidate["risk_loss_pct"]),
                "ts_open": str(next_ts),
                "signal_bar_index": int(i),
                "entry_bar_index": int(i + 1),
                "extra_trade_fields": candidate["extra_trade_fields"],
                **candidate["extra_position_fields"],
            }
            opened_this_bar += 1
            open_positions_count += 1

            if log_config.verbose_trades:
                prefix = _format_prefix(log_config.open_prefix, next_ts, trade_number)
                extra = ""
                if "fold" in candidate["extra_position_fields"]:
                    extra = f", fold={candidate['extra_position_fields']['fold']}"
                entry_price = candidate["entry_price"]
                print(
                    f"{prefix} 🔥 OPEN LONG: {candidate['symbol']} "
                    f"(Long={candidate['p_long']:.2f}{extra}) at {entry_price:.4f} | "
                    f"Size: {position_notional:.2f}$ Margin: {required_margin:.2f}$",
                    flush=True,
                )

    last_timestamp = pd.Timestamp(test_timestamps[-1])
    last_mark_prices = {}
    for symbol in symbols:
        last_exec = get_last_exec_row_on_or_before(all_raw[symbol]["main"], end_ts)
        if last_exec is not None:
            last_mark_prices[symbol] = float(last_exec["close"])

    final_month_key = last_timestamp.strftime("%Y-%m")
    _record_month(monthly_stats, final_month_key, balance, include_month_start_balance)

    for symbol, position in list(positions.items()):
        if position is None:
            continue
        mark_price = last_mark_prices.get(symbol)
        if mark_price is None or not np.isfinite(mark_price):
            continue

        exit_price = mark_price * (1 - config.slippage)
        pnl_clean, trade_profit, commission = compute_trade_outcome(position, exit_price, config)
        used_margin = max(0.0, used_margin - position["margin"])
        balance += trade_profit

        trade = {
            **position.get("extra_trade_fields", {}),
            "trade_number": int(position["trade_number"]),
            "sym": symbol,
            "direction": "LONG",
            "reason": "FINAL",
            "pnl_pct": float(pnl_clean),
            "pnl_abs": float(trade_profit),
            "commission": float(commission),
            "risk_loss_pct": float(position.get("risk_loss_pct", 0.0)),
            "ts": str(last_timestamp),
        }
        trades.append(trade)
        _add_trade_to_month(monthly_stats, final_month_key, trade_profit, pnl_clean)
        positions[symbol] = None

        if log_config.verbose_trades:
            prefix = _format_prefix(log_config.final_prefix, last_timestamp, position["trade_number"])
            print(
                f"{prefix} {_exit_emoji('FINAL')} CLOSE {symbol}: FINAL | "
                f"PnL={_format_pnl_pct(pnl_clean)} | Com={commission:.2f}$ | "
                f"Bal={_format_balance(balance)}",
                flush=True,
            )

    final_equity = compute_portfolio_equity(balance, positions, {**latest_mark_prices, **last_mark_prices}, config)
    equity_curve.append(final_equity)
    equity_timestamps.append(last_timestamp)
    peak_equity, max_drawdown = update_drawdown_stats(final_equity, peak_equity, max_drawdown)

    summary = summarize_performance(
        initial_balance=initial_balance,
        ending_balance=balance,
        trades=trades,
        equity_curve=equity_curve,
        equity_timestamps=equity_timestamps,
        max_drawdown=max_drawdown,
    )
    summary.update(
        {
            "period_start": str(test_timestamps[0]),
            "period_end": str(test_timestamps[-1]),
            "symbols": symbols,
            "ending_trade_number": int(next_trade_number - 1),
        }
    )

    return {
        "summary": summary,
        "ending_balance": float(balance),
        "next_trade_number": int(next_trade_number),
        "trades": trades,
        "equity_curve": equity_curve,
        "equity_timestamps": equity_timestamps,
        "monthly_stats": monthly_stats,
        "candidate_diagnostics": {key: int(value) for key, value in sorted(rejection_counts.items())},
        "test_timestamps": test_timestamps,
    }
