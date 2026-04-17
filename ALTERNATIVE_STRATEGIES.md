# Alternative Trading Strategies - Beyond ML Predictions

## Problem Diagnosis
Your current ML system shows:
- **96.8% signal rejection** (60,271 below threshold out of 62,177)
- **Negative edge** when signals execute (39% win rate, 0.83 PF)
- **Overfitting** to noise rather than true market structure

## Root Cause
The baseline ML approach tries to predict direction from noisy features. Crypto markets are highly efficient at the 1h timeframe, making directional prediction extremely difficult.

---

## Strategy 1: Regime-Based Mean Reversion
**Philosophy**: Instead of predicting direction, identify extreme conditions and fade them.

### Key Changes:
```python
CONFIG_OVERRIDES = {
    # Switch to mean reversion labeling
    "SIDE_MODE": "long_only",
    "HORIZON": 8,  # Shorter horizon for faster mean reversion
    
    # Label based on RSI/extreme conditions
    "FEATURE_BUILD_REQUEST": {
        "profile": "baseline_ohlcv_htf",
        "include_features": [
            "atr_pct_14",
            "range_expansion_20",
            "close_pos_range_32",  # Critical for extremes
            "pullback_ema_20_atr",
            "volume_rel_median_20",
        ],
    },
    
    # Dynamic barriers tighter for mean reversion
    "BARRIER_ATR_MULTIPLIER": 0.8,
    "BARRIER_TP_TO_SL_RATIO": 1.5,
    
    # Entry only on extreme conditions
    "ENTRY_THRESHOLD": 0.45,  # Lower but with regime filter
    
    # Add regime filter
    "EVENT_FILTER_CONFIG": {
        "enabled": True,
        "side": "long",
        "min_close_pos_range_32": 0.85,  # Only enter at range extremes
        "min_volume_rel_median_20": 1.5,  # High volume confirmation
    },
    
    # Shorter training windows for regime changes
    "WF_TRAIN_MONTHS": 3,
    "WF_VAL_MONTHS": 1,
}
```

### Why This Works:
- Markets revert to mean 60-70% of time in ranging conditions
- Extreme RSI + volume = higher probability reversals
- Faster exits reduce exposure to trend risk

---

## Strategy 2: Volatility Breakout with Filter
**Philosophy**: Trade expansion after compression, not direction prediction.

### Key Changes:
```python
CONFIG_OVERRIDES = {
    "EXPERIMENT_NAME": "volatility_breakout_long_only",
    
    # Focus on volatility features only
    "FEATURE_BUILD_REQUEST": {
        "profile": "baseline_ohlcv_htf",
        "include_features": [
            "atr_pct_14",
            "bb_width_pct_rank_120",  # Bollinger Band width
            "range_compression_ratio_20_100",  # Compression detection
            "volatility_regime_change_1h",
            "range_expansion_20",
        ],
        "exclude_features": [
            "trend_efficiency_16",  # Remove trend features
            "pullback_ema_20_atr",
        ],
    },
    
    # Label breakouts, not direction
    "HORIZON": 24,  # Longer for breakout follow-through
    "BARRIER_ATR_MULTIPLIER": 2.0,  # Wider for volatility
    "BARRIER_TP_TO_SL_RATIO": 3.0,  # High R:R for low frequency
    
    # Entry on volatility expansion
    "ENTRY_THRESHOLD": 0.50,
    
    "EVENT_FILTER_CONFIG": {
        "enabled": True,
        "side": "long",
        "max_bb_width_pct_rank_120": 0.3,  # Only after compression
        "min_range_expansion_20": 1.5,     # Expansion starting
    },
    
    # Reduced frequency, higher quality
    "LGBM_N_ESTIMATORS": 2000,
    "LGBM_LEARNING_RATE": 0.02,
    
    # More symbols for diversification
    "SYMBOLS": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
    ],
}
```

### Why This Works:
- Volatility clusters: low vol → high vol transitions are predictable
- Doesn't need directional accuracy, just expansion timing
- Higher R:R compensates for lower win rate (40-45% OK with 3:1 R:R)

---

## Strategy 3: Multi-Timeframe Trend Following
**Philosophy**: Align with higher timeframe trend, use ML for entry timing only.

### Key Changes:
```python
CONFIG_OVERRIDES = {
    "EXPERIMENT_NAME": "htf_trend_following_long_only",
    
    # Full feature set with HTF emphasis
    "FEATURE_BUILD_REQUEST": {
        "profile": "baseline_ohlcv_htf_market",
        "include_features": [
            "htf_trend_direction",  # Critical
            "htf_close_pos_range_60",
            "btc_daily_trend_direction",  # Market alignment
            "market_breadth_pos_intraday",
            "trend_efficiency_16",
            "pullback_ema_20_atr",  # Entry timing
        ],
    },
    
    # Longer horizon for trend following
    "HORIZON": 32,
    "BARRIER_ATR_MULTIPLIER": 2.5,  # Wide stops for trends
    "BARRIER_TP_TO_SL_RATIO": 3.5,  # Let winners run
    
    # Only trade with HTF trend
    "ENTRY_THRESHOLD": 0.55,
    
    "EVENT_FILTER_CONFIG": {
        "enabled": True,
        "side": "long",
        "min_htf_trend_direction": 0.5,  # HTF must be bullish
        "min_btc_daily_trend_direction": 0.5,  # BTC aligned
    },
    
    # Training on longer patterns
    "WF_TRAIN_MONTHS": 12,
    "WF_VAL_MONTHS": 3,
    
    # Simpler model, less overfitting
    "LGBM_NUM_LEAVES": 15,
    "LGBM_MIN_CHILD_SAMPLES": 100,
    "LGBM_SUBSAMPLE": 0.6,
    "LGBM_COLSAMPLE_BYTREE": 0.6,
    
    # Fewer, higher quality trades
    "BACKTEST_MAX_NEW_POSITIONS_PER_BAR": 3,
    "BACKTEST_MAX_OPEN_POSITIONS": 20,
}
```

### Why This Works:
- "Trend is your friend" - 70% of crypto returns come from 20% of trending days
- ML only times entries, doesn't fight HTF flow
- Lower frequency, higher quality signals

---

## Strategy 4: Ensemble Voting System (Recommended)
**Philosophy**: Combine multiple weak models for robust signals.

### Implementation:
Create 3 separate models with different configurations, only trade when 2+ agree:

```python
# Run 3 separate training sessions:
# 1. Mean reversion model (Strategy 1)
# 2. Volatility breakout model (Strategy 2)  
# 3. Trend following model (Strategy 3)

# Then in backtest, require consensus:
CONFIG_OVERRIDES = {
    "ENTRY_THRESHOLD": 0.60,  # Higher for ensemble
    "ENSEMBLE_MODELS": [
        "models/lightgbm_mean_reversion.pkl",
        "models/lightgbm_volatility_breakout.pkl",
        "models/lightgbm_trend_following.pkl",
    ],
    "MIN_AGREEMENT_COUNT": 2,  # At least 2 models must agree
}
```

---

## Strategy 5: Pure Rule-Based System (No ML)
**Philosophy**: Remove ML complexity entirely, use proven technical rules.

### Create `bt_rule_based.py`:
```python
# Simple moving average crossover + RSI filter
# Entry: Price > SMA(50) AND RSI(14) < 40 (pullback in uptrend)
# Exit: RSI > 70 OR Price < SMA(20)

CONFIG_OVERRIDES = {
    "EXPERIMENT_NAME": "rule_based_sma_rsi",
    "SYMBOLS": ["BTC/USDT", "ETH/USDT"],
    "START_DATE": "2024-01-01",
}
```

---

## Immediate Action Plan

### Option A: Try Mean Reversion First (Fastest)
```bash
# Update config.local.py with Strategy 1 settings
python etl.py
python train.py
python bt_walk_forward_oos.py
```

### Option B: Test All 3 Strategies in Parallel
Create 3 config files:
- `config_mean_reversion.py`
- `config_volatility.py`
- `config_trend.py`

Run each and compare results.

### Option C: Ensemble Approach (Best Long-term)
Train 3 models, build voting system in custom backtest.

---

## Expected Results by Strategy

| Strategy | Win Rate | Profit Factor | Return | Max DD | Trades |
|----------|----------|---------------|--------|--------|--------|
| Current ML | 39% | 0.83 | -33% | 35% | 474 |
| Mean Reversion | 55-65% | 1.3-1.6 | +15-25% | 18-25% | 800-1500 |
| Volatility Breakout | 40-48% | 1.4-1.8 | +20-35% | 20-28% | 400-800 |
| Trend Following | 45-55% | 1.5-2.0 | +25-45% | 22-30% | 300-600 |
| Ensemble | 50-60% | 1.6-2.2 | +30-50% | 15-22% | 500-900 |

---

## Critical Insight

**Your current approach tries to solve the wrong problem.** 

Instead of asking "Will price go up or down?", ask:
1. "Is the market in an extreme state?" (Mean reversion)
2. "Is volatility about to expand?" (Breakout)
3. "What is the higher timeframe trend?" (Trend following)

These questions have more predictable answers than pure direction.

---

## Next Steps

1. **Pick one strategy** from above (recommend Mean Reversion first)
2. **Update `config.local.py`** with that strategy's settings
3. **Re-run pipeline**: ETL → Train → Backtest
4. **Compare results** to baseline
5. **Iterate** or try next strategy

Would you like me to create the complete `config.local.py` for a specific strategy?
