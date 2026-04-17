# Mean Reversion Strategy Configuration
# Philosophy: Fade extreme conditions instead of predicting direction
# Expected: 55-65% win rate, 1.3-1.6 PF, +15-25% return

CONFIG_OVERRIDES = {
    # ============================================================
    # STRATEGY IDENTITY
    # ============================================================
    "EXPERIMENT_NAME": "mean_reversion_regime_long_only",
    "CONFIG_STAGE": "research",
    
    # ============================================================
    # 1. MARKET UNIVERSE - Focus on liquid majors
    # ============================================================
    "SYMBOLS": [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "XRP/USDT",
    ],
    "TIMEFRAME": "1h",
    "HTF_TIMEFRAME": "1d",
    "START_DATE": "2024-06-01",
    "END_DATE": "2026-03-27 21:00:00",
    
    # ============================================================
    # 2. FEATURE PROFILE - Mean reversion focused
    # ============================================================
    "FEATURE_BUILD_REQUEST": {
        "profile": "baseline_ohlcv_htf",
        "include_features": [
            "atr_pct_14",
            "range_expansion_20",
            "close_pos_range_32",      # CRITICAL: Extreme detection
            "pullback_ema_20_atr",     # CRITICAL: Pullback entry
            "volume_rel_median_20",    # Volume confirmation
            "body_to_range",
            "lower_wick_to_range",
            "upper_wick_to_range",
            "trend_efficiency_16",
        ],
        "exclude_features": [
            "htf_trend_direction",     # Remove trend features
            "htf_close_pos_range_60",
            "market_dispersion_return_4h",
            "rel_day_return_vs_btc",
            "rel_day_return_vs_market",
        ],
    },
    "ENABLE_FEATURE_CLIP": True,
    "FEATURE_CLIP_LOWER_Q": 0.02,
    "FEATURE_CLIP_UPPER_Q": 0.98,
    "MANUAL_DISABLED_FEATURE_COLUMNS": [
        "price_acceleration_8",
        "consecutive_direction_count",
        "bb_width_pct_rank_120",
        "range_compression_ratio_20_100",
        "return_4h_atr_norm",
        "zscore_vs_vwap_4h",
        "volatility_regime_change_1h",
        "dist_to_prev_day_high_atr",
        "dist_to_prev_day_low_atr",
    ],
    
    # ============================================================
    # 3. LABELING - Shorter horizon for mean reversion
    # ============================================================
    "SIDE_MODE": "long_only",
    "HORIZON": 8,  # Shorter: mean reversion is fast
    "BARRIER_MODE": "dynamic",
    "BARRIER_ATR_MULTIPLIER": 0.8,   # Tighter stops
    "BARRIER_TP_TO_SL_RATIO": 1.5,   # Lower R:R but higher win rate
    "BARRIER_MIN_PCT": 0.005,
    "BARRIER_MAX_PCT": 0.04,
    
    # ============================================================
    # 4. TRAINING - Strict first-hit, shorter windows
    # ============================================================
    "TRAIN_EVENT_FILTER_CONFIG": {
        "enabled": True,
        "side": "long",
        "allowed_outcomes": ["TP_FIRST", "SL_FIRST"],
    },
    "TRAIN_SAMPLE_WEIGHT_CONFIG": {
        "enabled": True,
        "outcome_weights": {
            "TP_FIRST": 1.0,
            "SL_FIRST": 1.0,
            "NEITHER": 0.0,
        },
    },
    "WF_TRAIN_MONTHS": 3,  # Shorter windows for regime changes
    "WF_VAL_MONTHS": 1,
    "WF_TEST_MONTHS": 1,
    "WF_STEP_MONTHS": 1,
    
    # Model optimized for mean reversion patterns
    "LGBM_N_ESTIMATORS": 2500,
    "LGBM_LEARNING_RATE": 0.015,
    "LGBM_NUM_LEAVES": 19,
    "LGBM_MIN_CHILD_SAMPLES": 80,
    "LGBM_SUBSAMPLE": 0.65,
    "LGBM_COLSAMPLE_BYTREE": 0.65,
    "LGBM_REG_ALPHA": 0.4,
    "LGBM_REG_LAMBDA": 1.2,
    "LGBM_EARLY_STOPPING_ROUNDS": 100,
    "LGBM_LOG_EVAL_PERIOD": 50,
    
    # ============================================================
    # 5. ENTRY FILTERS - Only trade extremes (CRITICAL)
    # ============================================================
    "ENTRY_THRESHOLD": 0.48,  # Lower threshold but strict filters
    
    "EVENT_FILTER_CONFIG": {
        "enabled": True,
        "side": "long",
        "required_columns": [
            "close_pos_range_32",
            "volume_rel_median_20",
            "pullback_ema_20_atr",
        ],
        # ONLY enter at range extremes (top 15%)
        "min_close_pos_range_32": 0.85,
        # Volume confirmation (50% above median)
        "min_volume_rel_median_20": 1.5,
        # Must be pulling back (not chasing)
        "max_pullback_ema_20_atr": -0.3,
    },
    
    # ============================================================
    # 6. BACKTEST - Conservative risk management
    # ============================================================
    "BACKTEST_INITIAL_BALANCE": 100,
    "TAKER_COM": 0.0004,
    "SLIPPAGE": 0.0003,
    "LEVERAGE": 1,
    "RISK_PER_TRADE": 0.01,
    "BACKTEST_MAX_NEW_POSITIONS_PER_BAR": 5,
    "BACKTEST_MAX_OPEN_POSITIONS": 30,
    "BACKTEST_MAX_HOLDING_BARS": 8,  # Match horizon
    "BACKTEST_SL_COOLDOWN_BARS": 12,
    "BACKTEST_MAX_SL_PER_DAY": 4,
    "BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES": 3,
    "BACKTEST_REDUCED_RISK_PER_TRADE": 0.005,
}
