# Optimized configuration for improved ML trading accuracy and stability
# Addresses: low win rate (39%), negative returns (-33.55%), low trade execution

CONFIG_OVERRIDES = {
    # ============================================================
    # 1. LOWER ENTRY THRESHOLD - Capture more signals for analysis
    # ============================================================
    # Reduced from 0.58 to 0.52 to increase trade execution rate
    # Current: 474 trades from 62,177 predictions (0.76% execution)
    # Target: 2-5% execution rate for better statistical significance
    "ENTRY_THRESHOLD": 0.52,
    
    # ============================================================
    # 2. STRICT TRAINING MODE - Remove noisy NEITHER outcomes
    # ============================================================
    # Only train on clear TP_FIRST and SL_FIRST outcomes
    # Excludes NEITHER cases that confuse the model
    "TRAIN_EVENT_FILTER_CONFIG": {
        "enabled": True,
        "side": "long",
        "allowed_outcomes": ["TP_FIRST", "SL_FIRST"],
    },
    
    # ============================================================
    # 3. IMPROVED FEATURE QUALITY - Richer profiles + aggressive clipping
    # ============================================================
    # Use full HTF+Market feature profile for better context
    "FEATURE_BUILD_REQUEST": {
        "profile": "baseline_ohlcv_htf_market",
        "include_features": [],
        "exclude_features": [],
        "exclude_blocks": [],
    },
    # More aggressive outlier removal
    "ENABLE_FEATURE_CLIP": True,
    "FEATURE_CLIP_LOWER_Q": 0.02,
    "FEATURE_CLIP_UPPER_Q": 0.98,
    # Remove manually disabled features that add noise
    "MANUAL_DISABLED_FEATURE_COLUMNS": [
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
    ],
    
    # ============================================================
    # 4. OPTIMIZED MODEL PARAMETERS - Reduce overfitting
    # ============================================================
    # Slower learning, more regularization, fewer trees
    "LGBM_N_ESTIMATORS": 3000,
    "LGBM_LEARNING_RATE": 0.005,
    "LGBM_NUM_LEAVES": 23,
    "LGBM_MIN_CHILD_SAMPLES": 60,
    "LGBM_SUBSAMPLE": 0.7,
    "LGBM_COLSAMPLE_BYTREE": 0.7,
    "LGBM_REG_ALPHA": 0.3,
    "LGBM_REG_LAMBDA": 1.0,
    "LGBM_EARLY_STOPPING_ROUNDS": 80,
    "LGBM_LOG_EVAL_PERIOD": 50,
    
    # ============================================================
    # 5. ADJUSTED BARRIERS - Wider stops, better R:R
    # ============================================================
    # Reduce noise exits with wider ATR-based stops
    "BARRIER_MODE": "dynamic",
    "BARRIER_ATR_MULTIPLIER": 1.5,      # Increased from 1.25
    "BARRIER_TP_TO_SL_RATIO": 2.5,      # Increased from 2.0
    "BARRIER_MIN_PCT": 0.006,           # Slightly lower minimum
    "BARRIER_MAX_PCT": 0.08,            # Higher maximum for volatile periods
    
    # ============================================================
    # 6. SIGNAL FILTERING - Enable execution filters
    # ============================================================
    # Filter out low-volatility and chaotic conditions
    "EVENT_FILTER_CONFIG": {
        "enabled": True,
        "side": "long",
        "required_columns": [
            "atr_pct_14",
            "trend_efficiency_16",
        ],
    },
    
    # ============================================================
    # 7. INCREASED TRAINING DATA - Longer walk-forward windows
    # ============================================================
    # Extend training period for more robust patterns
    "WF_TRAIN_MONTHS": 9,               # Increased from 6
    "WF_VAL_MONTHS": 2,                 # Increased from 1
    "WF_TEST_MONTHS": 1,
    "WF_STEP_MONTHS": 1,
    
    # ============================================================
    # 8. BACKTEST RISK MANAGEMENT - More conservative
    # ============================================================
    # Reduce position sizing during drawdowns
    "BACKTEST_REDUCE_RISK_AFTER_CONSECUTIVE_LOSSES": 3,
    "BACKTEST_REDUCED_RISK_PER_TRADE": 0.005,
    "BACKTEST_MAX_SL_PER_DAY": 5,       # Limit daily stop-outs
    "BACKTEST_SL_COOLDOWN_BARS": 24,    # Increased cooldown after SL
    
    # ============================================================
    # ADDITIONAL STABILITY IMPROVEMENTS
    # ============================================================
    # Enable sample weighting to prioritize clear outcomes
    "TRAIN_SAMPLE_WEIGHT_CONFIG": {
        "enabled": True,
        "outcome_weights": {
            "TP_FIRST": 1.0,
            "SL_FIRST": 1.0,
            "NEITHER": 0.0,             # Zero weight if not excluded
        },
    },
    
    # Focus on highest quality symbols only
    "SYMBOLS": [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "DOGE/USDT",
    ],
    
    # Extended date range for more training data
    "START_DATE": "2024-06-01",
    
    # Ensure production mode checks are active
    "CONFIG_STAGE": "research",
}
