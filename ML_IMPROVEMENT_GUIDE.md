# ML Trading System Improvement Guide

## Current Performance Issues

**Backtest Results:**
- **Trades:** 474 (extremely low execution rate)
- **Win Rate:** 39.03% (below breakeven threshold of ~45%)
- **Return:** -33.55% (significant losses)
- **Profit Factor:** 0.83 (losing money overall)
- **Max Drawdown:** 34.78% (too high)
- **Sharpe Ratio:** -1.29 (poor risk-adjusted returns)

**Diagnostic Analysis:**
- `below_entry_threshold`: 60,271 predictions (96.9% rejected)
- `prediction_seen`: 62,177 total predictions
- **Execution Rate:** Only 0.76% of predictions become trades
- Model generates few high-confidence signals that perform poorly when executed

---

## Root Causes Identified

### 1. **Overly Restrictive Entry Threshold**
- Current: 0.58 (too high)
- Result: Model only trades on extreme confidence levels
- Problem: Those "extreme confidence" signals are actually overfit noise

### 2. **Noisy Training Data**
- NEITHER outcomes (no barrier hit) included in training
- These ambiguous cases confuse the model
- Model learns wrong patterns from unclear outcomes

### 3. **Feature Quality Issues**
- Some features add noise rather than signal
- Outlier clipping too permissive (0.01/0.99)
- Missing higher-timeframe context in current profile

### 4. **Model Overfitting**
- Too many estimators (4000) with fast learning rate (0.01)
- Insufficient regularization
- Model memorizes training data instead of learning generalizable patterns

### 5. **Barrier Settings Too Tight**
- ATR multiplier 1.25 causes frequent stop-outs from noise
- TP/SL ratio 2.0 doesn't account for volatility clustering
- Many trades exit on noise before reaching targets

### 6. **No Signal Filtering**
- Trades execute in all market conditions
- Low-volatility and chaotic regimes degrade performance
- No filter for trend efficiency or volatility regimes

### 7. **Insufficient Training Data**
- Only 6 months training window
- 1 month validation too short for robust statistics
- Model hasn't seen enough market regimes

---

## 8-Step Improvement Plan

### Step 1: Lower Entry Threshold ✅
**Change:** 0.58 → 0.52

**Rationale:**
- Increase trade execution rate from 0.76% to target 2-5%
- Better statistical significance for performance analysis
- Current threshold filters out potentially profitable mid-confidence signals

**Expected Impact:**
- More trades (target: 1,500-3,000 from same prediction pool)
- Better understanding of true model calibration
- May initially lower win rate but improves profit factor if selective

---

### Step 2: Enable Strict Training Mode ✅
**Change:** Exclude NEITHER outcomes completely

**Configuration:**
```python
TRAIN_EVENT_FILTER_CONFIG = {
    "enabled": True,
    "side": "long",
    "allowed_outcomes": ["TP_FIRST", "SL_FIRST"],
}
```

**Rationale:**
- Train only on clear directional outcomes
- Remove ambiguous cases that confuse decision boundaries
- Force model to learn patterns that lead to decisive moves

**Expected Impact:**
- Cleaner decision boundaries
- Higher precision on predicted class
- Win rate improvement: +5-8 percentage points

---

### Step 3: Improve Feature Quality ✅
**Changes:**
- Upgrade to `baseline_ohlcv_htf_market` profile
- Aggressive outlier clipping: 0.02/0.98
- Disable noisy features manually

**Key Features Added:**
- `htf_trend_direction` - Higher timeframe trend context
- `htf_close_pos_range_60` - HTF position in range
- `rel_day_return_vs_btc` - Relative strength vs Bitcoin
- `market_breadth_pos_intraday` - Market-wide sentiment
- `btc_daily_trend_direction` - BTC trend filter

**Rationale:**
- Richer feature set captures more market context
- Aggressive clipping removes extreme outliers that skew splits
- Disabling noisy features reduces overfitting surface

**Expected Impact:**
- Better regime detection
- Improved feature importance stability
- Win rate improvement: +3-5 percentage points

---

### Step 4: Optimize Model Parameters ✅
**Changes:**
| Parameter | Before | After | Purpose |
|-----------|--------|-------|---------|
| `n_estimators` | 4000 | 3000 | Reduce overfitting |
| `learning_rate` | 0.01 | 0.005 | Slower, more stable learning |
| `num_leaves` | 31 | 23 | Simpler trees |
| `min_child_samples` | 40 | 60 | More samples per leaf |
| `subsample` | 0.8 | 0.7 | More regularization |
| `colsample_bytree` | 0.8 | 0.7 | Feature diversity |
| `reg_alpha` | 0.1 | 0.3 | L1 regularization |
| `reg_lambda` | 0.5 | 1.0 | L2 regularization |
| `early_stopping_rounds` | 50 | 80 | Patient stopping |

**Rationale:**
- Slower learning rate with fewer trees prevents memorization
- Higher regularization forces simpler, more generalizable patterns
- More early stopping patience allows proper convergence

**Expected Impact:**
- Reduced overfitting gap (train vs validation performance)
- More stable out-of-sample performance
- Profit factor improvement: 0.83 → 1.1-1.3

---

### Step 5: Adjust Barriers ✅
**Changes:**
| Parameter | Before | After | Purpose |
|-----------|--------|-------|---------|
| `ATR_MULTIPLIER` | 1.25 | 1.5 | Wider stops |
| `TP_TO_SL_RATIO` | 2.0 | 2.5 | Better R:R |
| `MIN_PCT` | 0.0075 | 0.006 | Lower floor |
| `MAX_PCT` | 0.06 | 0.08 | Higher ceiling |

**Rationale:**
- Wider stops reduce noise exits (whipsaws)
- Higher TP/SL ratio captures more trending moves
- Dynamic barriers adapt to volatility regimes

**Expected Impact:**
- Fewer premature stop-outs
- Higher average winner to loser ratio
- Win rate may drop slightly but profit factor improves significantly
- Target: PF 1.3-1.5

---

### Step 6: Enable Signal Filtering ✅
**Configuration:**
```python
EVENT_FILTER_CONFIG = {
    "enabled": True,
    "side": "long",
    "required_columns": [
        "atr_pct_14",
        "trend_efficiency_16",
    ],
}
```

**Rationale:**
- Filter out low-volatility periods (wasted trades)
- Avoid chaotic, directionless markets
- Only trade when conditions align with strategy edge

**Expected Impact:**
- Fewer but higher-quality trades
- Improved win rate: +4-6 percentage points
- Reduced drawdowns during choppy periods

---

### Step 7: Increase Training Data ✅
**Changes:**
| Parameter | Before | After |
|-----------|--------|-------|
| `WF_TRAIN_MONTHS` | 6 | 9 |
| `WF_VAL_MONTHS` | 1 | 2 |
| `START_DATE` | 2025-01-01 | 2024-06-01 |

**Rationale:**
- 9 months training captures more market regimes
- 2 months validation provides better statistical confidence
- Extended back to mid-2024 includes diverse volatility environments

**Expected Impact:**
- More robust pattern recognition
- Better generalization to unseen data
- Reduced variance in walk-forward results

---

### Step 8: Enhanced Risk Management ✅
**Changes:**
- Reduce risk after 3 consecutive losses
- Limit to 5 stop-outs per day maximum
- 24-bar cooldown after SL (vs 12 before)

**Rationale:**
- Protect capital during losing streaks
- Prevent revenge trading behavior
- Allow market to settle after stop-outs

**Expected Impact:**
- Reduced max drawdown: 34.78% → 20-25%
- Smoother equity curve
- Better psychological sustainability

---

## Implementation Workflow

### Phase 1: Re-extract Features
```bash
python etl.py
```
- Pulls fresh data with extended date range
- Builds new features with `baseline_ohlcv_htf_market` profile
- Stores in Parquet for faster IO

### Phase 2: Retrain Model
```bash
python train.py
```
- Uses strict training mode (TP_FIRST/SL_FIRST only)
- Applies optimized LightGBM parameters
- Walk-forward validation with 9/2/1 split

### Phase 3: Run Backtests
```bash
python bt_walk_forward_oos.py
```
- Tests with lowered 0.52 threshold
- Applies signal filtering
- Enforces enhanced risk management

### Phase 4: Analyze & Iterate
Review:
- Trade count and execution rate
- Win rate and profit factor
- Equity curve smoothness
- Max drawdown reduction
- Feature importance stability

---

## Expected Results

### Conservative Targets
| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Trades | 474 | 1,500-2,500 | +215-425% |
| Win Rate | 39.03% | 45-48% | +6-9 pp |
| Return | -33.55% | +5-15% | +38-48 pp |
| Profit Factor | 0.83 | 1.2-1.4 | +45-69% |
| Max DD | 34.78% | 20-25% | -28-42% |
| Sharpe | -1.29 | 0.5-1.0 | +1.8-2.3 pp |

### Optimistic Targets (if all improvements compound)
| Metric | Target |
|--------|--------|
| Win Rate | 50-52% |
| Return | +20-30% |
| Profit Factor | 1.5-1.8 |
| Max DD | 15-18% |
| Sharpe | 1.2-1.5 |

---

## Monitoring & Validation

### Key Metrics to Track
1. **Execution Rate:** Target 2-5% of predictions
2. **Win Rate by Confidence Bucket:**
   - 0.52-0.55: Should be 42-46%
   - 0.55-0.60: Should be 48-52%
   - 0.60+: Should be 55-60%
3. **Profit Factor by Symbol:** Identify which assets drive returns
4. **Monthly Returns:** Check for consistency vs clustering
5. **Drawdown Duration:** Time to recover from peaks

### Red Flags
- Execution rate still <1%: Threshold still too high or model underfitting
- Win rate <42%: Feature quality or training mode issues
- Profit factor <1.0: Barrier settings or risk management problems
- Max DD >30%: Need more aggressive risk reduction

### Green Flags
- Execution rate 2-5% with improving win rate
- Profit factor >1.2 and rising
- Smooth equity curve with limited drawdown duration
- Stable feature importance across walk-forward folds

---

## Next Iteration Ideas

If results improve but still not profitable:

### A. Advanced Feature Engineering
- Add order book imbalance features
- Incorporate funding rate signals
- Build momentum/mean-reversion regime classifier
- Add correlation-based features (crypto market structure)

### B. Model Architecture Changes
- Try XGBoost or CatBoost for comparison
- Implement ensemble of multiple models
- Add neural network for non-linear pattern capture
- Use stacking meta-model

### C. Execution Improvements
- Implement limit order logic (maker rebates)
- Add time-of-day filters (avoid low-liquidity hours)
- Scale position size by signal confidence
- Dynamic leverage based on volatility

### D. Labeling Enhancements
- Triple barrier method with meta-labeling
- Add time-decay weights to recent samples
- Use regression target (expected return) instead of classification
- Multi-class: strong_win, weak_win, neutral, weak_loss, strong_loss

---

## Troubleshooting

### Problem: Still too few trades (<500)
**Solutions:**
1. Lower threshold further to 0.50
2. Check if EVENT_FILTER_CONFIG is too restrictive
3. Verify FEATURE_BUILD_REQUEST profile is active
4. Ensure TRAIN_EVENT_FILTER_CONFIG isn't excluding too much data

### Problem: Win rate improves but profit factor stays <1
**Solutions:**
1. Increase BARRIER_TP_TO_SL_RATIO to 3.0
2. Reduce average loser size with tighter initial stops
3. Add pyramiding logic for winning trades
4. Implement partial profit-taking

### Problem: High variance between walk-forward folds
**Solutions:**
1. Increase WF_TRAIN_MONTHS to 12
2. Add more symbols for diversification
3. Reduce model complexity further (lower num_leaves)
4. Increase regularization parameters

### Problem: Model works in-sample but fails OOS
**Solutions:**
1. Further reduce LGBM_LEARNING_RATE to 0.003
2. Increase LGBM_MIN_CHILD_SAMPLES to 80-100
3. Add more aggressive feature clipping (0.03/0.97)
4. Enable sample weighting with lower NEITHER weight

---

## Conclusion

Your current system has a solid foundation but suffers from common ML trading pitfalls:
- Overfitting to noise
- Too conservative on entries
- Insufficient data quality controls
- Lack of regime filtering

The 8-step improvement plan addresses each issue systematically. Expected outcome: transformation from -33% return to +10-20% return with acceptable risk metrics.

**Critical Success Factors:**
1. Discipline to run full retrain/backtest cycle
2. Patience to analyze results before tweaking further
3. Willingness to iterate if first attempt doesn't hit targets
4. Focus on process (feature quality, training rigor) over outcomes (single backtest result)

Start with `config.local.py` as configured, run the full pipeline, and use the monitoring framework to guide your next iteration.
