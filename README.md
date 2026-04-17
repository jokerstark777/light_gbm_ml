# Light

`Light` is a local ML project for researching and backtesting crypto trading strategies on exchange candle (OHLCV) data. The project implements a complete pipeline from data ingestion through feature engineering, model training, and portfolio-level backtesting.

## Features

- **Multi-exchange support**: Bybit and Binance historical kline/candlestick data
- **Flexible storage**: SQLite for raw candles, Parquet/pyarrow for ML feature datasets
- **Comprehensive feature engineering**: Modular feature builders for trend, regime, volume, session context, HTF context, and more
- **Barrier-based labeling**: First-hit barrier outcomes with MFE/MAE tracking for both long and short setups
- **LightGBM classifier**: Optimized for imbalanced financial classification tasks
- **Walk-forward validation**: Support for single-split and walk-forward training/backtesting
- **Portfolio backtesting**: Multi-symbol execution engine with realistic costs, slippage, and risk management
- **Configurable experiments**: Preset configs and local overrides for reproducible research

## Stack

- Python 3.8+
- pandas / numpy
- LightGBM
- scikit-learn
- SQLite / SQLAlchemy
- Parquet / pyarrow
- matplotlib
- requests

## Project Structure

```text
light/
├── config.py                 # Main configuration and defaults
├── config.local.example.py   # Template for local overrides (gitignored)
├── etl.py                    # Data ingestion and feature preparation
├── train.py                  # Model training and validation
├── bt.py                     # Single-split backtest
├── bt_walk_forward.py        # Walk-forward backtest
├── bt_walk_forward_oos.py    # Out-of-sample walk-forward backtest
├── execution_engine.py       # Portfolio simulation and trade execution logic
├── signal_filter.py          # Entry signal filtering and event gates
├── requirements.txt
├── configs/
│   └── presets/              # Named experiment presets
├── src/
│   ├── contracts/            # Exchange interface contracts
│   ├── exchanges/            # Exchange-specific implementations (Bybit, Binance)
│   ├── features/             # Feature engineering modules and builders
│   ├── persistence/          # Storage abstraction (SQLite, Parquet)
│   └── types/                # Type definitions and data models
├── models/                   # Trained model artifacts (gitignored)
├── data/                     # Local data storage (gitignored)
└── backtest_charts/          # Generated equity curves and charts (gitignored)
```

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your experiment (optional)
# Edit config.py or create config.local.py from config.local.example.py

# Run the pipeline
python etl.py    # Download and prepare data
python train.py  # Train the model
python bt.py     # Run backtest on holdout period
```

## Configuration

### Default Config

Core settings live in `config.py`:
- Market universe (`SYMBOLS`, `ACTIVE_EXCHANGE`)
- Timeframes (`TIMEFRAME`, `HTF_TIMEFRAME`)
- Date ranges (`START_DATE`, `END_DATE`)
- Barrier parameters (`BARRIER_STOP_PCT`, `BARRIER_TAKE_PCT`)
- Trading costs (`TAKER_COM`, `MAKER_COM`, `SLIPPAGE`)
- Risk parameters (`RISK_PER_TRADE`, `LEVERAGE`)
- Feature build requests and model hyperparameters

### Presets

Named experiment presets live under `configs/presets/` and can be selected via environment variable:

```bash
# PowerShell
$env:LIGHT_LWTI_PRESET = "scaffold_long_only_1h"
python etl.py

# Bash/Linux
export LIGHT_LWTI_PRESET="scaffold_long_only_1h"
python etl.py
```

Preset files expose a `CONFIG_OVERRIDES` dict. Nested settings like `FEATURE_BUILD_REQUEST` are deep-merged with defaults, while scalar values like `SYMBOLS`, `TIMEFRAME`, and `ENTRY_THRESHOLD` replace them.

### Local Overrides

For machine-specific tweaks, copy `config.local.example.py` to `config.local.py` and edit only the values you want to change. This file is gitignored and applied after the selected preset.

You can also point to any custom config file:

```bash
# PowerShell
$env:LIGHT_LWTI_CONFIG = "configs/my_experiment.py"

# Bash/Linux
export LIGHT_LWTI_CONFIG="configs/my_experiment.py"

python etl.py
python train.py
python bt.py
```

## Pipeline

### `etl.py` — Data Ingestion & Feature Preparation

- Downloads raw OHLCV candles from the configured exchange (Bybit/Binance)
- Computes dynamic barrier levels (stop/take profit percentages)
- Builds first-hit outcome labels for long and short setups:
  - `target_long_label`, `target_long_outcome`, `target_long_pnl`
  - `target_long_exit_reason`, `target_long_mfe`, `target_long_mae`
  - `target_long_bars_to_tp`, `target_long_bars_to_sl`, `target_long_holding_bars`
  - (Same columns for short side)
- Applies modular feature builders based on `FEATURE_BUILD_REQUEST`
- Saves prepared datasets to:
  - SQLite `*_features` tables (default), or
  - Parquet files under `data/ml_features/` when `FEATURE_STORAGE="parquet"`

### `train.py` — Model Training

- Loads prepared feature datasets from SQLite or Parquet
- Constructs feature matrix from configured feature builders:
  - OHLCV baseline features
  - Trend, regime, and compression/expansion indicators
  - Entry location and interaction features
  - Market context, session context, volume flow
  - HTF (higher timeframe) context features
- Supports single train/test split or walk-forward cross-validation
- Optional event filtering (`TRAIN_EVENT_FILTER_CONFIG`) for strict TP-first vs SL-first training
- Saves model artifacts to `models/`:
  - Trained LightGBM model (`.pkl`)
  - Metrics JSON
  - Metadata (feature list, thresholds, config snapshot)
  - Feature importance chart

### `bt.py` — Single-Split Backtest

- Loads trained model and metadata
- Runs inference on holdout period features
- Simulates portfolio execution with:
  - Realistic taker/maker commissions and slippage
  - Position sizing based on `RISK_PER_TRADE` and stop distance
  - Signal filtering (event gates, cooldowns, max losses per day)
  - Barrier-based exits and max holding period
- Prints portfolio statistics (CAGR, Sharpe, max drawdown, win rate, etc.)
- Generates equity curve chart at `backtest_charts/equity_curve.png`

### `bt_walk_forward.py` / `bt_walk_forward_oos.py`

- Extend the backtest to walk-forward validation schemes
- Evaluate model performance across multiple out-of-sample periods
- Track rolling metrics and stability over time

## Execution Engine

The `execution_engine.py` module handles realistic portfolio simulation:

- **Position sizing**: Risk-based sizing using stop-loss distance
- **Cost modeling**: Taker/maker fees, slippage, and funding (if applicable)
- **Signal filtering**: Event gates, cooldown periods, daily loss limits
- **Trade lifecycle**: Entry → barrier monitoring → exit (TP/SL/time)
- **Portfolio aggregation**: Multi-symbol equity tracking and drawdown monitoring

## Signal Filtering

The `signal_filter.py` module provides entry signal conditioning:

- Event-based gating (only trade when specific conditions are met)
- Consecutive loss risk reduction
- Daily stop-loss limits
- Cooldown periods after exits

## Important Notes

1. **Run order**: Always run `etl.py` before `train.py`, and `train.py` before `bt.py`.
2. **Feature storage**: With `FEATURE_STORAGE="parquet"`, features are written to `data/ml_features/` instead of SQLite.
3. **Model artifacts**: After major config or feature pipeline changes, delete old models in `models/` and retrain.
4. **Data freshness**: Existing candles are incremental; re-running `etl.py` appends new data up to `END_DATE`.
5. **Git hygiene**: `config.local.py`, `data/`, `models/`, and `backtest_charts/` are gitignored.

## License

This project is for research and educational purposes. Use at your own risk.
