# Light

`Light` is a local ML project for researching and backtesting crypto strategies on exchange candle data.

The project now works in a raw-only mode:

- loads and stores historical candles in `SQLite`
- stores prepared ML feature datasets in `Parquet` when `FEATURE_STORAGE="parquet"`
- prepares per-symbol training datasets from raw `OHLCV`
- adds barrier columns and long-only labels
- trains a `LightGBM` model
- runs a portfolio backtest on the saved holdout window

## Stack

- Python
- pandas / numpy
- LightGBM
- scikit-learn
- SQLite
- Parquet / pyarrow
- matplotlib

## Project Structure

```text
light/
|-- config.py
|-- etl.py
|-- train.py
|-- bt.py
|-- requirements.txt
|-- models/
`-- data/
```

## Quick Start

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python etl.py
python train.py
python bt.py
```

## Config Presets

Stable defaults live in `config.py`. Named experiment presets live under `configs/presets/` and can be selected with `LIGHT_LWTI_PRESET`.

```powershell
$env:LIGHT_LWTI_PRESET = "research_btc_eth_1h"
python etl.py
python train.py
python bt.py
```

Preset files expose a `CONFIG_OVERRIDES` dict. Nested dict settings like `FEATURE_BUILD_REQUEST` are merged with defaults, while simple values like `SYMBOLS`, `TIMEFRAME`, and `ENTRY_THRESHOLD` replace the defaults.

For machine-local tweaks, copy `config.local.example.py` to `config.local.py` and edit only the values you want to change. `config.local.py` is ignored by git and is applied after the selected preset.

You can also point directly to another override file:

```powershell
$env:LIGHT_LWTI_CONFIG = "configs/my_experiment.py"
python etl.py
python train.py
python bt.py
```

## Pipeline

### `etl.py`

- loads raw candles from the configured exchange
- computes barrier columns
- builds explicit first-hit outcome columns for long and short setups:
  `target_long_label`, `target_long_outcome`, `target_long_pnl`, `target_long_exit_reason`,
  `target_long_mfe`, `target_long_mae`, `target_long_bars_to_tp`, `target_long_bars_to_sl`,
  `target_long_holding_bars`, `target_short_label`, `target_short_outcome`, `target_short_pnl`,
  `target_short_exit_reason`, `target_short_mfe`, `target_short_mae`, `target_short_bars_to_tp`,
  `target_short_bars_to_sl`, `target_short_holding_bars`
- saves prepared rows into `*_features` tables in SQLite

### `train.py`

- loads prepared tables from SQLite
- selects model inputs directly from saved dataset columns
- can keep the baseline target (`TP_FIRST` vs all non-TP) or enable
  `TRAIN_EVENT_FILTER_CONFIG` for strict `TP_FIRST` vs `SL_FIRST` training without `NEITHER`
- runs single split or walk-forward validation
- saves model, metrics, metadata, and feature importance

### `bt.py`

- loads the trained model and saved metadata
- uses prepared rows from SQLite for inference
- replays the saved holdout period
- prints portfolio statistics and writes `backtest_charts/equity_curve.png`

## Important Notes

- Run `etl.py` before `train.py` so the prepared `*_features` tables exist.
- With `FEATURE_STORAGE="parquet"`, `etl.py` writes prepared features under `data/ml_features/`.
- Run `train.py` before `bt.py` so model artifacts exist in `models/`.
- Existing model artifacts trained on the old feature pipeline should be considered stale after this cleanup.
