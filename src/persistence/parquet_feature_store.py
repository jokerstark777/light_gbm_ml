from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config as cfg
from src.types.common import Symbol

logger = logging.getLogger(__name__)


class ParquetFeatureStore:
    def __init__(
        self,
        base_dir: str | Path | None = None,
        exchange_code: str | None = None,
        timeframe: str | None = None,
        dataset_version: str | None = None,
    ) -> None:
        self.base_dir = Path(base_dir or getattr(cfg, "PARQUET_FEATURES_DIR", Path("data") / "ml_features"))
        self.exchange_code = self._safe_path_part(exchange_code or getattr(cfg, "ACTIVE_EXCHANGE", "bybit"))
        self.timeframe = self._safe_path_part(timeframe or getattr(cfg, "TIMEFRAME", "1h"))
        self.dataset_version = self._safe_path_part(
            dataset_version or getattr(cfg, "PARQUET_DATASET_VERSION", "default")
        )

    def save_features(self, symbol: str | Symbol, df: pd.DataFrame) -> None:
        path = self.feature_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = df.copy()
        if "timestamp" in output.columns:
            output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
        output.to_parquet(path, index=False)
        logger.info("%s prepared dataset saved to %s (%s rows)", self._symbol_name(symbol), path, len(output))

    def load_features(self, symbol: str | Symbol, columns: list[str] | None = None) -> pd.DataFrame:
        path = self.feature_path(symbol)
        symbol_name = self._symbol_name(symbol)
        if not path.exists():
            logger.warning("Skipping %s: Parquet feature file not found at %s", symbol_name, path)
            return pd.DataFrame()

        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception as exc:
            logger.warning("Skipping %s: failed reading Parquet features from %s: %s", symbol_name, path, exc)
            return pd.DataFrame()

        if frame.empty:
            logger.warning("Skipping %s: Parquet feature file is empty at %s", symbol_name, path)
            return frame

        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["symbol"] = symbol_name
        return frame

    def load_feature_dataset(self, symbols: list[str | Symbol]) -> pd.DataFrame:
        frames = [self.load_features(symbol) for symbol in symbols]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            raise RuntimeError(
                "No prepared Parquet feature files found. Run etl.py first or set FEATURE_STORAGE='sqlite'."
            )
        return pd.concat(frames, ignore_index=True)

    def feature_path(self, symbol: str | Symbol) -> Path:
        return (
            self.base_dir
            / f"exchange={self.exchange_code}"
            / f"timeframe={self.timeframe}"
            / f"version={self.dataset_version}"
            / f"{self._symbol_stub(symbol)}.parquet"
        )

    @staticmethod
    def _symbol_name(symbol: str | Symbol) -> str:
        return str(symbol) if isinstance(symbol, Symbol) else str(Symbol.from_string(symbol))

    @classmethod
    def _symbol_stub(cls, symbol: str | Symbol) -> str:
        return cls._symbol_name(symbol).replace("/", "_")

    @staticmethod
    def _safe_path_part(value: str) -> str:
        return str(value).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
