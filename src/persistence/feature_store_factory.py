from __future__ import annotations

import config as cfg
from src.persistence.parquet_feature_store import ParquetFeatureStore
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository


def create_feature_store(db_path: str | None = None, exchange_code: str | None = None):
    storage = str(getattr(cfg, "FEATURE_STORAGE", "sqlite")).strip().lower()
    if storage == "parquet":
        return ParquetFeatureStore(exchange_code=exchange_code)
    if storage == "sqlite":
        kwargs = {}
        if db_path is not None:
            kwargs["db_path"] = db_path
        if exchange_code is not None:
            kwargs["exchange_code"] = exchange_code
        return HistoricalKlineRepository(**kwargs)
    raise ValueError(f"Unsupported FEATURE_STORAGE: {storage}")
