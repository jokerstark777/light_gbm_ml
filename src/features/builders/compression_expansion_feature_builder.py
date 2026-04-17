from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class CompressionExpansionFeatureBuilder(FeatureBuilderContract):
    block_name = "compression_expansion"

    FEATURES = {
        "bb_width_pct_rank_120",
        "range_compression_ratio_20_100",
    }

    def __init__(self, timeframe_label: str = "1h", reference_timeframe: str | None = None) -> None:
        self.timeframe_label = timeframe_label
        self.reference_timeframe = reference_timeframe or timeframe_label

    def provides(self) -> set[str]:
        return set(self.FEATURES)

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        requested = self.FEATURES.intersection(requested_features)
        if not requested:
            return pd.DataFrame(index=context.main_frame.index)

        df = context.main_frame
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        true_range_pct = true_range / close.replace(0, np.nan)

        out = pd.DataFrame(index=df.index)

        if "bb_width_pct_rank_120" in requested:
            rolling_mean = close.rolling(20, min_periods=20).mean()
            rolling_std = close.rolling(20, min_periods=20).std()
            bb_width = (4.0 * rolling_std) / rolling_mean.replace(0, np.nan)
            out["bb_width_pct_rank_120"] = bb_width.rolling(120, min_periods=60).rank(pct=True)

        if "range_compression_ratio_20_100" in requested:
            short_range = true_range_pct.rolling(20, min_periods=20).median()
            long_range = true_range_pct.rolling(100, min_periods=60).median()
            out["range_compression_ratio_20_100"] = short_range / long_range.replace(0, np.nan)

        return out
