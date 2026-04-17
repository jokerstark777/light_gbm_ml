from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.indicators import timeframe_to_timedelta
from src.features.models.feature_context import FeatureContext


class HtfContextFeatureBuilder(FeatureBuilderContract):
    block_name = "htf_context"

    FEATURES = {
        "htf_trend_direction",
        "htf_close_pos_range_60",
        "htf_atr_pct_rank_90",
    }

    def provides(self) -> set[str]:
        return set(self.FEATURES)

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        requested = self.FEATURES.intersection(requested_features)
        if not requested or context.htf_frame is None or context.htf_frame.empty:
            return pd.DataFrame(index=context.main_frame.index)

        htf = context.htf_frame.copy().sort_values("timestamp").reset_index(drop=True)
        htf["timestamp"] = pd.to_datetime(htf["timestamp"], errors="coerce")
        htf = htf.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)
        if htf.empty:
            return pd.DataFrame(index=context.main_frame.index)

        close = htf["close"].astype(float)
        high = htf["high"].astype(float)
        low = htf["low"].astype(float)

        features = pd.DataFrame({"timestamp": htf["timestamp"]})
        htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "1d"))
        features["available_at"] = features["timestamp"] + timeframe_to_timedelta(htf_timeframe)

        if "htf_trend_direction" in requested:
            ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
            ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
            trend = np.select(
                [((close > ema_50) & (ema_20 > ema_50)), ((close < ema_50) & (ema_20 < ema_50))],
                [1.0, -1.0],
                default=0.0,
            )
            features["htf_trend_direction"] = trend

        if "htf_close_pos_range_60" in requested:
            rolling_low = low.rolling(60, min_periods=60).min()
            rolling_high = high.rolling(60, min_periods=60).max()
            features["htf_close_pos_range_60"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)

        if "htf_atr_pct_rank_90" in requested:
            prev_close = close.shift(1)
            true_range = pd.concat(
                [
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_pct = true_range.rolling(14, min_periods=14).mean() / close.replace(0, np.nan)
            features["htf_atr_pct_rank_90"] = atr_pct.rolling(90, min_periods=60).rank(pct=True)

        main = context.main_frame[["timestamp"]].copy()
        main["timestamp"] = pd.to_datetime(main["timestamp"], errors="coerce")

        aligned = pd.merge_asof(
            main.sort_values("timestamp"),
            features[["available_at", *sorted(requested)]].sort_values("available_at"),
            left_on="timestamp",
            right_on="available_at",
            direction="backward",
        )
        aligned = aligned.reindex(main.sort_values("timestamp").index).sort_index()
        return aligned[list(sorted(requested))]
