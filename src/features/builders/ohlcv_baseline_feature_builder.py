from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class OhlcvBaselineFeatureBuilder(FeatureBuilderContract):
    """Small leakage-safe state features for triple-barrier long entries."""

    block_name = "ohlcv_baseline"

    FEATURES = {
        "atr_pct_14",
        "volatility_regime_change_1h",
        "range_expansion_20",
        "return_4h_atr_norm",
        "zscore_vs_vwap_4h",
        "trend_efficiency_16",
        "close_pos_range_32",
        "pullback_ema_20_atr",
        "dist_to_prev_day_high_atr",
        "dist_to_prev_day_low_atr",
        "body_to_range",
        "lower_wick_to_range",
        "upper_wick_to_range",
        "volume_rel_median_20",
        "signed_volume_pressure_8",
    }

    def provides(self) -> set[str]:
        return set(self.FEATURES)

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        requested = self.FEATURES.intersection(requested_features)
        if not requested:
            return pd.DataFrame(index=context.main_frame.index)

        df = context.main_frame
        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        safe_range = (high - low).replace(0, np.nan)
        atr_14 = true_range.rolling(14, min_periods=14).mean()
        ret_1 = close.pct_change()

        out = pd.DataFrame(index=df.index)

        if "atr_pct_14" in requested:
            out["atr_pct_14"] = atr_14 / close.replace(0, np.nan)

        atr_pct_14 = atr_14 / close.replace(0, np.nan)

        if "volatility_regime_change_1h" in requested:
            baseline_atr_pct = atr_pct_14.rolling(96, min_periods=48).median()
            out["volatility_regime_change_1h"] = atr_pct_14 / baseline_atr_pct.replace(0, np.nan) - 1.0

        if "range_expansion_20" in requested:
            tr_pct = true_range / close.replace(0, np.nan)
            out["range_expansion_20"] = tr_pct / tr_pct.rolling(20, min_periods=20).median().replace(0, np.nan)

        if "return_4h_atr_norm" in requested:
            out["return_4h_atr_norm"] = (close / close.shift(4) - 1.0) / atr_pct_14.replace(0, np.nan)

        if "zscore_vs_vwap_4h" in requested:
            typical_price = (high + low + close) / 3.0
            rolling_vwap_4h = (
                (typical_price * volume).rolling(4, min_periods=4).sum()
                / volume.rolling(4, min_periods=4).sum().replace(0, np.nan)
            )
            out["zscore_vs_vwap_4h"] = (close - rolling_vwap_4h) / atr_14.replace(0, np.nan)

        if "trend_efficiency_16" in requested:
            net_move = (close / close.shift(16) - 1).abs()
            path = ret_1.abs().rolling(16, min_periods=16).sum()
            out["trend_efficiency_16"] = net_move / path.replace(0, np.nan)

        if "close_pos_range_32" in requested:
            rolling_low = low.rolling(32, min_periods=32).min()
            rolling_high = high.rolling(32, min_periods=32).max()
            out["close_pos_range_32"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)

        if "pullback_ema_20_atr" in requested:
            ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
            out["pullback_ema_20_atr"] = (close - ema_20) / atr_14.replace(0, np.nan)

        if "dist_to_prev_day_high_atr" in requested or "dist_to_prev_day_low_atr" in requested:
            daily_levels = self._previous_day_levels(df)
            if "dist_to_prev_day_high_atr" in requested:
                out["dist_to_prev_day_high_atr"] = (close - daily_levels["prev_day_high"]) / atr_14.replace(0, np.nan)
            if "dist_to_prev_day_low_atr" in requested:
                out["dist_to_prev_day_low_atr"] = (close - daily_levels["prev_day_low"]) / atr_14.replace(0, np.nan)

        if "body_to_range" in requested:
            out["body_to_range"] = (close - open_).abs() / safe_range

        if "lower_wick_to_range" in requested:
            out["lower_wick_to_range"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / safe_range

        if "upper_wick_to_range" in requested:
            out["upper_wick_to_range"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / safe_range

        if "volume_rel_median_20" in requested:
            out["volume_rel_median_20"] = volume / volume.rolling(20, min_periods=20).median().replace(0, np.nan)

        if "signed_volume_pressure_8" in requested:
            signed_volume = np.sign(ret_1.fillna(0.0)) * volume
            out["signed_volume_pressure_8"] = (
                signed_volume.rolling(8, min_periods=8).sum()
                / volume.rolling(8, min_periods=8).sum().replace(0, np.nan)
            )

        return out

    def _previous_day_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df[["timestamp", "high", "low"]].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
        frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
        daily = (
            frame.dropna()
            .set_index("timestamp")
            .resample("D")
            .agg({"high": "max", "low": "min"})
            .rename(columns={"high": "prev_day_high", "low": "prev_day_low"})
        )
        daily[["prev_day_high", "prev_day_low"]] = daily[["prev_day_high", "prev_day_low"]].shift(1)
        daily = daily.reset_index().rename(columns={"timestamp": "day"})

        output = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"], errors="coerce")}, index=df.index)
        output["day"] = output["timestamp"].dt.floor("D")
        output = output.merge(daily, on="day", how="left").set_index(df.index)
        return output[["prev_day_high", "prev_day_low"]]
