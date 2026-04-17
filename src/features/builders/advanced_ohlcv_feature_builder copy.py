from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class AdvancedOhlcvFeatureBuilder(FeatureBuilderContract):
    """Orthogonal OHLCV-derived features that complement the baseline set.

    Focus areas:
    - Orderflow proxy (CVD via Bishop's approximation)
    - Microstructure (wick imbalance, body dominance patterns)
    - Regime detection (return autocorrelation, rolling skewness)
    - Multi-scale price position
    - Liquidity proxy (Amihud illiquidity)
    - Momentum exhaustion (consecutive bars, price acceleration)
    """

    block_name = "advanced_ohlcv"

    FEATURES = {
        # Orderflow proxy
        "cvd_ratio_12",
        "cvd_slope_divergence_12",
        # Microstructure
        "wick_imbalance_ma_8",
        "body_dominance_ma_8",
        # Regime detection
        "return_autocorr_24",
        "return_skew_24",
        # Multi-scale price position
        "close_pos_range_8",
        "close_pos_range_96",
        # Liquidity proxy
        "amihud_illiquidity_14",
        # Momentum structure
        "consecutive_direction_count",
        "price_acceleration_8",
        # Volume anomaly
        "volume_zscore_20",
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

        candle_range = (high - low).replace(0, np.nan)
        ret_1 = close.pct_change()

        out = pd.DataFrame(index=df.index)

        # ── Orderflow proxy (Bishop's approximation of volume delta) ──
        # Splits each bar's volume into buy/sell using close position within the bar
        if "cvd_ratio_12" in requested or "cvd_slope_divergence_12" in requested:
            buy_vol = volume * (close - low) / candle_range
            sell_vol = volume * (high - close) / candle_range
            net_delta = buy_vol - sell_vol
            vol_sum_12 = volume.rolling(12, min_periods=12).sum().replace(0, np.nan)

            if "cvd_ratio_12" in requested:
                # Ratio of net delta to total volume over 12 bars: [-1, 1]
                out["cvd_ratio_12"] = net_delta.rolling(12, min_periods=12).sum() / vol_sum_12

            if "cvd_slope_divergence_12" in requested:
                # Divergence between price slope and CVD slope
                # Positive = price rising but selling pressure (bearish divergence)
                cumulative_delta = net_delta.cumsum()
                price_slope = self._rolling_slope(close, 12)
                cvd_slope = self._rolling_slope(cumulative_delta, 12)

                safe_cvd = cvd_slope.replace(0, np.nan)
                out["cvd_slope_divergence_12"] = price_slope / safe_cvd

        # ── Microstructure ──
        if "wick_imbalance_ma_8" in requested:
            # Bullish bars have larger lower wicks (buyers absorbing at lows)
            upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
            lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
            wick_imbalance = (lower_wick - upper_wick) / candle_range  # > 0 = bullish
            out["wick_imbalance_ma_8"] = wick_imbalance.rolling(8, min_periods=8).mean()

        if "body_dominance_ma_8" in requested:
            # Signed body ratio: (+) = bullish bars dominating, (-) = bearish
            signed_body = (close - open_) / candle_range
            out["body_dominance_ma_8"] = signed_body.rolling(8, min_periods=8).mean()

        # ── Regime detection ──
        if "return_autocorr_24" in requested:
            # Lag-1 autocorrelation of returns over 24 bars
            # > 0 = momentum regime, < 0 = mean-reversion regime
            out["return_autocorr_24"] = ret_1.rolling(24, min_periods=18).apply(
                self._autocorr_lag1, raw=True,
            )

        if "return_skew_24" in requested:
            # Rolling skewness of returns — negative skew before crashes
            out["return_skew_24"] = ret_1.rolling(24, min_periods=18).skew()

        # ── Multi-scale price position ──
        if "close_pos_range_8" in requested:
            # Short-term: where is price in the last 8 bars range
            rolling_low_8 = low.rolling(8, min_periods=8).min()
            rolling_high_8 = high.rolling(8, min_periods=8).max()
            out["close_pos_range_8"] = (close - rolling_low_8) / (rolling_high_8 - rolling_low_8).replace(0, np.nan)

        if "close_pos_range_96" in requested:
            # Long-term: where is price in the last 96 bars (4 days) range
            rolling_low_96 = low.rolling(96, min_periods=48).min()
            rolling_high_96 = high.rolling(96, min_periods=48).max()
            out["close_pos_range_96"] = (close - rolling_low_96) / (rolling_high_96 - rolling_low_96).replace(0, np.nan)

        # ── Liquidity proxy ──
        if "amihud_illiquidity_14" in requested:
            # Amihud illiquidity: |return| / volume — high = illiquid, thin market
            abs_ret = ret_1.abs()
            safe_vol = volume.replace(0, np.nan)
            raw_illiq = abs_ret / safe_vol
            out["amihud_illiquidity_14"] = raw_illiq.rolling(14, min_periods=14).mean()
            # Normalize to rank within rolling window for cross-asset comparability
            out["amihud_illiquidity_14"] = out["amihud_illiquidity_14"].rolling(
                96, min_periods=48,
            ).rank(pct=True)

        # ── Momentum structure ──
        if "consecutive_direction_count" in requested:
            # Count of consecutive green (+) or red (-) bars
            # Extreme values = momentum exhaustion
            direction = np.sign(ret_1.fillna(0.0)).values
            counts = np.zeros(len(direction), dtype=float)
            for i in range(1, len(direction)):
                if direction[i] == 0:
                    counts[i] = 0
                elif direction[i] == direction[i - 1]:
                    counts[i] = counts[i - 1] + direction[i]
                else:
                    counts[i] = direction[i]
            out["consecutive_direction_count"] = counts

        if "price_acceleration_8" in requested:
            # Second derivative of price: momentum of momentum
            # Negative after a strong move up = exhaustion
            momentum_4 = close.pct_change(4)
            out["price_acceleration_8"] = momentum_4.diff(4)

        # ── Volume anomaly ──
        if "volume_zscore_20" in requested:
            # Z-score of volume vs rolling stats — spikes = significant events
            vol_mean = volume.rolling(20, min_periods=20).mean()
            vol_std = volume.rolling(20, min_periods=20).std().replace(0, np.nan)
            out["volume_zscore_20"] = (volume - vol_mean) / vol_std

        return out

    @staticmethod
    def _autocorr_lag1(values: np.ndarray) -> float:
        """Pearson autocorrelation at lag 1."""
        n = len(values)
        if n < 4:
            return np.nan
        x = values[:-1]
        y = values[1:]
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        if len(x) < 4:
            return np.nan
        mx = np.mean(x)
        my = np.mean(y)
        sx = np.std(x, ddof=1)
        sy = np.std(y, ddof=1)
        if sx < 1e-15 or sy < 1e-15:
            return 0.0
        return float(np.mean((x - mx) * (y - my)) / (sx * sy))

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """OLS slope (normalized) over a rolling window."""
        x = np.arange(window, dtype=float)
        x -= x.mean()
        x_ss = (x * x).sum()
        if x_ss < 1e-15:
            return pd.Series(np.nan, index=series.index)

        def _slope(values: np.ndarray) -> float:
            if len(values) < window:
                return np.nan
            valid = ~np.isnan(values)
            if valid.sum() < window // 2:
                return np.nan
            y = np.where(valid, values, 0.0)
            return float((x * y).sum() / x_ss)

        return series.rolling(window, min_periods=window).apply(_slope, raw=True)
