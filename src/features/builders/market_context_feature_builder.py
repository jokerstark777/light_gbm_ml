from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class MarketContextFeatureBuilder(FeatureBuilderContract):
    block_name = "market_context"

    FEATURES = {
        "rel_day_return_vs_btc",
        "rel_day_return_vs_market",
        "market_dispersion_return_4h",
        "market_breadth_pos_intraday",
        "market_breadth_avg_sign_intraday",
        "btc_daily_trend_direction",
        "btc_intraday_return",
    }

    def provides(self) -> set[str]:
        return set(self.FEATURES)

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        requested = self.FEATURES.intersection(requested_features)
        if not requested or not context.market_frame_map:
            return pd.DataFrame(index=context.main_frame.index)

        btc_frame = self._find_btc_frame(context.market_frame_map)
        if btc_frame is None or btc_frame.empty:
            return pd.DataFrame(index=context.main_frame.index)

        main_state = self._intraday_state(context.main_frame)
        btc_state = self._intraday_state(btc_frame)
        market_state = self._equal_weight_market_state(context.market_frame_map)

        out = pd.DataFrame({"timestamp": self._normalize_timestamp(context.main_frame["timestamp"])})

        if "rel_day_return_vs_btc" in requested:
            btc_aligned = self._align_series(out["timestamp"], btc_state[["timestamp", "intraday_return"]])
            out["rel_day_return_vs_btc"] = main_state["intraday_return"].values - btc_aligned["intraday_return"].values

        if "rel_day_return_vs_market" in requested:
            market_aligned = self._align_series(out["timestamp"], market_state)
            out["rel_day_return_vs_market"] = main_state["intraday_return"].values - market_aligned["market_intraday_return"].values

        if "market_dispersion_return_4h" in requested:
            dispersion_state = self._market_dispersion_return_state(context.market_frame_map)
            dispersion_aligned = self._align_series(out["timestamp"], dispersion_state)
            out["market_dispersion_return_4h"] = dispersion_aligned["market_dispersion_return_4h"].values

        if "market_breadth_pos_intraday" in requested or "market_breadth_avg_sign_intraday" in requested:
            breadth_state = self._market_breadth_state(context.market_frame_map)
            breadth_aligned = self._align_series(out["timestamp"], breadth_state)
            if "market_breadth_pos_intraday" in requested:
                out["market_breadth_pos_intraday"] = breadth_aligned["market_breadth_pos_intraday"].values
            if "market_breadth_avg_sign_intraday" in requested:
                out["market_breadth_avg_sign_intraday"] = breadth_aligned["market_breadth_avg_sign_intraday"].values

        if "btc_intraday_return" in requested:
            btc_aligned = self._align_series(out["timestamp"], btc_state[["timestamp", "intraday_return"]])
            out["btc_intraday_return"] = btc_aligned["intraday_return"].values

        if "btc_daily_trend_direction" in requested:
            btc_daily_regime = self._btc_daily_regime(btc_frame)
            btc_daily_regime["available_at"] = self._normalize_timestamp(btc_daily_regime["available_at"])
            regime_aligned = pd.merge_asof(
                out[["timestamp"]].sort_values("timestamp"),
                btc_daily_regime.sort_values("available_at"),
                left_on="timestamp",
                right_on="available_at",
                direction="backward",
            )
            regime_aligned = regime_aligned.reindex(out.sort_values("timestamp").index).sort_index()
            out["btc_daily_trend_direction"] = regime_aligned["btc_daily_trend_direction"].values

        return out[list(sorted(requested))]

    def _find_btc_frame(self, market_frame_map: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        for symbol, frame in market_frame_map.items():
            if str(symbol).upper().replace("-", "/").startswith("BTC/"):
                return frame
        return None

    def _intraday_state(self, frame: pd.DataFrame) -> pd.DataFrame:
        state = frame[["timestamp", "open", "close"]].copy()
        state["timestamp"] = self._normalize_timestamp(state["timestamp"])
        state["open"] = pd.to_numeric(state["open"], errors="coerce")
        state["close"] = pd.to_numeric(state["close"], errors="coerce")
        state = state.dropna().sort_values("timestamp").reset_index(drop=True)
        day_key = state["timestamp"].dt.floor("D")
        day_open = state.groupby(day_key)["open"].transform("first")
        state["intraday_return"] = state["close"] / day_open.replace(0, np.nan) - 1.0
        return state[["timestamp", "intraday_return"]]

    def _equal_weight_market_state(self, market_frame_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for frame in market_frame_map.values():
            state = self._intraday_state(frame)
            if not state.empty:
                frames.append(state)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "market_intraday_return"])
        stacked = pd.concat(frames, ignore_index=True)
        market = (
            stacked.groupby("timestamp", as_index=False)["intraday_return"]
            .mean()
            .rename(columns={"intraday_return": "market_intraday_return"})
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return market

    def _market_breadth_state(self, market_frame_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for frame in market_frame_map.values():
            state = self._intraday_state(frame)
            if not state.empty:
                frames.append(state)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "market_breadth_pos_intraday",
                    "market_breadth_avg_sign_intraday",
                ]
            )
        stacked = pd.concat(frames, ignore_index=True)
        stacked["is_positive_intraday"] = (stacked["intraday_return"] > 0).astype(float)
        stacked["intraday_sign"] = np.sign(stacked["intraday_return"]).astype(float)
        breadth = (
            stacked.groupby("timestamp", as_index=False)
            .agg(
                market_breadth_pos_intraday=("is_positive_intraday", "mean"),
                market_breadth_avg_sign_intraday=("intraday_sign", "mean"),
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return breadth

    def _market_dispersion_return_state(self, market_frame_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for frame in market_frame_map.values():
            state = frame[["timestamp", "close"]].copy()
            state["timestamp"] = self._normalize_timestamp(state["timestamp"])
            state["close"] = pd.to_numeric(state["close"], errors="coerce")
            state = state.dropna().sort_values("timestamp").reset_index(drop=True)
            state["return_4h"] = state["close"] / state["close"].shift(4) - 1.0
            frames.append(state[["timestamp", "return_4h"]])
        if not frames:
            return pd.DataFrame(columns=["timestamp", "market_dispersion_return_4h"])
        stacked = pd.concat(frames, ignore_index=True)
        dispersion = (
            stacked.groupby("timestamp", as_index=False)["return_4h"]
            .std()
            .rename(columns={"return_4h": "market_dispersion_return_4h"})
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return dispersion

    def _btc_daily_regime(self, btc_frame: pd.DataFrame) -> pd.DataFrame:
        btc = btc_frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        btc["timestamp"] = self._normalize_timestamp(btc["timestamp"])
        for column in ["open", "high", "low", "close", "volume"]:
            btc[column] = pd.to_numeric(btc[column], errors="coerce")
        btc = btc.dropna().sort_values("timestamp").set_index("timestamp")
        daily = btc.resample("D").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        if daily.empty:
            return pd.DataFrame(columns=["available_at", "btc_daily_trend_direction"])

        close = daily["close"]
        ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        daily["btc_daily_trend_direction"] = np.select(
            [((close > ema_50) & (ema_20 > ema_50)), ((close < ema_50) & (ema_20 < ema_50))],
            [1.0, -1.0],
            default=0.0,
        )
        regime = daily[["btc_daily_trend_direction"]].reset_index()
        regime["available_at"] = regime["timestamp"] + pd.Timedelta(days=1)
        return regime[["available_at", "btc_daily_trend_direction"]]

    def _align_series(self, timestamps: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
        left = pd.DataFrame({"timestamp": self._normalize_timestamp(timestamps)})
        frame = frame.copy()
        frame["timestamp"] = self._normalize_timestamp(frame["timestamp"])
        aligned = pd.merge_asof(
            left.sort_values("timestamp"),
            frame.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        return aligned.reindex(left.sort_values("timestamp").index).sort_index()

    def _normalize_timestamp(self, values: pd.Series) -> pd.Series:
        return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")
