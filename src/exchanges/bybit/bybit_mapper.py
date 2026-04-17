from __future__ import annotations

from src.types.common import FundingRatePoint, HistoricalKline, OpenInterestPoint, PriceKline, Symbol


BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

BYBIT_OPEN_INTEREST_INTERVALS = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class BybitMapper:
    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        if isinstance(symbol, Symbol):
            return symbol
        return Symbol.from_string(symbol)

    def to_api_symbol(self, symbol: str | Symbol) -> str:
        normalized_symbol = self.normalize_symbol(symbol)
        return f"{normalized_symbol.base_asset}{normalized_symbol.quote_asset}"

    def to_interval(self, timeframe: str) -> str:
        interval = BYBIT_INTERVALS.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Bybit: {timeframe}")
        return interval

    def timeframe_to_ms(self, timeframe: str) -> int:
        timeframe_ms = TF_MS.get(timeframe)
        if timeframe_ms is None:
            raise ValueError(f"timeframe {timeframe} is missing in TF_MS")
        return timeframe_ms

    def to_open_interest_interval(self, timeframe: str) -> str:
        interval = BYBIT_OPEN_INTEREST_INTERVALS.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported open interest timeframe for Bybit: {timeframe}")
        return interval

    def to_klines(self, payload: dict) -> list[HistoricalKline]:
        candles = payload.get("result", {}).get("list", [])
        klines = [
            HistoricalKline(
                open_time=int(candle[0]),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
                quote_volume=float(candle[6]),
            )
            for candle in candles
        ]
        klines.sort(key=lambda candle: candle.open_time)
        return klines

    def to_price_klines(self, payload: dict) -> list[PriceKline]:
        candles = payload.get("result", {}).get("list", [])
        klines = [
            PriceKline(
                open_time=int(candle[0]),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
            )
            for candle in candles
        ]
        klines.sort(key=lambda candle: candle.open_time)
        return klines

    def to_funding_rates(self, payload: dict) -> list[FundingRatePoint]:
        rows = payload.get("result", {}).get("list", [])
        points = [
            FundingRatePoint(
                funding_time=int(row["fundingRateTimestamp"]),
                funding_rate=float(row["fundingRate"]),
            )
            for row in rows
        ]
        points.sort(key=lambda point: point.funding_time)
        return points

    def to_open_interest_points(self, payload: dict) -> list[OpenInterestPoint]:
        rows = payload.get("result", {}).get("list", [])
        points = [
            OpenInterestPoint(
                timestamp=int(row["timestamp"]),
                open_interest=float(row["openInterest"]),
            )
            for row in rows
        ]
        points.sort(key=lambda point: point.timestamp)
        return points
