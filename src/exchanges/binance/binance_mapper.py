from __future__ import annotations

from src.types.common import HistoricalKline, Symbol


BINANCE_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
    "1M": "1M",
}

TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


class BinanceMapper:
    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        if isinstance(symbol, Symbol):
            return symbol
        return Symbol.from_string(symbol)

    def to_api_symbol(self, symbol: str | Symbol) -> str:
        normalized_symbol = self.normalize_symbol(symbol)
        return f"{normalized_symbol.base_asset}{normalized_symbol.quote_asset}"

    def to_interval(self, timeframe: str) -> str:
        interval = BINANCE_INTERVALS.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Binance: {timeframe}")
        return interval

    def timeframe_to_ms(self, timeframe: str) -> int:
        timeframe_ms = TF_MS.get(timeframe)
        if timeframe_ms is None:
            raise ValueError(f"timeframe {timeframe} is missing in Binance TF_MS")
        return timeframe_ms

    def to_klines(self, payload: list) -> list[HistoricalKline]:
        klines = [
            HistoricalKline(
                open_time=int(candle[0]),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
                quote_volume=float(candle[7]),
            )
            for candle in payload
        ]
        klines.sort(key=lambda candle: candle.open_time)
        return klines
