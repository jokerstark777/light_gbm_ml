from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.binance.binance_adapter import BinanceAdapter
from src.exchanges.binance.binance_mapper import BinanceMapper
from src.types.common import HistoricalKline, Symbol


logger = logging.getLogger(__name__)


class BinanceService(ExchangeContract):
    def __init__(
        self,
        adapter: BinanceAdapter | None = None,
        mapper: BinanceMapper | None = None,
    ) -> None:
        self.adapter = adapter or BinanceAdapter()
        self.mapper = mapper or BinanceMapper()
        self._exchange_info_by_symbol: dict[str, dict] | None = None

    def get_exchange_code(self) -> str:
        return "binance"

    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        return self.mapper.normalize_symbol(symbol)

    def get_timeframe_ms(self, timeframe: str) -> int:
        return self.mapper.timeframe_to_ms(timeframe)

    def build_request_windows(
        self,
        start_ts: int,
        end_ts: int,
        timeframe_ms: int,
    ) -> list[tuple[int, int]]:
        if start_ts > end_ts:
            return []

        max_span_ms = timeframe_ms * max(self.adapter.limit - 1, 1)
        windows = []
        window_start = start_ts

        while window_start <= end_ts:
            window_end = min(window_start + max_span_ms, end_ts)
            windows.append((window_start, window_end))
            window_start = window_end + timeframe_ms

        return windows

    def get_symbol_metadata(self, symbol: str | Symbol) -> dict | None:
        if self._exchange_info_by_symbol is None:
            payload = self.adapter.fetch_exchange_info()
            symbols = payload.get("symbols", [])
            self._exchange_info_by_symbol = {
                symbol_row["symbol"]: symbol_row
                for symbol_row in symbols
                if symbol_row.get("status") == "TRADING"
            }

        api_symbol = self.mapper.to_api_symbol(symbol)
        return self._exchange_info_by_symbol.get(api_symbol)

    def clamp_start_ts_to_listing(
        self,
        symbol: str | Symbol,
        requested_start_ts: int,
    ) -> int:
        metadata = self.get_symbol_metadata(symbol)
        if metadata is None:
            return requested_start_ts

        onboard_ts = metadata.get("onboardDate")
        if onboard_ts is None:
            return requested_start_ts

        return max(requested_start_ts, int(onboard_ts))

    def fetch_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        normalized_symbol = self.normalize_symbol(symbol)
        start_ts = self.clamp_start_ts_to_listing(normalized_symbol, start_ts)
        timeframe_ms = self.get_timeframe_ms(timeframe)
        windows = self.build_request_windows(start_ts, end_ts, timeframe_ms)
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        interval = self.mapper.to_interval(timeframe)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_klines: list[HistoricalKline] = []

        logger.info(
            f"[{normalized_symbol}-{timeframe}] Binance backfill: {total_windows} windows, "
            f"limit={self.adapter.limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(
                    self.adapter.fetch_kline_window,
                    api_symbol,
                    interval,
                    window_start,
                    window_end,
                ): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                klines = self.mapper.to_klines(payload)
                if klines:
                    all_klines.extend(klines)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = klines[-1].open_time if klines else window_end
                    logger.info(
                        f"[{normalized_symbol}-{timeframe}] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        all_klines.sort(key=lambda candle: candle.open_time)
        return all_klines
