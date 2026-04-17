from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.bybit.bybit_adapter import BybitAdapter
from src.exchanges.bybit.bybit_mapper import BybitMapper
from src.types.common import FundingRatePoint, HistoricalKline, OpenInterestPoint, PriceKline, Symbol


logger = logging.getLogger(__name__)


class BybitService(ExchangeContract):
    def __init__(
        self,
        adapter: BybitAdapter | None = None,
        mapper: BybitMapper | None = None,
    ) -> None:
        self.adapter = adapter or BybitAdapter()
        self.mapper = mapper or BybitMapper()
        self._instrument_info_by_symbol: dict[str, dict | None] = {}

    def get_exchange_code(self) -> str:
        return "bybit"

    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        return self.mapper.normalize_symbol(symbol)

    def get_timeframe_ms(self, timeframe: str) -> int:
        return self.mapper.timeframe_to_ms(timeframe)

    def build_request_windows(
        self,
        start_ts: int,
        end_ts: int,
        timeframe_ms: int,
        limit: int | None = None,
    ) -> list[tuple[int, int]]:
        if start_ts > end_ts:
            return []

        request_limit = self.adapter.limit if limit is None else max(1, int(limit))
        max_span_ms = timeframe_ms * max(request_limit - 1, 1)
        windows = []
        window_start = start_ts

        while window_start <= end_ts:
            window_end = min(window_start + max_span_ms, end_ts)
            windows.append((window_start, window_end))
            window_start = window_end + timeframe_ms

        return windows

    def get_symbol_metadata(self, symbol: str | Symbol) -> dict | None:
        normalized_symbol = self.normalize_symbol(symbol)
        symbol_name = str(normalized_symbol)
        if symbol_name not in self._instrument_info_by_symbol:
            api_symbol = self.mapper.to_api_symbol(normalized_symbol)
            payload = self.adapter.fetch_instruments_info(api_symbol)
            rows = payload.get("result", {}).get("list", [])
            self._instrument_info_by_symbol[symbol_name] = rows[0] if rows else None
        return self._instrument_info_by_symbol[symbol_name]

    def clamp_start_ts_to_listing(
        self,
        symbol: str | Symbol,
        requested_start_ts: int,
    ) -> int:
        metadata = self.get_symbol_metadata(symbol)
        if metadata is None:
            return requested_start_ts

        launch_time = metadata.get("launchTime")
        if launch_time is None:
            return requested_start_ts
        return max(requested_start_ts, int(launch_time))

    def get_funding_interval_ms(self, symbol: str | Symbol) -> int:
        metadata = self.get_symbol_metadata(symbol)
        if metadata is None:
            return 8 * 60 * 60 * 1000

        funding_interval_minutes = metadata.get("fundingInterval")
        if funding_interval_minutes is None:
            return 8 * 60 * 60 * 1000
        return max(1, int(funding_interval_minutes)) * 60 * 1000

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
            f"[{normalized_symbol}-{timeframe}] Bybit backfill: {total_windows} windows, "
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

    def fetch_mark_price_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[PriceKline]:
        return self._fetch_price_dataset(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            dataset_name="mark",
            fetch_window_builder=lambda api_symbol, interval: (
                lambda window_start, window_end: self.adapter.fetch_mark_price_kline_window(
                    api_symbol,
                    interval,
                    window_start,
                    window_end,
                )
            ),
        )

    def fetch_index_price_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[PriceKline]:
        return self._fetch_price_dataset(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            dataset_name="index",
            fetch_window_builder=lambda api_symbol, interval: (
                lambda window_start, window_end: self.adapter.fetch_index_price_kline_window(
                    api_symbol,
                    interval,
                    window_start,
                    window_end,
                )
            ),
        )

    def fetch_premium_index_price_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[PriceKline]:
        return self._fetch_price_dataset(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            dataset_name="premium-index",
            fetch_window_builder=lambda api_symbol, interval: (
                lambda window_start, window_end: self.adapter.fetch_premium_index_price_kline_window(
                    api_symbol,
                    interval,
                    window_start,
                    window_end,
                )
            ),
        )

    def fetch_funding_rates(
        self,
        symbol: str | Symbol,
        start_ts: int,
        end_ts: int,
    ) -> list[FundingRatePoint]:
        normalized_symbol = self.normalize_symbol(symbol)
        start_ts = self.clamp_start_ts_to_listing(normalized_symbol, start_ts)
        interval_ms = self.get_funding_interval_ms(normalized_symbol)
        windows = self.build_request_windows(
            start_ts,
            end_ts,
            interval_ms,
            limit=self.adapter.funding_limit,
        )
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_points: list[FundingRatePoint] = []

        logger.info(
            f"[{normalized_symbol}-funding] Bybit backfill: {total_windows} windows, "
            f"limit={self.adapter.funding_limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(
                    self.adapter.fetch_funding_rate_window,
                    api_symbol,
                    window_start,
                    window_end,
                ): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                points = self.mapper.to_funding_rates(payload)
                if points:
                    all_points.extend(points)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = points[-1].funding_time if points else window_end
                    logger.info(
                        f"[{normalized_symbol}-funding] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        deduped_points = {point.funding_time: point for point in all_points}
        return [deduped_points[key] for key in sorted(deduped_points)]

    def fetch_open_interest(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[OpenInterestPoint]:
        normalized_symbol = self.normalize_symbol(symbol)
        start_ts = self.clamp_start_ts_to_listing(normalized_symbol, start_ts)
        interval_ms = self.get_timeframe_ms(timeframe)
        windows = self.build_request_windows(
            start_ts,
            end_ts,
            interval_ms,
            limit=self.adapter.open_interest_limit,
        )
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        interval_time = self.mapper.to_open_interest_interval(timeframe)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_points: list[OpenInterestPoint] = []

        logger.info(
            f"[{normalized_symbol}-{timeframe}-open-interest] Bybit backfill: {total_windows} windows, "
            f"limit={self.adapter.open_interest_limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(
                    self.adapter.fetch_open_interest_window,
                    api_symbol,
                    interval_time,
                    window_start,
                    window_end,
                ): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                points = self.mapper.to_open_interest_points(payload)
                if points:
                    all_points.extend(points)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = points[-1].timestamp if points else window_end
                    logger.info(
                        f"[{normalized_symbol}-{timeframe}-open-interest] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        deduped_points = {point.timestamp: point for point in all_points}
        return [deduped_points[key] for key in sorted(deduped_points)]

    def _fetch_price_dataset(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        dataset_name: str,
        fetch_window_builder,
    ) -> list[PriceKline]:
        normalized_symbol = self.normalize_symbol(symbol)
        start_ts = self.clamp_start_ts_to_listing(normalized_symbol, start_ts)
        timeframe_ms = self.get_timeframe_ms(timeframe)
        windows = self.build_request_windows(start_ts, end_ts, timeframe_ms)
        if not windows:
            return []

        api_symbol = self.mapper.to_api_symbol(normalized_symbol)
        interval = self.mapper.to_interval(timeframe)
        fetch_window = fetch_window_builder(api_symbol, interval)
        total_windows = len(windows)
        progress_step = max(1, total_windows // 10)
        all_klines: list[PriceKline] = []

        logger.info(
            f"[{normalized_symbol}-{timeframe}-{dataset_name}] Bybit backfill: {total_windows} windows, "
            f"limit={self.adapter.limit}, workers={min(self.adapter.max_workers, total_windows)}"
        )

        with ThreadPoolExecutor(max_workers=min(self.adapter.max_workers, total_windows)) as executor:
            future_to_window = {
                executor.submit(fetch_window, window_start, window_end): (window_start, window_end)
                for window_start, window_end in windows
            }

            for completed, future in enumerate(as_completed(future_to_window), start=1):
                _, window_end = future_to_window[future]
                payload = future.result()
                klines = self.mapper.to_price_klines(payload)
                if klines:
                    all_klines.extend(klines)

                if completed % progress_step == 0 or completed == total_windows:
                    progress_ts = klines[-1].open_time if klines else window_end
                    logger.info(
                        f"[{normalized_symbol}-{timeframe}-{dataset_name}] windows {completed}/{total_windows}, "
                        f"up to {datetime.fromtimestamp(progress_ts / 1000)}"
                    )

        deduped_klines = {candle.open_time: candle for candle in all_klines}
        return [deduped_klines[key] for key in sorted(deduped_klines)]
