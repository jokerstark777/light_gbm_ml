from __future__ import annotations

import logging
import threading
import time

import config as cfg
import requests
from requests.adapters import HTTPAdapter


logger = logging.getLogger(__name__)


class BybitAdapter:
    def __init__(self) -> None:
        self.kline_url = getattr(cfg, "BYBIT_KLINE_URL", "https://api.bybit.com/v5/market/kline")
        self.mark_price_kline_url = getattr(
            cfg,
            "BYBIT_MARK_PRICE_KLINE_URL",
            "https://api.bybit.com/v5/market/mark-price-kline",
        )
        self.index_price_kline_url = getattr(
            cfg,
            "BYBIT_INDEX_PRICE_KLINE_URL",
            "https://api.bybit.com/v5/market/index-price-kline",
        )
        self.premium_index_price_kline_url = getattr(
            cfg,
            "BYBIT_PREMIUM_INDEX_PRICE_KLINE_URL",
            "https://api.bybit.com/v5/market/premium-index-price-kline",
        )
        self.funding_history_url = getattr(
            cfg,
            "BYBIT_FUNDING_HISTORY_URL",
            "https://api.bybit.com/v5/market/funding/history",
        )
        self.open_interest_url = getattr(
            cfg,
            "BYBIT_OPEN_INTEREST_URL",
            "https://api.bybit.com/v5/market/open-interest",
        )
        self.instruments_info_url = getattr(
            cfg,
            "BYBIT_INSTRUMENTS_INFO_URL",
            "https://api.bybit.com/v5/market/instruments-info",
        )
        self.category = getattr(cfg, "BYBIT_CATEGORY", "linear")
        self.limit = min(1000, max(1, int(getattr(cfg, "BYBIT_LIMIT", 1000))))
        self.funding_limit = min(200, max(1, int(getattr(cfg, "BYBIT_FUNDING_LIMIT", 200))))
        self.open_interest_limit = min(200, max(1, int(getattr(cfg, "BYBIT_OPEN_INTEREST_LIMIT", 200))))
        self.timeout = float(getattr(cfg, "BYBIT_TIMEOUT", 20))
        self.retry_count = max(1, int(getattr(cfg, "BYBIT_RETRY_COUNT", 5)))
        self.retry_sleep = float(getattr(cfg, "BYBIT_RETRY_SLEEP", 0.33))
        self.max_workers = max(1, int(getattr(cfg, "BYBIT_MAX_WORKERS", 6)))
        self._thread_local = threading.local()

    def get_http_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=max(8, self.max_workers * 2),
                pool_maxsize=max(8, self.max_workers * 2),
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update({"User-Agent": "mlV2-etl-bybit/1.0"})
            self._thread_local.session = session
        return session

    def request_json(self, url: str, params: dict, request_name: str) -> dict:
        last_error = None

        for attempt in range(self.retry_count):
            try:
                response = self.get_http_session().get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                ret_code = payload.get("retCode")
                if ret_code == 0:
                    return payload

                last_error = RuntimeError(f"Bybit retCode={ret_code}, retMsg={payload.get('retMsg')}")
                if ret_code not in {10000, 10006, 10016}:
                    raise last_error
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc

            if attempt + 1 < self.retry_count:
                sleep_s = self.retry_sleep * (2 ** attempt)
                logger.warning(
                    f"[{request_name}] retry {attempt + 1}/{self.retry_count}: {last_error}"
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"Bybit request failed for {request_name}: {last_error}")

    def fetch_kline_window(
        self,
        api_symbol: str,
        interval: str,
        window_start: int,
        window_end: int,
    ) -> dict:
        params = {
            "category": self.category,
            "symbol": api_symbol,
            "interval": interval,
            "start": window_start,
            "end": window_end,
            "limit": self.limit,
        }
        return self.request_json(
            self.kline_url,
            params,
            f"{api_symbol}-{interval}-{window_start}-{window_end}",
        )

    def fetch_mark_price_kline_window(
        self,
        api_symbol: str,
        interval: str,
        window_start: int,
        window_end: int,
    ) -> dict:
        params = {
            "category": self.category,
            "symbol": api_symbol,
            "interval": interval,
            "start": window_start,
            "end": window_end,
            "limit": self.limit,
        }
        return self.request_json(
            self.mark_price_kline_url,
            params,
            f"mark-{api_symbol}-{interval}-{window_start}-{window_end}",
        )

    def fetch_index_price_kline_window(
        self,
        api_symbol: str,
        interval: str,
        window_start: int,
        window_end: int,
    ) -> dict:
        params = {
            "category": self.category,
            "symbol": api_symbol,
            "interval": interval,
            "start": window_start,
            "end": window_end,
            "limit": self.limit,
        }
        return self.request_json(
            self.index_price_kline_url,
            params,
            f"index-{api_symbol}-{interval}-{window_start}-{window_end}",
        )

    def fetch_premium_index_price_kline_window(
        self,
        api_symbol: str,
        interval: str,
        window_start: int,
        window_end: int,
    ) -> dict:
        params = {
            "category": self.category,
            "symbol": api_symbol,
            "interval": interval,
            "start": window_start,
            "end": window_end,
            "limit": self.limit,
        }
        return self.request_json(
            self.premium_index_price_kline_url,
            params,
            f"premium-index-{api_symbol}-{interval}-{window_start}-{window_end}",
        )

    def fetch_funding_rate_window(
        self,
        api_symbol: str,
        window_start: int,
        window_end: int,
    ) -> dict:
        params = {
            "category": self.category,
            "symbol": api_symbol,
            "startTime": window_start,
            "endTime": window_end,
            "limit": self.funding_limit,
        }
        return self.request_json(
            self.funding_history_url,
            params,
            f"funding-{api_symbol}-{window_start}-{window_end}",
        )

    def fetch_open_interest_window(
        self,
        api_symbol: str,
        interval_time: str,
        window_start: int,
        window_end: int,
    ) -> dict:
        params = {
            "category": self.category,
            "symbol": api_symbol,
            "intervalTime": interval_time,
            "startTime": window_start,
            "endTime": window_end,
            "limit": self.open_interest_limit,
        }
        return self.request_json(
            self.open_interest_url,
            params,
            f"open-interest-{api_symbol}-{interval_time}-{window_start}-{window_end}",
        )

    def fetch_instruments_info(
        self,
        api_symbol: str,
    ) -> dict:
        params = {
            "category": self.category,
            "symbol": api_symbol,
        }
        return self.request_json(
            self.instruments_info_url,
            params,
            f"instrument-info-{api_symbol}",
        )
