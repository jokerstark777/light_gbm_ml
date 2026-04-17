from __future__ import annotations

import logging
import threading
import time
from email.utils import parsedate_to_datetime

import config as cfg
import requests
from requests.adapters import HTTPAdapter


logger = logging.getLogger(__name__)


class BinanceAdapter:
    _rate_limit_lock = threading.Lock()
    _next_request_ts = 0.0
    _cooldown_until_ts = 0.0

    def __init__(self) -> None:
        self.base_url = getattr(cfg, "BINANCE_BASE_URL", "https://fapi.binance.com")
        self.kline_url = f"{self.base_url}/fapi/v1/klines"
        self.exchange_info_url = f"{self.base_url}/fapi/v1/exchangeInfo"
        self.limit = min(1500, max(1, int(getattr(cfg, "BINANCE_LIMIT", 1000))))
        self.timeout = float(getattr(cfg, "BINANCE_TIMEOUT", 20))
        self.retry_count = max(1, int(getattr(cfg, "BINANCE_RETRY_COUNT", 5)))
        self.retry_sleep = float(getattr(cfg, "BINANCE_RETRY_SLEEP", 0.3))
        self.max_workers = max(1, int(getattr(cfg, "BINANCE_MAX_WORKERS", 3)))
        self.request_weight_limit_per_minute = max(
            1,
            int(getattr(cfg, "BINANCE_REQUEST_WEIGHT_LIMIT_PER_MINUTE", 2400)),
        )
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
            session.headers.update({"User-Agent": "mlV2-etl-binance/1.0"})
            self._thread_local.session = session
        return session

    def request_json(self, url: str, params: dict | None, request_name: str) -> dict | list:
        last_error = None
        request_weight = self._resolve_request_weight(params)

        for attempt in range(self.retry_count):
            try:
                self._wait_for_request_slot(request_weight, request_name)
                response = self.get_http_session().get(url, params=params, timeout=self.timeout)
                if response.status_code in {418, 429}:
                    self._apply_rate_limit_cooldown(response, request_name)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc

            if attempt + 1 < self.retry_count:
                sleep_s = self.retry_sleep * (2 ** attempt)
                logger.warning(
                    f"[{request_name}] retry {attempt + 1}/{self.retry_count}: {last_error}"
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"Binance request failed for {request_name}: {last_error}")

    def _resolve_request_weight(self, params: dict | None) -> int:
        limit = int((params or {}).get("limit", 1))
        if limit < 100:
            return 1
        if limit < 500:
            return 2
        if limit <= 1000:
            return 5
        return 10

    def _wait_for_request_slot(self, request_weight: int, request_name: str) -> None:
        seconds_per_weight = 60.0 / float(self.request_weight_limit_per_minute)

        while True:
            with self._rate_limit_lock:
                now = time.time()
                wait_s = max(
                    self._cooldown_until_ts - now,
                    self._next_request_ts - now,
                    0.0,
                )
                if wait_s <= 0:
                    self._next_request_ts = now + (seconds_per_weight * request_weight)
                    return

            if wait_s >= 1.0:
                logger.warning(
                    f"[{request_name}] Binance cooldown active, sleeping {wait_s:.1f}s before next request"
                )
            time.sleep(min(wait_s, 1.0))

    def _apply_rate_limit_cooldown(self, response: requests.Response, request_name: str) -> None:
        retry_after_s = self._extract_retry_after_seconds(response)
        if retry_after_s is None:
            retry_after_s = 120.0 if response.status_code == 418 else 5.0

        cooldown_until = time.time() + retry_after_s
        with self._rate_limit_lock:
            self._cooldown_until_ts = max(self._cooldown_until_ts, cooldown_until)
            self._next_request_ts = max(self._next_request_ts, self._cooldown_until_ts)

        logger.warning(
            f"[{request_name}] Binance rate limit status={response.status_code}, "
            f"cooldown {retry_after_s:.1f}s"
        )

    @staticmethod
    def _extract_retry_after_seconds(response: requests.Response) -> float | None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after).timestamp()
                    return max(0.0, retry_at - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass

        retry_after_ms = response.headers.get("x-mbx-retry-after")
        if retry_after_ms:
            try:
                return max(0.0, float(retry_after_ms) / 1000.0)
            except ValueError:
                return None
        return None

    def fetch_exchange_info(self) -> dict:
        payload = self.request_json(
            self.exchange_info_url,
            params=None,
            request_name="binance-exchange-info",
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected Binance exchangeInfo payload")
        return payload

    def fetch_kline_window(
        self,
        api_symbol: str,
        interval: str,
        window_start: int,
        window_end: int,
    ) -> list:
        payload = self.request_json(
            self.kline_url,
            params={
                "symbol": api_symbol,
                "interval": interval,
                "startTime": window_start,
                "endTime": window_end,
                "limit": self.limit,
            },
            request_name=f"{api_symbol}-{interval}-{window_start}-{window_end}",
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Binance kline payload type for {api_symbol}: {type(payload).__name__}")
        return payload
