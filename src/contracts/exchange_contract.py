from __future__ import annotations

from abc import ABC, abstractmethod

from src.types.common import HistoricalKline, Symbol


class ExchangeContract(ABC):
    @abstractmethod
    def get_exchange_code(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def normalize_symbol(self, symbol: str | Symbol) -> Symbol:
        raise NotImplementedError

    @abstractmethod
    def get_timeframe_ms(self, timeframe: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str | Symbol,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[HistoricalKline]:
        raise NotImplementedError
