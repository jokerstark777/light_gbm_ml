from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Symbol:
    base_asset: str
    quote_asset: str

    @classmethod
    def from_string(cls, value: str) -> "Symbol":
        normalized = value.strip().upper()
        base_asset, quote_asset = normalized.split("/", maxsplit=1)
        return cls(base_asset=base_asset, quote_asset=quote_asset)

    def __str__(self) -> str:
        return f"{self.base_asset}/{self.quote_asset}"


@dataclass(frozen=True, slots=True)
class HistoricalKline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float


@dataclass(frozen=True, slots=True)
class PriceKline:
    open_time: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class FundingRatePoint:
    funding_time: int
    funding_rate: float


@dataclass(frozen=True, slots=True)
class OpenInterestPoint:
    timestamp: int
    open_interest: float
