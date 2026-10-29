"""Perpetual derivatives telemetry collector.

Collects public funding rates and open interest without private API credentials.
Used to detect extreme crowd positioning, leverage exhaustion, and funding squeeze.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from intelligence.external.collectors.orderbook_depth import (
    _DiskBackedSymbolCollector,
    _clean_dict_float,
    _resolve_binance_lookup_symbol,
)

logger = logging.getLogger("intelligence.external.derivatives_telemetry")


@dataclass(frozen=True)
class DerivativesTelemetry:
    """Normalized derivatives market state."""

    symbol: str
    funding_rate: float            # 8h rate (e.g. +0.0001 = 0.01% / 8h)
    predicted_funding_rate: float
    open_interest: float           # total open interest contracts/coins
    open_interest_usd: float       # total open interest in USD
    ts: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DerivativesTelemetry:
        if not isinstance(data, dict):
            data = {}
        raw_sym = data.get("symbol")
        sym_str = raw_sym.strip() if isinstance(raw_sym, str) else ""
        return cls(
            symbol=sym_str,
            funding_rate=_clean_dict_float(data, "funding_rate", 0.0),
            predicted_funding_rate=_clean_dict_float(data, "predicted_funding_rate", 0.0),
            open_interest=max(0.0, _clean_dict_float(data, "open_interest", 0.0)),
            open_interest_usd=max(0.0, _clean_dict_float(data, "open_interest_usd", 0.0)),
            ts=max(0.0, _clean_dict_float(data, "ts", 0.0)),
        )


class DerivativesTelemetryCollector(_DiskBackedSymbolCollector):
    """Fetches and caches public derivatives market metrics."""

    _DISK_FILE_PREFIX = "derivatives_telemetry"
    _SNAPSHOT_CLS = DerivativesTelemetry

    def __init__(
        self,
        cache_ttl_sec: float = 60.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 120.0,
    ) -> None:
        super().__init__(cache_ttl_sec, 60.0, cache_dir, disk_ttl_sec, 120.0)

    def fetch(
        self,
        symbol: str,
        allow_network: bool = True,
        now: Optional[float] = None,
    ) -> Optional[DerivativesTelemetry]:
        """Fetch funding rate and open interest from public Binance Futures endpoint."""
        symbol, now, cached, ready, should_fetch = self._prepare_fetch(symbol, allow_network, now)
        if not should_fetch or symbol is None:
            return ready

        # Use USDT-perp pair as proxy for non-USDT quotes
        lookup_sym = _resolve_binance_lookup_symbol(symbol)
        try:
            # 1. Premium index endpoint (funding rate)
            url_fund = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={lookup_sym}"
            data_fund = self._fetch_json_dict(url_fund, "premiumIndex", lookup_sym)

            funding_rate = _clean_dict_float(data_fund, "lastFundingRate", 0.0)
            next_funding = _clean_dict_float(data_fund, "nextFundingRate", funding_rate)

            # 2. Open interest endpoint
            url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={lookup_sym}"
            data_oi = self._fetch_json_dict(url_oi, "openInterest", lookup_sym)

            oi = max(0.0, _clean_dict_float(data_oi, "openInterest", 0.0))
            mark_price = max(0.0, _clean_dict_float(data_fund, "markPrice", 0.0))
            oi_usd = oi * mark_price

            telemetry = DerivativesTelemetry(
                symbol=symbol,
                funding_rate=funding_rate,
                predicted_funding_rate=next_funding,
                open_interest=oi,
                open_interest_usd=oi_usd,
                ts=now,
            )
            return self._commit_snapshot(symbol, now, telemetry)
        except Exception as e:
            logger.warning("Could not fetch derivatives telemetry for %s: %s", symbol, e)
            return self._fallback_stale(symbol, cached)

    def record_telemetry(self, telemetry: DerivativesTelemetry, save_to_disk: bool = False) -> None:
        """Inject telemetry directly (useful for tests and synthetic feeds)."""
        self._record_item(telemetry, save_to_disk=save_to_disk)
