"""Perpetual derivatives telemetry collector.

Collects public funding rates and open interest without private API credentials.
Used to detect extreme crowd positioning, leverage exhaustion, and funding squeeze.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

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


class DerivativesTelemetryCollector:
    """Fetches and caches public derivatives market metrics."""

    def __init__(self, cache_ttl_sec: float = 60.0) -> None:
        self.cache_ttl_sec = cache_ttl_sec
        self._cache: Dict[str, tuple[float, DerivativesTelemetry]] = {}

    def fetch(self, symbol: str) -> Optional[DerivativesTelemetry]:
        """Fetch funding rate and open interest from public Binance Futures endpoint."""
        symbol = symbol.upper()
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        # Use USDT-perp pair as proxy if USDC given
        lookup_sym = symbol.replace("USDC", "USDT")
        try:
            # 1. Premium index endpoint (funding rate)
            url_fund = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={lookup_sym}"
            req = urllib.request.Request(url_fund, headers={"User-Agent": "mptrade-intelligence/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data_fund = json.loads(resp.read().decode("utf-8"))

            funding_rate = float(data_fund.get("lastFundingRate") or 0.0)
            next_funding = float(data_fund.get("nextFundingRate") or funding_rate)

            # 2. Open interest endpoint
            url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={lookup_sym}"
            req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "mptrade-intelligence/1.0"})
            with urllib.request.urlopen(req_oi, timeout=5) as resp:
                data_oi = json.loads(resp.read().decode("utf-8"))

            oi = float(data_oi.get("openInterest") or 0.0)
            mark_price = float(data_fund.get("markPrice") or 0.0)
            oi_usd = oi * mark_price

            telemetry = DerivativesTelemetry(
                symbol=symbol,
                funding_rate=funding_rate,
                predicted_funding_rate=next_funding,
                open_interest=oi,
                open_interest_usd=oi_usd,
                ts=now,
            )
            self._cache[symbol] = (now, telemetry)
            return telemetry
        except Exception as e:
            logger.warning("Could not fetch derivatives telemetry for %s: %s", symbol, e)
            if cached:
                return cached[1]
            return None

    def record_telemetry(self, telemetry: DerivativesTelemetry) -> None:
        """Inject telemetry directly (useful for tests and synthetic feeds)."""
        self._cache[telemetry.symbol.upper()] = (telemetry.ts, telemetry)
