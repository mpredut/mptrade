"""Perpetual derivatives telemetry collector.

Collects public funding rates and open interest without private API credentials.
Used to detect extreme crowd positioning, leverage exhaustion, and funding squeeze.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from intelligence.external.collectors.orderbook_depth import (
    _resolve_binance_lookup_symbol,
    _symbol_variants,
)
from state_io import read_recovering_previous

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
        return cls(
            symbol=str(data.get("symbol", "")),
            funding_rate=float(data.get("funding_rate", 0.0)),
            predicted_funding_rate=float(data.get("predicted_funding_rate", 0.0)),
            open_interest=float(data.get("open_interest", 0.0)),
            open_interest_usd=float(data.get("open_interest_usd", 0.0)),
            ts=float(data.get("ts", 0.0)),
        )


class DerivativesTelemetryCollector:
    """Fetches and caches public derivatives market metrics."""

    def __init__(
        self,
        cache_ttl_sec: float = 60.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 120.0,
    ) -> None:
        self.cache_ttl_sec = cache_ttl_sec
        self.cache_dir = cache_dir
        self.disk_ttl_sec = disk_ttl_sec
        self._cache: Dict[str, tuple[float, DerivativesTelemetry]] = {}

    def _disk_path(self, symbol: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        cache_dir = self.cache_dir
        if cache_dir == "cachedb":
            try:
                from cachepaths import get_cachedb_dir
                cache_dir = get_cachedb_dir()
            except Exception:
                cache_dir = os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb")
        safe_sym = symbol.upper().strip().replace("/", "").replace("-", "").replace("_", "")
        return os.path.join(cache_dir, f"derivatives_telemetry_{safe_sym}.json")

    def _load_from_disk(self, symbol: str, now: float) -> Optional[DerivativesTelemetry]:
        from state_io import read_recovering_previous
        max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
        for cand in _symbol_variants(symbol):
            p = self._disk_path(cand)
            if not p:
                continue
            try:
                data = read_recovering_previous(p, json.loads, default=None)
                if not isinstance(data, dict):
                    continue
                telemetry = DerivativesTelemetry.from_dict(data)
                if (now - telemetry.ts) < max_age:
                    self._cache[symbol.upper()] = (telemetry.ts, telemetry)
                    return telemetry
            except Exception as e:
                logger.debug("Failed loading derivatives telemetry from %s: %s", p, e)
        return None

    def _save_to_disk(self, telemetry: DerivativesTelemetry) -> None:
        p = self._disk_path(telemetry.symbol)
        if not p:
            return
        try:
            from state_io import atomic_write_json
            atomic_write_json(p, telemetry.to_dict(), indent=2)
        except Exception as e:
            logger.debug("Failed writing derivatives telemetry to %s: %s", p, e)

    def fetch(self, symbol: str, allow_network: bool = True) -> Optional[DerivativesTelemetry]:
        """Fetch funding rate and open interest from public Binance Futures endpoint."""
        symbol = symbol.upper()
        now = time.time()
        cached = None
        for cand in _symbol_variants(symbol):
            cached = self._cache.get(cand)
            if cached is not None:
                break
        if cached:
            if isinstance(cached, tuple) and len(cached) >= 2:
                if (now - cached[0]) < self.cache_ttl_sec:
                    return cached[1]
            elif isinstance(cached, DerivativesTelemetry):
                if (now - cached.ts) < self.cache_ttl_sec:
                    return cached

        disk_snap = self._load_from_disk(symbol, now)
        if disk_snap is not None:
            return disk_snap

        if not allow_network:
            return None

        # Use USDT-perp pair as proxy for non-USDT quotes
        lookup_sym = _resolve_binance_lookup_symbol(symbol)
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
            self._save_to_disk(telemetry)
            return telemetry
        except Exception as e:
            logger.warning("Could not fetch derivatives telemetry for %s: %s", symbol, e)
            if cached:
                if isinstance(cached, tuple) and len(cached) >= 2:
                    return cached[1]
                if isinstance(cached, DerivativesTelemetry):
                    return cached
            for cand in _symbol_variants(symbol):
                p = self._disk_path(cand)
                if p:
                    try:
                        data = read_recovering_previous(p, json.loads, default=None)
                        if isinstance(data, dict):
                            return DerivativesTelemetry.from_dict(data)
                    except Exception:
                        pass
            return None

    def record_telemetry(self, telemetry: DerivativesTelemetry, save_to_disk: bool = False) -> None:
        """Inject telemetry directly (useful for tests and synthetic feeds)."""
        self._cache[telemetry.symbol.upper()] = (telemetry.ts, telemetry)
        if save_to_disk:
            self._save_to_disk(telemetry)
