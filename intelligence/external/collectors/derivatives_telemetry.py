"""Perpetual derivatives telemetry collector.

Collects public funding rates and open interest without private API credentials.
Used to detect extreme crowd positioning, leverage exhaustion, and funding squeeze.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from cachepaths import canonical_symbol_key
from intelligence.external.collectors.orderbook_depth import (
    _DiskBackedSymbolCollector,
    _clean_dict_float,
    _clean_snapshot_symbol,
    _extract_base_asset,
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
    mark_price: float = 0.0
    index_price: float = 0.0
    perp_spot_basis_bps: float = 0.0
    funding_annualized_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], symbol: str = "") -> DerivativesTelemetry:
        if not isinstance(data, dict):
            data = {}
        fr = _clean_dict_float(data, "funding_rate", 0.0)
        mark_px = _clean_dict_float(data, "mark_price", 0.0, "markPx", nonneg=True)
        idx_px = _clean_dict_float(data, "index_price", 0.0, "oraclePx", nonneg=True)
        basis_bps = _clean_dict_float(
            data,
            "perp_spot_basis_bps",
            round(((mark_px - idx_px) / idx_px) * 10000.0, 2) if idx_px > 0.0 and mark_px > 0.0 else 0.0,
        )
        ann_pct = _clean_dict_float(data, "funding_annualized_pct", round(fr * 3.0 * 365.0 * 100.0, 2))
        return cls(
            symbol=_clean_snapshot_symbol(data, symbol),
            funding_rate=fr,
            predicted_funding_rate=_clean_dict_float(data, "predicted_funding_rate", 0.0, "pred_funding_rate"),
            open_interest=_clean_dict_float(data, "open_interest", 0.0, nonneg=True),
            open_interest_usd=_clean_dict_float(data, "open_interest_usd", 0.0, "oi_usd", nonneg=True),
            ts=_clean_dict_float(data, "ts", 0.0, nonneg=True),
            mark_price=mark_px,
            index_price=idx_px,
            perp_spot_basis_bps=basis_bps,
            funding_annualized_pct=ann_pct,
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

    def _fetch_hyperliquid_derivatives(self, symbol: str, now: float) -> Optional[DerivativesTelemetry]:
        """Fetch native funding rate and open interest from Hyperliquid L1 metaAndAssetCtxs."""
        base_coin = _extract_base_asset(symbol)
        if not base_coin:
            return None
        raw = self._fetch_json_dict(
            "https://api.hyperliquid.xyz/info?type=metaAndAssetCtxs",
            "metaAndAssetCtxs",
            base_coin,
            timeout=4.0,
        )
        res_list = raw.get("result_list") if isinstance(raw, dict) else None
        if not isinstance(res_list, list) or len(res_list) < 2:
            return None
        meta, ctxs = res_list[0], res_list[1]
        universe = (meta.get("universe") if isinstance(meta, dict) else None) or []
        if not isinstance(universe, list) or not isinstance(ctxs, list):
            return None
        base_cf = base_coin.casefold()
        for idx, asset_info in enumerate(universe):
            if (
                isinstance(asset_info, dict)
                and str(asset_info.get("name") or "").casefold() == base_cf
                and idx < len(ctxs)
                and isinstance(ctxs[idx], dict)
            ):
                ctx = ctxs[idx]
                funding_rate = _clean_dict_float(ctx, "funding", 0.0)
                oi = max(0.0, _clean_dict_float(ctx, "openInterest", 0.0))
                mark_px = max(0.0, _clean_dict_float(ctx, "markPx", 0.0, "oraclePx"))
                oracle_px = max(0.0, _clean_dict_float(ctx, "oraclePx", mark_px))
                oi_usd = oi * mark_px
                basis_bps = round(((mark_px - oracle_px) / oracle_px) * 10000.0, 2) if oracle_px > 0.0 and mark_px > 0.0 else 0.0
                telemetry = DerivativesTelemetry(
                    symbol=symbol,
                    funding_rate=funding_rate,
                    predicted_funding_rate=funding_rate,
                    open_interest=oi,
                    open_interest_usd=oi_usd,
                    ts=now,
                    mark_price=mark_px,
                    index_price=oracle_px,
                    perp_spot_basis_bps=basis_bps,
                    funding_annualized_pct=round(funding_rate * 3.0 * 365.0 * 100.0, 2),
                )
                return self._commit_snapshot(symbol, now, telemetry)
        return None

    def fetch(
        self,
        symbol: str,
        allow_network: bool = True,
        now: Optional[float] = None,
    ) -> Optional[DerivativesTelemetry]:
        """Fetch funding rate and open interest from public Binance Futures or Hyperliquid L1 endpoint."""
        symbol, now, cached, ready, should_fetch = self._prepare_fetch(symbol, allow_network, now)
        if not should_fetch or symbol is None:
            return ready

        if canonical_symbol_key(symbol) == "HYPEUSDC":
            try:
                if (hl_tel := self._fetch_hyperliquid_derivatives(symbol, now)) is not None:
                    return hl_tel
            except Exception:
                pass

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
            index_price = max(0.0, _clean_dict_float(data_fund, "indexPrice", mark_price))
            oi_usd = oi * mark_price
            basis_bps = round(((mark_price - index_price) / index_price) * 10000.0, 2) if index_price > 0.0 and mark_price > 0.0 else 0.0

            telemetry = DerivativesTelemetry(
                symbol=symbol,
                funding_rate=funding_rate,
                predicted_funding_rate=next_funding,
                open_interest=oi,
                open_interest_usd=oi_usd,
                ts=now,
                mark_price=mark_price,
                index_price=index_price,
                perp_spot_basis_bps=basis_bps,
                funding_annualized_pct=round(funding_rate * 3.0 * 365.0 * 100.0, 2),
            )
            return self._commit_snapshot(symbol, now, telemetry)
        except Exception as e:
            if canonical_symbol_key(symbol) != "HYPEUSDC":
                try:
                    if (hl_tel := self._fetch_hyperliquid_derivatives(symbol, now)) is not None:
                        return hl_tel
                except Exception:
                    pass
            logger.warning("Could not fetch derivatives telemetry for %s: %s", symbol, e)
            return self._fallback_stale(symbol, cached)

    def record_telemetry(self, telemetry: DerivativesTelemetry, save_to_disk: bool = False) -> None:
        """Inject telemetry directly (useful for tests and synthetic feeds)."""
        self._record_item(telemetry, save_to_disk=save_to_disk)
