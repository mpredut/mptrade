"""Stateless public WebSocket telemetry streams & hub (Binance Futures markPrice, Hyperliquid activeAssetCtx, Binance Spot 5m kline flow).

Reuses the shared ``JsonPublicWsStreamBase`` reconnect loop from ``providers.ws_stream`` and
stateless protocol framing/parsing from ``providers.venue_ws_protocols`` (SSOT). All streams
hold lightweight thread-safe in-memory state consumed in 0 ms by external collectors with
transparent fallback to existing REST endpoints when warming up or disconnected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from botcore import (
    clean_upper_scalar,
    exact_pos_int,
    resolve_now_ts,
    safe_nonneg_float,
)
from cachepaths import (
    canonical_base_asset,
    clean_symbol_key,
    resolve_binance_lookup_symbol,
)
from instrument_registry import resolve_symbol_identity, venue_for_symbol
from providers.base import MarketDataProvider
from providers.venue_ws_protocols import BinancePublicWsMixin, HyperliquidWsMixin
from providers.ws_stream import JsonPublicWsStreamBase

logger = logging.getLogger("intelligence.external.telemetry_ws")

_KLINE_5M_STEP_MS: int = 300_000
_KLINE_5M_MAX_BARS: int = 4
_KLINE_5M_FRESH_LAG_MS: int = 360_000


class BinanceFuturesMarkPriceStream(JsonPublicWsStreamBase):
    """Live Binance USD-M Futures ``@markPrice@1s`` stream (funding rate + mark/index basis)."""

    _default_thread_name = "BinanceMarkPriceWS"

    def __init__(self, symbols: Sequence[str]) -> None:
        super().__init__()
        self._symbols: Tuple[str, ...] = tuple(
            dict.fromkeys(
                s
                for raw in symbols
                if (s := (clean_symbol_key(raw) or "").upper())
            )
        )
        # symbol -> (ts, mark_price, index_price, funding_rate)
        self._by_symbol: Dict[str, Tuple[float, float, float, float]] = {}

    def _resolve_ws_url(self) -> Optional[str]:
        return BinancePublicWsMixin.build_ws_mark_price_url(self._symbols, interval_sec=1)

    def _handle_payload(self, payload: Any, now_ts: float) -> bool:
        parsed = BinancePublicWsMixin.extract_ws_mark_price(payload)
        if parsed is None:
            return False
        sym = parsed["symbol"]
        with self._state_lock:
            self._by_symbol[sym] = (
                now_ts,
                parsed["mark_price"],
                parsed["index_price"],
                parsed["funding_rate"],
            )
        return True

    def get_mark_price(
        self,
        symbol: str,
        *,
        now: Optional[float] = None,
        max_age_sec: float = 15.0,
    ) -> Optional[Dict[str, float]]:
        """Return fresh ``{mark_price, index_price, funding_rate, ts}`` for ``symbol`` or ``None``."""
        lookup = (clean_symbol_key(symbol) or "").upper()
        if not lookup:
            return None
        eff_now = resolve_now_ts(now)
        with self._state_lock:
            if self._disconnected_since_ts > 0.0 and not self._connected:
                return None
            rec = self._by_symbol.get(lookup)
        if rec is None:
            return None
        ts_val, mark_px, index_px, fr = rec
        if not (0.0 <= (eff_now - ts_val) <= max_age_sec):
            return None
        return {
            "mark_price": mark_px,
            "index_price": index_px,
            "funding_rate": fr,
            "ts": ts_val,
        }

    def status(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        eff_now = resolve_now_ts(now)
        with self._state_lock:
            last_ts = self._last_msg_ts
            conn = self._connected
            ready = len(self._by_symbol)
        return {
            "running": bool(self.is_running),
            "connected": bool(conn),
            "tracked": len(self._symbols),
            "ready": ready,
            "last_msg_age_sec": round(max(0.0, eff_now - last_ts), 1) if last_ts > 0.0 else None,
        }


class HyperliquidAssetCtxStream(JsonPublicWsStreamBase):
    """Live Hyperliquid public ``activeAssetCtx`` stream (funding, open interest, mark & oracle)."""

    _default_thread_name = "HyperliquidAssetCtxWS"

    def __init__(self, coins: Sequence[str], *, app_ping_interval_sec: float = 45.0) -> None:
        super().__init__(app_ping_interval_sec=app_ping_interval_sec)
        self._coins: Tuple[str, ...] = tuple(
            dict.fromkeys(
                c
                for raw in coins
                if (c := (canonical_base_asset(raw) or clean_upper_scalar(raw) or ""))
            )
        )
        # coin -> (ts, ctx_dict)
        self._by_coin: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _resolve_ws_url(self) -> Optional[str]:
        return HyperliquidWsMixin.PUBLIC_WS_URL if self._coins else None

    def _build_subscribe_payloads(self) -> tuple[str, ...]:
        return HyperliquidWsMixin.build_ws_active_asset_ctx_subscriptions(self._coins)

    def _app_ping_payload(self) -> Optional[str]:
        return HyperliquidWsMixin.build_ws_ping_payload()

    def _handle_payload(self, payload: Any, now_ts: float) -> bool:
        parsed = HyperliquidWsMixin.extract_ws_active_asset_ctx(payload)
        if parsed is None:
            return False
        coin, ctx = parsed
        with self._state_lock:
            self._by_coin[coin] = (now_ts, ctx)
        return True

    def get_asset_ctx(
        self,
        symbol_or_coin: str,
        *,
        now: Optional[float] = None,
        max_age_sec: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Return a shallow copy of the fresh Hyperliquid ``ctx`` dict or ``None``."""
        coin = canonical_base_asset(symbol_or_coin) or clean_upper_scalar(symbol_or_coin)
        if not coin:
            return None
        eff_now = resolve_now_ts(now)
        with self._state_lock:
            if self._disconnected_since_ts > 0.0 and not self._connected:
                return None
            rec = self._by_coin.get(coin)
        if rec is None:
            return None
        ts_val, ctx = rec
        if not (0.0 <= (eff_now - ts_val) <= max_age_sec):
            return None
        return dict(ctx)

    def status(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        eff_now = resolve_now_ts(now)
        with self._state_lock:
            last_ts = self._last_msg_ts
            conn = self._connected
            ready = len(self._by_coin)
        return {
            "running": bool(self.is_running),
            "connected": bool(conn),
            "tracked": len(self._coins),
            "ready": ready,
            "last_msg_age_sec": round(max(0.0, eff_now - last_ts), 1) if last_ts > 0.0 else None,
        }


class BinanceSpotKline5mStream(JsonPublicWsStreamBase):
    """Live Binance Spot ``@kline_5m`` stream maintaining 3 contiguous 5m buckets (15m Spot CVD)."""

    _default_thread_name = "BinanceSpotKline5mWS"

    def __init__(self, symbols: Sequence[str]) -> None:
        super().__init__()
        self._symbols: Tuple[str, ...] = tuple(
            dict.fromkeys(
                s
                for raw in symbols
                if (s := (clean_symbol_key(raw) or "").upper())
            )
        )
        # symbol -> {open_time_ms: (tot_quote_usd, buy_quote_usd)}
        self._bars: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self._sym_ts: Dict[str, float] = {}

    def _resolve_ws_url(self) -> Optional[str]:
        return BinancePublicWsMixin.build_ws_spot_kline_5m_url(self._symbols)

    def _upsert_bar_locked(
        self,
        sym: str,
        open_ms: int,
        tot_q: float,
        buy_q: float,
        *,
        prefer_higher_vol: bool = False,
    ) -> None:
        bucket = self._bars.setdefault(sym, {})
        clamped_buy = min(tot_q, buy_q)
        if prefer_higher_vol and open_ms in bucket:
            prev_tot, prev_buy = bucket[open_ms]
            if prev_tot >= tot_q:
                return
        bucket[open_ms] = (tot_q, clamped_buy)
        if len(bucket) > _KLINE_5M_MAX_BARS:
            for old_ms in sorted(bucket)[:-_KLINE_5M_MAX_BARS]:
                bucket.pop(old_ms, None)

    def _handle_payload(self, payload: Any, now_ts: float) -> bool:
        parsed = BinancePublicWsMixin.extract_ws_kline_5m_flow(payload)
        if parsed is None:
            return False
        sym, open_ms, tot_q, buy_q = parsed
        with self._state_lock:
            self._upsert_bar_locked(sym, open_ms, tot_q, buy_q, prefer_higher_vol=False)
            self._sym_ts[sym] = now_ts
        return True

    def seed_from_rest_klines(
        self,
        symbol: str,
        raw_klines: Any,
        *,
        now: Optional[float] = None,
    ) -> int:
        """Seed 5m buckets from a REST ``/api/v3/klines?interval=5m&limit=3`` response without regressing newer WS bars."""
        sym = (clean_symbol_key(symbol) or "").upper()
        if not sym or not isinstance(raw_klines, list):
            return 0
        eff_now = resolve_now_ts(now)
        seeded = 0
        with self._state_lock:
            for row in raw_klines[-_KLINE_5M_MAX_BARS:]:
                if not isinstance(row, (list, tuple)) or len(row) < 11:
                    continue
                open_ms = exact_pos_int(row[0], default=None)
                tot_q = safe_nonneg_float(row[7], default=None)
                buy_q = safe_nonneg_float(row[10], default=None)
                if open_ms is None or tot_q is None or buy_q is None:
                    continue
                self._upsert_bar_locked(sym, open_ms, tot_q, buy_q, prefer_higher_vol=True)
                seeded += 1
            if seeded > 0:
                self._sym_ts[sym] = max(self._sym_ts.get(sym, 0.0), eff_now)
        return seeded

    def get_spot_flow_15m(
        self,
        symbol: str,
        *,
        now: Optional[float] = None,
        max_age_sec: float = 30.0,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(spot_cvd_15m_usd, spot_vol_15m_usd)`` from 3 contiguous live 5m bars, else ``None``."""
        sym = (clean_symbol_key(symbol) or "").upper()
        if not sym:
            return None
        eff_now = resolve_now_ts(now)
        with self._state_lock:
            if self._disconnected_since_ts > 0.0 and not self._connected:
                return None
            last_global = self._last_msg_ts
            sym_ts = self._sym_ts.get(sym, 0.0)
            # Require live WS activity (either global combined stream fresh while connected, or recent WS frame for sym)
            global_fresh = last_global > 0.0 and 0.0 <= (eff_now - last_global) <= max_age_sec
            sym_fresh = sym_ts > 0.0 and 0.0 <= (eff_now - sym_ts) <= max_age_sec
            if not (global_fresh or (not self._connected and last_global > 0.0 and sym_fresh)):
                return None
            bucket = self._bars.get(sym)
            if not bucket or len(bucket) < 3:
                return None
            latest_keys = sorted(bucket)[-3:]
            bars = [(k, bucket[k][0], bucket[k][1]) for k in latest_keys]

        # Enforce strict 5m contiguity (detects VPN disconnect gaps > 5m and triggers clean REST re-seed)
        if (bars[1][0] - bars[0][0]) != _KLINE_5M_STEP_MS or (bars[2][0] - bars[1][0]) != _KLINE_5M_STEP_MS:
            return None
        now_ms = eff_now * 1000.0
        newest_lag_ms = now_ms - bars[2][0]
        if newest_lag_ms < -60_000.0 or newest_lag_ms > _KLINE_5M_FRESH_LAG_MS:
            return None

        tot_quote = sum(b[1] for b in bars)
        buy_quote = sum(min(b[1], b[2]) for b in bars)
        return round(2.0 * buy_quote - tot_quote, 0), round(tot_quote, 0)

    def status(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        eff_now = resolve_now_ts(now)
        with self._state_lock:
            last_ts = self._last_msg_ts
            conn = self._connected
            ready_3bars = sum(1 for b in self._bars.values() if len(b) >= 3)
        return {
            "running": bool(self.is_running),
            "connected": bool(conn),
            "tracked": len(self._symbols),
            "ready": ready_3bars,
            "last_msg_age_sec": round(max(0.0, eff_now - last_ts), 1) if last_ts > 0.0 else None,
        }


class TelemetryWsHub:
    """Lifecycle and 0-ms read facade over the 3 lightweight stateless telemetry WebSocket streams."""

    def __init__(self, symbols: Sequence[str]) -> None:
        fut_syms: List[str] = []
        hl_coins: List[str] = []
        bn_spot_syms: List[str] = []
        seen_fut: set[str] = set()
        seen_hl: set[str] = set()
        seen_bn: set[str] = set()

        for raw_s in symbols:
            ident = resolve_symbol_identity(raw_s)
            if ident.is_rth_equity or ident.is_tokenized_equity:
                continue
            ven = (venue_for_symbol(raw_s, default="") or "").strip().lower()
            base = canonical_base_asset(ident.wire_symbol)
            if not base:
                continue
            if ident.has_crypto_perpetuals:
                if MarketDataProvider.has_native_l1_derivatives(ven):
                    if base not in seen_hl:
                        seen_hl.add(base)
                        hl_coins.append(base)
                else:
                    fut_s = resolve_binance_lookup_symbol(raw_s)
                    if fut_s and fut_s not in seen_fut:
                        seen_fut.add(fut_s)
                        fut_syms.append(fut_s)
            for spec_fn in (
                MarketDataProvider.resolve_spot_flow_15m_spec,
                MarketDataProvider.resolve_agg_spot_flow_15m_spec,
            ):
                req_spec, _ = spec_fn(raw_s, venue=ven, now_ts=0.0)
                if req_spec is None:
                    continue
                bn_sym = BinancePublicWsMixin.extract_kline_5m_symbol_from_url(req_spec[0])
                if bn_sym and bn_sym not in seen_bn:
                    seen_bn.add(bn_sym)
                    bn_spot_syms.append(bn_sym)

        self.mark_price_stream: Optional[BinanceFuturesMarkPriceStream] = (
            BinanceFuturesMarkPriceStream(fut_syms) if fut_syms else None
        )
        self.hl_asset_ctx_stream: Optional[HyperliquidAssetCtxStream] = (
            HyperliquidAssetCtxStream(hl_coins) if hl_coins else None
        )
        self.binance_spot_kline_stream: Optional[BinanceSpotKline5mStream] = (
            BinanceSpotKline5mStream(bn_spot_syms) if bn_spot_syms else None
        )
        self._lock = threading.Lock()

    def _all_streams(self) -> List[JsonPublicWsStreamBase]:
        return [
            st
            for st in (self.mark_price_stream, self.hl_asset_ctx_stream, self.binance_spot_kline_stream)
            if st is not None
        ]

    def start(self) -> "TelemetryWsHub":
        with self._lock:
            for st in self._all_streams():
                try:
                    st.start()
                except Exception as exc:
                    logger.debug("Could not start %s: %s", st.__class__.__name__, exc)
        return self

    def stop(self, timeout: float = 3.0) -> None:
        with self._lock:
            streams = self._all_streams()
        for st in streams:
            try:
                st.request_stop()
            except Exception:
                pass
        for st in streams:
            try:
                st.stop(timeout=timeout)
            except Exception:
                pass

    def get_binance_mark_price(
        self, symbol: str, *, now: Optional[float] = None, max_age_sec: float = 15.0
    ) -> Optional[Dict[str, float]]:
        st = self.mark_price_stream
        return st.get_mark_price(symbol, now=now, max_age_sec=max_age_sec) if st is not None else None

    def get_hyperliquid_asset_ctx(
        self, symbol: str, *, now: Optional[float] = None, max_age_sec: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        st = self.hl_asset_ctx_stream
        return st.get_asset_ctx(symbol, now=now, max_age_sec=max_age_sec) if st is not None else None

    def get_binance_spot_flow_15m(
        self, symbol: str, *, now: Optional[float] = None, max_age_sec: float = 30.0
    ) -> Optional[Tuple[float, float]]:
        st = self.binance_spot_kline_stream
        return st.get_spot_flow_15m(symbol, now=now, max_age_sec=max_age_sec) if st is not None else None

    def seed_binance_spot_bars(
        self, symbol: str, raw_klines: Any, *, now: Optional[float] = None
    ) -> int:
        st = self.binance_spot_kline_stream
        return st.seed_from_rest_klines(symbol, raw_klines, now=now) if st is not None else 0

    def status(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        eff_now = resolve_now_ts(now)
        out: Dict[str, Any] = {}
        if self.mark_price_stream is not None:
            out["binance_mark_price"] = self.mark_price_stream.status(now=eff_now)
        if self.hl_asset_ctx_stream is not None:
            out["hyperliquid_asset_ctx"] = self.hl_asset_ctx_stream.status(now=eff_now)
        if self.binance_spot_kline_stream is not None:
            out["binance_spot_kline_5m"] = self.binance_spot_kline_stream.status(now=eff_now)
        return out
