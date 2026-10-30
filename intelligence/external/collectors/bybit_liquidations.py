"""Bybit liquidation WebSocket stream collector.

Streams public Bybit USDT-perp forced liquidations in real-time.
Decoupled, zero-credential public endpoint accessible without API keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import websockets

from botcore import safe_float, safe_nonneg_float, safe_pos_float
from cachepaths import cache_symbol_variants
from instrument_registry import casefold_get
from intelligence.macro_sentiment_ssot import clean_symbol_key, parse_symbol_list
from providers.base import normalize_side

logger = logging.getLogger("intelligence.external.bybit_liquidations")

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

MAX_RETENTION_SEC = 3600.0
WS_RECV_TIMEOUT = 5.0
WS_APP_PING_SEC = 20.0
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_CLOSE_TIMEOUT = 2
WS_RETRY_INITIAL = 1.0
WS_RETRY_MAX = 60.0
STABLE_SESSION_SEC = 60.0


def _normalize_liq_side(raw_side: Any) -> Optional[str]:
    norm = normalize_side(raw_side)
    if norm == "BUY":
        return "LONG"
    if norm == "SELL":
        return "SHORT"
    return None


@dataclass(frozen=True)
class LiquidationSummary:
    """Aggregated liquidation metrics for a time window."""

    symbol: str
    window_sec: float
    long_liq_usd: float
    short_liq_usd: float
    net_usd: float
    capitulation_ratio: float
    squeeze_ratio: float
    events_count: int
    last_event_ts: float

    @property
    def total_usd(self) -> float:
        return self.long_liq_usd + self.short_liq_usd


class BybitLiquidationCollector:
    """Thread-safe background collector tracking rolling liquidation volumes from Bybit."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        retention_sec: float = MAX_RETENTION_SEC,
    ) -> None:
        self._symbols = parse_symbol_list(symbols) or list(DEFAULT_SYMBOLS)
        self._retention = safe_pos_float(retention_sec, default=MAX_RETENTION_SEC) or MAX_RETENTION_SEC
        # symbol -> deque[(ts, "LONG"|"SHORT", notional_usd)]
        self._events: Dict[str, Deque[Tuple[float, str, float]]] = {}
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._retry = WS_RETRY_INITIAL

    def start(self, name: str = "BybitLiquidationWS", daemon: bool = True) -> "BybitLiquidationCollector":
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return self
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name=name, daemon=daemon)
            self._thread.start()
        logger.info("Bybit liquidation collector started for %s", ",".join(self._symbols))
        return self

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        t_sec = safe_nonneg_float(timeout, default=8.0) or 8.0
        if (
            self._thread
            and self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=t_sec)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _worker(self) -> None:
        try:
            asyncio.run(self._run_with_reconnect())
        except Exception as e:
            logger.exception("Liquidation worker crash: %s", e)
        finally:
            logger.info("Liquidation worker exited")

    async def _run_with_reconnect(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                await self._session()
            except Exception as e:
                logger.warning("Bybit liquidation WS error: %s", e)
            if self._stop.is_set():
                break
            if time.time() - t0 >= STABLE_SESSION_SEC:
                self._retry = WS_RETRY_INITIAL
            logger.info("Bybit liquidation reconnecting in %.1fs", self._retry)
            await self._sleep(self._retry)
            self._retry = min(self._retry * 2, WS_RETRY_MAX)

    async def _session(self) -> None:
        async with websockets.connect(
            BYBIT_WS_URL,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            args = [f"allLiquidation.{s}" for s in self._symbols]
            await ws.send(json.dumps({"op": "subscribe", "args": args}))
            last_ping = time.time()
            while not self._stop.is_set():
                now = time.time()
                if now < last_ping:
                    last_ping = now
                if now - last_ping >= WS_APP_PING_SEC:
                    await ws.send(json.dumps({"op": "ping"}))
                    last_ping = now
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    return
                self._ingest(raw)

    async def _sleep(self, delay: float, step: float = 0.2) -> None:
        eff_delay = safe_pos_float(delay, 0.0)
        eff_step = safe_pos_float(step, 0.2)
        elapsed = 0.0
        while elapsed < eff_delay and not self._stop.is_set():
            await asyncio.sleep(min(eff_step, eff_delay - elapsed))
            elapsed += eff_step

    def _ingest(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(msg, dict):
            return
        topic = str(msg.get("topic") or "")
        if not topic.startswith("allLiquidation."):
            return
        now = time.time()
        data_items = msg.get("data")
        if not isinstance(data_items, list):
            return
        for item in data_items:
            if not isinstance(item, dict):
                continue
            v = safe_pos_float(item.get("v"), default=None)
            p = safe_pos_float(item.get("p"), default=None)
            if v is not None and p is not None:
                notional = safe_pos_float(v * p, default=None)
                if notional is not None:
                    self.record_event(item.get("s"), item.get("S"), notional, ts=now)

    def record_event(self, symbol: str, side: str, notional_usd: float, ts: Optional[float] = None) -> None:
        """Inject event directly (useful for tests and synthetic feeds)."""
        notional = safe_pos_float(notional_usd, default=None)
        if notional is None or not isinstance(symbol, str):
            return
        sym_norm = clean_symbol_key(symbol)
        if not sym_norm:
            return
        side_norm = _normalize_liq_side(side)
        if side_norm is None:
            return
        now = safe_pos_float(ts, default=None) if ts is not None else time.time()
        if now is None:
            return
        with self._lock:
            dq = self._events.setdefault(sym_norm, deque())
            dq.append((now, side_norm, notional))
            cutoff = now - self._retention
            while dq and (safe_float(dq[0][0], default=None) is None or dq[0][0] < cutoff):
                dq.popleft()

    def get_summary(
        self,
        symbol: str,
        window_sec: float = 300.0,
        now: Optional[float] = None,
    ) -> LiquidationSummary:
        """Aggregate liquidation metrics for symbol over window_sec."""
        symbol = clean_symbol_key(symbol)
        now_ts = safe_pos_float(now, default=None) or time.time()
        win_sec = safe_pos_float(window_sec, default=300.0) or 300.0
        cutoff = now_ts - win_sec
        long_usd = short_usd = 0.0
        events_n = 0
        last_ts = 0.0

        with self._lock:
            items = list(self._events.get(symbol, ()))
            if not items:
                for cand in cache_symbol_variants(symbol):
                    cand_k = clean_symbol_key(cand)
                    if cand_k and cand_k in self._events:
                        items = list(self._events[cand_k])
                        break

        for ts, side, notional in items:
            if safe_float(ts, default=None) is None or safe_pos_float(notional, default=None) is None:
                continue
            if ts < cutoff or ts > (now_ts + 5.0):
                continue
            last_ts = max(last_ts, ts)
            events_n += 1
            if side == "LONG":
                long_usd += notional
            else:
                short_usd += notional

        total = long_usd + short_usd
        cap_ratio = long_usd / total if total > 0 else 0.5
        sq_ratio = short_usd / total if total > 0 else 0.5

        return LiquidationSummary(
            symbol=symbol,
            window_sec=win_sec,
            long_liq_usd=long_usd,
            short_liq_usd=short_usd,
            net_usd=short_usd - long_usd,
            capitulation_ratio=cap_ratio,
            squeeze_ratio=sq_ratio,
            events_count=events_n,
            last_event_ts=last_ts,
        )

    def get_signal(self, symbol: str, window_sec: float = 300.0) -> Dict[str, float]:
        """Aggregate liquidations for ``symbol`` over the last ``window_sec`` as a dict."""
        symbol_str = str(symbol or "").strip().upper()
        sym_clean = clean_symbol_key(symbol_str)
        eff_win = safe_pos_float(window_sec, 300.0)
        now = time.time()
        cutoff = now - eff_win
        long_usd = short_usd = 0.0
        long_n = short_n = 0
        last_ts = 0.0
        with self._lock:
            items = list(casefold_get(self._events, sym_clean) or casefold_get(self._events, symbol_str) or ())
            if not items and symbol_str:
                for cand in cache_symbol_variants(
                    symbol_str, include_original=True, quotes=("USDT", "USDC", "USD")
                ):
                    cand_items = casefold_get(self._events, cand)
                    if cand_items:
                        items = list(cand_items)
                        break
        for ts, side, notional in items:
            if safe_float(ts, default=None) is None or ts < cutoff:
                continue
            last_ts = max(last_ts, ts)
            if side == "LONG":
                long_usd += notional
                long_n += 1
            else:
                short_usd += notional
                short_n += 1
        return {
            "symbol": symbol_str,
            "window_sec": eff_win,
            "long_liq_usd": long_usd,
            "short_liq_usd": short_usd,
            "long_count": long_n,
            "short_count": short_n,
            "net_usd": short_usd - long_usd,
            "age_sec": max(0.0, now - last_ts) if last_ts else float("inf"),
        }

    def capitulation_ratio(
        self,
        symbol: str,
        window_sec: float = 120.0,
        baseline_sec: float = 3600.0,
    ) -> float:
        """Intensity of recent LONG-liquidation burst relative to baseline rate."""
        eff_win = safe_pos_float(window_sec, 120.0)
        eff_base = max(safe_pos_float(baseline_sec, 3600.0), eff_win)
        recent = self.get_signal(symbol, eff_win)["long_liq_usd"]
        base = self.get_signal(symbol, eff_base)["long_liq_usd"]
        recent_rate = recent / max(eff_win, 1.0)
        base_rate = base / max(eff_base, 1.0)
        return recent_rate / base_rate if base_rate > 0 else 0.0
