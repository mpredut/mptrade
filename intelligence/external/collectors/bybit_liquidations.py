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
from typing import Deque, Dict, List, Optional, Tuple

import websockets

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
        self._symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        self._retention = retention_sec
        # symbol -> deque[(ts, "LONG"|"SHORT", notional_usd)]
        self._events: Dict[str, Deque[Tuple[float, str, float]]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._retry = WS_RETRY_INITIAL

    def start(self, name: str = "BybitLiquidationWS", daemon: bool = True) -> "BybitLiquidationCollector":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name=name, daemon=daemon)
        self._thread.start()
        logger.info("Bybit liquidation collector started for %s", ",".join(self._symbols))
        return self

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

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
        elapsed = 0.0
        while elapsed < delay and not self._stop.is_set():
            await asyncio.sleep(min(step, delay - elapsed))
            elapsed += step

    def _ingest(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        topic = msg.get("topic", "")
        if not topic.startswith("allLiquidation."):
            return
        now = time.time()
        for item in msg.get("data", []) or []:
            try:
                symbol = item["s"]
                # Bybit: Buy = long liquidated, Sell = short liquidated
                side = "LONG" if item["S"] == "Buy" else "SHORT"
                notional = float(item["v"]) * float(item["p"])
            except (KeyError, ValueError, TypeError):
                continue
            with self._lock:
                dq = self._events.setdefault(symbol, deque())
                dq.append((now, side, notional))
                cutoff = now - self._retention
                while dq and dq[0][0] < cutoff:
                    dq.popleft()

    def record_event(self, symbol: str, side: str, notional_usd: float, ts: Optional[float] = None) -> None:
        """Inject event directly (useful for tests and synthetic feeds)."""
        now = ts or time.time()
        side_norm = "LONG" if side.upper() in ("LONG", "BUY") else "SHORT"
        with self._lock:
            dq = self._events.setdefault(symbol.upper(), deque())
            dq.append((now, side_norm, float(notional_usd)))
            cutoff = now - self._retention
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def get_summary(self, symbol: str, window_sec: float = 300.0) -> LiquidationSummary:
        """Aggregate liquidation metrics for symbol over window_sec."""
        symbol = symbol.upper()
        now = time.time()
        cutoff = now - window_sec
        long_usd = short_usd = 0.0
        events_n = 0
        last_ts = 0.0

        with self._lock:
            items = list(self._events.get(symbol, ()))

        for ts, side, notional in items:
            if ts < cutoff:
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
            window_sec=window_sec,
            long_liq_usd=long_usd,
            short_liq_usd=short_usd,
            net_usd=short_usd - long_usd,
            capitulation_ratio=cap_ratio,
            squeeze_ratio=sq_ratio,
            events_count=events_n,
            last_event_ts=last_ts,
        )
