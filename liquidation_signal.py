"""
Liquidation signal from Bybit (free, public, no API key)
========================================================

Streams Bybit USDT-perp forced-liquidation orders and turns them into a rolling
per-symbol "capitulation" signal for the spot strategies (DCA / trailing re-buy):
a burst of LONG liquidations means longs are being force-closed (forced selling =
a panic dip), which is exactly the bottom the trailing re-buy wants to catch. A
burst of SHORT liquidations means shorts are being squeezed up.

Why Bybit and not Binance / Coinglass:
  - Coinglass' liquidation map/heatmap is a paid endpoint (Professional, $699/mo).
  - Binance's free futures liquidation stream (fstream !forceOrder@arr) is
    GEO-BLOCKED from the Romanian ISP (verified: even aggTrade returns nothing
    from a direct RO IP). It might work through the PIA exit, but that couples the
    signal to the (currently flaky) VPN.
  - Bybit's public stream works from a direct RO IP, so the signal is FREE and
    DECOUPLED from PIA. Liquidations are market-wide, so Bybit's BTC liquidations
    are a fine proxy for "is BTC cascading right now" regardless of which venue
    or quote currency (USDC/USDT) the fleet actually trades.

Source: wss://stream.bybit.com/v5/public/linear , topic allLiquidation.<symbol>.

Bybit side semantics (IMPORTANT, opposite of Binance): the `S` field is the
liquidated POSITION side -- "Buy" = a LONG was liquidated (forced selling, down),
"Sell" = a SHORT was liquidated (squeeze, up). Internally we normalise to
LONG / SHORT so the rest of the code is unambiguous.

Caveat: Bybit is one venue; use this as a RELATIVE burst/direction signal (see
capitulation_ratio), not an exact market-wide USD figure.

Observe only: this module produces a signal, it never places orders. Run it
standalone to watch it live before we wire it into the strategies:

    python liquidation_signal.py BTCUSDT TAOUSDT ETHUSDT
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import websockets

logger = logging.getLogger("liquidation")

BYBIT_URL = "wss://stream.bybit.com/v5/public/linear"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

MAX_RETENTION_SEC = 3600.0     # Keep raw events at most this long; windows are subsets.
WS_RECV_TIMEOUT = 5.0
WS_APP_PING_SEC = 20.0         # Bybit closes idle connections; send {"op":"ping"} periodically.
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_CLOSE_TIMEOUT = 2
WS_RETRY_INITIAL = 1.0
WS_RETRY_MAX = 60.0
STABLE_SESSION_SEC = 60.0       # Reset backoff only after a session stays up this long.

_LONG = "LONG"     # a long position was liquidated -> forced selling (down / panic)
_SHORT = "SHORT"   # a short position was liquidated -> forced buying (up / squeeze)


class LiquidationSignal:
    """Background thread holding a rolling window of Bybit liquidations and
    exposing an aggregated per-symbol signal. Thread-safe."""

    def __init__(self, symbols: Optional[List[str]] = None,
                 retention_sec: float = MAX_RETENTION_SEC):
        self._symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        self._retention = retention_sec
        # symbol -> deque[(ts, LONG|SHORT, notional_usd)]
        self._events: Dict[str, Deque[Tuple[float, str, float]]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._retry = WS_RETRY_INITIAL

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self, name: str = "LiquidationWS", daemon: bool = True) -> "LiquidationSignal":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name=name, daemon=daemon)
        self._thread.start()
        logger.info("liquidation signal started (%s)", ",".join(self._symbols))
        return self

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── worker / reconnect ───────────────────────────────────────────────────
    def _worker(self) -> None:
        try:
            asyncio.run(self._run_with_reconnect())
        except Exception as e:  # defensive: never let the thread die silently
            logger.exception("liquidation thread crashed: %s", e)
        finally:
            logger.info("liquidation thread exited")

    async def _run_with_reconnect(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                await self._session()
            except Exception as e:
                logger.warning("liquidation WS error: %s", e)
            if self._stop.is_set():
                break
            if time.time() - t0 >= STABLE_SESSION_SEC:
                self._retry = WS_RETRY_INITIAL
            logger.info("liquidation WS reconnect in %.1fs", self._retry)
            await self._sleep(self._retry)
            self._retry = min(self._retry * 2, WS_RETRY_MAX)

    async def _session(self) -> None:
        async with websockets.connect(
            BYBIT_URL, ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT, close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            args = [f"allLiquidation.{s}" for s in self._symbols]
            await ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.info("liquidation subscribed: %s", ", ".join(args))
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
                    logger.info("liquidation stream closed by server; reconnecting")
                    return
                self._ingest(raw)

    async def _sleep(self, delay: float, step: float = 0.2) -> None:
        elapsed = 0.0
        while elapsed < delay and not self._stop.is_set():
            await asyncio.sleep(min(step, delay - elapsed))
            elapsed += step

    # ── ingest / query ───────────────────────────────────────────────────────
    def _ingest(self, raw: str) -> None:
        # Liquidation msg: {"topic":"allLiquidation.BTCUSDT","type":"snapshot","ts":..,
        #                   "data":[{"T":..,"s":"BTCUSDT","S":"Buy","v":"0.5","p":"77000"}]}
        # Subscribe acks / pongs are ignored (no "allLiquidation." topic).
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
                # Bybit: "Buy" = long liquidated, "Sell" = short liquidated.
                side = _LONG if item["S"] == "Buy" else _SHORT
                notional = float(item["v"]) * float(item["p"])
            except (KeyError, ValueError, TypeError):
                continue
            with self._lock:
                dq = self._events.setdefault(symbol, deque())
                dq.append((now, side, notional))
                cutoff = now - self._retention
                while dq and dq[0][0] < cutoff:
                    dq.popleft()

    def get_signal(self, symbol: str, window_sec: float = 300.0) -> Dict[str, float]:
        """Aggregate liquidations for ``symbol`` over the last ``window_sec``.

        long_liq_usd  = longs liquidated (forced selling -> downward / panic)
        short_liq_usd = shorts liquidated (forced buying -> upward squeeze)
        net_usd       = short_liq_usd - long_liq_usd  (>0 squeeze up, <0 capitulation)
        """
        symbol = symbol.upper()
        now = time.time()
        cutoff = now - window_sec
        long_usd = short_usd = 0.0
        long_n = short_n = 0
        last_ts = 0.0
        with self._lock:
            items = list(self._events.get(symbol, ()))
        for ts, side, notional in items:
            if ts < cutoff:
                continue
            last_ts = max(last_ts, ts)
            if side == _LONG:
                long_usd += notional
                long_n += 1
            else:
                short_usd += notional
                short_n += 1
        return {
            "symbol": symbol,
            "window_sec": window_sec,
            "long_liq_usd": long_usd,
            "short_liq_usd": short_usd,
            "long_count": long_n,
            "short_count": short_n,
            "net_usd": short_usd - long_usd,
            "age_sec": (now - last_ts) if last_ts else float("inf"),
        }

    def capitulation_ratio(self, symbol: str,
                           window_sec: float = 120.0,
                           baseline_sec: float = 3600.0) -> float:
        """Intensity of the recent LONG-liquidation (panic-selling) burst relative
        to the average rate over ``baseline_sec``. >1 = above average; a threshold
        like >=3 is a reasonable "capitulation" trigger, but the strategy owns the
        final number. Returns 0.0 until there is enough baseline history."""
        recent = self.get_signal(symbol, window_sec)["long_liq_usd"]
        base = self.get_signal(symbol, baseline_sec)["long_liq_usd"]
        recent_rate = recent / max(window_sec, 1.0)
        base_rate = base / max(baseline_sec, 1.0)
        return recent_rate / base_rate if base_rate > 0 else 0.0


# Shared lazy singleton, mirroring bapi_ws.get_ws_manager().
_singleton: Optional[LiquidationSignal] = None
_singleton_lock = threading.Lock()


def get_liquidation_signal(symbols: Optional[List[str]] = None) -> LiquidationSignal:
    """Return the shared liquidation signal, starting its stream on first use.
    ``symbols`` is only honoured on the first call that creates the singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = LiquidationSignal(symbols=symbols)
        if not _singleton.is_running:
            _singleton.start()
    return _singleton


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s")
    watch = [s.upper() for s in sys.argv[1:]] or DEFAULT_SYMBOLS
    sig = get_liquidation_signal(symbols=watch)
    print(f"Watching Bybit liquidations for: {', '.join(watch)}  (Ctrl-C to stop)")
    print("long(panic sell) = longs liquidated; short(squeeze) = shorts liquidated\n")
    try:
        while True:
            time.sleep(10)
            for s in watch:
                w60 = sig.get_signal(s, 60)
                w300 = sig.get_signal(s, 300)
                ratio = sig.capitulation_ratio(s)
                print(
                    f"{s:10s} 60s long=${w60['long_liq_usd']:>12,.0f} "
                    f"short=${w60['short_liq_usd']:>12,.0f} | "
                    f"5m long=${w300['long_liq_usd']:>13,.0f} | "
                    f"capitulation x{ratio:.1f}")
            print("")
    except KeyboardInterrupt:
        sig.stop()
        print("stopped")
