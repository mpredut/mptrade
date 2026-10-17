"""
Binance WebSocket — shared infrastructure and two streams
==========================================================

BinanceWSBase
    Shared thread and asyncio loop, reconnect backoff, clean shutdown, and
    interruptible sleep. Subclasses implement only ``_connect_and_run``.

BinancePriceStream       (public, market data)
    Combined stream: one socket, one thread, N ticker symbols, dynamic subscriptions.
    Historical alias: ``BinanceWebSocketManager``.
    URL: wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/...

BinanceAccountStream     (private user-data execution reports)
    WS-API with Ed25519 ``session.logon`` and keepalive. Events are delivered through
    callbacks, avoiding a dependency on ``cacheManager`` and its import cycle.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
import websockets

from typing import Dict, Optional, Set, Callable

import symbols as sym
import utils as u

try:
    from keys.apikeys import api_key_ws
except (ModuleNotFoundError, ImportError):
    # Read-only and test imports must not require secrets. The private stream
    # refuses to start below when the key is unavailable.
    api_key_ws = os.environ.get("BINANCE_API_KEY_WS", "")

logger = logging.getLogger("binance.ws")

# ─── Constants ────────────────────────────────────────────────────────────────

WS_BASE_URL      = "wss://stream.binance.com:9443/stream"   # market data
WS_API_URL       = "wss://ws-api.binance.com:443/ws-api/v3"  # user-data (auth)
WS_MAX_STREAMS   = 1024        # Binance limit per connection
WS_RECV_TIMEOUT  = 3.0         # Must be lower than WS_STOP_TIMEOUT (market).
WS_USERDATA_RECV_TIMEOUT = 30.0
WS_STOP_TIMEOUT  = 8.0
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT  = 10
WS_CLOSE_TIMEOUT = 2
WS_RETRY_INITIAL = 1.0
WS_RETRY_MAX     = 60.0
WS_USERDATA_KEEPALIVE_SEC = 30 * 60
WS_USERDATA_LOSS_TIMEOUT_SEC = 40


class _Cmd:
    SUBSCRIBE   = "SUBSCRIBE"
    UNSUBSCRIBE = "UNSUBSCRIBE"


# ══════════════════════════════════════════════════════════════════════════════
# Shared base: thread, asyncio, reconnect backoff, and clean shutdown.
# ══════════════════════════════════════════════════════════════════════════════

class BinanceWSBase:
    # Reset backoff only after a session remains stable for this duration.
    # Prevent repeated logins from causing a reconnect storm and hitting rate limits.
    STABLE_SESSION_SEC = 60.0

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._retry_delay = WS_RETRY_INITIAL
        self._start_lock = threading.Lock()

    # Thread lifecycle.
    def start(self, name: str = "BinanceWS", daemon: bool = True) -> "BinanceWSBase":
        with self._start_lock:          # [F2] concurrent start() calls are safe
            if self.is_running:
                logger.info("%s already running", name)
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._thread_worker, name=name, daemon=daemon)
            self._thread.start()
        logger.info("%s started", name)
        return self

    def stop(self, timeout: float = WS_STOP_TIMEOUT) -> bool:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("WS thread did not stop within %.1fs", timeout)
                return False
        return True

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _thread_worker(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as e:
            if not self._is_shutdown_error(e):
                logger.exception("WS thread crashed: %s", e)
        finally:
            logger.info("WS thread exited")

    # Generic reconnect loop.
    async def _main(self) -> None:
        """Reconnect around ``_connect_and_run`` by default.

        Subclasses may override this when setup must occur in the correct event
        loop, such as creation of an asynchronous queue.
        """
        await self._run_with_reconnect(self._connect_and_run)

    async def _run_with_reconnect(self, body: Callable) -> None:
        # Reconnect with backoff. Sleep for ``_retry_delay`` between every attempt,
        # including clean disconnects, as the rate-limit safeguard
        # to prevent reconnect storms. Reset backoff only after a stable session;
        # repeated flapping therefore increases the delay.
        while not self._stop_event.is_set():
            t0 = time.time()
            exc = None
            try:
                await body()                          # Return or raise when the session ends.
            except Exception as e:
                exc = e
                if self._is_shutdown_error(e):
                    break
                logger.warning("WS error: %s", e)
            self._on_session_end(exc)
            if self._stop_event.is_set():
                break
            if time.time() - t0 >= self.STABLE_SESSION_SEC:
                self._reset_backoff()                 # A stable session restores the short delay.
            logger.info("WS reconnect in %.1fs", self._retry_delay)
            await self._interruptible_sleep(self._retry_delay)
            self._retry_delay = min(self._retry_delay * 2, WS_RETRY_MAX)
        logger.info("WS reconnect loop stopped")

    def _reset_backoff(self) -> None:
        self._retry_delay = WS_RETRY_INITIAL

    def _on_session_end(self, exc: Optional[Exception]) -> None:
        """Handle session completion; ``exc`` is None after a clean termination."""
        pass

    async def _connect_and_run(self) -> None:
        """Run one complete connection; subclasses return or raise on disconnect."""
        raise NotImplementedError

    # ─── Shared helpers ──────────────────────────────────────────────────────
    async def _interruptible_sleep(self, delay: float, step: float = 0.2) -> None:
        elapsed = 0.0
        while elapsed < delay and not self._stop_event.is_set():
            await asyncio.sleep(min(step, delay - elapsed))
            elapsed += step

    @staticmethod
    def _is_shutdown_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, RuntimeError) and (
            "cannot schedule new futures after shutdown" in msg
            or "event loop is closed" in msg
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Market data — Combined Stream (public)
# ══════════════════════════════════════════════════════════════════════════════

class BinancePriceStream(BinanceWSBase):
    """
    Public market-data stream: live prices through one Combined Stream thread,
    one WebSocket, and N multiplexed symbols. Symmetric with BinanceAccountStream.

    Public API:
        start()/stop(), add_symbol(s)/remove_symbol(s), get_price(s),
        get_all_prices(), subscribe(sub)/unsubscribe(sub), running_symbols.
    """

    def __init__(self, symbols: Optional[list] = None):
        super().__init__()
        self._prices: Dict[str, float] = {}
        self._subscribed: Set[str]     = set()
        self._lock = threading.Lock()
        self._cmd_queue: Optional[asyncio.Queue] = None
        self._req_id = 0
        self._subscribers = []
        if symbols:
            for s in symbols:
                self._subscribed.add(s.upper())

    def start(self, name: str = "BinanceWS", daemon: bool = False) -> "BinancePriceStream":
        # Preserve the original non-daemon market-manager thread behavior.
        super().start(name=name, daemon=daemon)
        return self

    # ─── Subscription management (observers) ─────────────────────────────────
    def subscribe(self, subscriber) -> None:
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _notify_subscribers(self, symbol: str, items) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub.on_items_update(symbol, items)
            except Exception as e:
                logger.error("Subscriber notify error: %s", e)

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _stream_name(self, symbol: str) -> str:
        return f"{symbol.lower()}@ticker"

    def _build_url(self, symbols: Set[str]) -> str:
        streams = "/".join(self._stream_name(s) for s in sorted(symbols))
        return f"{WS_BASE_URL}?streams={streams}"

    # ─── Async core ───────────────────────────────────────────────────────────
    async def _main(self) -> None:
        self._cmd_queue = asyncio.Queue()   # Create it in the event loop that uses it.
        await self._run_with_reconnect(self._connect_and_run)

    async def _connect_and_run(self) -> None:
        # Wait for symbols here rather than treating the wait as a session. This avoids
        # increasing backoff before a real connection session occurs.
        while not self._stop_event.is_set():
            with self._lock:
                current_symbols = set(self._subscribed)
            if current_symbols:
                break
            await self._interruptible_sleep(1.0)
        if self._stop_event.is_set() or not current_symbols:
            return
        url = self._build_url(current_symbols)
        logger.info("Connecting to combined stream (%d symbols)...", len(current_symbols))
        async with websockets.connect(
            url, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT,
            close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            logger.info("Connected. Streams active: %d", len(current_symbols))
            await self._session(ws)             # ``_run_with_reconnect`` manages backoff.

    async def _session(self, ws) -> None:
        """Run receive and command loops concurrently within one session.

        When either task terminates, cancel both so ``_connect_and_run`` reconnects.
        """
        recv_task = asyncio.create_task(self._recv_loop(ws))
        cmd_task  = asyncio.create_task(self._cmd_loop(ws))
        done, pending = await asyncio.wait(
            [recv_task, cmd_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for task in done:
            exc = task.exception()
            if exc and not self._stop_event.is_set():
                logger.warning("Session task ended with: %s", exc)

    async def _recv_loop(self, ws) -> None:
        while not self._stop_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                self._process_message(raw)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server, reconnecting...")
                break
            except json.JSONDecodeError as e:
                logger.warning("JSON decode error: %s", e)
                continue

    async def _cmd_loop(self, ws) -> None:
        while not self._stop_event.is_set():
            try:
                cmd = await asyncio.wait_for(self._cmd_queue.get(), timeout=WS_RECV_TIMEOUT)
            except asyncio.TimeoutError:
                continue
            payload = json.dumps({
                "method": cmd["method"], "params": [cmd["stream"]], "id": self._next_id()})
            try:
                await ws.send(payload)
                logger.debug("%s: %s", cmd["method"], cmd["stream"])
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection lost while sending %s for %s", cmd["method"], cmd["stream"])
                break
            except Exception as e:
                logger.error("Failed to send %s: %s", cmd["method"], e)

    def _process_message(self, raw: str) -> None:
        """{"stream": "btcusdc@ticker", "data": {"s": "BTCUSDC", "c": "65432.10", ...}}"""
        try:
            envelope = json.loads(raw)
            if "result" in envelope:                    # subscribe/unsubscribe acknowledgment
                logger.debug("WS ack: %s", envelope)
                return
            data   = envelope.get("data", envelope)
            symbol = data["s"]
            price  = float(data["c"])
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"invalid ticker price: {price!r}")
            raw_ts = data.get("E") or data.get("C")
            with self._lock:
                self._prices[symbol] = price
            items = [price, int(raw_ts)] if raw_ts is not None else [price]
            self._notify_subscribers(symbol, items)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning("process_message error: %s | raw: %s", e, raw[:120])

    # ─── Enqueue command (thread-safe) ────────────────────────────────────────
    def _enqueue_cmd(self, method: str, symbol: str) -> None:
        if self._cmd_queue is None:
            logger.warning("Manager not started, cannot send %s for %s", method, symbol)
            return
        self._cmd_queue.put_nowait({"method": method, "stream": self._stream_name(symbol)})

    # ─── Public API ───────────────────────────────────────────────────────────
    def add_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._subscribed:
                logger.debug("[%s] Already subscribed", symbol)
                return False
            if len(self._subscribed) >= WS_MAX_STREAMS:
                logger.warning("Max streams (%d) reached, cannot add %s", WS_MAX_STREAMS, symbol)
                return False
            self._subscribed.add(symbol)
        self._enqueue_cmd(_Cmd.SUBSCRIBE, symbol)
        return True

    def remove_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._subscribed:
                logger.debug("[%s] Not subscribed", symbol)
                return False
            self._subscribed.discard(symbol)
            self._prices.pop(symbol, None)
        self._enqueue_cmd(_Cmd.UNSUBSCRIBE, symbol)
        return True

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol.upper())

    def get_all_prices(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._prices)

    @property
    def running_symbols(self) -> Set[str]:
        with self._lock:
            return set(self._subscribed)


# ══════════════════════════════════════════════════════════════════════════════
#  User data — private WS-API session.logon and execution reports
# ══════════════════════════════════════════════════════════════════════════════

class BinanceAccountStream(BinanceWSBase):
    """
    Authenticated user-data stream. Events are delivered through callbacks so the
    stream does not depend on ``cacheManager`` or create an import cycle:
        on_event(payload)      — real event (executionReport, etc.)
        on_available(bool)     — WebSocket available (key and library present)
        on_healthy()           — a live signal (event/ping) was received
        on_unhealthy()         — lost connection or expired watchdog
    """

    def __init__(self, on_event: Callable,
                 on_available: Optional[Callable] = None,
                 on_healthy: Optional[Callable] = None,
                 on_unhealthy: Optional[Callable] = None,
                 keepalive_sec: float = WS_USERDATA_KEEPALIVE_SEC,
                 loss_timeout_sec: float = WS_USERDATA_LOSS_TIMEOUT_SEC):
        super().__init__()
        self.on_event = on_event
        self.on_available = on_available or (lambda *_: None)
        self.on_healthy = on_healthy or (lambda: None)
        self.on_unhealthy = on_unhealthy or (lambda: None)
        self.keepalive_sec = keepalive_sec
        self.loss_timeout_sec = loss_timeout_sec
        self._signing_key = u._load_ed25519_signing_key()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._health_lock = threading.Lock()
        self._last_event_ts = 0.0
        self._available = False
        self._healthy = False

    # Local health state and callback propagation.
    def _mark_available(self, value: bool) -> None:
        with self._health_lock:
            self._available = value
        self.on_available(value)

    def _mark_event(self) -> None:
        with self._health_lock:
            self._last_event_ts = time.time()
            self._healthy = True
        self.on_healthy()

    def _mark_unhealthy(self) -> None:
        with self._health_lock:
            self._healthy = False
        self.on_unhealthy()

    def start(self, name: str = "WSUserData", daemon: bool = True) -> "BinanceAccountStream":
        super().start(name=name, daemon=daemon)
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="WSUserDataWatchdog", daemon=True)
            self._watchdog_thread.start()
        return self

    def stop(self, timeout: float = WS_STOP_TIMEOUT) -> bool:
        ok = super().stop(timeout=timeout)              # Set stop event and join run thread.
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=timeout)  # Stop the watchdog cleanly too.
        return ok

    # ─── logon semnat (login + keepalive) ─────────────────────────────────────
    def _signed_logon_msg(self, msg_id: str) -> str:
        timestamp = int(time.time() * 1000)
        params_str = f"apiKey={api_key_ws}&timestamp={timestamp}"
        signature = u._sign_ed25519(self._signing_key, params_str)
        return json.dumps({
            "id": msg_id, "method": "session.logon",
            "params": {"apiKey": api_key_ws, "timestamp": timestamp, "signature": signature}})

    @staticmethod
    def _classify(event: dict):
        """Return ``(kind, payload)`` for ping, command response, or unpacked event."""
        if "id" in event:
            return ("ping" if event.get("id") == "ping" else "response"), event
        if "event" in event:
            return "event", event["event"]
        return "event", event

    async def _main(self) -> None:
        if self._signing_key is None or not api_key_ws:
            self._mark_available(False); self._mark_unhealthy()
            logger.error("[WS] The Ed25519/API key is missing -> falling back to polling.")
            return
        self._mark_available(True)
        await self._run_with_reconnect(self._connect_and_run)

    async def _connect_and_run(self) -> None:
        async with websockets.connect(
            WS_API_URL, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_INTERVAL) as ws:
            # Login
            await ws.send(self._signed_logon_msg("login"))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if resp.get("status") != 200:
                raise RuntimeError(f"login failed: {resp}")     # The base reconnect loop applies backoff.
            logger.info("[WS] ✅ Login OK")
            # Do not reset backoff here. The reconnect loop does so only after a stable
            # session, protecting the connection rate limit from reconnect storms.

            await ws.send(json.dumps({"id": "sub", "method": "userDataStream.subscribe"}))
            self._mark_event()
            last_keepalive = last_ping = time.time()

            while not self._stop_event.is_set():
                now = time.time()
                if now - last_keepalive >= self.keepalive_sec:
                    await ws.send(self._signed_logon_msg("keepalive"))
                    last_keepalive = now
                if now - last_ping >= self.loss_timeout_sec / 2:
                    await ws.send(json.dumps({"id": "ping", "method": "ping"}))
                    last_ping = now

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_USERDATA_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("[WS] Heartbeat (no events)")
                    continue

                event = json.loads(raw)
                logger.debug("[WS RAW] %s", raw[:200])
                kind, payload = self._classify(event)
                if kind == "ping":
                    self._mark_event()
                    continue
                if kind == "response":
                    if event.get("status") not in (None, 200):
                        logger.warning("[WS] Error response id=%s status=%s: %s",
                                       event.get("id"), event.get("status"), event)
                        if event.get("id") in ("sub", "keepalive"):
                            raise RuntimeError(f"{event.get('id')} failed: {event}")
                    continue
                self._mark_event()
                self.on_event(payload)

    def _on_session_end(self, exc: Optional[Exception]) -> None:
        # Mark every failed session unhealthy so polling becomes the fallback.
        if exc is not None:
            self._mark_unhealthy()

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            with self._health_lock:
                age = now - self._last_event_ts if self._last_event_ts else float("inf")
                available, healthy = self._available, self._healthy
            if available and healthy and age > self.loss_timeout_sec:
                logger.warning("[WS][WARN] No WS events for %ds -> falling back to polling.", int(age))
                self._mark_unhealthy()
            self._stop_event.wait(5)


# Compatibility aliases retained for earlier code and tests.
BinanceWebSocketManager = BinancePriceStream   # nume generic original
BinanceMarketStream     = BinancePriceStream   # nume intermediar
BinanceUserDataStream   = BinanceAccountStream  # termenul oficial Binance pt user-data


# ─── Entry point market data (singleton partajat, start LAZY) ──────────────────

bapi_ws_manager = BinancePriceStream(symbols=sym.symbols)   # Importing does not open a socket.


def get_ws_manager() -> BinancePriceStream:
    """Return the market-data stream, starting it lazily on the first request."""
    if not bapi_ws_manager.is_running:
        bapi_ws_manager.start()
    return bapi_ws_manager


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s")
    manager = get_ws_manager()
    try:
        while True:
            time.sleep(5)
            prices = manager.get_all_prices()
            print(f"Prices received: {len(prices)} / {len(manager.running_symbols)}")
            for symbol, price in list(prices.items())[:3]:
                print(f"  {symbol}: {price:.4f}")
    except KeyboardInterrupt:
        print("Shutting down...")
        manager.stop()
