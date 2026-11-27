"""Venue-agnostic WebSocket reconnect stream and HTTP connect-retry primitives."""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Set
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    import uvloop as _uvloop
except ImportError:  # pragma: no cover
    _uvloop = None

try:
    import websockets as _websockets
except ImportError:  # pragma: no cover
    _websockets = None

from botcore import (
    _parse_bool_flag,
    async_interruptible_sleep,
    clamp_event_timestamp_ms,
    clean_scalar_str,
    clean_upper_scalar,
    is_worker_thread_alive,
    resolve_now_ts,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_int,
    set_stop_event,
)
from instrument_registry import resolve_kraken_tokenized_pair
from state_io import parse_json_payload


def mount_connect_retry(
    session,
    *,
    attempts: int = 3,
    backoff: float = 0.4,
    read: int = 0,
    status_forcelist=None,
    raise_on_status: bool = False,
    allowed_methods=None,
) -> bool:
    """Mount an order-safe Retry adapter on ``session`` for https:// and http:// (best-effort)."""
    try:
        if session is None or isinstance(status_forcelist, (bool, str, bytes, dict)):
            return False
        att_i = safe_nonneg_int(attempts, default=None)
        read_i = safe_nonneg_int(read, default=None)
        bo_f = safe_nonneg_float(backoff, default=None)
        if att_i is None or read_i is None or bo_f is None:
            return False
        forcelist = None
        if status_forcelist:
            forcelist = []
            for code in status_forcelist:
                c_i = safe_pos_int(code, default=None)
                if c_i is None:
                    return False
                forcelist.append(c_i)
        retry_kw = {
            "total": att_i,
            "connect": att_i,
            "read": read_i,
            "status": att_i if forcelist else 0,
            "redirect": 0,
            "backoff_factor": bo_f,
            "status_forcelist": forcelist if forcelist else None,
            "raise_on_status": _parse_bool_flag(raise_on_status, default=False),
        }
        if allowed_methods is not None:
            retry_kw["allowed_methods"] = allowed_methods
        retry = Retry(**retry_kw)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return True
    except Exception:  # noqa: BLE001 -- retries are best-effort; the timeout still guards
        return False


WS_RECV_TIMEOUT = 0.5
WS_STOP_TIMEOUT = 8.0
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_CLOSE_TIMEOUT = 0.5
WS_RETRY_INITIAL = 1.0
WS_RETRY_MAX = 60.0

_ws_logger = logging.getLogger("providers.ws")


def run_ws_event_loop(coro) -> None:
    """Run ``coro`` on a ``uvloop`` event loop when available, falling back to ``asyncio.run``."""
    loop_factory = getattr(_uvloop, "new_event_loop", None)
    if callable(loop_factory) and getattr(asyncio.run, "__module__", "") == "asyncio.runners":
        asyncio.run(coro, loop_factory=loop_factory)
    else:
        asyncio.run(coro)


class AsyncReconnectStreamBase(ABC):
    """Shared thread, asyncio loop, reconnect backoff, and clean shutdown for venue WebSocket streams."""

    STABLE_SESSION_SEC = 60.0
    _default_thread_name = "BinanceWS"

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self._retry_delay = WS_RETRY_INITIAL
        self._start_lock = threading.Lock()

    def start(self, name: Optional[str] = None, daemon: bool = True) -> "AsyncReconnectStreamBase":
        eff_name = clean_scalar_str(name) or self._default_thread_name
        eff_daemon = _parse_bool_flag(daemon, default=True)
        with self._start_lock:
            if self.is_running:
                _ws_logger.info("%s already running", eff_name)
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._thread_worker, name=eff_name, daemon=eff_daemon
            )
            self._thread.start()
        _ws_logger.info("%s started", eff_name)
        return self

    request_stop = set_stop_event

    def stop(self, timeout: float = WS_STOP_TIMEOUT) -> bool:
        self.request_stop()
        loop = getattr(self, "_loop", None)
        task = getattr(self, "_main_task", None)
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        eff_timeout = safe_nonneg_float(timeout, default=WS_STOP_TIMEOUT * 1.0)
        if (
            self._thread
            and self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=eff_timeout)
            if self._thread.is_alive():
                _ws_logger.warning("WS thread did not stop within %.1fs", eff_timeout)
                return False
        return True

    is_running = property(is_worker_thread_alive)

    async def _main(self) -> None:
        await self._run_with_reconnect(self._connect_and_run)

    def _thread_worker(self) -> None:
        try:
            run_ws_event_loop(self._main())
        except (asyncio.CancelledError, Exception) as exc:
            if not isinstance(exc, asyncio.CancelledError) and not self._is_shutdown_error(exc):
                _ws_logger.exception("WS thread crashed: %s", exc)
        finally:
            self._loop = None
            self._main_task = None
            _ws_logger.info("WS thread exited")

    async def _run_with_reconnect(self, body: Callable) -> None:
        try:
            self._loop = asyncio.get_running_loop()
            self._main_task = asyncio.current_task()
        except RuntimeError:
            pass
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            exc = None
            try:
                await body()
            except asyncio.CancelledError:
                if self._stop_event.is_set():
                    break
                raise
            except Exception as err:
                exc = err
                if self._is_shutdown_error(err):
                    break
                _ws_logger.warning("WS error: %s", err)
            self._on_session_end(exc)
            if self._stop_event.is_set():
                break
            if time.monotonic() - t0 >= self.STABLE_SESSION_SEC:
                self._retry_delay = WS_RETRY_INITIAL
            _ws_logger.info("WS reconnect in %.1fs", self._retry_delay)
            await self._interruptible_sleep(self._retry_delay)
            self._retry_delay = min(self._retry_delay * 2, WS_RETRY_MAX)
        _ws_logger.info("WS reconnect loop stopped")

    def _on_session_end(self, exc: Optional[Exception]) -> None:
        pass

    @abstractmethod
    async def _connect_and_run(self) -> None:
        """Connect to the venue WebSocket and run one session until disconnect."""

    async def _interruptible_sleep(self, delay: float, step: float = 0.2) -> None:
        await async_interruptible_sleep(self._stop_event, delay, step, sleep_fn=asyncio.sleep)

    @staticmethod
    def _is_shutdown_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, RuntimeError) and (
            "cannot schedule new futures after shutdown" in msg
            or "event loop is closed" in msg
        )


class PublicPriceStreamBase(AsyncReconnectStreamBase):
    """Shared base class for public WebSocket price streams across crypto venues."""

    _MAX_STREAMS: Optional[int] = None

    @staticmethod
    def _normalize_tick_timestamp_ms(
        raw_ts: object,
        *,
        now_ms: Optional[int] = None,
        strict_invalid: bool = False,
    ) -> int:
        """Normalize any venue tick timestamp (s/ms/us/ns/ISO) and clamp future/invalid values to ``now_ms``."""
        return clamp_event_timestamp_ms(raw_ts, now_ms=now_ms, strict_invalid=strict_invalid)

    @staticmethod
    def _norm_ws_symbol(symbol: object) -> Optional[str]:
        raw_clean = clean_scalar_str(symbol)
        if raw_clean is None or (
            "/" in raw_clean and any(clean_scalar_str(p) is None for p in raw_clean.split("/"))
        ):
            return None
        compact = raw_clean.replace("/", "")
        tok_pair, is_tok = resolve_kraken_tokenized_pair(compact)
        if is_tok and tok_pair:
            return tok_pair
        return clean_upper_scalar(compact)

    def __init__(self, symbols: Optional[list] = None, *, stale_reconnect_sec: float = 0.0):
        super().__init__()
        self._prices: Dict[str, float] = {}
        self._subscribed: Set[str] = set()
        self._lock = threading.Lock()
        self._subscribers = []
        self._stale_reconnect_sec: float = (
            safe_nonneg_float(stale_reconnect_sec, default=0.0) or 0.0
        )
        self._last_price_mono: float = 0.0
        self._last_price_mono_by_symbol: Dict[str, float] = {}
        self._symbol_subscribed_mono: Dict[str, float] = {}
        if isinstance(symbols, str):
            raw_syms = [symbols]
        elif symbols is None or isinstance(symbols, (bool, bytes, dict)) or not isinstance(symbols, Iterable):
            raw_syms = []
        else:
            raw_syms = list(symbols)
        norm_list = []
        for s in raw_syms:
            norm_s = self._norm_ws_symbol(s)
            if norm_s and norm_s not in self._subscribed:
                self._subscribed.add(norm_s)
                norm_list.append(norm_s)
        self._symbols = tuple(norm_list)

    def subscribe(self, subscriber) -> None:
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _active_staleness_symbols(self) -> Set[str]:
        """Return symbols actively expected on this WebSocket connection for per-symbol staleness checks."""
        return set(self._subscribed)

    def _mark_price_data_received(self, symbol: Optional[str] = None) -> None:
        now_m = time.monotonic()
        with self._lock:
            self._last_price_mono = now_m
            if symbol:
                self._last_price_mono_by_symbol[symbol] = now_m

    def _is_price_stream_stale(self, session_start_mono: float) -> bool:
        limit = safe_nonneg_float(getattr(self, "_stale_reconnect_sec", 0.0), default=0.0) or 0.0
        if limit <= 0.0:
            return False
        now_m = time.monotonic()
        with self._lock:
            active_syms = self._active_staleness_symbols()
            if not active_syms:
                return False
            by_sym = self._last_price_mono_by_symbol
            if not by_sym and self._last_price_mono > 0.0:
                last_mono = max(self._last_price_mono, session_start_mono)
                return (now_m - last_mono) >= limit
            sub_mono = self._symbol_subscribed_mono
            for sym_k in active_syms:
                sym_start = max(session_start_mono, sub_mono.get(sym_k, 0.0))
                sym_last = max(by_sym.get(sym_k, 0.0), sym_start)
                if (now_m - sym_last) >= limit:
                    return True
            return False

    def _check_data_staleness_or_reconnect(
        self,
        session_start_mono: float,
        *,
        stream_logger: Optional[logging.Logger] = None,
    ) -> bool:
        """Single SSOT watchdog invariant: log and return True when the stream exceeded ``_stale_reconnect_sec``."""
        if not self._is_price_stream_stale(session_start_mono):
            return False
        (stream_logger or _ws_logger).warning(
            "%s silent for %.0fs without ticker data; forcing reconnect",
            self.__class__.__name__,
            self._stale_reconnect_sec,
        )
        return True

    def _notify_subscribers(self, symbol: str, items) -> None:
        now_m = time.monotonic()
        with self._lock:
            self._last_price_mono = now_m
            self._last_price_mono_by_symbol[symbol] = now_m
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub.on_items_update(symbol, items)
            except Exception as exc:
                _ws_logger.error("Subscriber notify error: %s", exc)

    def add_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not norm_s:
            return False
        with self._lock:
            if norm_s in self._subscribed:
                _ws_logger.debug("[%s] Already subscribed", norm_s)
                return False
            max_streams = safe_pos_int(self._MAX_STREAMS, default=None)
            if max_streams is not None and len(self._subscribed) >= max_streams:
                _ws_logger.warning("Max streams (%d) reached, cannot add %s", max_streams, norm_s)
                return False
            self._subscribed.add(norm_s)
            self._symbols = tuple(dict.fromkeys((*self._symbols, norm_s)))
            self._symbol_subscribed_mono[norm_s] = time.monotonic()
        return True

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        with self._lock:
            if not norm_s or norm_s not in self._subscribed:
                return False
            self._subscribed.discard(norm_s)
            self._symbols = tuple(s for s in self._symbols if s != norm_s)
            self._prices.pop(norm_s, None)
            self._last_price_mono_by_symbol.pop(norm_s, None)
            self._symbol_subscribed_mono.pop(norm_s, None)
        return True

    def get_price(self, symbol: str) -> Optional[float]:
        norm_s = self._norm_ws_symbol(symbol)
        with self._lock:
            return self._prices.get(norm_s) if norm_s else None

    def get_all_prices(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._prices)

    @property
    def running_symbols(self) -> Set[str]:
        with self._lock:
            return set(self._subscribed)


class JsonPublicWsStreamBase(AsyncReconnectStreamBase):
    """Shared Level-2 base for stateless public JSON WebSocket streams (connect, subscribe, ping, freshness)."""

    _default_thread_name = "PublicJsonWS"

    def __init__(
        self,
        *,
        app_ping_interval_sec: float = 0.0,
        stale_reconnect_sec: float = 0.0,
    ):
        super().__init__()
        self._state_lock = threading.Lock()
        self._last_msg_ts: float = 0.0
        self._connected: bool = False
        self._disconnected_since_ts: float = 0.0
        self._app_ping_interval_sec: float = (
            safe_nonneg_float(app_ping_interval_sec, default=0.0) or 0.0
        )
        self._stale_reconnect_sec: float = (
            safe_nonneg_float(stale_reconnect_sec, default=0.0) or 0.0
        )

    @abstractmethod
    def _resolve_ws_url(self) -> Optional[str]:
        """Return the target public WebSocket URL or ``None`` if nothing is configured to stream."""

    def _build_subscribe_payloads(self) -> tuple[str, ...]:
        """Return zero or more JSON subscribe frames sent immediately after connecting."""
        return ()

    def _app_ping_payload(self) -> Optional[str]:
        """Return optional application-level ping frame sent every ``_app_ping_interval_sec``."""
        return None

    @abstractmethod
    def _handle_payload(self, payload: Any, now_ts: float) -> bool:
        """Process a decoded JSON WebSocket payload; return True when domain state updated."""

    def ingest_raw_message(self, raw: Any, *, now_ts: Optional[float] = None) -> bool:
        """Decode one raw JSON WebSocket message and dispatch to ``_handle_payload``."""
        if not raw:
            return False
        payload = parse_json_payload(raw, default=None) if isinstance(raw, (str, bytes)) else raw
        if payload is None:
            return False
        eff_now = resolve_now_ts(now_ts)
        updated = False
        try:
            updated = bool(self._handle_payload(payload, eff_now))
        except Exception as exc:
            _ws_logger.debug("%s payload error: %s", self.__class__.__name__, exc)
            return False
        if updated:
            with self._state_lock:
                self._last_msg_ts = eff_now
                self._disconnected_since_ts = 0.0
        return updated

    def last_message_ts(self) -> float:
        with self._state_lock:
            return self._last_msg_ts

    def is_connected(self) -> bool:
        with self._state_lock:
            return self._connected

    def is_fresh(self, max_age_sec: float, *, now: Optional[float] = None) -> bool:
        """Return True when the stream has received a message within ``max_age_sec`` and is not disconnected."""
        max_age = safe_nonneg_float(max_age_sec, default=0.0) or 0.0
        if max_age <= 0.0:
            return False
        eff_now = resolve_now_ts(now)
        with self._state_lock:
            if self._disconnected_since_ts > 0.0 and not self._connected:
                return False
            last_ts = self._last_msg_ts
        return last_ts > 0.0 and 0.0 <= (eff_now - last_ts) <= max_age

    def _on_session_end(self, exc: Optional[Exception]) -> None:
        _ = exc
        with self._state_lock:
            self._connected = False
            self._disconnected_since_ts = time.time()

    async def _connect_and_run(self) -> None:
        if _websockets is None:
            _ws_logger.warning("%s: websockets package unavailable", self.__class__.__name__)
            self._stop_event.set()
            return
        ws_url = clean_scalar_str(self._resolve_ws_url())
        if not ws_url:
            await self._interruptible_sleep(5.0)
            return
        async with _websockets.connect(
            ws_url,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            for sub_msg in self._build_subscribe_payloads():
                if sub_msg:
                    await ws.send(sub_msg)
            with self._state_lock:
                self._connected = True
                self._disconnected_since_ts = 0.0
            last_ping = time.monotonic()
            last_data_mono = last_ping
            ping_payload = self._app_ping_payload() if self._app_ping_interval_sec > 0.0 else None
            while not self._stop_event.is_set():
                now_mono = time.monotonic()
                if self._stale_reconnect_sec > 0.0 and (now_mono - last_data_mono) >= self._stale_reconnect_sec:
                    _ws_logger.warning(
                        "%s silent for %.0fs (no data frames); reconnecting...",
                        self.__class__.__name__,
                        now_mono - last_data_mono,
                    )
                    break
                if ping_payload and (now_mono - last_ping) >= self._app_ping_interval_sec:
                    try:
                        await ws.send(ping_payload)
                    except Exception:
                        break
                    last_ping = now_mono
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    continue
                except _websockets.exceptions.ConnectionClosed:
                    break
                except Exception as exc:
                    _ws_logger.warning("%s recv error (%s); reconnecting...", self.__class__.__name__, exc)
                    break
                if self.ingest_raw_message(raw):
                    last_data_mono = time.monotonic()
