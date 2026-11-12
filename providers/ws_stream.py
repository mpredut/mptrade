"""Venue-agnostic WebSocket reconnect stream and HTTP connect-retry primitives."""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Set
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    import uvloop as _uvloop
except ImportError:  # pragma: no cover
    _uvloop = None

from botcore import (
    _parse_bool_flag,
    async_interruptible_sleep,
    is_worker_thread_alive,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_int,
    set_stop_event,
)


def mount_connect_retry(
    session,
    *,
    attempts: int = 3,
    backoff: float = 0.4,
    read: int = 0,
    status_forcelist=None,
    raise_on_status: bool = False,
) -> bool:
    """Mount an order-safe Retry adapter on ``session`` for https:// and http:// (best-effort)."""
    try:
        if session is None or isinstance(status_forcelist, (bool, str, bytes)):
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
        retry = Retry(
            total=att_i,
            connect=att_i,
            read=read_i,
            status=att_i if forcelist else 0,
            redirect=0,
            backoff_factor=bo_f,
            status_forcelist=forcelist if forcelist else None,
            raise_on_status=_parse_bool_flag(raise_on_status, default=False),
        )
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
        eff_name = (
            str(name).strip()
            if isinstance(name, str) and name.strip()
            else self._default_thread_name
        )
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

    def _thread_worker(self) -> None:
        try:
            run_ws_event_loop(self._run_with_reconnect(self._connect_and_run))
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
            t0 = time.time()
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
            if time.time() - t0 >= self.STABLE_SESSION_SEC:
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
    def _norm_ws_symbol(symbol: object) -> Optional[str]:
        if not isinstance(symbol, str):
            return None
        s = symbol.strip().upper().replace("/", "")
        return s or None

    def __init__(self, symbols: Optional[list] = None):
        super().__init__()
        self._prices: Dict[str, float] = {}
        self._subscribed: Set[str] = set()
        self._lock = threading.Lock()
        self._subscribers = []
        if isinstance(symbols, str):
            raw_syms = [symbols]
        elif symbols is None or isinstance(symbols, (bool, bytes, dict)):
            raw_syms = []
        else:
            try:
                raw_syms = list(symbols)
            except TypeError:
                raw_syms = []
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

    def _notify_subscribers(self, symbol: str, items) -> None:
        with self._lock:
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
            if self._MAX_STREAMS is not None and len(self._subscribed) >= self._MAX_STREAMS:
                _ws_logger.warning("Max streams (%d) reached, cannot add %s", self._MAX_STREAMS, norm_s)
                return False
            self._subscribed.add(norm_s)
            self._symbols = tuple(dict.fromkeys((*self._symbols, norm_s)))
        return True

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        with self._lock:
            if not norm_s or norm_s not in self._subscribed:
                return False
            self._subscribed.discard(norm_s)
            self._symbols = tuple(s for s in self._symbols if s != norm_s)
            self._prices.pop(norm_s, None)
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
