"""Public venue price WebSocket streams and multi-venue REST fallback poller."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import builtins
import concurrent.futures as _futures
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional
import websockets as _websockets

from botcore import (
    PRICE_FALLBACK_HTTP_DEADLINE_SEC,
    PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC,
    PRICE_STREAM_POLL_INTERVAL_SEC,
    _parse_bool_flag,
    clean_scalar_str,
    clean_upper_scalar,
    is_auto_start_websockets_enabled,
    is_live_network_backfill_enabled,
    is_rth_gate_enabled,
    is_us_extended_hours_open,
    is_us_rth_open,
    resolve_now_ms,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    clean_symbol_key,
    is_equity_venue,
    is_primary_stream_venue,
    normalize_venue_name,
)
from instrument_registry import is_equity_symbol, is_symbol_market_open
from state_io import parse_json_payload
from .base import MarketDataProvider
from .ssot_bridge import live_price_fetch_context
from .ws_stream import (
    PublicPriceStreamBase,
    WS_CLOSE_TIMEOUT,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
    WS_RECV_TIMEOUT,
)

_NB_FETCH_DEADLINE_SEC = PRICE_FALLBACK_HTTP_DEADLINE_SEC

__all__ = [
    "HyperliquidPriceStream",
    "InFlightRestSlot",
    "KrakenPriceStream",
    "NonBinanceTrendPoller",
    "NonBlockingRestSlotTracker",
    "VenuePriceStreamBase",
    "_NB_FETCH_DEADLINE_SEC",
    "_NonBinanceTrendPoller",
    "_VenuePriceStreamBase",
    "_fetch_price_with_deadline",
    "_start_nonbinance_trend_poller",
    "fetch_price_with_deadline",
    "start_nonbinance_trend_poller",
]


class VenuePriceStreamBase(PublicPriceStreamBase, ABC):
    """Shared public WebSocket price stream base for non-Binance venues."""

    MIN_SAME_PRICE_EMIT_MS = 1000
    _stream_init_hooks: list[Callable[..., None]] = []
    _VENUE_STREAM_SPECS: dict[str, tuple[str, Callable[..., "VenuePriceStreamBase"]]] = {}

    @classmethod
    def register_venue_stream(
        cls,
        venue: str,
        ws_thread_name: str,
        builder: Callable[..., "VenuePriceStreamBase"],
    ) -> None:
        """Declaratively register a public WebSocket stream builder for ``venue``."""
        v_key = normalize_venue_name(venue, default="")
        if v_key:
            cls._VENUE_STREAM_SPECS[v_key] = (str(ws_thread_name), builder)

    @classmethod
    def iter_registered_venue_streams(
        cls,
    ) -> tuple[tuple[str, str, Callable[..., "VenuePriceStreamBase"]], ...]:
        """Return ``(venue_key, ws_thread_name, builder)`` for all registered venue price streams."""
        return tuple(
            (v_key, ws_name, builder)
            for v_key, (ws_name, builder) in cls._VENUE_STREAM_SPECS.items()
        )

    _VENUE_NAME: str = ""

    def __init__(
        self,
        symbols=None,
        *,
        ws_url: Optional[str] = None,
        stale_reconnect_sec: float = 0.0,
    ):
        prov_cls = (
            MarketDataProvider.resolve_venue_provider_class(self._VENUE_NAME)
            if self._VENUE_NAME
            else None
        )
        super().__init__(symbols=symbols, stale_reconnect_sec=stale_reconnect_sec)
        self._prov_cls = prov_cls
        self.ws_url = clean_scalar_str(ws_url) or getattr(prov_cls, "PUBLIC_WS_URL", "")
        self._last_emit_ts_ms: dict[str, int] = {}
        self._last_emit_price: dict[str, float] = {}
        self._extra_subscribe_builders: list[Callable[[], tuple[str, ...]]] = []
        self._channel_handlers: dict[str, Callable[[object, int], None]] = {}
        self._tick_listeners: list[Callable[[int], None]] = []

    @classmethod
    def register_stream_init_hook(cls, hook: Callable[..., None]) -> None:
        if "_stream_init_hooks" not in cls.__dict__:
            cls._stream_init_hooks = []
        if hook not in cls._stream_init_hooks:
            cls._stream_init_hooks.append(hook)

    def register_channel_handler(
        self,
        channel: str,
        handler: Callable[[object, int], None],
        *,
        subscribe_builder: Optional[Callable[[], tuple[str, ...]]] = None,
        on_tick: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._channel_handlers[str(channel)] = handler
        if callable(subscribe_builder):
            self._extra_subscribe_builders.append(subscribe_builder)
        if callable(on_tick):
            self._tick_listeners.append(on_tick)

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not super().remove_symbol(symbol):
            return False
        if norm_s:
            with self._lock:
                self._last_emit_ts_ms.pop(norm_s, None)
                self._last_emit_price.pop(norm_s, None)
        return True

    def _notify_subscribers(self, symbol: str, items) -> None:
        ts_ms = None
        px = None
        if (
            isinstance(items, list)
            and items
            and isinstance(items[-1], (list, tuple))
            and len(items[-1]) >= 2
        ):
            ts_ms = safe_pos_int(items[-1][0], default=None)
            px = safe_pos_float(items[-1][1], default=None)
        now_m = time.monotonic()
        with self._lock:
            self._last_price_mono = now_m
            self._last_price_mono_by_symbol[symbol] = now_m
            subscribers = list(self._subscribers)
            if ts_ms is not None and px is not None:
                prev_ts = self._last_emit_ts_ms.get(symbol)
                prev_px = self._last_emit_price.get(symbol)
                if (
                    prev_ts is not None
                    and prev_px == px
                    and 0 <= (ts_ms - prev_ts) < self.MIN_SAME_PRICE_EMIT_MS
                ):
                    now_s = time.time()
                    for sub in subscribers:
                        ws_map = getattr(sub, "_ws_stream_event_ts_per_symbol", None)
                        if isinstance(ws_map, dict):
                            ws_map[symbol] = now_s
                    return
                self._last_emit_ts_ms[symbol] = ts_ms
                self._last_emit_price[symbol] = px
        for sub in subscribers:
            try:
                sub.on_items_update(symbol, items)
            except Exception as exc:
                builtins.print(
                    f"[{self.__class__.__name__}] subscriber error ({symbol}): {exc}"
                )

    async def _connect_and_run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                has_symbols = bool(self._symbols)
            if has_symbols:
                break
            await self._interruptible_sleep(1.0)
        if not self._symbols or self._stop_event.is_set():
            return
        async with _websockets.connect(
            self.ws_url,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            sub_msg = self._build_subscribe_payload()
            if sub_msg:
                await ws.send(sub_msg)
            for extra_msg in self._build_extra_subscribe_payloads():
                if extra_msg:
                    await ws.send(extra_msg)
            session_start = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    raw = None
                except _websockets.exceptions.ConnectionClosed:
                    break
                except Exception:
                    raw = None
                if raw is not None:
                    try:
                        self._process_message(raw)
                    except Exception:
                        pass
                if self._check_data_staleness_or_reconnect(session_start):
                    break

    @abstractmethod
    def _build_subscribe_payload(self) -> Optional[str]:
        """Build the venue-specific WebSocket subscription message."""

    def _build_extra_subscribe_payloads(self) -> tuple[str, ...]:
        payloads: list[str] = []
        for builder in self._extra_subscribe_builders:
            built = builder()
            if built:
                payloads.extend(p for p in built if p)
        return tuple(payloads)

    @abstractmethod
    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        """Process one incoming venue WebSocket message."""


_VenuePriceStreamBase = VenuePriceStreamBase


class HyperliquidPriceStream(VenuePriceStreamBase):
    """Real-time Hyperliquid WebSocket price stream delegating protocol logic to ``HyperliquidProvider``."""

    _VENUE_NAME = "hyperliquid"

    def __init__(
        self,
        symbols=None,
        *,
        market_api=None,
        ws_url: Optional[str] = None,
        stale_reconnect_sec: float = 120.0,
        **extension_kwargs,
    ):
        super().__init__(symbols=symbols, ws_url=ws_url, stale_reconnect_sec=stale_reconnect_sec)
        self.market_api = market_api
        self._resolved_spot_keys: dict[str, tuple[str, ...]] = {}
        for hook in getattr(self, "_stream_init_hooks", ()):
            hook(self, **extension_kwargs)

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        removed = super().remove_symbol(symbol)
        if removed and norm_s:
            with self._lock:
                self._resolved_spot_keys.pop(norm_s, None)
        return removed

    def _candidate_mids_keys(self, symbol: str) -> tuple[str, ...]:
        with self._lock:
            cached = self._resolved_spot_keys.get(symbol)
        if cached:
            return cached
        prov_cls = self._prov_cls
        resolved = (
            prov_cls.resolve_ws_spot_keys(symbol, market_api=self.market_api)
            if prov_cls is not None
            else (symbol,)
        )
        with self._lock:
            self._resolved_spot_keys[symbol] = resolved
        return resolved

    def _build_subscribe_payload(self) -> str:
        prov_cls = self._prov_cls
        return prov_cls.build_ws_all_mids_subscription() if prov_cls is not None else ""

    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        try:
            envelope = parse_json_payload(raw) if isinstance(raw, str) else raw
            if not isinstance(envelope, dict):
                return
            ts_ms = resolve_now_ms(now_ms)
            channel = envelope.get("channel")
            handler = self._channel_handlers.get(str(channel or ""))
            if handler is not None:
                handler(envelope.get("data"), ts_ms=ts_ms)
                return
            if (prov_cls := self._prov_cls) is None:
                return
            with self._lock:
                symbols = self._symbols
            prices = prov_cls.extract_ws_all_mids_prices(
                envelope, symbols, candidate_mids_fn=self._candidate_mids_keys
            )
            if prices is None:
                return
            for s_sym, px in prices.items():
                with self._lock:
                    self._prices[s_sym] = px
                self._notify_subscribers(s_sym, [[ts_ms, px]])
            for tick_fn in self._tick_listeners:
                tick_fn(ts_ms)
        except Exception:
            return


class KrakenPriceStream(VenuePriceStreamBase):
    """Real-time Kraken public WebSocket stream delegating protocol logic to ``KrakenProvider``."""

    _VENUE_NAME = "kraken"

    def __init__(
        self,
        symbols=None,
        *,
        ws_url: Optional[str] = None,
        stale_reconnect_sec: float = 300.0,
    ):
        super().__init__(symbols=symbols, ws_url=ws_url, stale_reconnect_sec=stale_reconnect_sec)
        self._ws_pairs: list[str] = []
        self._pair_to_symbol: dict[str, str] = {}
        for s_sym in self._symbols:
            self._register_symbol_pair(s_sym)

    def _active_staleness_symbols(self) -> set[str]:
        """Only enforce WS staleness on pairs actually subscribed to Kraken's public ticker channel."""
        return set(self._pair_to_symbol.values())

    def _register_symbol_pair(self, s_sym: str) -> None:
        prov_cls = self._prov_cls
        if prov_cls is not None:
            prov_cls.register_ws_ticker_symbol_pair(
                s_sym, self._ws_pairs, self._pair_to_symbol
            )

    def add_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not super().add_symbol(symbol) or not norm_s:
            return False
        with self._lock:
            self._register_symbol_pair(norm_s)
        return True

    def remove_symbol(self, symbol: str) -> bool:
        if not super().remove_symbol(symbol):
            return False
        with self._lock:
            self._ws_pairs.clear()
            self._pair_to_symbol.clear()
            for s_sym in self._symbols:
                self._register_symbol_pair(s_sym)
        return True

    def _build_subscribe_payload(self) -> str:
        prov_cls = self._prov_cls
        if prov_cls is None:
            return ""
        with self._lock:
            pairs = list(self._ws_pairs)
        return prov_cls.build_ws_ticker_subscription(pairs)

    def _resolve_symbol(self, pair_label: str) -> Optional[str]:
        if (raw_u := clean_upper_scalar(pair_label)) is None:
            return None
        with self._lock:
            return self._pair_to_symbol.get(raw_u) or self._pair_to_symbol.get(
                clean_symbol_key(raw_u)
            )

    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        try:
            if (prov_cls := self._prov_cls) is None:
                return
            msg = parse_json_payload(raw) if isinstance(raw, str) else raw
            ts_ms = resolve_now_ms(now_ms)
            kind, s_sym, price = prov_cls.parse_ws_public_ticker_message(
                msg, resolve_symbol_fn=self._resolve_symbol
            )
            if kind == "keepalive":
                with self._lock:
                    known_prices = list(self._prices.items())
                for sym_key, px in known_prices:
                    self._notify_subscribers(sym_key, [[ts_ms, px]])
                return
            if kind == "tick" and s_sym and price is not None:
                with self._lock:
                    self._prices[s_sym] = price
                self._notify_subscribers(s_sym, [[ts_ms, price]])
        except Exception:
            return


VenuePriceStreamBase.register_venue_stream(
    "hyperliquid",
    "HyperliquidPriceWS",
    lambda syms, cpm=None: HyperliquidPriceStream(
        syms, market_api=getattr(cpm, "market_api", None)
    ),
)
VenuePriceStreamBase.register_venue_stream(
    "kraken",
    "KrakenPriceWS",
    lambda syms, cpm=None: KrakenPriceStream(syms),
)


def fetch_price_with_deadline(
    market_api,
    symbol,
    pool,
    deadline_sec=_NB_FETCH_DEADLINE_SEC,
    provider_name=None,
    prefer_public_extended=False,
):
    """Bound the poller's wait while the provider owns the network-call timeout."""
    kwargs = {"symbol": symbol}
    if provider_name is not None and not isinstance(provider_name, bool):
        if isinstance(provider_name, str):
            if (pn := clean_scalar_str(provider_name)) is not None:
                kwargs["provider_name"] = pn
        else:
            kwargs["provider_name"] = provider_name
    use_ext = _parse_bool_flag(prefer_public_extended, False)

    def _poll():
        with live_price_fetch_context():
            if use_ext:
                try:
                    return market_api.get_current_price(**kwargs, prefer_public_extended=True)
                except TypeError:
                    pass
            return market_api.get_current_price(**kwargs)

    fut = pool.submit(_poll)
    eff_deadline = safe_pos_float(deadline_sec, default=_NB_FETCH_DEADLINE_SEC) or _NB_FETCH_DEADLINE_SEC
    try:
        return fut.result(timeout=eff_deadline)
    except _futures.TimeoutError:
        fut.cancel()
        raise


_fetch_price_with_deadline = fetch_price_with_deadline


@dataclass(slots=True)
class InFlightRestSlot:
    """Single per-symbol in-flight REST fallback request tracked without blocking poll loops."""

    future: _futures.Future
    submit_mono: float
    submit_wall_ts: float
    meta: Any = None


class NonBlockingRestSlotTracker:
    """Thread-safe SSOT tracker for non-blocking per-symbol REST fallback slots.

    Guarantees at most one active REST request per symbol without blocking poll loops on a
    short deadline: loops revisit slots non-blockingly, harvest completed results or real
    socket errors, and evict/reset genuinely wedged requests only after ``hard_timeout_sec``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, InFlightRestSlot] = {}

    @property
    def slots(self) -> dict[str, InFlightRestSlot]:
        return self._slots

    def active_symbols(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._slots.keys())

    def has_active(self) -> bool:
        with self._lock:
            return bool(self._slots)

    def has_done(self) -> bool:
        with self._lock:
            return any(slot.future.done() for slot in self._slots.values())

    def next_hard_timeout_remaining(
        self, *, now_mono: float, hard_timeout_sec: float
    ) -> Optional[float]:
        with self._lock:
            if not self._slots:
                return None
            return min(
                max(0.0, (slot.submit_mono + hard_timeout_sec) - now_mono)
                for slot in self._slots.values()
            )

    def inspect_and_harvest(
        self,
        symbol: str,
        *,
        now_mono: float,
        hard_timeout_sec: float,
    ) -> tuple[str, Any, Optional[InFlightRestSlot]]:
        """Inspect ``symbol``'s slot without blocking.

        Returns ``(state, payload_or_exc, slot)`` where ``state`` is one of:
        ``IDLE``, ``IN_FLIGHT``, ``HARD_TIMEOUT``, ``DONE_OK``, ``DONE_ERR``, ``DONE_CANCELLED``.
        """
        with self._lock:
            slot = self._slots.get(symbol)
            if slot is None:
                return "IDLE", None, None
            if not slot.future.done():
                if (now_mono - slot.submit_mono) >= hard_timeout_sec:
                    self._slots.pop(symbol, None)
                    slot.future.cancel()
                    return "HARD_TIMEOUT", None, slot
                return "IN_FLIGHT", None, slot
            self._slots.pop(symbol, None)
        if slot.future.cancelled():
            return "DONE_CANCELLED", None, slot
        try:
            return "DONE_OK", slot.future.result(), slot
        except Exception as exc:  # noqa: BLE001
            return "DONE_ERR", exc, slot

    def submit(
        self,
        symbol: str,
        pool: _futures.ThreadPoolExecutor,
        fn: Callable[[], Any],
        *,
        now_mono: float,
        now_wall_ts: float,
        meta: Any = None,
        on_done: Optional[Callable[[_futures.Future], None]] = None,
    ) -> Optional[InFlightRestSlot]:
        """Submit ``fn`` for ``symbol`` if idle; returns the new slot or ``None`` if already active."""
        with self._lock:
            existing = self._slots.get(symbol)
            if existing is not None and not existing.future.done():
                return None
            try:
                fut = pool.submit(fn)
            except RuntimeError:
                return None
            slot = InFlightRestSlot(
                future=fut,
                submit_mono=now_mono,
                submit_wall_ts=now_wall_ts,
                meta=meta,
            )
            self._slots[symbol] = slot
        if callable(on_done):
            fut.add_done_callback(on_done)
        return slot

    def clear(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                slot.future.cancel()
            self._slots.clear()


class NonBinanceTrendPoller:
    """Own the non-Binance WebSocket streams, fallback poller thread, and executor lifecycle."""

    def __init__(
        self,
        thread,
        pool,
        stop_event,
        ws_streams=None,
        *,
        cpm=None,
        delegated_symbols=(),
        wake_event: Optional[threading.Event] = None,
        slot_tracker: Optional[NonBlockingRestSlotTracker] = None,
    ):
        self.thread = thread
        self.pool = pool
        self.stop_event = stop_event
        self.ws_streams = tuple(ws_streams or ())
        self._cpm = cpm
        self._delegated_symbols = tuple(delegated_symbols or ())
        self._wake_event = wake_event
        self._slot_tracker = slot_tracker

    def request_stop(self) -> None:
        self.stop_event.set()
        if self._wake_event is not None:
            self._wake_event.set()
        if self._slot_tracker is not None:
            self._slot_tracker.clear()
        delegated = getattr(self._cpm, "_delegated_poller_symbols", None)
        if isinstance(delegated, set):
            for sym_key in self._delegated_symbols:
                delegated.discard(sym_key)
                delegated.discard(str(sym_key).strip().upper())
        for ws in self.ws_streams:
            try:
                req_stop = getattr(ws, "request_stop", None)
                if callable(req_stop):
                    req_stop()
                else:
                    stop_ev = getattr(ws, "_stop_event", None)
                    if stop_ev is not None:
                        stop_ev.set()
            except Exception:
                pass

    def stop(self, timeout=5.0):
        self.request_stop()
        eff_timeout = safe_nonneg_float(timeout, default=5.0)
        total_deadline = time.monotonic() + eff_timeout
        ws_budget = min(2.0, eff_timeout)
        ws_deadline = time.monotonic() + ws_budget
        for ws in self.ws_streams:
            try:
                rem = min(eff_timeout, max(0.05, ws_deadline - time.monotonic())) if eff_timeout > 0.0 else 0.0
                ws.stop(timeout=rem)
            except Exception:
                pass
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=max(0.0, total_deadline - time.monotonic()))
        self.pool.shutdown(wait=False, cancel_futures=True)
        try:
            for th in list(getattr(self.pool, "_threads", ())):
                _futures.thread._threads_queues.pop(th, None)
        except Exception:
            pass
        return not self.thread.is_alive()


_NonBinanceTrendPoller = NonBinanceTrendPoller


def start_nonbinance_trend_poller(
    cpm,
    symbols,
    interval_sec=PRICE_STREAM_POLL_INTERVAL_SEC,
    fetch_deadline_sec=_NB_FETCH_DEADLINE_SEC,
    provider_names=None,
    *,
    start_ws=None,
    hard_timeout_sec=None,
):
    """Feed short trends for non-Binance symbols using WebSocket-first + non-blocking REST fallback."""
    if isinstance(symbols, (str, bytes, bool, dict)) or not hasattr(symbols, "__iter__"):
        raise ValueError("symbols must be an iterable of symbol strings")
    if provider_names is not None and not hasattr(provider_names, "items"):
        raise ValueError("provider_names must be a mapping")
    symbols = tuple(clean_scalar_str(symbol) or "" for symbol in symbols)
    eff_interval = safe_pos_float(interval_sec, default=PRICE_STREAM_POLL_INTERVAL_SEC) or PRICE_STREAM_POLL_INTERVAL_SEC
    _ = safe_pos_float(fetch_deadline_sec, default=_NB_FETCH_DEADLINE_SEC)
    eff_hard_timeout = (
        safe_pos_float(hard_timeout_sec, default=PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC)
        or PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC
    )
    bindings = {
        sym_k: (clean_scalar_str(provider_name) or "")
        for symbol, provider_name in (provider_names.items() if provider_names is not None else ())
        if (sym_k := clean_upper_scalar(symbol)) is not None
    }
    missing = [
        symbol
        for symbol in symbols
        if (
            not symbol
            or not bindings.get(symbol.upper())
            or is_primary_stream_venue(bindings[symbol.upper()])
        )
    ]
    if missing:
        raise ValueError(
            "explicit provider bindings are required for non-Binance symbols: "
            + ", ".join(missing)
        )
    cpm.bind_providers(bindings)
    delegated_set = getattr(cpm, "_delegated_poller_symbols", None)
    if isinstance(delegated_set, set):
        for sym_item in symbols:
            delegated_set.add(sym_item)
            delegated_set.add(sym_item.upper())

    if start_ws is None:
        enable_ws = (
            is_auto_start_websockets_enabled()
            and is_live_network_backfill_enabled()
        )
    else:
        enable_ws = _parse_bool_flag(start_ws, default=False)

    ws_streams = []
    if enable_ws and hasattr(cpm, "on_items_update"):
        for venue_key, ws_name, build_ws in (
            VenuePriceStreamBase.iter_registered_venue_streams()
        ):
            v_syms = [
                s
                for s in symbols
                if normalize_venue_name(bindings[s.upper()], default="") == venue_key
            ]
            if v_syms:
                ws_inst = build_ws(v_syms, cpm=cpm)
                if hasattr(cpm, "attach_ws_manager"):
                    cpm.attach_ws_manager(ws_inst)
                else:
                    ws_inst.subscribe(cpm)
                ws_inst.start(name=ws_name, daemon=True)
                ws_streams.append(ws_inst)

    pool = _futures.ThreadPoolExecutor(
        max_workers=max(3, len(symbols) + 2), thread_name_prefix="NBTrendFetch"
    )
    stop_event = threading.Event()
    wake_event = threading.Event()
    slot_tracker = NonBlockingRestSlotTracker()

    has_cls_mark_ws = callable(getattr(type(cpm), "mark_symbol_ws_event", None))

    def run():
        rth_seeded: set[str] = set()
        empty_backoff_ts: dict[str, float] = {}
        err_backoff_until: dict[str, float] = {}
        err_streaks: dict[str, int] = {}
        last_equity_poll_ts: dict[str, float] = {}
        next_poll_mono: dict[str, float] = {}
        equity_min_gap_sec = max(eff_interval, 6.0)
        equity_ext_gap_sec = max(60.0, equity_min_gap_sec)
        cpm_cache = getattr(cpm, "cache", None)
        extract_fn = getattr(cpm, "extract_price_and_ts", None)
        if isinstance(cpm_cache, dict) and callable(extract_fn):
            for s in symbols:
                try:
                    cached_px, _ = extract_fn(cpm_cache.get(s))
                    if safe_pos_float(cached_px, default=None) is not None:
                        rth_seeded.add(s)
                except Exception:
                    pass
        if ws_streams:
            for _ in range(20):
                ws_health_fn = getattr(cpm, "is_real_ws_healthy", None)
                if stop_event.is_set() or (
                    callable(ws_health_fn) and all(_parse_bool_flag(ws_health_fn(s), False) for s in symbols)
                ):
                    break
                stop_event.wait(0.1)

        def _record_poll_error_backoff(sym_item: str, exc: object, *, now_mono_val: Optional[float] = None) -> float:
            streak = err_streaks.get(sym_item, 0) + 1
            err_streaks[sym_item] = streak
            err_l = str(exc or "").lower()
            if "429" in err_l or "rate limit" in err_l or "too many requests" in err_l:
                delay_s = min(120.0, max(15.0, eff_interval * 4.0) * (2.0 ** min(streak - 1, 3)))
            else:
                delay_s = min(30.0, max(eff_interval * 2.0, 0.5) * (2.0 ** min(streak - 1, 5)))
            ref_mono = time.monotonic() if now_mono_val is None else now_mono_val
            err_backoff_until[sym_item] = ref_mono + delay_s
            return delay_s

        def _was_ws_fresh_or_preempted(sym_item: str, submit_wall_ts: float, now_wall_ts: float) -> bool:
            ws_health_fn = getattr(cpm, "is_real_ws_healthy", None)
            if callable(ws_health_fn) and _parse_bool_flag(ws_health_fn(sym_item), False):
                return True
            ws_stream_map = getattr(cpm, "_ws_stream_event_ts_per_symbol", None)
            if isinstance(ws_stream_map, dict):
                ws_cands = (
                    ts_v
                    for raw_ts in (ws_stream_map.get(sym_item), ws_stream_map.get(sym_item.upper()))
                    if (ts_v := safe_pos_float(raw_ts, default=None)) is not None
                    and ts_v <= now_wall_ts + 60.0
                )
                ws_ts = max(ws_cands, default=0.0)
                if ws_ts > 0.0 and ws_ts >= submit_wall_ts:
                    return True
            return False

        def _apply_fetched_price(sym_item: str, raw_px: Any, *, is_eq: bool) -> None:
            pf = safe_pos_float(raw_px, default=None)
            now_done = time.time()
            now_done_mono = time.monotonic()
            if pf is not None:
                rth_seeded.add(sym_item)
                empty_backoff_ts.pop(sym_item, None)
                err_streaks.pop(sym_item, None)
                err_backoff_until.pop(sym_item, None)
                if is_eq:
                    last_equity_poll_ts[sym_item] = now_done_mono
                if has_cls_mark_ws:
                    cpm.mark_symbol_ws_event(sym_item, now_done)
                elif isinstance(ws_ts_map := getattr(cpm, "_ws_last_event_ts_per_symbol", None), dict):
                    ws_ts_map[sym_item] = now_done
                push_fn = getattr(cpm, "push_price", None)
                if callable(push_fn):
                    push_fn(sym_item, pf)
            else:
                empty_backoff_ts[sym_item] = now_done_mono

        def _mark_eq_idle(sym_item: str, ts_val: float) -> None:
            if has_cls_mark_ws:
                cpm.mark_symbol_ws_event(sym_item, ts_val)
            elif isinstance(ws_ts_map := getattr(cpm, "_ws_last_event_ts_per_symbol", None), dict):
                ws_ts_map[sym_item] = ts_val

        def _cached_ts_ms(sym_item: str) -> Optional[float]:
            if isinstance(cpm_cache, dict) and callable(extract_fn):
                try:
                    _, ts_v = extract_fn(cpm_cache.get(sym_item))
                    return safe_pos_float(ts_v, default=None)
                except Exception:
                    return None
            return None

        while not stop_event.is_set():
            wake_event.clear()
            now_mono = time.monotonic()
            now_ts = time.time()
            for s in symbols:
                if stop_event.is_set():
                    break
                state, res_val, slot = slot_tracker.inspect_and_harvest(
                    s, now_mono=now_mono, hard_timeout_sec=eff_hard_timeout
                )
                if state == "IN_FLIGHT":
                    continue
                if state == "HARD_TIMEOUT":
                    builtins.print(
                        f"[NB-trend] {s}: REST hard-timeout >{eff_hard_timeout:.1f}s -- resetting slot"
                    )
                elif state == "DONE_ERR":
                    delay_s = _record_poll_error_backoff(s, res_val, now_mono_val=now_mono)
                    builtins.print(f"[NB-trend] {s}: {res_val} (backoff {delay_s:.1f}s)")
                    continue
                elif state == "DONE_OK" and slot is not None:
                    if _was_ws_fresh_or_preempted(s, slot.submit_wall_ts, now_ts):
                        empty_backoff_ts.pop(s, None)
                        err_streaks.pop(s, None)
                        err_backoff_until.pop(s, None)
                        next_poll_mono[s] = now_mono + eff_interval
                        continue
                    if safe_pos_float(res_val, default=None) is None:
                        empty_backoff_ts[s] = now_mono
                        next_poll_mono[s] = now_mono + eff_interval
                        continue
                    if (now_mono - slot.submit_mono) <= eff_hard_timeout:
                        _apply_fetched_price(s, res_val, is_eq=bool(slot.meta))
                        next_poll_mono[s] = now_mono + eff_interval
                        continue

                ws_health_fn = getattr(cpm, "is_real_ws_healthy", None)
                if callable(ws_health_fn) and _parse_bool_flag(ws_health_fn(s), False):
                    continue
                if now_mono < err_backoff_until.get(s, 0.0):
                    continue
                if now_mono < next_poll_mono.get(s, 0.0):
                    continue
                prov_name = bindings[s.upper()]
                is_eq = is_equity_venue(prov_name) or is_equity_symbol(s)
                rth_gate_active = is_rth_gate_enabled(
                    getattr(cpm, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
                )
                last_empty = empty_backoff_ts.get(s)
                market_open = not rth_gate_active or is_symbol_market_open(s, now_ts)
                is_ext_open = bool(is_eq and rth_gate_active and not market_open and is_us_extended_hours_open(now_ts))
                if rth_gate_active and (market_open or is_ext_open) and last_empty is not None and (now_mono - last_empty) < 60.0:
                    continue
                last_eq_mono = last_equity_poll_ts.get(s)
                if is_ext_open:
                    cur_ts_ms = _cached_ts_ms(s)
                    ext_cache_fresh = cur_ts_ms is not None and -5_000.0 <= (now_ts * 1000.0 - cur_ts_ms) < (equity_ext_gap_sec * 1000.0)
                    if (s in rth_seeded and last_eq_mono is not None and (now_mono - last_eq_mono) < equity_ext_gap_sec) or ext_cache_fresh:
                        _mark_eq_idle(s, now_ts)
                        continue
                elif (
                    not market_open and (s in rth_seeded or (last_empty is not None and (now_mono - last_empty) < 300.0))
                ) or (is_eq and s in rth_seeded and last_eq_mono is not None and (now_mono - last_eq_mono) < equity_min_gap_sec):
                    _mark_eq_idle(s, now_ts)
                    continue
                # Prefer live public equity quotes (Yahoo RTH + PRE/POST) over authenticated broker portfolio marks
                # during both US RTH and extended hours to prevent multi-minute broker-mark freezes on held tickers
                # while falling back automatically to the provider's broker mark if the public quote is unavailable.
                prefer_public_eq = bool(is_eq and (market_open or is_ext_open))

                def _poll_item(_sym=s, _prov=prov_name, _ext=prefer_public_eq):
                    with live_price_fetch_context():
                        if _ext:
                            try:
                                return cpm.market_api.get_current_price(
                                    symbol=_sym, provider_name=_prov, prefer_public_extended=True
                                )
                            except TypeError:
                                pass
                        return cpm.market_api.get_current_price(symbol=_sym, provider_name=_prov)

                if (
                    slot_tracker.submit(
                        s,
                        pool,
                        _poll_item,
                        now_mono=now_mono,
                        now_wall_ts=now_ts,
                        meta=is_eq,
                        on_done=lambda _f: wake_event.set(),
                    )
                    is not None
                ):
                    next_poll_mono[s] = now_mono + eff_interval

            if stop_event.is_set():
                break
            if slot_tracker.has_done():
                continue
            now_after = time.monotonic()
            wait_s = eff_interval
            next_ht = slot_tracker.next_hard_timeout_remaining(
                now_mono=now_after, hard_timeout_sec=eff_hard_timeout
            )
            if next_ht is not None:
                wait_s = min(wait_s, max(0.005, next_ht))
            active_now = set(slot_tracker.active_symbols())
            for s in symbols:
                if s in active_now:
                    continue
                due_at = next_poll_mono.get(s, 0.0)
                if due_at > now_after:
                    wait_s = min(wait_s, max(0.005, due_at - now_after))
            wake_event.wait(timeout=wait_s)

    t = threading.Thread(target=run, name="NonBinanceTrendPoller", daemon=True)
    t.start()
    return NonBinanceTrendPoller(
        t,
        pool,
        stop_event,
        ws_streams=ws_streams,
        cpm=cpm,
        delegated_symbols=symbols,
        wake_event=wake_event,
        slot_tracker=slot_tracker,
    )


_start_nonbinance_trend_poller = start_nonbinance_trend_poller
