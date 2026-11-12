"""Public venue price WebSocket streams and multi-venue REST fallback poller."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import builtins
import concurrent.futures as _futures
import threading
import time
from typing import Any, Callable, Optional
import websockets as _websockets

from botcore import (
    PRICE_FALLBACK_HTTP_DEADLINE_SEC,
    PRICE_STREAM_POLL_INTERVAL_SEC,
    _parse_bool_flag,
    is_auto_start_websockets_enabled,
    is_live_network_backfill_enabled,
    is_rth_gate_enabled,
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
    "KrakenPriceStream",
    "NonBinanceTrendPoller",
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

    def __init__(self, symbols=None, *, ws_url: Optional[str] = None):
        prov_cls = (
            MarketDataProvider.resolve_venue_provider_class(self._VENUE_NAME)
            if self._VENUE_NAME
            else None
        )
        super().__init__(symbols=symbols)
        self._prov_cls = prov_cls
        self.ws_url = ws_url or getattr(prov_cls, "PUBLIC_WS_URL", "")
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
        with self._lock:
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
            while not self._stop_event.is_set():
                try:
                    self._process_message(
                        await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                    )
                except asyncio.TimeoutError:
                    continue
                except _websockets.exceptions.ConnectionClosed:
                    break
                except Exception:
                    continue

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
        **extension_kwargs,
    ):
        super().__init__(symbols=symbols, ws_url=ws_url)
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

    def __init__(self, symbols=None, *, ws_url: Optional[str] = None):
        super().__init__(symbols=symbols, ws_url=ws_url)
        self._ws_pairs: list[str] = []
        self._pair_to_symbol: dict[str, str] = {}
        for s_sym in self._symbols:
            self._register_symbol_pair(s_sym)

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
        raw_u = str(pair_label or "").strip().upper()
        if not raw_u:
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
):
    """Bound the poller's wait while the provider owns the network-call timeout."""
    kwargs = {"symbol": symbol}
    if provider_name is not None:
        kwargs["provider_name"] = provider_name

    def _poll():
        with live_price_fetch_context():
            return market_api.get_current_price(**kwargs)

    fut = pool.submit(_poll)
    eff_deadline = safe_pos_float(deadline_sec, default=_NB_FETCH_DEADLINE_SEC) or _NB_FETCH_DEADLINE_SEC
    try:
        return fut.result(timeout=eff_deadline)
    except _futures.TimeoutError:
        fut.cancel()
        raise


_fetch_price_with_deadline = fetch_price_with_deadline


class NonBinanceTrendPoller:
    """Own the non-Binance WebSocket streams, fallback poller thread, and executor lifecycle."""

    def __init__(self, thread, pool, stop_event, ws_streams=None, *, cpm=None, delegated_symbols=()):
        self.thread = thread
        self.pool = pool
        self.stop_event = stop_event
        self.ws_streams = tuple(ws_streams or ())
        self._cpm = cpm
        self._delegated_symbols = tuple(delegated_symbols or ())

    def request_stop(self) -> None:
        self.stop_event.set()
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
):
    """Feed short trends for non-Binance symbols using WebSocket-first + REST fallback."""
    if isinstance(symbols, (str, bytes, bool)) or not hasattr(symbols, "__iter__"):
        raise ValueError("symbols must be an iterable of symbol strings")
    symbols = tuple("" if isinstance(symbol, bool) else str(symbol or "").strip() for symbol in symbols)
    eff_interval = safe_pos_float(interval_sec, default=PRICE_STREAM_POLL_INTERVAL_SEC) or PRICE_STREAM_POLL_INTERVAL_SEC
    eff_deadline = safe_pos_float(fetch_deadline_sec, default=_NB_FETCH_DEADLINE_SEC) or _NB_FETCH_DEADLINE_SEC
    bindings = {
        str(symbol or "").strip().upper(): str(provider_name or "").strip()
        for symbol, provider_name in dict(provider_names or {}).items()
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
        max_workers=max(2, len(symbols)), thread_name_prefix="NBTrendFetch"
    )
    stop_event = threading.Event()

    cpm_cls = type(cpm)
    has_cls_mark_ws = callable(getattr(cpm_cls, "mark_symbol_ws_event", None))

    def _mark_symbol_ws_ts(sym_key: str, ts_val: float) -> None:
        if has_cls_mark_ws:
            cpm.mark_symbol_ws_event(sym_key, ts_val)
            return
        ws_ts_map = getattr(cpm, "_ws_last_event_ts_per_symbol", None)
        if isinstance(ws_ts_map, dict):
            ws_ts_map[sym_key] = ts_val

    def run():
        rth_seeded: set[str] = set()
        empty_backoff_ts: dict[str, float] = {}
        last_equity_poll_ts: dict[str, float] = {}
        equity_min_gap_sec = max(eff_interval, 6.0)
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

        def _apply_fetched_price(sym_item: str, raw_px: Any, *, is_eq: bool) -> None:
            pf = safe_pos_float(raw_px, default=None)
            now_done = time.time()
            if pf is not None:
                rth_seeded.add(sym_item)
                empty_backoff_ts.pop(sym_item, None)
                if is_eq:
                    last_equity_poll_ts[sym_item] = now_done
                _mark_symbol_ws_ts(sym_item, now_done)
                push_fn = getattr(cpm, "push_price", None)
                if callable(push_fn):
                    push_fn(sym_item, pf)
            else:
                empty_backoff_ts[sym_item] = now_done

        while not stop_event.is_set():
            pending_polls: list[tuple[str, str, bool]] = []
            for s in symbols:
                if stop_event.is_set():
                    break
                ws_health_fn = getattr(cpm, "is_real_ws_healthy", None)
                if callable(ws_health_fn) and _parse_bool_flag(ws_health_fn(s), False):
                    continue
                prov_name = bindings[s.upper()]
                is_eq = is_equity_venue(prov_name) or is_equity_symbol(s)
                rth_gate_active = is_rth_gate_enabled(
                    getattr(cpm, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
                )
                now_ts = time.time()
                last_empty = empty_backoff_ts.get(s, 0.0)
                market_open = not rth_gate_active or is_symbol_market_open(s, now_ts)
                if rth_gate_active and market_open and last_empty > 0.0 and (now_ts - last_empty) < 60.0:
                    continue
                if (
                    not market_open and (s in rth_seeded or (last_empty > 0.0 and (now_ts - last_empty) < 300.0))
                ) or (is_eq and s in rth_seeded and (now_ts - last_equity_poll_ts.get(s, 0.0)) < equity_min_gap_sec):
                    _mark_symbol_ws_ts(s, now_ts)
                    continue
                pending_polls.append((s, prov_name, is_eq))

            if len(pending_polls) == 1:
                s_one, prov_one, is_eq_one = pending_polls[0]
                try:
                    p = fetch_price_with_deadline(
                        cpm.market_api,
                        s_one,
                        pool,
                        eff_deadline,
                        provider_name=prov_one,
                    )
                    _apply_fetched_price(s_one, p, is_eq=is_eq_one)
                except _futures.TimeoutError:
                    builtins.print(
                        f"[NB-trend] {s_one}: fetch BLOCKED >{eff_deadline}s "
                        f"(poll deadline -- probably DNS or the network) -- skipping, retrying next cycle"
                    )
                except Exception as _e:
                    builtins.print(f"[NB-trend] {s_one}: {_e}")
            elif len(pending_polls) > 1:
                fut_map = {}
                for s_item, prov_item, is_eq_item in pending_polls:
                    def _poll_item(_sym=s_item, _prov=prov_item):
                        with live_price_fetch_context():
                            return cpm.market_api.get_current_price(symbol=_sym, provider_name=_prov)
                    fut_map[pool.submit(_poll_item)] = (s_item, is_eq_item)
                done, not_done = _futures.wait(fut_map.keys(), timeout=eff_deadline)
                for fut in done:
                    s_done, is_eq_done = fut_map[fut]
                    try:
                        _apply_fetched_price(s_done, fut.result(), is_eq=is_eq_done)
                    except Exception as _e:
                        builtins.print(f"[NB-trend] {s_done}: {_e}")
                for fut in not_done:
                    fut.cancel()
                    s_slow, _ = fut_map[fut]
                    builtins.print(
                        f"[NB-trend] {s_slow}: fetch BLOCKED >{eff_deadline}s "
                        f"(poll deadline -- probably DNS or the network) -- skipping, retrying next cycle"
                    )
            stop_event.wait(eff_interval)

    t = threading.Thread(target=run, name="NonBinanceTrendPoller", daemon=True)
    t.start()
    return NonBinanceTrendPoller(
        t,
        pool,
        stop_event,
        ws_streams=ws_streams,
        cpm=cpm,
        delegated_symbols=symbols,
    )


_start_nonbinance_trend_poller = start_nonbinance_trend_poller
